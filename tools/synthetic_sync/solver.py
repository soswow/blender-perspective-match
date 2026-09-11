"""Translate evidence to Sync inputs; never pass the truth record to Sync."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types

import numpy as np


def load_core():
    """Allow numerical replay outside Blender without executing add-on registration."""
    if "match_perspective" not in sys.modules:
        root = Path(__file__).resolve().parents[2]
        package = types.ModuleType("match_perspective")
        package.__path__ = [str(root)]
        package.__file__ = str(root / "__init__.py")
        sys.modules["match_perspective"] = package
    from match_perspective import core
    from match_perspective.core import sync
    return core, sync


def calibration(camera):
    core, _sync = load_core()
    return core.Calibration(
        core.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"], camera["cy"], camera["width"], camera["height"]),
        np.array(camera["rotation"], dtype=float), np.array(camera["center"], dtype=float),
    )


def solver_arguments(request: dict) -> dict:
    """Convert only the request; copy arrays because the solver may modify them."""
    _core, sync = load_core()
    points = {p["id"]: p for p in request["points"]}
    arguments = dict(
        matches=[sync.SyncMatchInput(c["id"], calibration(c)) for c in request["cameras"]],
        observations=[sync.SyncObservation(**o, on_ground=points[o["landmark_id"]]["ground"],
            protect_outlier=o["weight"] > 1) for o in request["observations"]],
        known_world={p["id"]: np.array(p["known"], dtype=float) for p in points.values() if p["known"] is not None},
        line_observations=[sync.SyncLineObservation(**o) for o in request["line_observations"]],
        known_lines={line["id"]: tuple(np.array(p, dtype=float) for p in line["known"])
                     for line in request["lines"] if line["known"] is not None},
        fixed_similarities={key: sync.SimilarityTransform(float(value["scale"]), np.array(value["rotation"]),
            np.array(value["translation"])) for key, value in request["fixed_similarities"].items()},
    )
    for key in ("anchor_id", "lock_rotation", "lock_translation", "ground_slack", "known_3d_slack",
                "mirror_pairs", "mirror_plane", "mirror_slack", "parallel_pairs", "plane_groups",
                "plane_slack"):
        arguments[key] = deepcopy(request[key]) if key in request else (None if key != "plane_groups" else [])
    for key in ("mirror_pairs", "parallel_pairs"):
        arguments[key] = [tuple(pair) for pair in arguments[key]]
    if arguments["plane_groups"]:
        arguments["plane_groups"] = [tuple(item) for item in arguments["plane_groups"]]
    # Synthetic cameras default to the UI's Solve role. Forward the same explicit
    # sets as scene preparation, including its recovered-camera 3D thaw stage.
    location_ids = request.get("location_match_ids")
    arguments["location_match_ids"] = set(
        (camera["id"] for camera in request["cameras"]) if location_ids is None else location_ids
    )
    arguments["readonly_match_ids"] = set(request.get("readonly_match_ids") or [])
    return arguments


def result_record(result, cameras: list[dict]) -> dict:
    """Resolve private poses into world cameras independently of projection code."""
    recovered = {}
    for camera in cameras:
        similarity = result.similarities.get(camera["id"])
        if similarity is None:
            continue
        rotation = np.asarray(similarity.rotation)
        recovered[camera["id"]] = dict(camera,
            center=(similarity.scale * rotation @ camera["center"] + similarity.translation).tolist(),
            rotation=(np.asarray(camera["rotation"]) @ rotation.T).tolist())
    return dict(success=bool(result.success), message=result.message,
        reported_rmse_px=float(result.mean_reprojection_px), cameras=recovered,
        landmarks={key: np.asarray(value).tolist() for key, value in result.landmarks.items()},
        line_segments={key: [np.asarray(p).tolist() for p in value] for key, value in result.line_segments.items()},
        line_support_angles_deg=dict(getattr(result,"line_support_angles_deg",{})),
        weak_line_ids=list(getattr(result,"weak_line_ids",[])),
        plane_seeded_landmark_ids=list(getattr(result,"plane_seeded_landmark_ids",[])))


def fingerprint(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment(root=None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                revision=revision.stdout.strip(), working_tree_dirty=bool(dirty.stdout.strip()))


def legacy_plane_arguments(arguments, supported):
    """Omit only inactive plane defaults when replaying pre-plane solver revisions."""
    missing = {"plane_groups", "plane_slack"} - set(supported)
    if missing and (arguments.get("plane_groups") or arguments.get("plane_slack") not in (None, 0.0)):
        raise ValueError("Historical solver cannot preserve active plane settings")
    return {key: value for key, value in arguments.items() if key not in missing}, sorted(missing)


def solve(request: dict, *, use_cache=False, allow_legacy_plane_defaults=False) -> dict:
    _core, sync = load_core()
    arguments = solver_arguments(request)
    omitted = []
    if allow_legacy_plane_defaults:
        arguments, omitted = legacy_plane_arguments(arguments, inspect.signature(sync.solve_landmark_sync).parameters)
    started = time.perf_counter()
    try:
        result = sync.solve_landmark_sync(**arguments, use_pose_cache=use_cache)
    except Exception as error:
        return dict(success=False, message=str(error), exception=traceback.format_exc(),
                    reported_rmse_px=0.0, cameras={}, landmarks={}, line_segments={},
                    elapsed_s=time.perf_counter()-started, request_sha256=fingerprint(request),
                    environment=environment(), use_cache=use_cache)
    # Record the actual intrinsics used, including any solver-side repair.
    used_cameras = deepcopy(request["cameras"])
    for camera, match in zip(used_cameras, arguments["matches"]):
        camera.update(fx=match.calibration.intrinsics.fx, fy=match.calibration.intrinsics.fy)
    record = result_record(result, used_cameras)
    record.update(elapsed_s=time.perf_counter() - started, request_sha256=fingerprint(request),
                  environment=environment(), use_cache=use_cache)
    if omitted:
        record["omitted_legacy_defaults"] = omitted
    return record
