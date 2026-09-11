"""Translate evidence to Sync inputs; never pass the truth record to Sync."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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
                "mirror_pairs", "mirror_plane", "mirror_slack", "parallel_pairs"):
        arguments[key] = deepcopy(request[key])
    for key in ("mirror_pairs", "parallel_pairs"):
        arguments[key] = [tuple(pair) for pair in arguments[key]]
    if request.get("location_match_ids") is not None:
        arguments["location_match_ids"] = set(request["location_match_ids"])
    if request.get("readonly_match_ids") is not None:
        arguments["readonly_match_ids"] = set(request["readonly_match_ids"])
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
        weak_line_ids=list(getattr(result,"weak_line_ids",[])))


def fingerprint(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment() -> dict:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                revision=revision.stdout.strip(), working_tree_dirty=bool(dirty.stdout.strip()))


def solve(request: dict, *, use_cache=False) -> dict:
    _core, sync = load_core()
    arguments = solver_arguments(request)
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
    return record
