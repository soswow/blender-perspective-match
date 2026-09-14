"""Bounded Sync initializer replay and native Refine→Solve→Solve sequence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.solver import environment, load_core, result_record, solver_arguments


def _sources(out: Path) -> str:
    sources = sorted((ROOT / "core" / "sync").glob("*.py"))
    sources += sorted((ROOT / "core").glob("*.py"))
    sources += [ROOT / "scene" / "__init__.py",
                ROOT / "properties" / "__init__.py", Path(__file__)]
    sources += sorted((ROOT / "tools" / "synthetic_sync").glob("focal_*.py"))
    sources += [ROOT / "tools" / "synthetic_sync" / "blender_case.py",
                ROOT / "tools" / "synthetic_sync" / "verify_focal_constraints_blender.py",
                ROOT / "tools" / "synthetic_sync" / "budget.py",
                ROOT / "tools" / "synthetic_sync" / "solver.py",
                ROOT / "tools" / "synthetic_sync" / "evaluation.py",
                ROOT / "tools" / "synthetic_sync" / "scenarios.py"]
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(ROOT)
        payload = path.read_bytes()
        digest.update(str(relative).encode() + b"\0" + payload)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Source changed after numerical sequence began: {relative}")
        target.write_bytes(payload)
    return digest.hexdigest()


def _sequence_case(*, mirror_kind: str = "fixed") -> dict:
    """Build varied point/line/plane evidence with an optional mirror."""
    from tools.synthetic_sync import focal_constraints, focal_line_constraints, focal_lines

    case = focal_line_constraints.generate()
    if mirror_kind != "none":
        mirror = (focal_lines.generate() if mirror_kind == "live" else
                  focal_constraints.generate("mirror-hard-offcenter-exact"))
        assert [item["id"] for item in case["truth"]["cameras"]] == [
            item["id"] for item in mirror["truth"]["cameras"]
        ]
        reference_id = mirror["request"].get("mirror_landmark_id")
        mirror_ids = {
            key for key in mirror["truth"]["points"]
            if key.startswith("mirror_") or key == reference_id
        }
        assert not mirror_ids.intersection(case["truth"]["points"])
        case["truth"]["points"].update({
            key: value for key, value in mirror["truth"]["points"].items()
            if key in mirror_ids
        })
        case["truth"]["oracle_pixels"].update({
            key: value for key, value in mirror["truth"]["oracle_pixels"].items()
            if key in mirror_ids
        })
        case["request"]["points"].extend(
            item for item in mirror["request"]["points"] if item["id"] in mirror_ids
        )
        case["request"]["observations"].extend(
            item for item in mirror["request"]["observations"]
            if item["landmark_id"] in mirror_ids
        )
        case["request"]["mirror_pairs"] = deepcopy(mirror["request"]["mirror_pairs"])
        case["request"]["mirror_plane"] = deepcopy(mirror["request"]["mirror_plane"])
        case["request"]["mirror_landmark_id"] = mirror["request"].get("mirror_landmark_id")
        case["request"]["mirror_slack"] = mirror["request"]["mirror_slack"]
        case["truth"]["mirror_plane"] = deepcopy(mirror["truth"]["mirror_plane"])
        if mirror_kind == "fixed":
            # An off-center fixed mirror supplies metric scale; a live point
            # reference moves with the free reconstruction.
            case["expectation"]["scale_gauge"] = mirror["expectation"]["scale_gauge"]
        if mirror_kind == "live":
            assert not set(mirror["truth"]["lines"]).intersection(case["truth"]["lines"])
            case["truth"]["lines"].update(deepcopy(mirror["truth"]["lines"]))
            case["truth"]["line_oracle_pixels"].update(
                deepcopy(mirror["truth"]["line_oracle_pixels"]))
            case["request"]["lines"].extend(deepcopy(mirror["request"]["lines"]))
            case["request"]["line_observations"].extend(
                deepcopy(mirror["request"]["line_observations"]))
    rng = __import__("numpy").random.default_rng(20260914)
    for item in case["request"]["observations"]:
        item["u"] += float(rng.normal(0.0, 0.18))
        item["v"] += float(rng.normal(0.0, 0.18))
    for item in case["request"]["line_observations"]:
        for key in ("u1", "v1", "u2", "v2"):
            item[key] += float(rng.normal(0.0, 0.18))
    for camera in case["request"]["cameras"]:
        camera["fx"] *= 1.04
        camera["fy"] *= 1.04
    case["name"] = f"sync-continuation-{mirror_kind}-mirror"
    _validate_sequence_case(case)
    return case


def _validate_sequence_case(case: dict) -> None:
    """Check the composite fixture against its independent projection oracle."""
    import numpy as np
    from tools.synthetic_sync.geometry import project

    request, truth = case["request"], case["truth"]
    if case["expectation"]["scale_gauge"] != _fixture_scale_gauge(case):
        raise ValueError("Fixture scale gauge contradicts its world constraints")
    cameras = {item["id"]: item for item in truth["cameras"]}
    points = {key: np.asarray(value, float) for key, value in truth["points"].items()}
    lines = {key: np.asarray(value, float) for key, value in truth["lines"].items()}
    if (set(points) & set(truth["holdouts"]) or set(points) & set(lines)
            or set(lines) & set(truth["holdouts"])):
        raise ValueError("Training points, lines and holdouts must be disjoint")
    if set(cameras) != {item["id"] for item in request["cameras"]}:
        raise ValueError("Stored and oracle cameras differ")
    for key, views in truth["oracle_pixels"].items():
        if key not in points or set(views) != set(cameras):
            raise ValueError("Training point oracle has incomplete support")
        for camera_id, expected in views.items():
            actual, depth = project([points[key]], cameras[camera_id])
            if depth[0] <= 0 or not np.allclose(actual[0], expected, atol=1e-9, rtol=0):
                raise ValueError("Training point oracle is inconsistent")
    for key, position in truth["holdouts"].items():
        for camera_id, expected in truth["holdout_pixels"][key].items():
            actual, depth = project([position], cameras[camera_id])
            if depth[0] <= 0 or not np.allclose(actual[0], expected, atol=1e-9, rtol=0):
                raise ValueError("Withheld point oracle is inconsistent")
    for item in request["observations"]:
        target = truth["oracle_pixels"][item["landmark_id"]][item["match_id"]]
        if (np.linalg.norm(np.asarray((item["u"], item["v"])) - target) >
                5.0 * case["pick_sigma_px"]):
            raise ValueError("A point pick exceeds the declared fixture noise")
    for key, views in truth["line_oracle_pixels"].items():
        if key not in lines or set(views) != set(cameras):
            raise ValueError("Line oracle has incomplete support")
        for camera_id, expected in views.items():
            observed = np.asarray(expected, float)
            if observed.shape != (2, 2):
                raise ValueError("Line oracle has invalid stroke endpoints")
            camera = cameras[camera_id]
            # Different views intentionally mark different physical intervals.
            # Their pixels must still lie on the projected infinite truth line.
            exact, depth = project(lines[key], camera)
            if min(depth) <= 0:
                raise ValueError("A truth line is behind a camera")
            direction = exact[1] - exact[0]
            normal = np.array((-direction[1], direction[0])) / np.linalg.norm(direction)
            if np.max(np.abs((observed - exact[0]) @ normal)) > 1e-8:
                raise ValueError("Line stroke oracle is inconsistent")
    for item in request["line_observations"]:
        target = truth["line_oracle_pixels"][item["landmark_id"]][item["match_id"]]
        observed = np.asarray(((item["u1"], item["v1"]),
                               (item["u2"], item["v2"])))
        if (np.max(np.linalg.norm(observed - target, axis=1)) >
                5.0 * case["pick_sigma_px"]):
            raise ValueError("A line stroke exceeds the declared fixture noise")
    relations = _relation_geometry(case, dict(
        landmarks=truth["points"], line_segments=truth["lines"]))
    for bucket in relations["plane_buckets"].values():
        if (bucket["point_rms_world"] > 1e-9 or
                bucket["line_max_distance_world"] > 1e-9 or
                bucket["line_max_direction_sine"] > 1e-9):
            raise ValueError("Truth violates one of its separate plane buckets")
    if (max(relations["parallel_direction_sine"].values(), default=0.0) > 1e-9 or
            relations["mirror_max_gap_world"] > 1e-9 or
            relations["mirror_line_max_gap_world"] > 1e-9 or
            relations["mirror_line_max_direction_sine"] > 1e-9):
        raise ValueError("Truth violates a parallel or mirror relation")


def _fixture_scale_gauge(case: dict) -> str:
    """A hard fixed mirror off the anchor supplies metric scale."""
    import numpy as np

    request = case["request"]
    plane = request.get("mirror_plane")
    if (plane is None or not request.get("mirror_pairs") or
            request.get("mirror_landmark_id") is not None or
            request.get("mirror_slack", 0.0) > 0):
        return "free"
    center = next(camera["center"] for camera in case["truth"]["cameras"]
                  if camera["id"] == request["anchor_id"])
    normal = np.asarray(plane[1], float)
    offset = abs(float(normal @ (np.asarray(plane[0], float) - center)))
    return "anchor-plane-offset" if offset > 1e-8 else "free"


def _relation_geometry(case: dict, record: dict) -> dict:
    """Measure each relation in its own bucket and infinite-line geometry."""
    import numpy as np

    landmarks = {key: np.asarray(value, float)
                 for key, value in record["landmarks"].items()}
    segments = {key: np.asarray(value, float)
                for key, value in record["line_segments"].items()}
    point_ids = {item["id"] for item in case["request"]["points"]}
    groups = {}
    for landmark_id, axis, bucket in case["request"]["plane_groups"]:
        groups.setdefault((axis, int(bucket)), []).append(landmark_id)
    buckets = {}
    all_point_distances = []
    for (axis, bucket), members in sorted(groups.items()):
        positions = np.asarray([landmarks[key] for key in members
                                if key in point_ids], float)
        if axis == "FREE":
            origin = positions.mean(axis=0)
            _u, _s, vh = np.linalg.svd(positions - origin, full_matrices=False)
            normal = vh[-1]
        else:
            normal = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[axis]]
            origin = positions.mean(axis=0)
        distances = (positions - origin) @ normal
        all_point_distances.extend(distances.tolist())
        line_distance = []
        line_directions = []
        for key in members:
            if key not in segments:
                continue
            ends = segments[key]
            line_distance.append(abs(float((ends.mean(axis=0) - origin) @ normal)))
            direction = ends[1] - ends[0]
            line_directions.append(abs(float(direction @ normal)) /
                                   float(np.linalg.norm(direction)))
        buckets[f"{axis}:{bucket}"] = dict(
            point_count=len(positions), line_count=len(line_directions),
            point_rms_world=float(np.sqrt(np.mean(np.square(distances)))),
            line_max_distance_world=max(line_distance, default=0.0),
            line_max_direction_sine=max(line_directions, default=0.0))

    parallel_gaps = {}
    axis_directions = {"WORLD_AXIS_X": np.array((1., 0., 0.)),
                       "WORLD_AXIS_Y": np.array((0., 1., 0.)),
                       "WORLD_AXIS_Z": np.array((0., 0., 1.))}
    for left, right in case["request"]["parallel_pairs"]:
        first = segments[left][1] - segments[left][0]
        second = (axis_directions[right] if right in axis_directions
                  else segments[right][1] - segments[right][0])
        parallel_gaps[f"{left}:{right}"] = float(np.linalg.norm(np.cross(first, second)) /
                                                   (np.linalg.norm(first) * np.linalg.norm(second)))

    mirror_point_gaps = []
    mirror_line_gaps = []
    mirror_line_directions = []
    if case["request"]["mirror_pairs"]:
        origin, normal = (np.asarray(value, float) for value in
                          case["request"]["mirror_plane"])
        normal /= np.linalg.norm(normal)
        reference = case["request"].get("mirror_landmark_id")
        if reference:
            origin = landmarks[reference]
        def reflect(value):
            return value - 2.0 * float((value - origin) @ normal) * normal
        for left, right in case["request"]["mirror_pairs"]:
            if left in point_ids and right in point_ids:
                mirror_point_gaps.append(float(np.linalg.norm(
                    reflect(landmarks[left]) - landmarks[right])))
            else:
                first, second = segments[left], segments[right]
                first_midpoint = reflect(first.mean(axis=0))
                first_direction = reflect(first[1]) - reflect(first[0])
                second_midpoint = second.mean(axis=0)
                second_direction = second[1] - second[0]
                second_direction /= np.linalg.norm(second_direction)
                mirror_line_gaps.append(float(np.linalg.norm(np.cross(
                    first_midpoint - second_midpoint, second_direction))))
                mirror_line_directions.append(float(np.linalg.norm(np.cross(
                    first_direction / np.linalg.norm(first_direction),
                    second_direction))))
    return dict(
        plane_buckets=buckets,
        plane_rms_world=(float(np.sqrt(np.mean(np.square(all_point_distances))))
                         if all_point_distances else None),
        parallel_direction_sine=parallel_gaps,
        mirror_max_gap_world=max(mirror_point_gaps, default=0.0),
        mirror_line_max_gap_world=max(mirror_line_gaps, default=0.0),
        mirror_line_max_direction_sine=max(mirror_line_directions, default=0.0),
    )


def _assess_sequence_case(case: dict, record: dict) -> dict:
    """Assess disjoint truth and each separate world relation bucket."""
    import numpy as np
    from tools.synthetic_sync.focal_constraints import assess

    point_ids = {item["id"] for item in case["request"]["points"]}
    point_case = deepcopy(case)
    point_case["request"]["plane_groups"] = []
    point_case["request"]["mirror_pairs"] = [
        pair for pair in case["request"]["mirror_pairs"]
        if pair[0] in point_ids and pair[1] in point_ids]
    reference = case["request"].get("mirror_landmark_id")
    if reference:
        point_case["request"]["mirror_plane"][0] = record["landmarks"][reference]
    assessment = assess(point_case, record, validated=True)
    segments = {key: np.asarray(value, float)
                for key, value in record["line_segments"].items()}
    assessment.update(_relation_geometry(case, record))

    anchor_id = case["request"]["anchor_id"]
    fitted_anchor = np.asarray(record["cameras"][anchor_id]["center"], float)
    true_cameras = {item["id"]: item for item in case["truth"]["cameras"]}
    true_anchor = np.asarray(true_cameras[anchor_id]["center"], float)
    factor = assessment["alignment_scale"]
    line_positions = []
    line_angles = []
    for key, truth_ends in case["truth"]["lines"].items():
        fitted = np.asarray(segments[key], float)
        if case["expectation"]["scale_gauge"] == "free":
            fitted = true_anchor + (fitted - fitted_anchor) / factor
        wanted = np.asarray(truth_ends, float)
        wanted_direction = wanted[1] - wanted[0]
        wanted_direction /= np.linalg.norm(wanted_direction)
        fitted_direction = fitted[1] - fitted[0]
        fitted_direction /= np.linalg.norm(fitted_direction)
        line_positions.append(float(np.linalg.norm(np.cross(
            fitted.mean(axis=0) - wanted[0], wanted_direction))))
        line_angles.append(float(np.degrees(np.arccos(np.clip(
            abs(float(fitted_direction @ wanted_direction)), 0.0, 1.0)))))
    assessment["line_position_max_world"] = max(line_positions, default=0.0)
    assessment["line_direction_max_deg"] = max(line_angles, default=0.0)
    assessment["accuracy_flags"] = [
        flag for flag, failed in (
            ("Focal error exceeds 2%", max(abs(value) for value in
                                           assessment["focal_relative"].values()) > 0.02),
            ("Withheld point RMS exceeds 1px", assessment["withheld_rmse_px"] > 1.0),
        ) if failed]
    relation_flags = []
    for label, values in assessment["plane_buckets"].items():
        if values["point_rms_world"] > 0.001:
            relation_flags.append(f"{label}: point plane RMS exceeds 0.001 world units")
        if values["line_max_distance_world"] > 0.01:
            relation_flags.append(f"{label}: line plane gap exceeds 0.01 world units")
        if values["line_max_direction_sine"] > 0.01:
            relation_flags.append(f"{label}: line plane direction sine exceeds 0.01")
    for label, value in assessment["parallel_direction_sine"].items():
        if value > 0.01:
            relation_flags.append(f"{label}: parallel direction sine exceeds 0.01")
    for field in ("mirror_max_gap_world", "mirror_line_max_gap_world"):
        if assessment[field] > 0.01:
            relation_flags.append(f"{field} exceeds 0.01 world units")
    if assessment["mirror_line_max_direction_sine"] > 0.01:
        relation_flags.append("Mirror line direction sine exceeds 0.01")
    assessment["relation_flags"] = relation_flags
    assessment["accuracy_passed"] = not assessment["accuracy_flags"]
    assessment["relation_passed"] = not relation_flags
    return assessment


def _bundle_record(case: dict, outcome: dict) -> dict:
    """Resolve a direct bundle's own private calibrations and root transforms."""
    import numpy as np

    result = outcome["sync_result"]
    cameras = {}
    for truth in case["truth"]["cameras"]:
        key = truth["id"]
        calibration = outcome["calibrations"][key]
        similarity = result["similarities"][key]
        intrinsics = calibration["intrinsics"]
        root_rotation = np.asarray(similarity["rotation"], float)
        private_center = np.asarray(calibration["camera_center"], float)
        cameras[key] = dict(
            truth, fx=intrinsics["fx"], fy=intrinsics["fy"],
            cx=intrinsics["cx"], cy=intrinsics["cy"],
            center=(similarity["scale"] * root_rotation @ private_center +
                    similarity["translation"]).tolist(),
            rotation=(np.asarray(calibration["rotation_w2c"], float) @
                      root_rotation.T).tolist(),
        )
    return dict(success=result["success"], message=result["message"],
                reported_rmse_px=result["mean_reprojection_px"],
                cameras=cameras, landmarks=result["landmarks"],
                line_segments=result["line_segments"])


def _reassess_bundle_archive(out: Path) -> None:
    """Reassess one saved direct bundle without another numerical call."""
    case = json.loads((out / "joint-case.json").read_text())
    archived_scale_gauge = case["expectation"]["scale_gauge"]
    case["expectation"]["scale_gauge"] = _fixture_scale_gauge(case)
    _validate_sequence_case(case)
    rows = [json.loads(line) for line in (out / "joint-ledger.jsonl").read_text().splitlines()]
    started = {row["attempt"] for row in rows
               if row["kind"] == "started" and row["label"].startswith("bundle:")}
    completed = [row for row in rows
                 if row["kind"] == "completed" and row["attempt"] in started]
    if len(completed) != 1:
        raise ValueError("Expected exactly one completed direct bundle in this archive")
    outcome = completed[0]["result"]
    record = _bundle_record(case, outcome)
    assessment = _assess_sequence_case(case, record)
    report = dict(
        source_sha256=rows[0]["metadata"]["source_sha256"],
        assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archived_scale_gauge=archived_scale_gauge,
        assessed_scale_gauge=case["expectation"]["scale_gauge"],
        attempt=completed[0]["attempt"],
        accepted_by_bundle=outcome["accepted"],
        candidate_from_bundle=outcome["candidate"] is not None,
        recovered=record, assessment=assessment,
    )
    (out / "bundle-reassessment.json").write_text(json.dumps(
        report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(
        accepted_by_bundle=report["accepted_by_bundle"],
        withheld_rmse_px=assessment["withheld_rmse_px"],
        accuracy_flags=assessment["accuracy_flags"],
        relation_flags=assessment["relation_flags"],
    ), indent=2))


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


@contextmanager
def _count_inner_calls(budget: ExperimentBudget, stage_label: list[str]):
    """Reserve each Sync and common bundle call before it enters native code."""
    from match_perspective.core import focal_bundle, lens_refine, sync
    from match_perspective.core.sync.request import json_values

    original_sync = sync.solve_landmark_sync
    original_bundle = focal_bundle.fit_independent_focals

    def counted(kind, original, args, kwargs):
        count = sum(row.get("kind") == "started" and
                    row.get("label", "").startswith(kind + ":")
                    for row in budget.rows)
        metadata = budget.header["metadata"]
        limit = int(metadata["max_sync_calls" if kind == "sync" else "max_bundle_calls"])
        if count >= limit:
            raise RuntimeError(f"{kind} numerical call budget exhausted")
        values = dict(args=args, kwargs={
            key: value for key, value in kwargs.items() if not callable(value)
        }, callbacks={
            key: True for key, value in kwargs.items() if callable(value)
        })
        exact = _finite_json(json_values(values))
        label = f"{kind}:{stage_label[0]}:{count + 1}"
        with budget.attempt(label, exact) as attempt:
            outcome = original(*args, **kwargs)
            attempt.complete(_finite_json(json_values(outcome)))
        return outcome

    def counted_sync(*args, **kwargs):
        return counted("sync", original_sync, args, kwargs)

    def counted_bundle(*args, **kwargs):
        return counted("bundle", original_bundle, args, kwargs)

    with patch.object(sync, "solve_landmark_sync", counted_sync), \
         patch.object(lens_refine, "fit_independent_focals", counted_bundle), \
         patch.object(focal_bundle, "fit_independent_focals", counted_bundle):
        yield


def _execute_current_stage(stage: str, out: Path):
    """Run one real product route on the currently open scene."""
    import bpy
    from match_perspective import scene
    from match_perspective.core.sync.request import json_values

    if stage == "refine":
        prep = scene.prepare_lens_refine(bpy.context)
        (out / "refine-inputs.json").write_text(json.dumps(
            _finite_json(json_values(prep.solver_kwargs())),
            indent=2, allow_nan=False) + "\n")
        refined = scene.run_lens_refine(prep)
        if refined.improved and refined.sync_result is not None:
            _message, applied = scene.apply_lens_refine_result(
                bpy.context, refined, prep)
            if applied is None or not applied.success:
                raise RuntimeError("Refine did not apply a successful Sync result")
            return applied
        (out / "refine-refusal.json").write_text(json.dumps(
            _finite_json(json_values(refined)), indent=2) + "\n")
        raise RuntimeError(f"Refine declined: {refined.refusal_reason}")
    prep = scene.prepare_diagnose_sync(bpy.context)
    result = scene.run_solve_sync(prep)
    return scene.apply_solve_sync_result(bpy.context, prep, result)


def _stage_directory(out: Path, stage: str) -> None:
    """Allow Refine after read-only preparation but never replay a started job."""
    if stage == "prepare":
        out.mkdir(parents=True, exist_ok=False)
    elif stage == "refine":
        if not out.exists():
            out.mkdir(parents=True)
        elif ((out / "joint-ledger.jsonl").exists() or
              (out / "refine-request.json").exists()):
            raise ValueError("Refine already started in this archive")
    elif not out.is_dir():
        raise ValueError("Run the Refine stage first")


def _run_native_stage(
    out: Path, stage: str, *, case_name: str,
    max_sync_calls: int | None, max_bundle_calls: int | None,
    active_seconds: float | None,
) -> None:
    import bpy
    import numpy as np
    from tools.synthetic_sync.blender_case import register_extension
    from tools.synthetic_sync.verify_focal_constraints_blender import build

    _stage_directory(out, stage)
    register_extension()
    from match_perspective import properties, scene
    from match_perspective.core import focal_bundle, lens_refine, sync
    from match_perspective.core.sync.request import json_values
    mirror_kind = {"joint-mirror": "fixed", "free-gauge": "none",
                   "live-mirror": "live"}[case_name]
    case = _sequence_case(mirror_kind=mirror_kind)
    case_file = out / "joint-case.json"
    exact_case = json.dumps(case, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if case_file.exists() and case_file.read_text() != exact_case:
        raise ValueError("Generated numerical evidence changed")
    case_file.write_text(exact_case)
    source_sha256 = _sources(out)
    if stage in {"prepare", "refine"}:
        build(case, out)
        if case_name == "live-mirror":
            space = properties.workspace(bpy.context)
            space.mirror_origin = "LANDMARK"
            space.mirror_landmark = case["request"]["mirror_landmark_id"]
    else:
        prior = "refine" if stage == "solve1" else "solve1"
        bpy.ops.wm.open_mainfile(filepath=str(out / f"{prior}.blend"))

    if stage == "prepare":
        lens = scene.prepare_lens_refine(bpy.context)
        request = scene.collect_sync_request(bpy.context)
        prepared = dict(
            case=case_name, point_picks=len(request.observations),
            line_strokes=len(request.line_observations or ()),
            mirror_pairs=request.mirror_pairs,
            mirror_landmark_id=request.mirror_landmark_id,
            plane_groups=request.plane_groups,
            parallel_pairs=request.parallel_pairs,
            independent_fov=lens.estimate_focal_from_points and not lens.share_lens,
        )
        (out / "prepared.json").write_text(json.dumps(
            prepared, indent=2, allow_nan=False) + "\n")
        print(json.dumps(prepared, indent=2))
        return

    if (max_sync_calls is None or max_bundle_calls is None or
            active_seconds is None or max_sync_calls < 0 or
            max_bundle_calls < 0 or max_sync_calls + max_bundle_calls == 0 or
            not math.isfinite(active_seconds) or active_seconds <= 0):
        raise ValueError("Numerical stages require explicit remaining call and active-time caps")

    request_before = scene.collect_sync_request(bpy.context)
    (out / f"{stage}-request.json").write_text(
        json.dumps(request_before.to_record(), indent=2, allow_nan=False) + "\n")
    metadata = dict(case=case_name, source_sha256=source_sha256,
                    max_sync_calls=max_sync_calls, max_bundle_calls=max_bundle_calls,
                    stages=["refine", "solve1", "solve2"],
                    environment=environment(ROOT))
    with ExperimentBudget(
        out / "joint-ledger.jsonl", metadata=metadata,
        max_calls=max_sync_calls + max_bundle_calls,
        wall_seconds=active_seconds, per_call_seconds=180,
    ) as budget:
        with _count_inner_calls(budget, [stage]):
            result = _execute_current_stage(stage, out)

    post_request = scene.collect_sync_request(bpy.context)
    applied_calibrations = (
        post_request.initial_solution.calibrations
        if post_request.initial_solution is not None else result.calibrations
    )
    cameras = case["truth"]["cameras"]
    record = result_record(result, cameras, calibrations=applied_calibrations or None)
    assessment, assessment_error = None, None
    try:
        assessment = _assess_sequence_case(case, record)
    except (KeyError, ValueError, AssertionError, np.linalg.LinAlgError) as error:
        assessment_error = f"{type(error).__name__}: {error}"
    report = dict(stage=stage, source_sha256=source_sha256,
                  result=_finite_json(json_values(result)),
                  post_request=post_request.to_record(),
                  recovered=record, assessment=assessment,
                  assessment_error=assessment_error,
                  post_evidence_sha256=post_request.evidence_sha256(),
                  post_seed_certified=bool(
                      post_request.initial_solution is not None and
                      post_request.initial_solution.evidence_sha256 ==
                      post_request.evidence_sha256()),
                  applied_status=properties.workspace(bpy.context).sync_status)
    (out / f"{stage}-result.json").write_text(json.dumps(
        report, indent=2, allow_nan=False) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{stage}.blend"))
    print(json.dumps(dict(stage=stage, success=result.success,
                          reported_rmse_px=result.mean_reprojection_px,
                          joint_objective=result.joint_final_objective,
                          assessment=assessment,
                          assessment_error=assessment_error,
                          post_seed_certified=report["post_seed_certified"]), indent=2))


def _compare_private_endpoints(out: Path) -> dict:
    """Score captured applied endpoints with the first endpoint's frozen weights."""
    import numpy as np
    from match_perspective.core.joint_fit_score import (
        JointFitScorer, supported_joint_request,
    )
    from match_perspective.core.focal_projection import project_stroke_line
    from match_perspective.core.sync.projection import project_private_point
    from match_perspective.core.sync.request import SyncSolveRequest, json_values
    from match_perspective.core.sync.solve import solution_result_from_seed

    names = ("refine", "solve1", "solve2")
    reports = [json.loads((out / f"{name}-result.json").read_text())
               for name in names]
    requests = [SyncSolveRequest.from_record(row["post_request"])
                for row in reports]
    seeds = [request.initial_solution for request in requests]
    if any(seed is None or seed.diagnostics is None or
           seed.evidence_sha256 != request.evidence_sha256()
           for request, seed in zip(requests, seeds)):
        raise AssertionError("An applied endpoint has no certified solution seed")
    frozen_weights = seeds[0].diagnostics.joint_point_weights
    if frozen_weights is None and requests[0].observations:
        raise AssertionError("Refine has no persisted effective point weights")
    def weight_map(seed):
        return {(camera, point): weight for camera, point, weight in
                seed.diagnostics.joint_point_weights or ()}
    if any(weight_map(seed) != weight_map(seeds[0]) for seed in seeds[1:]):
        raise AssertionError("Effective point weights changed between applied endpoints")
    endpoints = [solution_result_from_seed(seed) for seed in seeds]

    def structural_evidence(request):
        values = request._inputs(include_seed=False)
        for match in values["matches"]:
            match.pop("calibration")
        return values

    first_structure = structural_evidence(requests[0])
    if any(structural_evidence(request) != first_structure
           for request in requests[1:]):
        raise AssertionError("Saved-project inputs changed between fitted endpoints")

    float32_epsilon = float(np.finfo(np.float32).eps)
    consistency = []
    for name, report, seed in zip(names, reports, seeds):
        numerical = report["result"]
        largest_ratio = 0.0

        def check(label, actual, applied):
            nonlocal largest_ratio
            left = np.asarray(actual, dtype=np.float64)
            right = np.asarray(applied, dtype=np.float64)
            if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise AssertionError(f"{name}: invalid applied {label}")
            scale = max(float(np.max(np.abs(left))) if left.size else 0.0,
                        float(np.max(np.abs(right))) if right.size else 0.0, 1.0)
            tolerance = 8.0 * float32_epsilon * scale
            difference = float(np.max(np.abs(left - right))) if left.size else 0.0
            largest_ratio = max(largest_ratio, difference / tolerance)
            if difference > tolerance:
                raise AssertionError(
                    f"{name}: numerical {label} differs from applied state by "
                    f"{difference:g} (> {tolerance:g})")

        if set(numerical["similarities"]) != set(seed.similarities):
            raise AssertionError(f"{name}: camera support changed during apply")
        if set(numerical["landmarks"]) != set(seed.landmarks):
            raise AssertionError(f"{name}: point support changed during apply")
        if set(numerical["line_segments"]) != set(seed.line_segments):
            raise AssertionError(f"{name}: line support changed during apply")
        for key, applied in seed.similarities.items():
            item = numerical["similarities"][key]
            check(f"camera {key} scale", item["scale"], applied.scale)
            check(f"camera {key} root rotation", item["rotation"], applied.rotation)
            check(f"camera {key} root translation", item["translation"],
                  applied.translation)
        for key, applied in seed.calibrations.items():
            item = numerical["calibrations"].get(key)
            if item is None:
                raise AssertionError(f"{name}: camera {key} calibration missing from result")
            intrinsics = item["intrinsics"]
            check(f"camera {key} intrinsics",
                  [intrinsics[field] for field in ("fx", "fy", "cx", "cy")],
                  [getattr(applied.intrinsics, field) for field in
                   ("fx", "fy", "cx", "cy")])
            check(f"camera {key} private rotation",
                  item["rotation_w2c"], applied.rotation_w2c)
            check(f"camera {key} private center",
                  item["camera_center"], applied.camera_center)
            check(f"camera {key} division distortion",
                  item["division_lambda"], applied.division_lambda)
            check(f"camera {key} Brown distortion",
                  item["brown_conrady"], applied.brown_conrady)
            if (bool(item["lambda_saturated"]) != bool(applied.lambda_saturated)
                    or int(intrinsics["image_width"]) != applied.intrinsics.image_width
                    or int(intrinsics["image_height"]) != applied.intrinsics.image_height):
                raise AssertionError(f"{name}: camera {key} calibration flags changed")
        for key, applied in seed.landmarks.items():
            check(f"landmark {key}", numerical["landmarks"][key], applied)
        for key, applied in seed.line_segments.items():
            check(f"line {key}", numerical["line_segments"][key], applied)
        check("headline RMSE", numerical["mean_reprojection_px"],
              seed.diagnostics.mean_reprojection_px)
        for field in ("point_rmse_px", "line_rmse_px",
                      "joint_initial_objective", "joint_final_objective",
                      "joint_mirror_offset_m"):
            value = numerical.get(field)
            applied = getattr(seed.diagnostics, field)
            if value is None or applied is None:
                if value != applied:
                    raise AssertionError(f"{name}: applied {field} changed")
            else:
                check(field, value, applied)
        consistency.append(dict(stage=name, max_float32_tolerance_fraction=largest_ratio))

    supported, baseline_coverage = supported_joint_request(requests[0], endpoints[0])
    scorer = JointFitScorer(supported, endpoints[0],
                            calibrations=seeds[0].calibrations,
                            frozen_point_weights=frozen_weights)
    scored = []
    for name, endpoint, seed in zip(names, endpoints, seeds):
        score = scorer.score(endpoint, calibrations=seed.calibrations)
        _current, coverage = supported_joint_request(requests[0], endpoint)
        scored.append(dict(
            stage=name, valid=score.valid, reason=score.reason,
            objective=score.objective,
            point_rmse_px=score.point_rmse_px,
            line_rmse_px=score.line_rmse_px,
            supported_observations=score.supported_observations,
            coverage=coverage,
        ))
    if not all(item["valid"] for item in scored):
        raise AssertionError("An applied endpoint cannot score on first-stage evidence")
    if len({item["supported_observations"] for item in scored}) != 1:
        raise AssertionError("Applied endpoints lost common point or line support")

    def projections(result, calibrations):
        points = {}
        for item in supported.observations:
            point = result.landmarks[item.landmark_id]
            similarity = result.similarities[item.match_id]
            image = project_private_point(
                similarity.inverse_point(point), calibrations[item.match_id])
            if image is None:
                raise AssertionError("An applied point cannot project")
            points[(item.match_id, item.landmark_id)] = np.asarray(image, float)
        lines = {}
        for item in supported.line_observations or ():
            start, end = result.line_segments[item.landmark_id]
            calibration = calibrations[item.match_id]
            similarity = result.similarities[item.match_id]
            ends = np.asarray(((item.u1, item.v1), (item.u2, item.v2)), float)
            line = project_stroke_line(
                0.5 * (start + end), end - start,
                calibration.rotation_w2c @ similarity.rotation.T,
                similarity.transform_point(calibration.camera_center),
                calibration.intrinsics, ends,
                division_lambda=calibration.division_lambda,
                brown_conrady=calibration.brown_conrady)
            if line is None:
                raise AssertionError("An applied line cannot project")
            lines[(item.match_id, item.landmark_id)] = ends @ line[:2] + line[2]
        return points, lines

    projected = [projections(result, seed.calibrations)
                 for result, seed in zip(endpoints, seeds)]
    movements = []
    for left, right in ((0, 1), (1, 2), (0, 2)):
        point_delta = [float(np.linalg.norm(projected[left][0][key] -
                                            projected[right][0][key]))
                       for key in projected[left][0]]
        line_delta = []
        for key, before in projected[left][1].items():
            after = projected[right][1][key]
            delta = (np.abs(before - after) if
                     np.linalg.norm(before - after) <= np.linalg.norm(before + after)
                     else np.abs(before + after))
            line_delta.extend(float(value) for value in delta)
        movements.append(dict(
            from_stage=names[left], to_stage=names[right],
            point_projection_max_px=max(point_delta, default=0.0),
            point_projection_median_px=float(np.median(point_delta)) if point_delta else 0.0,
            line_overlay_max_px=max(line_delta, default=0.0),
            line_overlay_median_px=float(np.median(line_delta)) if line_delta else 0.0,
        ))
    summary = dict(
        scorer_basis="first applied Refine endpoint; frozen point weights and same evidence",
        effective_point_weights_preserved=True,
        objective_nonincreasing=all(
            right["objective"] <= left["objective"] +
            1e-9 * max(left["objective"], 1.0)
            for left, right in zip(scored, scored[1:])),
        baseline_coverage=baseline_coverage,
        stages=scored, movements=movements,
        numerical_vs_applied=consistency,
    )
    (out / "same-evidence-comparison.json").write_text(json.dumps(
        _finite_json(json_values(summary)), indent=2, allow_nan=False) + "\n")
    return summary


def _run_private_sequence(
    out: Path, blend: Path, *, max_sync_calls: int,
    max_bundle_calls: int, active_seconds: float,
) -> None:
    """Replay one saved project in memory and leave its source bytes unchanged."""
    import bpy
    from tools.synthetic_sync.blender_case import register_extension

    source = blend.resolve(strict=True)
    if not source.is_file():
        raise ValueError("The supplied Blender project is not a file")
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Private numerical archives must stay outside the repository")
    out.mkdir(parents=True, exist_ok=False)
    before_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    source_sha256 = _sources(out)
    register_extension()
    from match_perspective import properties, scene
    from match_perspective.core.sync.request import json_values

    metadata = dict(
        case="read-only-saved-project", source_sha256=source_sha256,
        blend_sha256=before_sha256,
        max_sync_calls=max_sync_calls, max_bundle_calls=max_bundle_calls,
        stages=["refine", "solve1", "solve2"],
        blender_version=bpy.app.version_string,
        blender_build_hash=str(bpy.app.build_hash),
        environment=environment(ROOT),
    )
    try:
        bpy.ops.wm.open_mainfile(filepath=str(source))
        stage_label = ["refine"]
        with ExperimentBudget(
            out / "private-ledger.jsonl", metadata=metadata,
            max_calls=max_sync_calls + max_bundle_calls,
            wall_seconds=active_seconds, per_call_seconds=180,
        ) as budget, _count_inner_calls(budget, stage_label):
            for stage in ("refine", "solve1", "solve2"):
                stage_label[0] = stage
                request_before = scene.collect_sync_request(bpy.context)
                (out / f"{stage}-request.json").write_text(json.dumps(
                    request_before.to_record(), indent=2, allow_nan=False) + "\n")
                result = _execute_current_stage(stage, out)
                post_request = scene.collect_sync_request(bpy.context)
                report = dict(
                    stage=stage,
                    result=_finite_json(json_values(result)),
                    post_request=post_request.to_record(),
                    post_seed_certified=bool(
                        post_request.initial_solution is not None and
                        post_request.initial_solution.evidence_sha256 ==
                        post_request.evidence_sha256()),
                    applied_status=properties.workspace(bpy.context).sync_status,
                )
                (out / f"{stage}-result.json").write_text(json.dumps(
                    report, indent=2, allow_nan=False) + "\n")
                print(json.dumps(dict(
                    stage=stage, success=result.success,
                    reported_rmse_px=result.mean_reprojection_px,
                    joint_objective=result.joint_final_objective,
                    post_seed_certified=report["post_seed_certified"],
                ), indent=2))
        comparison = _compare_private_endpoints(out)
        print(json.dumps(dict(
            same_evidence_objectives=[row["objective"] for row in comparison["stages"]],
            projection_movements=comparison["movements"],
            numerical_vs_applied=comparison["numerical_vs_applied"],
        ), indent=2))
    finally:
        after_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        (out / "source-file-check.json").write_text(json.dumps(dict(
            before_sha256=before_sha256, after_sha256=after_sha256,
            unchanged=before_sha256 == after_sha256,
        ), indent=2) + "\n")
        if before_sha256 != after_sha256:
            raise AssertionError("The source Blender project changed during replay")


def _run_outer_blender(out: Path, *, outer_seconds: float) -> None:
    """Enforce a process deadline even when a native call delays Python signals."""
    blender = os.environ.get(
        "BLENDER_BIN", "/Applications/Blender 5.1.app/Contents/MacOS/blender")
    command = [
        blender, "--factory-startup", "--disable-autoexec", "-b",
        "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
        "--", *sys.argv[1:],
    ]
    status = "failed"
    exit_code = None
    elapsed_s = None
    import time
    started = time.monotonic()
    output = ""
    try:
        child_environment = dict(os.environ)
        child_environment["PM_SYNC_CONTINUATION_OUTER"] = "1"
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=outer_seconds, text=True, check=False,
            env=child_environment,
        )
        elapsed_s = time.monotonic() - started
        exit_code = process.returncode
        status = "completed" if exit_code == 0 else "failed"
        output = process.stdout
        print(output, end="")
    except subprocess.TimeoutExpired as error:
        elapsed_s = time.monotonic() - started
        status = "timed_out"
        captured = error.stdout or b""
        output = (captured.decode(errors="replace") if isinstance(captured, bytes)
                  else captured)
        print(output, end="")
    finally:
        out.mkdir(parents=True, exist_ok=True)
        (out / "process-output.log").write_text(output)
        (out / "process-exit.json").write_text(json.dumps(dict(
            status=status, exit_code=exit_code, elapsed_s=elapsed_s,
            outer_seconds=outer_seconds,
            autoexec_disabled=True, saved_blend=False if "--blend" in sys.argv else None,
        ), indent=2) + "\n")
    if status == "timed_out":
        raise SystemExit(124)
    if exit_code != 0:
        raise SystemExit(exit_code or 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "refine", "solve1", "solve2"))
    parser.add_argument("--case", choices=("joint-mirror", "free-gauge", "live-mirror"),
                        default="joint-mirror")
    parser.add_argument("--blend", type=Path,
                        help="Replay one supplied Blender project without saving it")
    parser.add_argument("--reassess-bundle", action="store_true",
                        help="Assess a saved direct bundle with its own fitted calibrations")
    parser.add_argument("--max-sync-calls", type=int)
    parser.add_argument("--max-bundle-calls", type=int)
    parser.add_argument("--active-seconds", type=float)
    parser.add_argument("--outer-seconds", type=float,
                        help="Hard process deadline for a single private replay")
    cli = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(cli)
    out = args.out
    if args.reassess_bundle:
        if args.stage is not None or args.blend is not None:
            parser.error("--reassess-bundle reads an existing generated archive")
        _reassess_bundle_archive(out)
        return
    if args.blend is not None and out.resolve().is_relative_to(ROOT.resolve()):
        parser.error("Private numerical archives must stay outside the repository")
    if args.blend is not None and (
        args.outer_seconds is None or not math.isfinite(args.outer_seconds)
        or args.outer_seconds <= (args.active_seconds or 0) + 15.0
    ):
        parser.error("--blend requires an outer deadline above active time plus 15 seconds")
    if (args.stage is not None or args.blend is not None) and importlib.util.find_spec("bpy") is None:
        _run_outer_blender(out, outer_seconds=(
            args.outer_seconds if args.blend is not None else 180.0))
        return
    if ((args.stage in {"refine", "solve1", "solve2"} or args.blend is not None)
            and os.environ.get("PM_SYNC_CONTINUATION_OUTER") != "1"):
        parser.error("Run numerical stages through the outer Python process wrapper")
    if args.blend is not None:
        if args.stage is not None:
            parser.error("--blend runs the entire in-memory sequence; omit --stage")
        if (args.max_sync_calls is None or args.max_bundle_calls is None or
                args.active_seconds is None):
            parser.error("--blend requires explicit numerical call and active-time caps")
        if (args.max_sync_calls < 0 or args.max_bundle_calls < 0 or
                args.max_sync_calls + args.max_bundle_calls == 0 or
                not math.isfinite(args.active_seconds) or args.active_seconds <= 0):
            parser.error("Private numerical caps must be finite and positive overall")
        _run_private_sequence(
            out, args.blend, max_sync_calls=args.max_sync_calls,
            max_bundle_calls=args.max_bundle_calls,
            active_seconds=args.active_seconds,
        )
        return
    if args.stage:
        _run_native_stage(out, args.stage, case_name=args.case,
                          max_sync_calls=args.max_sync_calls,
                          max_bundle_calls=args.max_bundle_calls,
                          active_seconds=args.active_seconds)
        return
    out.mkdir(parents=True, exist_ok=False)
    case = generate("mixed_lines", seed=0, noise_px=0.0)
    (out / "case.json").write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
    source_sha256 = _sources(out)
    _core, sync = load_core()
    from match_perspective.core.sync.request import SyncSolveRequest
    from match_perspective.core.sync import solve as solve_module

    metadata = dict(
        case="mixed_lines-seed-0", source_sha256=source_sha256,
        environment=environment(ROOT), max_sync_calls=6,
    )
    request = SyncSolveRequest(**solver_arguments(case["request"]))
    results = {}
    with ExperimentBudget(
        out / "sync-ledger.jsonl", metadata=metadata,
        max_calls=6, wall_seconds=600, per_call_seconds=120,
    ) as budget:
        for label in ("cold", "seeded", "edited"):
            if label == "seeded":
                previous = results["cold"]["live"]
                request.initial_solution = sync.SyncSolutionSeed(
                    calibrations={item.match_id: deepcopy(item.calibration) for item in request.matches},
                    similarities=deepcopy(previous.similarities),
                    landmarks=deepcopy(previous.landmarks),
                    line_segments=deepcopy(previous.line_segments),
                    evidence_sha256=request.evidence_sha256(),
                )
            elif label == "edited":
                request.observations[0].u += 1.0
            exact = request.to_record()
            (out / f"{label}-request.json").write_text(json.dumps(exact, indent=2) + "\n")
            registration_seed_ids = []
            original_register = solve_module._register_from_relative_pose

            def trace_registration(*args, **kwargs):
                registration_seed_ids.extend(sorted((kwargs.get("initial_similarities") or {}).keys()))
                return original_register(*args, **kwargs)

            with budget.attempt(label, exact) as attempt:
                with patch.object(solve_module, "_register_from_relative_pose", trace_registration):
                    solved = sync.solve_landmark_sync(**request.solver_kwargs())
                record = result_record(solved, case["request"]["cameras"])
                assessment = evaluate(case, record)
                attempt.complete(dict(result=record, assessment=assessment,
                                      registration_seed_ids=registration_seed_ids))
            (out / f"{label}-result.json").write_text(json.dumps(
                dict(result=record, assessment=assessment,
                     registration_seed_ids=registration_seed_ids), indent=2, allow_nan=False,
            ) + "\n")
            results[label] = dict(live=solved, record=record, assessment=assessment,
                                  registration_seed_ids=registration_seed_ids)
            if not solved.success or not assessment["passed"]:
                break
    summary = {
        label: dict(success=item["live"].success,
                    reported_rmse_px=item["live"].mean_reprojection_px,
                    geometry_passed=item["assessment"]["passed"],
                    registration_seed_ids=item["registration_seed_ids"],
                    violations=item["assessment"]["violations"])
        for label, item in results.items()
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        len(results) != 3
        or not all(item["assessment"]["passed"] for item in results.values())
        or set(results["seeded"]["registration_seed_ids"]) != set(results["cold"]["live"].similarities)
    ):
        raise SystemExit("Sync continuation assessment failed")


if __name__ == "__main__":
    main()
