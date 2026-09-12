"""Measure line-midpoint gauge conditioning without modifying production Sync."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.accepted_recovery import observe
from tools.synthetic_sync.constraint_interactions import evaluate_interactions, interaction_cases
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project as independent_project
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, environment, load_core, solve


ACCEPTED_CASE = Path(__file__).parent / "cases/accepted-recovery.json"
SHIFT_DISTANCES = (1.0, 100.0, -9000.0, 9000.0)


def sampled_line_projection(point, direction, camera, similarity):
    """Historical finite-span projector retained as the diagnostic baseline."""
    from match_perspective.core.sync import projection

    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    projected = []
    for scale in (0.25, 0.5, 1.0, 2.0, 8.0, 32.0):
        for sample in (point - scale * unit, point + scale * unit):
            private = similarity.inverse_point(sample)
            image_point = projection.project_private_point(private, camera)
            if image_point is None:
                continue
            projected.append((float(image_point[0]), float(image_point[1])))
            if len(projected) >= 2:
                line = projection._image_line_homogeneous(
                    projected[0][0], projected[0][1],
                    projected[-1][0], projected[-1][1],
                )
                if line is not None:
                    return line
    return None


def analytic_line_projection(point, direction, camera, similarity):
    """Diagnostic pinhole projection through the 3D line's viewing plane."""
    if camera.has_distortion:
        return _ORIGINAL_LINE_PROJECTION(point, direction, camera, similarity)
    world_direction = np.asarray(direction, dtype=float).copy()
    norm = float(np.linalg.norm(world_direction))
    if norm <= 1e-12:
        return None
    world_direction /= norm
    center = similarity.transform_point(camera.camera_center)
    from_center = np.asarray(point, dtype=float) - center
    # Center the representative point on the closest point to the camera.
    # This limits cancellation when its input slid far along the same line.
    perpendicular = from_center - float(from_center @ world_direction) * world_direction
    normal_world = np.cross(perpendicular, world_direction)
    if float(np.linalg.norm(normal_world)) <= 1e-10:
        return None
    world_to_camera = camera.rotation_w2c @ similarity.rotation.T
    direction_camera = world_to_camera @ world_direction
    depth = float((world_to_camera @ from_center)[2]) / max(float(similarity.scale), 1e-12)
    if abs(float(direction_camera[2])) <= 1e-12 and depth <= 1e-8:
        return None
    normal_camera = world_to_camera @ normal_world
    intrinsics = camera.intrinsics
    a = float(normal_camera[0]) / float(intrinsics.fx)
    b = float(normal_camera[1]) / float(intrinsics.fy)
    c = float(normal_camera[2]) - intrinsics.cx * a - intrinsics.cy * b
    scale = float(np.hypot(a, b))
    if scale <= 1e-12:
        return None
    return np.array((a, b, c), dtype=float) / scale


_ORIGINAL_LINE_PROJECTION = None


def projection_checks(case: dict) -> dict:
    """Check the prototype against independently projected image samples."""
    load_core()
    from match_perspective.core.sync import projection, SimilarityTransform

    global _ORIGINAL_LINE_PROJECTION
    _ORIGINAL_LINE_PROJECTION = sampled_line_projection
    camera = calibration(case["request"]["cameras"][0])
    endpoints = np.asarray(case["truth"]["lines"]["edge_1"], dtype=float)
    original_point = endpoints.mean(axis=0)
    original_direction = endpoints[1] - endpoints[0]
    theta = 0.37
    rotation = np.array(((np.cos(theta), -np.sin(theta), 0.0),
                         (np.sin(theta), np.cos(theta), 0.0), (0.0, 0.0, 1.0)))
    similarities = {
        "identity": SimilarityTransform(),
        "scaled_rotated_translated": SimilarityTransform(1.7, rotation, np.array((0.3, -0.2, 0.5))),
    }
    checks = {}
    for label, similarity in similarities.items():
        point = similarity.transform_point(original_point)
        direction = similarity.rotation @ original_direction
        line = analytic_line_projection(point, direction, camera, similarity)
        if line is None:
            raise AssertionError(f"{label}: ordinary line did not project")
        sampled = []
        private_camera = case["request"]["cameras"][0]
        for offset in (-0.5, 0.5):
            private = similarity.inverse_point(point + offset * direction)
            uv, depths = independent_project([private], private_camera)
            if depths[0] <= 1e-8:
                raise AssertionError(f"{label}: sample landed behind camera")
            sampled.append(abs(float(line @ np.array((uv[0, 0], uv[0, 1], 1.0)))))
        shifted = {}
        unit = direction / np.linalg.norm(direction)
        for distance in SHIFT_DISTANCES:
            displaced = point + distance * unit
            moved = analytic_line_projection(displaced, direction, camera, similarity)
            old = _ORIGINAL_LINE_PROJECTION(displaced, direction, camera, similarity)
            shifted[str(distance)] = {
                "analytic_line_delta": None if moved is None else float(min(np.linalg.norm(moved - line), np.linalg.norm(moved + line))),
                "sampled_projector_returned": bool(old is not None),
            }
        checks[label] = {"max_sample_pixel_line_distance": max(sampled), "shift": shifted}
    # A line through the optical center has no unique image line.
    center = camera.camera_center
    checks["degenerate_camera_crossing"] = analytic_line_projection(center, np.array((1.0, 0.0, 0.0)), camera, similarities["identity"]) is None
    behind = center + camera.rotation_w2c.T @ np.array((0.0, 0.0, -2.0))
    parallel = camera.rotation_w2c.T @ np.array((1.0, 0.0, 0.0))
    checks["wholly_behind_parallel"] = analytic_line_projection(behind, parallel, camera, similarities["identity"]) is None
    direction_input = np.array((2.0, 0.0, 3.0), dtype=float)
    direction_before = direction_input.copy()
    analytic_line_projection(original_point, direction_input, camera, similarities["identity"])
    checks["direction_input_unchanged"] = bool(np.array_equal(direction_input, direction_before))
    if not all(checks[name] for name in ("degenerate_camera_crossing", "wholly_behind_parallel", "direction_input_unchanged")):
        raise AssertionError("analytic projection mishandled degeneracy or modified input")
    if any(info["max_sample_pixel_line_distance"] > 1e-8 or
           any(value["analytic_line_delta"] is None or value["analytic_line_delta"] > 1e-7
               for value in info["shift"].values()) for info in (checks["identity"], checks["scaled_rotated_translated"])):
        raise AssertionError("analytic projection failed sample or along-line invariance")
    return checks


def _free_line_base(kwargs: dict) -> int:
    stride = (0 if kwargs["lock_scale"] else 1) + (0 if kwargs["lock_rotation"] else 3) + (0 if kwargs["lock_translation"] else 3)
    return len(kwargs["free_match_ids"]) * stride + 3 * len(kwargs["free_landmark_ids"])


def _line_positions(params: np.ndarray, kwargs: dict) -> dict[str, np.ndarray]:
    base = _free_line_base(kwargs)
    return {line_id: params[base + 3 * i:base + 3 * i + 3].copy()
            for i, line_id in enumerate(kwargs["free_line_ids"])}


def _unit_directions(kwargs: dict) -> dict[str, np.ndarray]:
    return {line_id: np.asarray(direction, dtype=float) / np.linalg.norm(direction)
            for line_id, direction in kwargs["fixed_line_directions"].items()
            if np.linalg.norm(direction) > 1e-12}


def _residual_shift_check(params: np.ndarray, kwargs: dict, residual_fn) -> dict:
    """Compare every active weighted residual after moving one infinite line."""
    positions = _line_positions(params, kwargs)
    directions = _unit_directions(kwargs)
    base = residual_fn(params, **kwargs)
    start = _free_line_base(kwargs)
    checks = {}
    for i, line_id in enumerate(kwargs["free_line_ids"]):
        if line_id not in directions:
            continue
        checks[line_id] = {}
        for distance in SHIFT_DISTANCES:
            trial = params.copy()
            trial[start + 3 * i:start + 3 * i + 3] += distance * directions[line_id]
            shifted = residual_fn(trial, **kwargs)
            checks[line_id][str(distance)] = {
                "max_weighted_residual_change": float(np.max(np.abs(shifted - base))),
                "squared_cost_change": float(shifted @ shifted - base @ base),
            }
    return {"initial_midpoints": {key: value.tolist() for key, value in positions.items()}, "shifts": checks}


@contextmanager
def measure_ba(mode: str):
    """Patch only BA-call instrumentation and line-pixel Jacobian columns."""
    load_core()
    ba = importlib.import_module("match_perspective.core.sync.ba")
    solve_module = importlib.import_module("match_perspective.core.sync.solve")
    original_bundle = solve_module._bundle_adjust_registration
    original_residual = ba._ba_residual_vector
    original_jacobian = ba._jacobian_ba
    calls: list[dict] = []
    active: dict | None = None

    def wrapped_residual(params, **kwargs):
        result = original_residual(params, **kwargs)
        if active is not None:
            active["residual_evaluations"] += 1
            cost = float(result @ result)
            if active["initial_cost"] is None:
                active["initial_cost"] = cost
            active["minimum_evaluated_cost"] = min(active["minimum_evaluated_cost"], cost)
            for line_id, point in _line_positions(params, kwargs).items():
                active["maximum_midpoint_norms"][line_id] = max(
                    active["maximum_midpoint_norms"].get(line_id, 0.0),
                    float(np.linalg.norm(point)),
                )
        return result

    def wrapped_jacobian(params, kwargs):
        matrix = original_jacobian(params, kwargs)
        if active is not None:
            active["jacobian_evaluations"] += 1
            if not active["invariance"] and kwargs["free_line_ids"]:
                probe_started = perf_counter()
                active["invariance"] = _residual_shift_check(params, kwargs, original_residual)
                active["diagnostic_probe_seconds"] += perf_counter() - probe_started
        if mode == "project-line-pixel" and kwargs["free_line_ids"]:
            # Line observation rows are appended last in _ba_raw_residuals_and_jacobian.
            # Other (point/plane/mirror) rows retain their actual derivatives.
            base = _free_line_base(kwargs)
            indices = {line_id: i for i, line_id in enumerate(kwargs["free_line_ids"])}
            directions = _unit_directions(kwargs)
            line_rows = matrix.shape[0] - 2 * len(kwargs["line_constraints"])
            for j, (line_id, _point, _direction, _observation) in enumerate(kwargs["line_constraints"]):
                i = indices.get(line_id)
                direction = directions.get(line_id)
                if i is None or direction is None:
                    continue
                columns = slice(base + 3 * i, base + 3 * i + 3)
                rows = slice(line_rows + 2 * j, line_rows + 2 * j + 2)
                block = matrix[rows, columns]
                matrix[rows, columns] = block - np.outer(block @ direction, direction)
        return matrix

    def wrapped_bundle(*args, **kwargs):
        nonlocal active
        parent = active
        active = {
            "index": len(calls),
            "max_iterations": int(kwargs.get("max_iterations", 20)),
            "line_constraint_ids": sorted({item[0] for item in args[8]}),
            "residual_evaluations": 0,
            "jacobian_evaluations": 0,
            "diagnostic_probe_seconds": 0.0,
            "initial_cost": None,
            "minimum_evaluated_cost": float("inf"),
            "maximum_midpoint_norms": {},
            "invariance": {},
        }
        start = perf_counter()
        try:
            output = original_bundle(*args, **kwargs)
            active["returned_midpoint_norms"] = {
                line_id: float(np.linalg.norm((np.asarray(ends[0]) + np.asarray(ends[1])) / 2))
                for line_id, ends in output[2].items()
            }
            return output
        finally:
            active["seconds"] = perf_counter() - start
            active["seconds_excluding_shift_probes"] = active["seconds"] - active["diagnostic_probe_seconds"]
            calls.append(active)
            active = parent

    with ExitStack() as stack:
        stack.enter_context(patch.object(ba, "_ba_residual_vector", side_effect=wrapped_residual))
        stack.enter_context(patch.object(ba, "_jacobian_ba", side_effect=wrapped_jacobian))
        stack.enter_context(patch.object(solve_module, "_bundle_adjust_registration", side_effect=wrapped_bundle))
        if mode in {"baseline", "project-line-pixel", "analytic-line"}:
            projection = importlib.import_module("match_perspective.core.sync.projection")
            global _ORIGINAL_LINE_PROJECTION
            _ORIGINAL_LINE_PROJECTION = sampled_line_projection
            projector = analytic_line_projection if mode == "analytic-line" else sampled_line_projection
            stack.enter_context(patch.object(projection, "_project_world_line_to_image", side_effect=projector))
        yield calls


def _finite_extent(record: dict) -> dict[str, dict]:
    output = {}
    for line_id, ends in record["line_segments"].items():
        a, b = np.asarray(ends, dtype=float)
        output[line_id] = {
            "length": float(np.linalg.norm(b-a)),
            "midpoint_norm": float(np.linalg.norm((a+b)/2)),
            "endpoints": [a.tolist(), b.tolist()],
        }
    return output


def run_case(case: dict, mode: str, *, accepted: bool) -> dict:
    started = perf_counter()
    with measure_ba(mode) as calls:
        if accepted:
            result, trace = observe(deepcopy(case), freeze=False)
        else:
            result = solve(deepcopy(case["request"]))
            trace = None
    elapsed = perf_counter() - started
    assessment = evaluate(case, result) if accepted else evaluate_interactions(case, result)
    return {
        "case": case["name"],
        "mode": mode,
        "seconds": elapsed,
        "ba_calls": calls,
        "accepted_stage": None if trace is None else {
            "candidate_accepted": trace.get("candidate_accepted"),
            "before": trace.get("before"),
            "candidate": trace.get("candidate"),
            "after_guard": trace.get("after_guard"),
            "after_final_line_rebuild": trace.get("after_final_line_rebuild"),
        },
        "assessment": assessment,
        "finite_extent": _finite_extent(result),
        "record": result,
    }


def _line_distance(ends_a, ends_b) -> dict:
    a = np.asarray(ends_a, dtype=float)
    b = np.asarray(ends_b, dtype=float)
    direction_a = (a[1] - a[0]) / np.linalg.norm(a[1] - a[0])
    direction_b = (b[1] - b[0]) / np.linalg.norm(b[1] - b[0])
    return {
        "direction_sine": float(np.linalg.norm(np.cross(direction_a, direction_b))),
        "infinite_line_offset": float(np.linalg.norm(np.cross((b.mean(axis=0) - a.mean(axis=0)), direction_a))),
        "finite_endpoint_max_delta": float(np.max(np.linalg.norm(a-b, axis=1))),
        "length_delta": float(abs(np.linalg.norm(a[1]-a[0]) - np.linalg.norm(b[1]-b[0]))),
    }


def compare(left: dict, right: dict) -> dict:
    a, b = left["record"], right["record"]
    if not a.get("request_sha256") or a["request_sha256"] != b.get("request_sha256"):
        raise ValueError("Cannot compare runs with different or missing request fingerprints")
    cameras = set(a["cameras"]) & set(b["cameras"])
    lines = set(a["line_segments"]) & set(b["line_segments"])
    return {
        "support_changes": {field: {
            "lost": sorted(set(a.get(field, {})) - set(b.get(field, {}))),
            "gained": sorted(set(b.get(field, {})) - set(a.get(field, {}))),
        } for field in ("cameras", "landmarks", "line_segments")},
        "reported_rmse_delta_px": float(b["reported_rmse_px"] - a["reported_rmse_px"]),
        "camera_center_max_delta": max((float(np.linalg.norm(np.asarray(a["cameras"][key]["center"]) - np.asarray(b["cameras"][key]["center"]))) for key in cameras), default=None),
        "lines": {key: _line_distance(a["line_segments"][key], b["line_segments"][key]) for key in sorted(lines)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analytic", action="store_true", help="Run the three-solve analytic-projection follow-up")
    parser.add_argument("--baseline-dir", type=Path, help="Six-solve baseline output for analytic comparisons")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 for pinned-thread timing")
    args.out.mkdir(parents=True, exist_ok=False)
    accepted = read_case(ACCEPTED_CASE)
    interaction = interaction_cases()[0]
    checks = projection_checks(accepted)
    (args.out / "projection-checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    if args.analytic and args.baseline_dir is None:
        parser.error("--analytic requires --baseline-dir")
    protocol = {
        "environment": environment(),
        "baseline_projection": "historical finite-span sampler copied from core/sync/projection.py at 16a6cb6",
        "accepted_case_sha256": hashlib.sha256(ACCEPTED_CASE.read_bytes()).hexdigest(),
        "solve_count": 3 if args.analytic else 6,
        "order": (["accepted analytic", "interaction analytic", "accepted analytic repeat"] if args.analytic else
                  ["accepted baseline", "accepted prototype", "interaction baseline", "interaction prototype", "accepted prototype repeat", "accepted baseline repeat"]),
        "prototype": ("analytic pinhole infinite-line projection; retain sampled projection for distorted calibration" if args.analytic else
                      "project only line-pixel midpoint Jacobian rows orthogonal to each fixed line direction; retain every residual and all plane/mirror derivatives"),
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    schedule = ([
        ("accepted-analytic", accepted, "analytic-line", True),
        ("interaction-analytic", interaction, "analytic-line", False),
        ("accepted-analytic-repeat", accepted, "analytic-line", True),
    ] if args.analytic else [
        ("accepted-baseline", accepted, "baseline", True),
        ("accepted-prototype", accepted, "project-line-pixel", True),
        ("interaction-baseline", interaction, "baseline", False),
        ("interaction-prototype", interaction, "project-line-pixel", False),
        ("accepted-prototype-repeat", accepted, "project-line-pixel", True),
        ("accepted-baseline-repeat", accepted, "baseline", True),
    ])
    records = {}
    for label, case, mode, stage in schedule:
        run = run_case(case, mode, accepted=stage)
        records[label] = run
        (args.out / f"{label}.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        print(label, "seconds", round(run["seconds"], 4), "passed", run["assessment"]["passed"],
              "calls", [(c["residual_evaluations"], c["jacobian_evaluations"]) for c in run["ba_calls"]], flush=True)
    if args.analytic:
        baseline = {name: json.loads((args.baseline_dir / f"{name}.json").read_text())
                    for name in ("accepted-baseline", "interaction-baseline")}
        comparisons = {
            "accepted": compare(baseline["accepted-baseline"], records["accepted-analytic"]),
            "interaction": compare(baseline["interaction-baseline"], records["interaction-analytic"]),
            "accepted_analytic_repeat": compare(records["accepted-analytic"], records["accepted-analytic-repeat"]),
        }
    else:
        comparisons = {
            "accepted": compare(records["accepted-baseline"], records["accepted-prototype"]),
            "interaction": compare(records["interaction-baseline"], records["interaction-prototype"]),
            "accepted_baseline_repeat": compare(records["accepted-baseline"], records["accepted-baseline-repeat"]),
            "accepted_prototype_repeat": compare(records["accepted-prototype"], records["accepted-prototype-repeat"]),
        }
    (args.out / "comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    print("comparisons", json.dumps(comparisons, sort_keys=True), flush=True)
    return 0 if all(run["assessment"]["passed"] for run in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
