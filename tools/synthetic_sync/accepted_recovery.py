"""Trace one accepted recovered-3D candidate and its paired freeze control."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import environment, load_core, result_record, solve


STAGE_CAMERA = "view_2"
FREE_LINE = "edge_1"
PLANE_AXIS = "Y"
PLANE_BUCKET = 1
WORLD_AXES = {
    "WORLD_AXIS_X": np.array((1.0, 0.0, 0.0)),
    "WORLD_AXIS_Y": np.array((0.0, 1.0, 0.0)),
    "WORLD_AXIS_Z": np.array((0.0, 0.0, 1.0)),
}


def accepted_stage_case(
    *, seed: int = 0, noise_px: float = 0.3, known_bias: float = 0.03
) -> dict:
    """Extend the established positive stage-only control with line evidence."""
    case = generate("mixed_lines", seed, noise_px)
    case.update(name=f"accepted-recovery-{seed}", family="accepted_recovery")
    plane_points = [item["id"] for item in case["request"]["points"][:4]]
    known_points = plane_points[:3]
    case["request"].update(
        ground_slack=0.02,
        known_3d_slack=0.08,
        plane_slack=0.0,
        plane_groups=[
            *[(landmark_id, PLANE_AXIS, PLANE_BUCKET) for landmark_id in plane_points],
            (FREE_LINE, PLANE_AXIS, PLANE_BUCKET),
        ],
    )
    for point in case["request"]["points"]:
        if point["id"] in known_points:
            point["known"] = list(case["truth"]["points"][point["id"]])
            point["known"][2] += known_bias
    case["truth"]["planes"] = [
        {
            "axis": PLANE_AXIS,
            "bucket": PLANE_BUCKET,
            "origin": [0.0, -0.9, 0.0],
            "normal": [0.0, 1.0, 0.0],
        }
    ]
    case["expectation"].update(
        required_lines=[FREE_LINE],
        line_angle_deg=1.0,
        line_fraction=0.02,
        # This is an independent absolute-plane accuracy check. Hard plane
        # membership is separately visible as equal per-member distances.
        plane_max_distance=0.005,
    )
    case["measurement_edit"] = {
        "kind": "soft_known_3d_bias",
        "landmark_ids": known_points,
        "offset": [0.0, 0.0, known_bias],
        "stage_routing": (
            f"stage-only: explicitly mark already posed {STAGE_CAMERA} recovered"
        ),
        "line_strokes": "unchanged",
    }
    return case


def _snapshot(state, case: dict) -> dict:
    synthetic = SimpleNamespace(
        success=True,
        message="stage snapshot",
        mean_reprojection_px=0.0,
        similarities=state.similarities,
        landmarks=state.landmarks,
        line_segments=state.line_segments,
        line_support_angles_deg={},
        weak_line_ids=[],
        plane_seeded_landmark_ids=[],
    )
    record = result_record(synthetic, case["request"]["cameras"])
    assessment = evaluate(case, record)
    segment = np.asarray(record["line_segments"][FREE_LINE], dtype=float)
    z_range = [float(value) for value in sorted(segment[:, 2])]
    direction = segment[1] - segment[0]
    direction /= np.linalg.norm(direction)
    target_name = next(
        right
        for left, right in case["request"]["parallel_pairs"]
        if left == FREE_LINE and right in WORLD_AXES
    )
    direction_sine = float(np.linalg.norm(np.cross(direction, WORLD_AXES[target_name])))
    return {
        "record": record,
        "independent": {
            "passed": assessment["passed"],
            "violations": assessment["violations"],
            "cameras": {
                key: {
                    field: values[field]
                    for field in (
                        "holdout_rmse_px",
                        "holdout_max_px",
                        "rotation_deg",
                        "center_fraction",
                    )
                }
                for key, values in assessment["cameras"].items()
            },
            "line": assessment["geometry"]["lines"].get(FREE_LINE),
            "line_endpoint_z_range": z_range,
            "declared_axis": target_name,
            "declared_axis_direction_sine": direction_sine,
            "declared_axis_angle_deg": float(
                np.degrees(np.arcsin(np.clip(direction_sine, 0.0, 1.0)))
            ),
            "plane_distances": assessment.get("plane_distances", {}),
        },
        "known_truth_error": {
            key: float(
                np.linalg.norm(
                    np.asarray(state.landmarks[key])
                    - np.asarray(case["truth"]["points"][key])
                )
            )
            for key in case["measurement_edit"]["landmark_ids"]
            if key in state.landmarks
        },
        "kept_joint_geometry": bool(getattr(state, "kept_joint_geometry", False)),
    }


def _max_geometry_delta(left: dict, right: dict) -> dict:
    left_record, right_record = left["record"], right["record"]
    point_ids = set(left_record["landmarks"]) & set(right_record["landmarks"])
    line_ids = set(left_record["line_segments"]) & set(right_record["line_segments"])
    camera_ids = set(left_record["cameras"]) & set(right_record["cameras"])
    return {
        "landmark": max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(left_record["landmarks"][key])
                        - np.asarray(right_record["landmarks"][key])
                    )
                )
                for key in point_ids
            ),
            default=0.0,
        ),
        "line_endpoint": max(
            (
                float(
                    np.max(
                        np.linalg.norm(
                            np.asarray(left_record["line_segments"][key])
                            - np.asarray(right_record["line_segments"][key]),
                            axis=1,
                        )
                    )
                )
                for key in line_ids
            ),
            default=0.0,
        ),
        "camera_center": max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(left_record["cameras"][key]["center"])
                        - np.asarray(right_record["cameras"][key]["center"])
                    )
                )
                for key in camera_ids
            ),
            default=0.0,
        ),
    }


def _plane_member_spread(snapshot: dict, case: dict) -> float:
    """Signed spread of declared members, including both free-line endpoints."""
    plane = next(
        plane
        for plane in case["truth"]["planes"]
        if (plane["axis"], plane["bucket"]) == (PLANE_AXIS, PLANE_BUCKET)
    )
    normal = np.asarray(plane["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    origin = np.asarray(plane["origin"], dtype=float)
    record = snapshot["record"]
    signed = []
    for key, axis, bucket in case["request"]["plane_groups"]:
        if (axis, bucket) != (PLANE_AXIS, PLANE_BUCKET):
            continue
        positions = record["line_segments"].get(key)
        if positions is None and key in record["landmarks"]:
            positions = [record["landmarks"][key]]
        for position in positions or ():
            signed.append(float((np.asarray(position, dtype=float) - origin) @ normal))
    return max(signed) - min(signed)


def observe(case: dict, *, freeze: bool) -> tuple[dict, dict]:
    """Trace before/candidate/commit/final rebuild around the one stage."""
    load_core()
    module = importlib.import_module("match_perspective.core.sync.solve")
    original_thaw = module._thaw_recovered_location
    original_refine = module._refine_recovered_location
    original_rebuild = module._rebuild_free_line_segments
    trace: dict = {
        "freeze_control": freeze,
        "stage_control_recovered": [STAGE_CAMERA],
        "routing": "stage-only",
    }
    awaiting_final_rebuild = set()

    def refine(candidate):
        original_refine(candidate)
        trace["candidate"] = _snapshot(candidate, case)

    def thaw(state):
        if STAGE_CAMERA not in state.similarities:
            raise ValueError("Stage-only control requires the camera to be posed")
        state.recovered = [STAGE_CAMERA]
        trace["before"] = _snapshot(state, case)
        if not freeze:
            original_thaw(state)
        trace["after_guard"] = _snapshot(state, case)
        awaiting_final_rebuild.add(id(state))

    def rebuild(state):
        original_rebuild(state)
        if id(state) in awaiting_final_rebuild:
            trace["after_final_line_rebuild"] = _snapshot(state, case)
            awaiting_final_rebuild.remove(id(state))

    with (
        patch.object(module, "_refine_recovered_location", side_effect=refine),
        patch.object(module, "_thaw_recovered_location", side_effect=thaw),
        patch.object(module, "_rebuild_free_line_segments", side_effect=rebuild),
    ):
        result = solve(case["request"])

    if "before" in trace:
        trace["stage_delta"] = _max_geometry_delta(
            trace["before"], trace["after_guard"]
        )
        trace["final_rebuild_delta"] = _max_geometry_delta(
            trace["after_guard"], trace["after_final_line_rebuild"]
        )
        trace["candidate_accepted"] = bool(
            not freeze
            and "candidate" in trace
            and not trace["after_guard"]["kept_joint_geometry"]
            and any(value > 1.0e-10 for value in trace["stage_delta"].values())
        )
    return result, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-px", type=float, default=0.3)
    parser.add_argument("--known-bias", type=float, default=0.03)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.known_bias) or args.known_bias <= 0.0:
        parser.error("known bias must be finite and positive")

    case = (
        read_case(args.case)
        if args.case
        else accepted_stage_case(
            seed=args.seed, noise_px=args.noise_px, known_bias=args.known_bias
        )
    )
    args.out.mkdir(parents=True, exist_ok=False)
    case_path = args.out / "case.json"
    write_case(case, case_path)
    protocol = {
        "environment": environment(),
        "solve_count": 2,
        "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "runs": ["production", "freeze-control"],
        "paired_boundary": (
            "identical input; freeze control bypasses only "
            "_thaw_recovered_location; both retain the final free-line rebuild"
        ),
        "routing": (
            f"stage-only: explicitly mark already posed {STAGE_CAMERA} recovered"
        ),
        "acceptance_rule": (
            "candidate is counted only when the guard commits a numerically changed state"
        ),
    }
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    reports = []
    records = {}
    for label, freeze in (("production", False), ("freeze-control", True)):
        result, trace = observe(case, freeze=freeze)
        assessment = evaluate(case, result)
        records[label] = {"result": result, "assessment": assessment, "trace": trace}
        (args.out / f"{label}.json").write_text(
            json.dumps(records[label], indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        display = deepcopy(case)
        display["name"] += "-" + label
        reports.append((display, result, assessment))
        print(
            label,
            "accepted=" + repr(trace.get("candidate_accepted")),
            "final_axis_angle_deg="
            + repr(
                trace.get("after_final_line_rebuild", {})
                .get("independent", {})
                .get("declared_axis_angle_deg")
            ),
            "violations=" + repr(assessment["violations"]),
            flush=True,
        )
    write_report(reports, args.out / "report.html")

    production, frozen = records["production"], records["freeze-control"]
    production_trace = production["trace"]
    known_ids = case["measurement_edit"]["landmark_ids"]
    known_improved = all(
        production_trace["after_guard"]["known_truth_error"][key]
        < production_trace["before"]["known_truth_error"][key]
        for key in known_ids
    )
    exact_final_snapshot = all(
        production["result"][field]
        == production_trace["after_final_line_rebuild"]["record"][field]
        for field in ("cameras", "landmarks", "line_segments")
    )
    candidate_coplanar = _plane_member_spread(production_trace["candidate"], case) < 1.0e-5
    final_coplanar = (
        _plane_member_spread(production_trace["after_final_line_rebuild"], case) < 1.0e-5
    )
    hard_parallel_preserved = all(
        snapshot["independent"]["declared_axis_direction_sine"] <= 1.0e-8
        for snapshot in (
            production_trace["candidate"],
            production_trace["after_final_line_rebuild"],
            frozen["trace"]["after_final_line_rebuild"],
        )
    )
    reference_z = sorted(
        float(point[2]) for point in case["truth"]["lines"][FREE_LINE]
    )
    final_z = production_trace["after_final_line_rebuild"]["independent"][
        "line_endpoint_z_range"
    ]
    final_segment_extent = max(
        abs(value - expected) for value, expected in zip(final_z, reference_z)
    ) < 0.05
    valid = (
        production["result"].get("success")
        and frozen["result"].get("success")
        and production["assessment"]["passed"]
        and frozen["assessment"]["passed"]
        and production_trace.get("candidate_accepted") is True
        and production_trace["candidate"]["independent"]["passed"]
        and candidate_coplanar
        and final_coplanar
        and hard_parallel_preserved
        and final_segment_extent
        and known_improved
        and frozen["trace"].get("candidate_accepted") is False
        and not any(frozen["trace"]["stage_delta"].values())
        and production["result"]["request_sha256"]
        == frozen["result"]["request_sha256"]
        and exact_final_snapshot
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
