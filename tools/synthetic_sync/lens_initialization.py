"""Compare warm and cold Sync initialization at the true focal candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.lens_support import all_point_error, lens_case
from tools.synthetic_sync.scenarios import read_case, write_case
from tools.synthetic_sync.solver import environment, load_core, result_record, solver_arguments


def _request(case, focal_scale, initial_similarities):
    """Build one complete, serializable Sync request from the synthetic input."""
    load_core()
    from match_perspective.core import lens_refine, sync

    arguments = solver_arguments(case["request"])
    arguments["matches"] = [
        sync.SyncMatchInput(
            match.match_id,
            lens_refine.calibration_scaled_keep_pose(
                match.calibration, focal_scale,
            ),
        )
        for match in arguments["matches"]
    ]
    arguments["initial_similarities"] = initial_similarities
    return sync.SyncSolveRequest(**arguments)


def _run(case, request):
    """Run one exact request and retain both solver and independent metrics."""
    from match_perspective.core import sync

    started = time.perf_counter()
    result = sync.solve_landmark_sync(**request.solver_kwargs())
    elapsed_s = time.perf_counter() - started
    cameras = []
    for match in request.matches:
        intrinsics = match.calibration.intrinsics
        cameras.append(dict(
            id=match.match_id,
            width=intrinsics.image_width,
            height=intrinsics.image_height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            center=match.calibration.camera_center.tolist(),
            rotation=match.calibration.rotation_w2c.tolist(),
        ))
    record = result_record(result, cameras)
    return result, dict(
        record=record,
        assessment=evaluate(case, record),
        all_points=all_point_error(case, record),
        elapsed_s=elapsed_s,
    )


def _identity_deviation(similarities):
    """Summarize how far a warm map lies from identity."""
    rows = {}
    for match_id, similarity in similarities.items():
        rows[match_id] = dict(
            scale_delta=abs(float(similarity.scale) - 1.0),
            rotation_frobenius=float(np.linalg.norm(similarity.rotation - np.eye(3))),
            translation_norm=float(np.linalg.norm(similarity.translation)),
        )
    return rows


def _summary(row):
    cameras = row["assessment"]["cameras"]
    return dict(
        success=row["record"]["success"],
        message=row["record"]["message"],
        reported_rmse_px=row["record"]["reported_rmse_px"],
        all_supported_rmse_px=row["all_points"]["rmse_px"],
        camera_count=len(row["record"]["cameras"]),
        landmark_count=len(row["record"]["landmarks"]),
        oracle_passed=row["assessment"]["passed"],
        oracle_violations=row["assessment"]["violations"],
        camera_metric_basis=(
            "solver refusal placeholder similarities"
            if not row["record"]["success"]
            else "accepted solved similarities"
        ),
        withheld_camera_rmse_px={
            key: value["holdout_rmse_px"] for key, value in cameras.items()
        },
    )


def run(seed=0, case=None):
    """Return an unlocked refused-start baseline and paired true-focal solves."""
    if case is None:
        case = lens_case("refused_start", seed)
        case["name"] = f"lens-initialization-{seed}"
        case["request"]["fixed_similarities"] = {}
        case["investigation"] = dict(
            source="lens_case('refused_start')",
            edit="removed explicit pose locks",
            true_focal_scale=0.5,
        )

    initial_request = _request(case, 1.0, None)
    initial_record = initial_request.to_record()
    initial_result, initial = _run(case, initial_request)

    warm_request = _request(case, 0.5, initial_result.similarities)
    cold_request = _request(case, 0.5, None)
    warm_record = warm_request.to_record()
    cold_record = cold_request.to_record()

    truth_cameras = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    for match in cold_request.matches:
        truth = truth_cameras[match.match_id]
        intrinsics = match.calibration.intrinsics
        if not np.allclose(
            [intrinsics.fx, intrinsics.fy], [truth["fx"], truth["fy"]],
            rtol=0.0, atol=1e-9,
        ):
            raise ValueError("This experiment requires initially doubled focal lengths")

    warm_inputs = deepcopy(warm_record["inputs"])
    cold_inputs = deepcopy(cold_record["inputs"])
    del warm_inputs["initial_similarities"]
    del cold_inputs["initial_similarities"]
    if warm_inputs != cold_inputs:
        raise AssertionError("Warm and cold true-focal evidence differs")

    _warm_result, warm = _run(case, warm_request)
    _cold_result, cold = _run(case, cold_request)
    return case, {
        "environment": environment(),
        "solve_count": 3,
        "warm_cold_inputs_identical_except_initial_similarities": True,
        "refused_camera_metrics_are_placeholder_identities": True,
        "warm_start_identity_deviation": _identity_deviation(
            initial_result.similarities,
        ),
        "summary": {
            "initial_doubled_focal_unlocked": _summary(initial),
            "true_focal_warm": _summary(warm),
            "true_focal_cold": _summary(cold),
        },
        "details": {
            "initial_doubled_focal_unlocked": initial,
            "true_focal_warm": warm,
            "true_focal_cold": cold,
        },
        "requests": {
            "initial": initial_record,
            "true_focal_warm": warm_record,
            "true_focal_cold": cold_record,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=Path,
                        help="Replay an exact case JSON instead of regenerating it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)

    case, report = run(args.seed, read_case(args.case) if args.case else None)
    write_case(case, args.out / "case.json")
    for label, record in report.pop("requests").items():
        (args.out / f"request-{label}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (args.out / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for label, row in report["summary"].items():
        print(
            label,
            "success", row["success"],
            "reported", row["reported_rmse_px"],
            "all picks", row["all_supported_rmse_px"],
            "withheld", row["withheld_camera_rmse_px"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
