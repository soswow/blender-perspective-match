"""Measure lens-search evidence coverage against independent withheld geometry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import environment, load_core, result_record, solver_arguments


def lens_case(kind="noisy_view", seed=0):
    """Keep exact reference geometry; vary only noisy picks or initial focal bias."""
    case = generate("known_3d", seed, 0.)
    case.update(name=f"lens-{kind}-{seed}", family="lens_support")
    if kind == "noisy_view":
        rng = np.random.default_rng(7)
        for observation in case["request"]["observations"]:
            if observation["match_id"] == "view_2":
                observation["u"] += float(rng.normal(0., 30.))
                observation["v"] += float(rng.normal(0., 30.))
        case["measurement_edit"] = dict(camera="view_2", noise_seed=7, noise_px=30.)
        span = .18
    elif kind in {"recovery", "refused_start", "refused_improving"}:
        scale = .875 if kind == "recovery" else .5
        for camera in case["request"]["cameras"]:
            camera["fx"] /= scale
            camera["fy"] /= scale
        case["initial_focal_scale"] = 1. / scale
        span = .125 if kind == "refused_improving" else 1. - scale
        if kind != "recovery":
            truth = {camera["id"]: camera for camera in case["truth"]["cameras"]}
            for camera in case["request"]["cameras"][1:]:
                target = truth[camera["id"]]
                rotation = np.asarray(target["rotation"]).T @ np.asarray(camera["rotation"])
                case["request"]["fixed_similarities"][camera["id"]] = dict(scale=1.,
                    rotation=rotation.tolist(),
                    translation=(np.asarray(target["center"])-rotation @ camera["center"]).tolist())
    else:
        raise ValueError(f"Unknown lens case: {kind}")
    case["lens_search"] = dict(share_lens=True, fx_span=span, coarse_samples=3, refine_samples=0)
    return case


def all_point_error(case, record):
    """Score every projectable supported pick with the independent projector."""
    squared, support = [], []
    invalid = []
    for observation in case["request"]["observations"]:
        match_id, point_id = observation["match_id"], observation["landmark_id"]
        if match_id not in record["cameras"] or point_id not in record["landmarks"]:
            continue
        uv, depth = project([record["landmarks"][point_id]], record["cameras"][match_id])
        key = [match_id, point_id]
        support.append(key)
        if depth[0] <= 0 or not np.isfinite(uv).all():
            invalid.append(key)
            continue
        squared.append(float(np.sum((uv[0]-[observation["u"], observation["v"]])**2)))
    return dict(rmse_px=float(np.sqrt(np.mean(squared))) if squared and not invalid else None,
                support=support, invalid=invalid)


def run_search(case):
    """Trace the real outer search without replacing its numerical evaluations."""
    load_core()
    from match_perspective.core import lens_refine
    arguments = solver_arguments(case["request"])
    matches = arguments.pop("matches")
    lens_inputs = [lens_refine.MatchLensInput(match.match_id, {}, match.calibration.intrinsics,
                  base_calibration=match.calibration) for match in matches]
    traces, numerical_results = [], []
    original = lens_refine._run_sync

    def observe(calibrations, *args, **kwargs):
        result = original(calibrations, *args, **kwargs)
        cameras = deepcopy(case["request"]["cameras"])
        for camera in cameras:
            calibration = calibrations[camera["id"]]
            camera.update(fx=calibration.intrinsics.fx, fy=calibration.intrinsics.fy,
                          center=calibration.camera_center.tolist(), rotation=calibration.rotation_w2c.tolist())
        record = result_record(result, cameras)
        traces.append(dict(record=record, assessment=evaluate(case, record),
                           all_points=all_point_error(case, record),
                           per_match_rmse_px=result.per_match_rmse_px))
        numerical_results.append(result)
        return result

    with patch.object(lens_refine, "_run_sync", observe):
        result = lens_refine.refine_lenses_from_landmarks(lens_inputs, **arguments, **case["lens_search"])
    selected = next(trace for trace, numerical in zip(traces, numerical_results)
                    if numerical is result.sync_result)
    initial = traces[0]
    selected_ids = set(selected["record"]["cameras"])
    selected_support = set(map(tuple, selected["all_points"]["support"]))
    return dict(environment=environment(), initial=initial, selected=selected, trials=traces,
                initial_cost=result.initial_cost, final_cost=result.final_cost, improved=result.improved,
                message=result.message, fx_deltas=result.fx_deltas,
                lost_cameras=sorted(set(initial["record"]["cameras"])-selected_ids),
                lost_point_picks=sorted(set(map(tuple, initial["all_points"]["support"]))-selected_support))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--kind", choices=("noisy_view", "recovery", "refused_start", "refused_improving"), default="noisy_view")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    case = read_case(args.case) if args.case else lens_case(args.kind)
    write_case(case, args.out/"case.json")
    report = run_search(case)
    (args.out/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for label in ("initial", "selected"):
        trace = report[label]
        holdout = {key: round(value["holdout_rmse_px"], 5)
                   for key, value in trace["assessment"]["cameras"].items()}
        print(label, "success", trace["record"]["success"], "headline", trace["record"]["reported_rmse_px"],
              "all picks", trace["all_points"]["rmse_px"], "holdout", holdout)
    print(report["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
