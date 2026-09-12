"""Replay frozen Sync starts with unchanged or counterfactual bundle relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import focal_constraint_trial as trial
from . import focal_constraint_reliability as fixture
from .budget import ExperimentBudget
from .solver import calibration, load_core, solver_arguments


PAIRS = (("weak-axis-removed", "weak-axis-hard"),
         ("weak-axis-hard", "weak-axis-removed"),
         ("weak-axis-removed", "weak-axis-wrong-member"),
         ("weak-free-removed", "weak-free-hard"),
         ("weak-mirror-removed", "weak-mirror-hard"),
         ("weak-mirror-removed", "weak-mirror-tilted"),
         ("weak-mirror-anchor-rotated-removed", "weak-mirror-anchor-rotated")) + tuple(
             (name, name) for name in fixture.CASE_NAMES)


def run(out: Path, source: Path, pairs: tuple[tuple[str, str], ...], *,
        fixed_frame=False, max_iterations=None):
    trial.SOURCE_PATHS = tuple(sorted(set(trial.SOURCE_PATHS + (
        Path("tools/synthetic_sync/focal_constraint_reliability.py"),
        Path("tools/synthetic_sync/focal_constraint_saved_start.py"),
        Path("tools/synthetic_sync/geometry.py"),
    ))))
    trial.BUNDLE_LIMITS = dict(max_calls=len(pairs), per_call_seconds=30,
                               wall_seconds=30 * len(pairs))
    metadata = trial._prepare_output(out)
    metadata["sync_source"] = str(source.resolve())
    load_core()
    from match_perspective.core import focal_bundle
    iterations = focal_bundle.MAX_ITERATIONS if max_iterations is None else max_iterations
    metadata.update(fixed_frame=fixed_frame, max_iterations=iterations)
    original_basis = focal_bundle.orientation_basis

    with (ExperimentBudget(out / "bundle-ledger.jsonl", metadata=metadata,
                           **trial.BUNDLE_LIMITS) as budget,
          patch.object(focal_bundle, "MAX_ITERATIONS", iterations),
          patch.object(focal_bundle, "orientation_basis",
                       (lambda *_args: np.empty((0, 3))) if fixed_frame else original_basis)):
        for start_name, target_name in pairs:
            start_case = json.loads((fixture.ROOT / f"{start_name}.json").read_text())
            target_case = json.loads((fixture.ROOT / f"{target_name}.json").read_text())
            fixture.validate(start_case)
            fixture.validate(target_case)
            if start_case["request"]["observations"] != target_case["request"]["observations"]:
                raise ValueError("Crossed cases differ in image picks")
            initial, world = trial._replay_initial(start_case, source)
            request = target_case["request"]
            calibrations = {item["id"]: calibration(item) for item in request["cameras"]}
            arguments = solver_arguments(request)
            trial_input = dict(start_name=start_name, target_name=target_name,
                               request=request, archived_initial_world=world,
                               pick_sigma_px=target_case["pick_sigma_px"],
                               fixed_frame=fixed_frame, max_iterations=iterations)
            endpoints = []
            with budget.attempt(f"{start_name}:to:{target_name}", trial_input) as attempt:
                outcome = focal_bundle.fit_independent_focals(
                    calibrations, arguments["observations"], initial,
                    anchor_id=request["anchor_id"], pick_sigma_px=target_case["pick_sigma_px"],
                    fx_span=0.4, plane_groups=arguments["plane_groups"],
                    plane_slack=arguments["plane_slack"],
                    mirror_pairs=arguments["mirror_pairs"],
                    mirror_plane=arguments["mirror_plane"],
                    mirror_slack=arguments["mirror_slack"],
                    diagnostic_callback=endpoints.append)
                record = trial._outcome_record(outcome, request["cameras"])
                record["endpoint"] = endpoints[-1] if endpoints else None
                attempt.complete(trial._finite_json(record))
            summary = dict(start_name=start_name, target_name=target_name,
                           accepted=outcome.accepted, reason=outcome.reason,
                           initial_rmse_px=outcome.initial_rmse_px,
                           fitted_rmse_px=outcome.fitted_rmse_px,
                           fixed_frame=fixed_frame, max_iterations=iterations,
                           iterations=endpoints[-1]["iterations"] if endpoints else None)
            if outcome.accepted:
                summary["independent"] = fixture.assess(target_case, record["fitted"])
                summary["focal_interval_truth_coverage"] = {
                    camera["id"]: bool(outcome.intervals_px[camera["id"]][0] <= camera["fx"] <=
                                       outcome.intervals_px[camera["id"]][1])
                    for camera in target_case["truth"]["cameras"]}
            (out / f"{start_name}-to-{target_name}.json").write_text(
                json.dumps(trial._finite_json(summary), sort_keys=True, indent=2) + "\n")
            print(json.dumps(trial._finite_json(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        choices=sorted(fixture.CASE_NAMES))
    parser.add_argument("--fixed-frame", action="store_true",
                        help="Diagnostic control: disable the fitted common rotation")
    parser.add_argument("--max-iterations", type=int,
                        help="Diagnostic iteration cap (1–300); retains the time limit")
    options = parser.parse_args()
    if options.max_iterations is not None and not 1 <= options.max_iterations <= 300:
        parser.error("--max-iterations must be between 1 and 300")
    pairs = tuple(tuple(pair) for pair in options.pair)
    if any(pair not in PAIRS for pair in pairs):
        raise ValueError("Pair is not in the frozen cross-start design")
    run(options.out, options.source, pairs, fixed_frame=options.fixed_frame,
        max_iterations=options.max_iterations)
