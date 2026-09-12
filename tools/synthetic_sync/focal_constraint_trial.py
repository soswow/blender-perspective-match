"""Ledgered public point-FOV trials on frozen plane and mirror cases.

Every actual inner Sync and bundle call is reserved separately. Run this via
an outer process timeout (for example, ``gtimeout 720 python3 -m ...``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from unittest import mock

import numpy as np

from .budget import ExperimentBudget
from .focal_constraints import CASE_NAMES, ROOT as CASE_ROOT, assess, validate
from .solver import calibration, environment, load_core, result_record, solver_arguments


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = tuple(sorted(
    [*(path.relative_to(ROOT) for path in (ROOT / "core").glob("**/*.py")),
     Path("tools/synthetic_sync/focal_constraints.py"),
     Path("tools/synthetic_sync/focal_constraint_trial.py"),
     Path("tools/synthetic_sync/solver.py"),
     Path("tools/synthetic_sync/budget.py")]
))
SYNC_LIMITS = dict(max_calls=12, per_call_seconds=120, wall_seconds=600)
BUNDLE_LIMITS = dict(max_calls=18, per_call_seconds=30, wall_seconds=540)
INITIAL_CASES = (
    "free-hard", "free-hard-removed", "axis-hard", "axis-hard-removed",
    "mirror-hard-offcenter", "mirror-through-anchor", "mirror-removed",
)


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _source_hash() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        digest.update(str(relative).encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _prepare_output(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source_sha256 = _source_hash()
    metadata = dict(source_sha256=source_sha256, environment=environment(ROOT),
                    sync_limits=SYNC_LIMITS, bundle_limits=BUNDLE_LIMITS,
                    case_schema=1)
    manifest = out / "source-runtime.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Trial source/runtime changed; start a new output directory")
    else:
        manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
            for relative in SOURCE_PATHS:
                archive.add(ROOT / relative, arcname=str(relative))
    return metadata


def _outcome_record(outcome, request_cameras: list[dict]) -> dict:
    if not outcome.accepted or outcome.sync_result is None:
        return dict(accepted=False, reason=outcome.reason,
                    initial_rmse_px=(outcome.initial_rmse_px if
                                     outcome.initial_rmse_px < float("inf") else None),
                    fitted_rmse_px=(outcome.fitted_rmse_px if
                                    outcome.fitted_rmse_px < float("inf") else None))
    cameras = []
    for camera in request_cameras:
        updated = dict(camera)
        k = outcome.calibrations[camera["id"]].intrinsics
        updated.update(fx=float(k.fx), fy=float(k.fy))
        cameras.append(updated)
    return dict(accepted=True, reason="", intervals_px=outcome.intervals_px,
                initial_rmse_px=outcome.initial_rmse_px,
                fitted_rmse_px=outcome.fitted_rmse_px,
                fitted=result_record(outcome.sync_result, cameras))


def run(out: Path, names: tuple[str, ...], *, stop_on_positive_failure: bool = True) -> None:
    """Replay named cases in order, with no unreserved numerical attempts."""
    if any(name not in CASE_NAMES for name in names):
        raise ValueError("Unknown focal constraint case")
    metadata = _prepare_output(out)
    _core, sync = load_core()
    from match_perspective.core import lens_refine

    original_sync = lens_refine._run_sync
    original_bundle = lens_refine.fit_independent_focals
    with (ExperimentBudget(out / "sync-ledger.jsonl", metadata=metadata, **SYNC_LIMITS) as sync_budget,
          ExperimentBudget(out / "bundle-ledger.jsonl", metadata=metadata, **BUNDLE_LIMITS) as bundle_budget):
        for name in names:
            case = json.loads((CASE_ROOT / f"{name}.json").read_text())
            validate(case)
            request = case["request"]
            arguments = solver_arguments(request)
            match_inputs = []
            for camera in request["cameras"]:
                cal = calibration(camera)
                match_inputs.append(lens_refine.MatchLensInput(
                    camera["id"], {}, cal.intrinsics,
                    base_calibration=cal, freeze_focal=False))

            def ledgered_sync(*args, **kwargs):
                with sync_budget.attempt(f"{name}:inner-sync", request) as attempt:
                    result = original_sync(*args, **kwargs)
                    attempt.complete(_finite_json(result_record(result, request["cameras"])))
                return result

            def ledgered_bundle(*args, **kwargs):
                initial = args[2]
                trial_input = dict(request=request,
                                   initial=result_record(initial, request["cameras"]),
                                   pick_sigma_px=case["pick_sigma_px"])
                with bundle_budget.attempt(f"{name}:joint-bundle", trial_input) as attempt:
                    outcome = original_bundle(*args, **kwargs)
                    attempt.complete(_finite_json(_outcome_record(outcome, request["cameras"])))
                return outcome

            with (mock.patch.object(lens_refine, "_run_sync", side_effect=ledgered_sync),
                  mock.patch.object(lens_refine, "fit_independent_focals", side_effect=ledgered_bundle)):
                result = lens_refine.refine_lenses_from_landmarks(
                    match_inputs, arguments["observations"], anchor_id=request["anchor_id"],
                    share_lens=False, estimate_focal_from_points=True,
                    pick_sigma_px=case["pick_sigma_px"], fx_span=0.4,
                    plane_groups=arguments["plane_groups"], plane_slack=arguments["plane_slack"],
                    mirror_pairs=arguments["mirror_pairs"], mirror_plane=arguments["mirror_plane"],
                    mirror_slack=arguments["mirror_slack"])
            accepted = bool(result.improved and not result.refusal_reason)
            summary = dict(name=name, accepted=accepted, message=result.message,
                           refusal_reason=result.refusal_reason,
                           behavior=case["expectation"]["behavior"],
                           initial_sync_success=bool(result.sync_result.success),
                           initial_cost=result.initial_cost, final_cost=result.final_cost)
            if accepted:
                summary["focal_intervals_px"] = result.focal_intervals
                summary["focal_interval_truth_coverage"] = {
                    camera["id"]: bool(result.focal_intervals[camera["id"]][0] <= camera["fx"] <=
                                       result.focal_intervals[camera["id"]][1])
                    for camera in case["truth"]["cameras"]}
                fitted_cameras = []
                for camera in request["cameras"]:
                    updated = dict(camera)
                    updated.update(fx=result.calibrations[camera["id"]].intrinsics.fx,
                                   fy=result.calibrations[camera["id"]].intrinsics.fy)
                    fitted_cameras.append(updated)
                record = result_record(result.sync_result, fitted_cameras)
                summary["independent"] = assess(case, dict(
                    cameras=record["cameras"], landmarks=record["landmarks"]))
            (out / f"{name}-summary.json").write_text(
                json.dumps(_finite_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(_finite_json(summary), sort_keys=True), flush=True)
            if stop_on_positive_failure and not accepted and case["expectation"]["behavior"] == "reference_geometry":
                break


def _replay_initial(case: dict, source: Path):
    """Recover archived world poses as private-to-world similarities of scale one."""
    from match_perspective.core import sync

    name = case["name"]
    entries = [json.loads(line) for line in (source / "sync-ledger.jsonl").read_text().splitlines()]
    starts = [item for item in entries if item.get("kind") == "started" and
              item.get("label") == f"{name}:inner-sync"]
    if len(starts) != 1 or starts[0]["request"] != case["request"]:
        raise ValueError("Archived Sync request does not match the frozen case")
    completed = [item for item in entries if item.get("kind") == "completed" and
                 item.get("key") == starts[0]["key"]]
    if len(completed) != 1 or not completed[0]["result"]["success"]:
        raise ValueError("Archived Sync did not complete successfully")
    record = completed[0]["result"]
    similarities = {}
    for private in case["request"]["cameras"]:
        recovered = record["cameras"][private["id"]]
        private_rotation = np.asarray(private["rotation"], float)
        world_rotation = np.asarray(recovered["rotation"], float)
        rotation = world_rotation.T @ private_rotation
        translation = np.asarray(recovered["center"], float) - rotation @ np.asarray(
            private["center"], float)
        similarities[private["id"]] = sync.SimilarityTransform(1.0, rotation, translation)
    initial = sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(point, float) for key, point in record["landmarks"].items()},
        mean_reprojection_px=record["reported_rmse_px"], per_match_rmse_px={},
        per_landmark_rmse_px={}, message="archived successful Sync world state",
        success=True)
    return initial, record


def replay_bundles(out: Path, names: tuple[str, ...], source: Path) -> None:
    """Exercise final bundle source on saved Sync geometry, with no new Sync call."""
    load_core()
    from match_perspective.core import focal_bundle

    if any(name not in CASE_NAMES for name in names):
        raise ValueError("Unknown focal constraint case")
    metadata = _prepare_output(out)
    metadata["replay_sync_source"] = str(source.resolve())
    with ExperimentBudget(out / "bundle-ledger.jsonl", metadata=metadata,
                          **BUNDLE_LIMITS) as budget:
        for name in names:
            case = json.loads((CASE_ROOT / f"{name}.json").read_text())
            validate(case)
            request = case["request"]
            initial, archived_world = _replay_initial(case, source)
            calibrations = {item["id"]: calibration(item) for item in request["cameras"]}
            arguments = solver_arguments(request)
            trial_input = dict(request=request, archived_initial_world=archived_world,
                               reconstruction="private-to-world similarity scale one, exact archived world center/rotation",
                               pick_sigma_px=case["pick_sigma_px"])
            with budget.attempt(f"{name}:final-bundle-replay", trial_input) as attempt:
                outcome = focal_bundle.fit_independent_focals(
                    calibrations, arguments["observations"], initial,
                    anchor_id=request["anchor_id"], pick_sigma_px=case["pick_sigma_px"],
                    fx_span=0.4, plane_groups=arguments["plane_groups"],
                    plane_slack=arguments["plane_slack"],
                    mirror_pairs=arguments["mirror_pairs"],
                    mirror_plane=arguments["mirror_plane"],
                    mirror_slack=arguments["mirror_slack"])
                attempt.complete(_finite_json(_outcome_record(outcome, request["cameras"])))
            summary = dict(name=name, accepted=outcome.accepted, reason=outcome.reason,
                           initial_rmse_px=outcome.initial_rmse_px,
                           fitted_rmse_px=outcome.fitted_rmse_px)
            if outcome.accepted:
                record = _outcome_record(outcome, request["cameras"])["fitted"]
                summary["independent"] = assess(case, dict(
                    cameras=record["cameras"], landmarks=record["landmarks"]))
            (out / f"{name}-summary.json").write_text(
                json.dumps(_finite_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(_finite_json(summary), sort_keys=True), flush=True)
            if not outcome.accepted:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=INITIAL_CASES)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--replay-sync-from", type=Path,
                        help="Replay bundle only from this completed Sync ledger directory")
    options = parser.parse_args()
    if options.replay_sync_from:
        replay_bundles(options.out, tuple(options.cases), options.replay_sync_from)
    else:
        run(options.out, tuple(options.cases), stop_on_positive_failure=not options.continue_on_failure)
