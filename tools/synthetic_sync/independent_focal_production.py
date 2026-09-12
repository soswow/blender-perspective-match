"""Replay frozen point-focal cases through fresh Sync and the NumPy bundle fit.

Every numerical call is reserved with exact input and source bytes before use.
Truth is read only after both the acceptance decision and result are captured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import environment, fingerprint, load_core, result_record, solver_arguments


ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).resolve().parent / "cases" / "independent-focal-production"
CASES = ("no-vp-mixed-guessedK", "no-vp-shared-guessedK", "noisy-mixed",
         "noisy-shared", "four-view", "no-vp-weak-baseline-guessedK",
         "no-vp-pure-rotation-guessedK", "planar")
SOURCE_FILES = ("core/focal_bundle.py", "core/lens_refine.py",
                "core/sync/pose.py", "core/sync/ba.py",
                "tools/synthetic_sync/independent_focal_production.py")


def _sync_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "core").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _snapshot_sources(out: Path) -> None:
    source = out / "source"
    source.mkdir(parents=True, exist_ok=False)
    for relative in SOURCE_FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def _record_fit(case: dict, initial, *, sigma: float) -> dict:
    from match_perspective.core.focal_bundle import fit_independent_focals
    arguments = solver_arguments(case["request"])
    calibrations = {item.match_id: item.calibration for item in arguments["matches"]}
    result = fit_independent_focals(calibrations, arguments["observations"], initial,
                                    anchor_id=arguments["anchor_id"], pick_sigma_px=sigma)
    record = (result_record(result.sync_result, case["request"]["cameras"],
                            calibrations=result.calibrations) if result.accepted else
        dict(success=False, message=result.reason, cameras={}, landmarks={},
             reported_rmse_px=result.fitted_rmse_px))
    return dict(accepted=result.accepted, reason=result.reason,
                intervals_px=result.intervals_px,
                initial_rmse_px=(result.initial_rmse_px if
                                 math.isfinite(result.initial_rmse_px) else None),
                fitted_rmse_px=(result.fitted_rmse_px if math.isfinite(result.fitted_rmse_px)
                                else None), record=dict(record,
                    reported_rmse_px=(record["reported_rmse_px"] if
                                       math.isfinite(record["reported_rmse_px"]) else None)))


def run(out: Path, selected: tuple[str, ...], *, sigma: float) -> None:
    _core, sync = load_core()
    out.mkdir(parents=True, exist_ok=False)
    _snapshot_sources(out)
    source = dict(sync_source_sha256=_sync_fingerprint(), environment=environment(ROOT),
                  executable=sys.executable, sigma_px=sigma,
                  cases=list(selected), numerical_limits=dict(
                      max_inner_sync=12, per_sync_seconds=120,
                      cumulative_sync_seconds=600, max_bundle=12,
                      max_bundle_iterations=100, max_bundle_seconds=30,
                      max_total_bundle_seconds=300, outer_seconds=420))
    (out / "source-runtime.json").write_text(json.dumps(source, indent=2,
                                                       allow_nan=False) + "\n")
    bundle_ledger = out / "bundle-ledger.jsonl"
    bundle_seconds = 0.0
    bundle_count = 0
    with ExperimentBudget(out / "sync-ledger.jsonl", metadata=source,
                              max_calls=12, per_call_seconds=120,
                              wall_seconds=600) as budget, bundle_ledger.open("x") as log:
            for name in selected:
                case = read_case(FROZEN / f"{name}.json")
                request = case["request"]
                arguments = solver_arguments(request)
                with budget.attempt(name, request) as attempt:
                    initial = sync.solve_landmark_sync(**arguments)
                    attempt.complete(result_record(initial, request["cameras"]))
                if bundle_count >= 12 or bundle_seconds >= 300.0:
                    raise RuntimeError("Bundle fit budget exhausted")
                trial = dict(name=name, request=request,
                             initial=result_record(initial, request["cameras"]),
                             request_sha256=fingerprint(request), source_runtime=source)
                log.write(json.dumps(dict(kind="started", trial=trial),
                                     sort_keys=True, allow_nan=False) + "\n")
                log.flush(); os.fsync(log.fileno())
                bundle_count += 1
                start = time.monotonic()
                try:
                    fitted = _record_fit(case, initial, sigma=sigma)
                except BaseException as exc:
                    log.write(json.dumps(dict(kind="failed", name=name,
                                              exception=type(exc).__name__, detail=str(exc))) + "\n")
                    log.flush(); os.fsync(log.fileno())
                    raise
                elapsed = time.monotonic() - start
                bundle_seconds += elapsed
                log.write(json.dumps(dict(kind="completed", name=name, fitted=fitted,
                                          elapsed_s=elapsed, total_bundle_seconds=bundle_seconds),
                                     sort_keys=True, allow_nan=False) + "\n")
                log.flush(); os.fsync(log.fileno())
                # The oracle enters only after the product decision was recorded.
                assessment = assess(case, fitted["record"]) if name not in ("planar", "four-view") else None
                report = dict(name=name, request_sha256=fingerprint(request), fitted=fitted,
                              assessment=assessment, elapsed_s=elapsed)
                (out / f"{name}.json").write_text(json.dumps(report, indent=2,
                                                           allow_nan=False) + "\n")
                print(name, fitted["accepted"], fitted["reason"],
                      round(elapsed, 3), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    run(args.out, tuple(args.case or CASES), sigma=args.sigma)


if __name__ == "__main__":
    main()
