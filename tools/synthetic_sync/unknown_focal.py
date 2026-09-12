"""Ledgered no-VP focal trials on frozen 2D-only cases.

Truth is used only by assess(), after the requested candidate has been solved.
Run each process under an outer timeout; the ledger caps inner Sync calls.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import environment, fingerprint, load_core, result_record, solve, solver_arguments


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASE_NAMES = (
    "no-vp-shared-guessedK", "no-vp-mixed-guessedK",
    "no-vp-weak-baseline-trueK", "no-vp-weak-baseline-guessedK",
    "no-vp-pure-rotation-trueK", "no-vp-pure-rotation-guessedK",
)
METADATA = dict(plan="unknown-focal-2026-09-12", cases=list(CASE_NAMES),
                max_calls=16, per_call_seconds=180, wall_seconds=900,
                truth_use="assessment only; never candidate selection")


def source_identity() -> dict:
    """Bind each candidate to the numerical code, harness, and runtime."""
    paths = sorted((ROOT / "core").rglob("*.py"))
    paths += [HERE / name for name in (
        "budget.py", "unknown_focal.py", "no_vp_bootstrap.py", "solver.py",
        "evaluation.py", "geometry.py", "scenarios.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    threads = {key: os.environ.get(key) for key in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return dict(source_sha256=digest.hexdigest(), environment=environment(ROOT),
                executable=sys.executable, python=platform.python_version(),
                numpy=np.__version__, opencv=opencv, threads=threads)


def trial_request(case: dict, scale: float) -> dict:
    request = deepcopy(case["request"])
    for camera in request["cameras"]:
        camera["fx"] *= scale
        camera["fy"] *= scale
    return request


def run(case_name: str, scale: float, out: Path) -> dict:
    if case_name not in CASE_NAMES or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Select a frozen case and positive finite common focal scale")
    case = read_case(HERE / "cases" / (case_name + ".json"))
    validate_fixture(case)
    request = trial_request(case, scale)
    trial = dict(request=request, source_runtime=source_identity(), scale=scale)
    label = f"{case_name}-scale-{scale:g}"
    out.mkdir(parents=True, exist_ok=True)
    with ExperimentBudget(out / "ledger.jsonl", metadata=METADATA,
                          max_calls=16, per_call_seconds=180, wall_seconds=900) as budget:
        record = budget.cached(label, trial)
        if record is None:
            with budget.attempt(label, trial) as attempt:
                record = solve(request)
                attempt.complete(record)
    evaluation_case = deepcopy(case)
    evaluation_case["request"] = request
    assessment = assess(evaluation_case, record)
    result = dict(label=label, case_file=str((HERE / "cases" / (case_name + ".json")).relative_to(ROOT)),
                  scale=scale, request_sha256=fingerprint(request), record=record,
                  assessment=assessment)
    (out / (label + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return dict(label=label, classification=assessment["classification"],
                success=record["success"], message=record["message"],
                rmse_px=record["reported_rmse_px"], elapsed_s=record["elapsed_s"],
                focal_relative_error=assessment["focal_relative_error"],
                violations=assessment["independent_geometry"]["violations"])


def shared_search(out: Path) -> dict:
    """Exercise the actual Same Lens API with five budgeted inner solves."""
    load_core()
    from match_perspective.core import lens_refine

    case = read_case(HERE / "cases" / "no-vp-shared-guessedK.json")
    validate_fixture(case)
    arguments = solver_arguments(case["request"])
    matches = arguments.pop("matches")
    inputs = [lens_refine.MatchLensInput(match.match_id, {}, match.calibration.intrinsics,
              base_calibration=match.calibration) for match in matches]
    search = dict(share_lens=True, fx_span=0.25, coarse_samples=3, refine_samples=3)
    assert lens_refine.estimate_refine_evaluation_count(
        len(inputs), share_lens=True, coarse_samples=3, refine_samples=3) == 5
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "actual-shared-search.json"
    if output_path.exists():
        raise FileExistsError("The live search must not silently rerun its inner solves")
    source = source_identity()
    trials = []
    original = lens_refine._run_sync

    with ExperimentBudget(out / "ledger.jsonl", metadata=METADATA,
                          max_calls=16, per_call_seconds=180, wall_seconds=900) as budget:
        def observe(calibrations, *args, **kwargs):
            request = deepcopy(case["request"])
            for camera in request["cameras"]:
                calibration = calibrations[camera["id"]]
                camera.update(fx=float(calibration.intrinsics.fx),
                              fy=float(calibration.intrinsics.fy))
            warm = args[7] if len(args) > 7 else None
            warm_record = {key: dict(scale=float(value.scale),
                           rotation=np.asarray(value.rotation).tolist(),
                           translation=np.asarray(value.translation).tolist())
                           for key, value in (warm or {}).items()}
            trial_key = dict(request=request, initial_similarities=warm_record,
                             source_runtime=source, search=search)
            index = len(trials)
            with budget.attempt(f"actual-shared-search-{index}", trial_key) as attempt:
                started = time.perf_counter()
                numerical = original(calibrations, *args, **kwargs)
                record = result_record(numerical, request["cameras"])
                record["elapsed_s"] = time.perf_counter() - started
                attempt.complete(record)
            evaluation_case = deepcopy(case)
            evaluation_case["request"] = request
            assessment = assess(evaluation_case, record)
            trials.append(dict(index=index, scale=request["cameras"][0]["fx"] /
                               case["request"]["cameras"][0]["fx"],
                               request_sha256=fingerprint(request),
                               warm_start=bool(warm_record), record=record,
                               assessment=assessment, numerical=numerical))
            return numerical

        with patch.object(lens_refine, "_run_sync", observe):
            result = lens_refine.refine_lenses_from_landmarks(
                inputs, **arguments, **search)
    selected = next(row["index"] for row in trials if row["numerical"] is result.sync_result)
    serializable = [{key: value for key, value in row.items() if key != "numerical"}
                    for row in trials]
    report = dict(search=search, selected_index=selected,
                  initial_cost=result.initial_cost, final_cost=result.final_cost,
                  improved=result.improved, message=result.message,
                  trials=serializable)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return dict(selected_index=selected, improved=result.improved,
                message=result.message, initial_cost=result.initial_cost,
                final_cost=result.final_cost,
                trials=[dict(scale=row["scale"], success=row["record"]["success"],
                             fitted_rmse_px=row["record"]["reported_rmse_px"],
                             classification=row["assessment"]["classification"],
                             focal_error=row["assessment"]["focal_relative_error"])
                        for row in trials])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-search", action="store_true")
    args = parser.parse_args()
    if not args.shared_search and args.case is None:
        parser.error("--case is required for an individual Sync trial")
    result = shared_search(args.out) if args.shared_search else run(args.case, args.scale, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
