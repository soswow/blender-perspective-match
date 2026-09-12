"""Frozen noisy-pick, three-start independent-focal experiment.

The new request is solved through ordinary Sync before focal BA. Truth is only
read after a candidate and its input-only diagnostics have been persisted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.independent_focal import CASES, input_only, optimize, source_identity
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import fingerprint, solve


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "cases" / "independent-focal-noise"
SIGMA_PX = 0.5
SEEDS = {name: 20260912 + i for i, name in enumerate(CASES)}
START_SCALES = (0.7, 1.0, 1.3)
LIMITS = dict(per_case_residual=10000, per_case_jacobian=300,
              per_case_seconds=30., total_residual=120000,
              total_jacobian=3600, total_seconds=300.)
MAX_RUNS = 12
MAX_NFEV = 300
INNER_METADATA = dict(plan="independent-focal-noise-2026-09-12",
                      names=list(CASES), sigma_px=SIGMA_PX, seeds=SEEDS,
                      purpose="fresh noisy-pick startup only")


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def noise_source() -> dict:
    identity = source_identity()
    digest = hashlib.sha256()
    digest.update(identity["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    identity["source_sha256"] = digest.hexdigest()
    return identity


def frozen_case(name: str) -> dict:
    original = read_case(HERE / "cases" / f"{name}.json")
    validate_fixture(original)
    case = deepcopy(original)
    rng = np.random.default_rng(SEEDS[name])
    for observation in case["request"]["observations"]:
        observation["u"] += float(rng.normal(0., SIGMA_PX))
        observation["v"] += float(rng.normal(0., SIGMA_PX))
    case.update(name=f"{name}-noise05", noise_px=SIGMA_PX)
    case["source"] = dict(case["source"], independent_coordinate_noise=dict(
        distribution="normal", sigma_px=SIGMA_PX, seed=SEEDS[name]))
    validate_fixture(case)
    return case


def freeze() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    manifest = dict(names=list(CASES), seeds=SEEDS, sigma_px=SIGMA_PX,
                    start_scales=START_SCALES, limits=LIMITS, max_runs=MAX_RUNS,
                    max_nfev=MAX_NFEV, inner_metadata=INNER_METADATA,
                    source_runtime=noise_source(), cases={})
    for name in CASES:
        case = frozen_case(name)
        path = CORPUS / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        manifest["cases"][name] = dict(
            original_request_sha256=fingerprint(read_case(HERE / "cases" / f"{name}.json")["request"]),
            noisy_request_sha256=fingerprint(case["request"]),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def read_frozen(name: str) -> tuple[dict, dict]:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    path = CORPUS / f"{name}.json"
    case = read_case(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["cases"][name]["file_sha256"]
    assert fingerprint(case["request"]) == manifest["cases"][name]["noisy_request_sha256"]
    assert case == frozen_case(name)
    return case, manifest


def initialize() -> None:
    source = noise_source()
    with ExperimentBudget(CORPUS / "sync-ledger.jsonl", metadata=INNER_METADATA,
                          max_calls=6, per_call_seconds=120, wall_seconds=600) as budget:
        for name in CASES:
            case, _ = read_frozen(name)
            request = case["request"]
            trial = dict(request=request, source_runtime=source)
            with budget.attempt(name, trial) as attempt:
                record = solve(request)
                attempt.complete(record)
            output = CORPUS / f"{name}-initial.json"
            if output.exists():
                raise FileExistsError(output)
            output.write_text(json.dumps(dict(name=name, request_sha256=fingerprint(request),
                record=record, assessment=assess(case, record)), indent=2, allow_nan=False) + "\n")


def fit() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    if tuple(manifest["start_scales"]) != START_SCALES or manifest["limits"] != LIMITS:
        raise ValueError("The frozen optimizer plan changed")
    total = dict(residual=0, jacobian=0, started=time.monotonic())
    source = noise_source()
    ledger = CORPUS / "optimizer-ledger.jsonl"
    if ledger.exists():
        raise FileExistsError("A separate output directory is needed for another experiment")
    rows = []
    with ledger.open("x") as log:
        for name in CASES:
            case, _ = read_frozen(name)
            initial = json.loads((CORPUS / f"{name}-initial.json").read_text())
            request, baseline = input_only(case["request"], initial["record"])
            if initial["request_sha256"] != fingerprint(request) or baseline["request_sha256"] != fingerprint(request):
                raise ValueError("Fresh initializer request changed")
            for scale in START_SCALES:
                label = f"{name}-start-{scale:g}"
                trial = dict(name=name, scale=scale, request=request, baseline=baseline,
                             source_runtime=source, limits=LIMITS, max_nfev=MAX_NFEV)
                log.write(canonical(dict(kind="started", label=label, trial=trial)) + "\n")
                log.flush()
                os.fsync(log.fileno())
                try:
                    result = optimize(request, baseline, total, initial_focal_scale=scale,
                                      max_nfev=MAX_NFEV, limits=LIMITS)
                except BaseException as exc:
                    log.write(canonical(dict(kind="failed", label=label,
                        error_type=type(exc).__name__, error=str(exc))) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    raise
                log.write(canonical(dict(kind="completed", label=label, result=result,
                    total_work={k: v for k, v in total.items() if k != "started"})) + "\n")
                log.flush()
                os.fsync(log.fileno())
                # Independent truth enters only after the input-only result is fixed.
                assessment = assess(case, result["record"])
                report = dict(label=label, name=name, start_scale=scale,
                    request_sha256=fingerprint(request), source_runtime=source,
                    limits=LIMITS, max_nfev=MAX_NFEV, **result, assessment=assessment)
                (CORPUS / f"{label}.json").write_text(json.dumps(report, indent=2,
                    allow_nan=False) + "\n")
                rows.append(dict(label=label, success=result["record"]["success"],
                    rmse_px=result["record"]["reported_rmse_px"],
                    focal_error=assessment["focal_relative_error"],
                    withheld_passed=assessment["independent_geometry"]["passed"],
                    work=result["optimizer"]))
    (CORPUS / "summary.json").write_text(json.dumps(dict(rows=rows, limits=LIMITS,
        max_runs=MAX_RUNS, source_runtime=source,
        total_work={k: v for k, v in total.items() if k != "started"}),
        indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2, allow_nan=False))


def diagnose() -> None:
    """Measure local focal sensitivity at saved states without optimizing."""
    from tools.synthetic_sync.independent_focal_sensitivity import diagnose as local_sensitivity

    output = CORPUS / "diagnostics.json"
    ledger = CORPUS / "diagnostics-ledger.jsonl"
    if output.exists() or ledger.exists():
        raise FileExistsError("Read-only sensitivity cannot be silently rerun")
    source = noise_source()
    rows = []
    start = time.monotonic()
    with ledger.open("x") as log:
        for name in CASES:
            case, _ = read_frozen(name)
            baseline = json.loads((CORPUS / f"{name}-initial.json").read_text())["record"]
            request, baseline = input_only(case["request"], baseline)
            for scale in START_SCALES:
                label = f"{name}-start-{scale:g}"
                candidate = json.loads((CORPUS / f"{label}.json").read_text())["record"]
                trial = dict(label=label, request=request, baseline=baseline,
                             candidate=candidate, source_runtime=source,
                             assumed_coordinate_sigma_px=SIGMA_PX)
                log.write(canonical(dict(kind="started", trial=trial)) + "\n")
                log.flush()
                os.fsync(log.fileno())
                counters = dict(residual=0, jacobian=0, started=time.monotonic())
                result = local_sensitivity(request, baseline, candidate, counters)
                rows.append(dict(label=label, **result))
                log.write(canonical(dict(kind="completed", label=label,
                    result=result, work={k: v for k, v in counters.items() if k != "started"})) + "\n")
                log.flush()
                os.fsync(log.fileno())
                if time.monotonic()-start > 30.:
                    raise TimeoutError("Global read-only diagnostic deadline exceeded")
    output.write_text(json.dumps(dict(rows=rows, source_runtime=source,
        elapsed_s=time.monotonic()-start), indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "initialize", "fit", "diagnose"))
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "initialize":
        initialize()
    elif args.stage == "diagnose":
        diagnose()
    else:
        signal.signal(signal.SIGALRM, lambda *_: sys.exit("Outer optimizer deadline exceeded"))
        signal.setitimer(signal.ITIMER_REAL, 360.)
        try:
            fit()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
