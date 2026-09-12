"""One frozen dense-TRF control of sparse focal convergence on noisy requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.independent_focal import CASES, input_only, optimize
from tools.synthetic_sync.independent_focal_noise import CORPUS, noise_source, read_frozen
from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.solver import fingerprint


LIMITS = dict(per_case_residual=30000, per_case_jacobian=300,
              per_case_seconds=30., total_residual=120000,
              total_jacobian=1200, total_seconds=120.)
MAX_NFEV = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.diagnose:
        diagnose(args.out)
        return
    ledger = args.out / "optimizer-ledger.jsonl"
    if ledger.exists():
        raise FileExistsError("Dense trial cannot be silently rerun")
    source = noise_source()
    digest = hashlib.sha256()
    digest.update(source["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    source["source_sha256"] = digest.hexdigest()
    total = dict(residual=0, jacobian=0, started=time.monotonic())
    rows = []
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("Outer dense deadline exceeded"))
    signal.setitimer(signal.ITIMER_REAL, 180.)
    try:
        with ledger.open("x") as log:
            for name in CASES:
                case, _ = read_frozen(name)
                initial = json.loads((CORPUS / f"{name}-initial.json").read_text())
                request, baseline = input_only(case["request"], initial["record"])
                if initial["request_sha256"] != fingerprint(request) or baseline["request_sha256"] != fingerprint(request):
                    raise ValueError("Noisy fresh initializer changed")
                label = name + "-dense-exact-start-1"
                trial = dict(label=label, request=request, baseline=baseline,
                    start_scale=1., backend="dense-finite-difference-TRF-exact",
                    max_nfev=MAX_NFEV, limits=LIMITS, source_runtime=source)
                log.write(json.dumps(dict(kind="started", trial=trial), sort_keys=True,
                    allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                try:
                    fitted = optimize(request, baseline, total, initial_focal_scale=1.,
                        max_nfev=MAX_NFEV, limits=LIMITS, dense_exact=True)
                except BaseException as exc:
                    log.write(json.dumps(dict(kind="failed", label=label,
                        error_type=type(exc).__name__, error=str(exc)), allow_nan=False) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    raise
                log.write(json.dumps(dict(kind="completed", label=label, result=fitted,
                    total_work={k: v for k, v in total.items() if k != "started"}),
                    sort_keys=True, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                assessment = assess(case, fitted["record"])
                sparse = json.loads((CORPUS / f"{name}-start-1.json").read_text())
                report = dict(label=label, request_sha256=fingerprint(request),
                    source_runtime=source, limits=LIMITS, max_nfev=MAX_NFEV,
                    **fitted, assessment=assessment,
                    sparse_control=dict(label=sparse["label"],
                        fitted_rmse_px=sparse["record"]["reported_rmse_px"],
                        converged=sparse["record"]["success"]))
                (args.out / f"{label}.json").write_text(json.dumps(report, indent=2,
                    allow_nan=False) + "\n")
                rows.append(dict(label=label, success=fitted["record"]["success"],
                    rmse_px=fitted["record"]["reported_rmse_px"],
                    sparse_rmse_px=sparse["record"]["reported_rmse_px"],
                    focal_error=assessment["focal_relative_error"],
                    withheld_passed=assessment["independent_geometry"]["passed"],
                    work=fitted["optimizer"]))
        (args.out / "summary.json").write_text(json.dumps(dict(rows=rows,
            total_work={k: v for k, v in total.items() if k != "started"},
            limits=LIMITS, source_runtime=source), indent=2, allow_nan=False) + "\n")
        print(json.dumps(rows, indent=2, allow_nan=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def diagnose(out: Path) -> None:
    """Read-only local focal uncertainty at the saved dense candidates."""
    from tools.synthetic_sync.independent_focal_sensitivity import diagnose as sensitivity

    output = out / "diagnostics.json"
    ledger = out / "diagnostics-ledger.jsonl"
    if output.exists() or ledger.exists():
        raise FileExistsError("Dense candidate diagnostics cannot be silently rerun")
    source = noise_source()
    digest = hashlib.sha256()
    digest.update(source["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    source["source_sha256"] = digest.hexdigest()
    rows = []
    with ledger.open("x") as log:
        for name in CASES:
            case, _ = read_frozen(name)
            initial = json.loads((CORPUS / f"{name}-initial.json").read_text())
            request, baseline = input_only(case["request"], initial["record"])
            candidate = json.loads((out / f"{name}-dense-exact-start-1.json").read_text())["record"]
            trial = dict(name=name, request=request, baseline=baseline,
                         candidate=candidate, source_runtime=source,
                         assumed_coordinate_sigma_px=0.5)
            log.write(json.dumps(dict(kind="started", trial=trial), sort_keys=True,
                allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            counters = dict(residual=0, jacobian=0, started=time.monotonic())
            result = sensitivity(request, baseline, candidate, counters)
            log.write(json.dumps(dict(kind="completed", name=name, result=result,
                work={k: v for k, v in counters.items() if k != "started"}),
                sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            rows.append(dict(name=name, **result))
    output.write_text(json.dumps(dict(rows=rows, source_runtime=source),
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
