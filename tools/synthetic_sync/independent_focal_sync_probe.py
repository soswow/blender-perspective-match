"""Ledgered stage traces for the calibrated noisy no-VP Sync failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.independent_focal_noise import CORPUS, noise_source
from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.no_vp_startup_trace import traced_solve
from tools.synthetic_sync.scenarios import read_case


OUT = CORPUS / "sync-investigation"
METADATA = dict(plan="noisy-calibrated-startup-fix-2026-09-12",
                max_calls=8, per_call_seconds=120, wall_seconds=600,
                outer_process_seconds=720)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mixed", "shared"), required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    name = "no-vp-" + args.case + "-guessedK"
    case = read_case(CORPUS / "true-k-controls" / f"{name}.json")
    source = noise_source()
    digest = hashlib.sha256()
    digest.update(source["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    digest.update((Path(__file__).parent / "no_vp_startup_trace.py").read_bytes())
    source["source_sha256"] = digest.hexdigest()
    trial = dict(label=args.label, request=case["request"], source_runtime=source)
    with ExperimentBudget(OUT / "ledger.jsonl", metadata=METADATA,
                          max_calls=8, per_call_seconds=120, wall_seconds=600) as budget:
        with budget.attempt(args.label, trial) as attempt:
            result = traced_solve(case["request"])
            attempt.complete(result)
    result["assessment"] = assess(case, result["record"])
    output = OUT / f"{args.label}.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(label=args.label, output=str(output),
        success=result["record"]["success"],
        rmse_px=result["record"]["reported_rmse_px"],
        stages=[dict(stage=row["stage"], n=row.get("n"),
            rmse_px=row.get("rmse_px"), active=row.get("active"),
            skipped=row.get("skipped")) for row in result["trace"]]),
        indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
