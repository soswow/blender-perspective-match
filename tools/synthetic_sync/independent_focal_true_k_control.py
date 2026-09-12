"""Two fresh noisy-pick Sync controls with saved calibrated intrinsics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.independent_focal_noise import CORPUS, noise_source, read_frozen
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import fingerprint, solve


HERE = Path(__file__).resolve().parent
OUT = CORPUS / "true-k-controls"
NAMES = ("no-vp-mixed-guessedK", "no-vp-shared-guessedK")
METADATA = dict(plan="noisy-true-K-startup-2026-09-12", names=list(NAMES),
                max_calls=2, per_call_seconds=120, wall_seconds=240,
                truth_use="calibrated K is an explicit experimental input; oracle assesses only")


def source_identity() -> dict:
    source = noise_source()
    digest = hashlib.sha256()
    digest.update(source["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    source["source_sha256"] = digest.hexdigest()
    return source


def control_case(name: str) -> dict:
    noisy, _ = read_frozen(name)
    true_name = name.replace("guessedK", "trueK")
    calibrated = read_case(HERE / "cases" / f"{true_name}.json")
    case = deepcopy(noisy)
    for stored, known in zip(case["request"]["cameras"], calibrated["request"]["cameras"]):
        assert stored["id"] == known["id"]
        assert stored["center"] == known["center"] and stored["rotation"] == known["rotation"]
        stored["fx"], stored["fy"] = known["fx"], known["fy"]
    assert [(o["match_id"], o["landmark_id"]) for o in case["request"]["observations"]] == [
        (o["match_id"], o["landmark_id"]) for o in calibrated["request"]["observations"]]
    case["name"] = name.replace("guessedK", "trueK") + "-noise05"
    case["source"] = dict(case["source"], focal_input="saved true-K request, not solver output")
    validate_fixture(case)
    return case


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata=METADATA, source_runtime=source_identity(), cases={})
    for name in NAMES:
        case = control_case(name)
        path = OUT / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        manifest["cases"][name] = dict(request_sha256=fingerprint(case["request"]),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def run() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["metadata"] != METADATA:
        raise ValueError("Frozen control plan changed")
    source = source_identity()
    with ExperimentBudget(OUT / "sync-ledger.jsonl", metadata=METADATA,
                          max_calls=2, per_call_seconds=120, wall_seconds=240) as budget:
        for name in NAMES:
            path = OUT / f"{name}.json"
            case = read_case(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["cases"][name]["file_sha256"]
            request = case["request"]
            assert fingerprint(request) == manifest["cases"][name]["request_sha256"]
            with budget.attempt(name, dict(request=request, source_runtime=source)) as attempt:
                record = solve(request)
                attempt.complete(record)
            assessment = assess(case, record)
            (OUT / f"{name}-result.json").write_text(json.dumps(dict(name=name,
                request_sha256=fingerprint(request), record=record,
                assessment=assessment), indent=2, allow_nan=False) + "\n")
            print(json.dumps(dict(name=name, success=record["success"],
                rmse_px=record["reported_rmse_px"],
                classification=assessment["classification"]), allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else run)()


if __name__ == "__main__":
    main()
