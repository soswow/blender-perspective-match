"""One bounded truth-focal control for an archived noisy continuation fixture."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.solver import environment, load_core, result_record
from tools.synthetic_sync.sync_continuation import (
    _assess_sequence_case, _finite_json, _sources,
)


def run(source: Path, out: Path, *, active_seconds: float,
        replacement_of: Path | None = None) -> None:
    """Hold the generated picks/relations fixed and optimize at oracle focal lengths."""
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Numerical archive must stay outside the repository")
    out.mkdir(parents=True, exist_ok=False)
    source_hash = _sources(out)
    script_copy = out / "source" / "tools" / "synthetic_sync" / Path(__file__).name
    script_copy.write_bytes(Path(__file__).read_bytes())
    source_hash = hashlib.sha256(
        (source_hash + hashlib.sha256(script_copy.read_bytes()).hexdigest()).encode()
    ).hexdigest()
    load_core()
    from match_perspective.core import focal_bundle
    from match_perspective.core.joint_fit_score import JointFitScorer, supported_joint_request
    from match_perspective.core.sync.request import SyncSolveRequest, json_values
    from match_perspective.core.sync.solve import solution_result_from_seed

    case = json.loads((source / "joint-case.json").read_text())
    report = json.loads((source / "refine-result.json").read_text())
    request = SyncSolveRequest.from_record(report["post_request"])
    seed = request.initial_solution
    if (seed is None or seed.diagnostics is None or
            seed.evidence_sha256 != request.evidence_sha256()):
        raise ValueError("Source Refine endpoint is not certified")
    initial = solution_result_from_seed(seed)
    if initial is None:
        raise ValueError("Source Refine endpoint cannot be reconstructed")
    truth = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    if set(truth) != {item.match_id for item in request.matches}:
        raise ValueError("Truth and fitted camera sets differ")
    fitted_focals = {}
    for match in request.matches:
        key = match.match_id
        fitted_focals[key] = [match.calibration.intrinsics.fx,
                              match.calibration.intrinsics.fy]
        match.calibration.intrinsics.fx = float(truth[key]["fx"])
        match.calibration.intrinsics.fy = float(truth[key]["fy"])
        initial.calibrations[key] = deepcopy(match.calibration)
    request.initial_solution = None
    supported, coverage = supported_joint_request(request, initial)
    frozen = seed.diagnostics.joint_point_weights
    if frozen is None and supported.observations:
        raise ValueError("Source Refine effective point weights are absent")
    scorer = JointFitScorer(
        supported, initial, calibrations=initial.calibrations,
        frozen_point_weights=frozen,
    )
    before = scorer.score(initial, calibrations=initial.calibrations)
    if not before.valid or scorer.weight_refusal:
        raise ValueError(f"Truth-focal start cannot score: {before.reason or scorer.weight_refusal}")
    metadata = dict(
        case="paired-truth-focal-control", source_sha256=source_hash,
        source_archive=str(source.resolve()), source_refine_sha256=hashlib.sha256(
            (source / "refine-result.json").read_bytes()).hexdigest(),
        replacement_of=(str(replacement_of.resolve()) if replacement_of else None),
        replacement_reason=("Prior wrapper-only record omitted the inner bundle endpoint"
                            if replacement_of else None),
        max_sync_calls=0, max_bundle_calls=1, environment=environment(ROOT),
    )
    (out / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    (out / "request.json").write_text(json.dumps(request.to_record(), indent=2) + "\n")
    (out / "start.json").write_text(json.dumps(_finite_json(json_values(dict(
        initial=initial, fitted_focals=fitted_focals,
        truth_focals={key: [value["fx"], value["fy"]] for key, value in truth.items()},
        before=before, coverage=coverage, frozen_weights=frozen,
    ))), indent=2) + "\n")
    raw_runtime = []
    raw_record = {}
    original_fit = focal_bundle.fit_independent_focals

    def capture_inner(*args, **kwargs):
        if raw_runtime:
            raise RuntimeError("Truth-focal control must make exactly one inner fit")
        raw_record["inputs"] = _finite_json(json_values(dict(
            args=args, kwargs={key: value for key, value in kwargs.items()
                               if not callable(value)},
            callbacks={key: True for key, value in kwargs.items()
                       if callable(value)},
        )))
        inner = original_fit(*args, **kwargs)
        raw_runtime.append(inner)
        raw_record["outcome"] = _finite_json(json_values(inner))
        return inner

    with ExperimentBudget(
        out / "ledger.jsonl", metadata=metadata, max_calls=1,
        wall_seconds=active_seconds, per_call_seconds=180,
    ) as budget:
        with budget.attempt("bundle:truth-focal:1", dict(
            request=request.to_record(), initial=_finite_json(json_values(initial)),
            frozen_weights=frozen,
        )) as attempt:
            with patch.object(focal_bundle, "fit_independent_focals", capture_inner):
                outcome = focal_bundle.refine_fixed_focals(
                    request, initial, frozen_point_weights=frozen,
                )
            attempt.complete(dict(wrapper=_finite_json(json_values(outcome)),
                                  raw_fit=raw_record))
    result = dict(
        accepted=outcome.accepted, reason=outcome.reason,
        initial_objective=outcome.initial_objective,
        fitted_objective=outcome.fitted_objective,
        initial_point_rmse_px=before.point_rmse_px,
        initial_line_rmse_px=before.line_rmse_px,
        coverage=coverage,
    )
    if outcome.accepted and outcome.sync_result is not None:
        final = scorer.score(outcome.sync_result, calibrations=outcome.calibrations)
        record = result_record(
            outcome.sync_result, case["truth"]["cameras"],
            calibrations=outcome.calibrations,
        )
        result.update(final_score=_finite_json(json_values(final)),
                      recovered=record,
                      assessment=_assess_sequence_case(case, record))
    if raw_runtime and raw_runtime[0].accepted and raw_runtime[0].sync_result is not None:
        raw = raw_runtime[0]
        raw_score = scorer.score(raw.sync_result, calibrations=raw.calibrations)
        raw_recovered = result_record(
            raw.sync_result, case["truth"]["cameras"],
            calibrations=raw.calibrations,
        )
        result["raw_inner"] = dict(
            accepted=True, public_score=_finite_json(json_values(raw_score)),
            recovered=raw_recovered,
            assessment=_assess_sequence_case(case, raw_recovered),
        )
    (out / "result.json").write_text(json.dumps(_finite_json(result), indent=2) + "\n")
    print(json.dumps(dict(
        accepted=result["accepted"], reason=result["reason"],
        initial_objective=result["initial_objective"],
        fitted_objective=result["fitted_objective"],
        withheld_rmse_px=(result.get("assessment") or {}).get("withheld_rmse_px"),
        accuracy_flags=(result.get("assessment") or {}).get("accuracy_flags"),
        raw_inner_accepted=(result.get("raw_inner") or {}).get("accepted"),
        raw_public_objective=((result.get("raw_inner") or {}).get("public_score") or {}).get("objective"),
        raw_withheld_rmse_px=((result.get("raw_inner") or {}).get("assessment") or {}).get("withheld_rmse_px"),
    ), indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--active-seconds", type=float, required=True)
    parser.add_argument("--outer-seconds", type=float, default=180.0)
    parser.add_argument("--replacement-of", type=Path)
    args = parser.parse_args()
    if os.environ.get("PM_KNOWN_FOCAL_CHILD") != "1":
        env = dict(os.environ, PM_KNOWN_FOCAL_CHILD="1")
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=args.outer_seconds, check=False, env=env,
            )
            print(process.stdout, end="")
            exit_code = process.returncode
        except subprocess.TimeoutExpired as error:
            print(error.stdout or "", end="")
            exit_code = 124
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "process-exit.json").write_text(json.dumps(dict(
            exit_code=exit_code, elapsed_s=time.monotonic() - started,
            outer_seconds=args.outer_seconds,
        ), indent=2) + "\n")
        raise SystemExit(exit_code)
    run(args.source, args.out, active_seconds=args.active_seconds,
        replacement_of=args.replacement_of)


if __name__ == "__main__":
    main()
