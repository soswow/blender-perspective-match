"""Shrink a forbidden-geometry regression against separate old/fixed checkouts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.scenarios import read_case, write_case
from tools.synthetic_sync.solver import environment, fingerprint, solve


def removable_points(case):
    """Preserve all cameras, references, constraints, expected features and truth."""
    request, expectation = case["request"], case["expectation"]
    protected = set(expectation.get("required_points", [])) | set(expectation.get("excluded_points", []))
    protected.update(key for pair in request["mirror_pairs"] for key in pair)
    protected.update(item[0] for item in request.get("plane_groups") or [])
    return [point["id"] for point in request["points"]
            if not point["ground"] and point["known"] is None and point["id"] not in protected]


def drop_points(case, names):
    """Remove whole optional landmarks and their observations, without changing truth."""
    names = set(names)
    if not names <= set(removable_points(case)):
        raise ValueError("Cannot remove a protected or absent landmark")
    candidate = deepcopy(case)
    for field, key in (("points", "id"), ("observations", "landmark_id")):
        candidate["request"][field] = [item for item in candidate["request"][field] if item[key] not in names]
    return candidate


def forbidden_geometry(case, result):
    """Identify forbidden output only when every other accuracy contract passes."""
    if not result["success"] or result.get("exception"):
        return None
    signature = {kind: sorted(set(case["expectation"].get("excluded_" + kind, [])) & set(result[field]))
                 for kind, field in (("points", "landmarks"), ("lines", "line_segments"))}
    if not any(signature.values()):
        return None
    control = deepcopy(case)
    control["expectation"].update(excluded_points=[], excluded_lines=[])
    return signature if evaluate(control, result)["passed"] else None


def reduce_case(case, accepts, *, max_attempts=40):
    """Try deterministic chunk deletions; stop at the budget or single-point limit."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    current, trace, granularity = deepcopy(case), [], 2
    while (remaining := removable_points(current)) and len(trace) < max_attempts:
        size = max(1, (len(remaining) + granularity - 1) // granularity)
        accepted = False
        for start in range(0, len(remaining), size):
            if len(trace) == max_attempts:
                break
            removed = remaining[start:start+size]
            candidate = drop_points(current, removed)
            accepted = bool(accepts(candidate, len(trace)+1))
            trace.append(dict(attempt=len(trace)+1, removed=removed, accepted=accepted,
                              request_sha256=fingerprint(candidate["request"])))
            if accepted:
                current = candidate
                granularity = max(2, granularity-1)
                break
        if not accepted:
            if size == 1 and len(trace) < max_attempts:
                break
            granularity = min(len(remaining), granularity*2)
    exhausted = bool(removable_points(current)) and len(trace) == max_attempts
    return current, dict(attempts=len(trace), budget_exhausted=exhausted,
                         locally_irreducible=not exhausted, trace=trace)


def source_digest(root, directory):
    digest = hashlib.sha256()
    for path in sorted((root / directory).rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def worker(root, case_path, output):
    """Load numerical code from one checkout in an otherwise fresh interpreter."""
    before = source_digest(root, "core")
    package = types.ModuleType("match_perspective")
    package.__path__, package.__file__ = [str(root)], str(root / "__init__.py")
    sys.modules["match_perspective"] = package
    result = solve(read_case(case_path)["request"], use_cache=False)
    result["harness_environment"] = result["environment"]
    result["environment"] = environment(root)
    result["numerical_source_sha256"] = source_digest(root, "core")
    if before != result["numerical_source_sha256"]:
        raise RuntimeError("Numerical source changed during solve")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--fixed-root", type=Path, default=ROOT)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--solve-timeout", type=float, default=60, help="Seconds per child; timeout aborts reduction")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_root:
        worker(args.worker_root.resolve(), args.case, args.out)
        return 0
    if args.baseline_root is None or args.max_attempts < 1 or args.solve_timeout <= 0:
        parser.error("Supply --baseline-root and positive attempt/time limits")
    case = read_case(args.case)
    if case["expectation"]["gauge"] != "anchor" or case["expectation"]["outcome"] != "solve":
        parser.error("This reducer requires an anchor-frame solve contract")
    if not any(case["expectation"].get("excluded_"+kind) for kind in ("points", "lines")):
        parser.error("This reducer requires explicitly named forbidden geometry")
    roots = dict(baseline=args.baseline_root.resolve(), fixed=args.fixed_root.resolve())
    for root in roots.values():
        if not (root / "core/sync/solve.py").is_file():
            parser.error(f"Not a Sync checkout: {root}")
    sources = {key: source_digest(root, "core") for key, root in roots.items()}
    harness = source_digest(ROOT, "tools/synthetic_sync")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = dict(predicate="same forbidden geometry; all other accuracy checks pass; fixed checkout passes",
        protected="all cameras, ground/known points, mirrors, lines, roles, truth and expectations",
        max_attempts=args.max_attempts, solve_timeout_s=args.solve_timeout,
        roots={key: str(root) for key, root in roots.items()}, numerical_sources=sources,
        harness_source_sha256=harness, environments={key: environment(root) for key, root in roots.items()})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    write_case(case, args.out / "original.json")

    def run_pair(candidate, label):
        if source_digest(ROOT, "tools/synthetic_sync") != harness:
            raise RuntimeError("Harness source changed during reduction")
        folder = args.out / label
        folder.mkdir()
        case_path = folder / "case.json"
        write_case(candidate, case_path)
        results = {}
        for key, root in roots.items():
            if source_digest(root, "core") != sources[key]:
                raise RuntimeError(f"{key} numerical source changed during reduction")
            output = folder / (key+".json")
            with (folder / (key+".log")).open("w") as log:
                try:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-root", str(root),
                        "--case", str(case_path.resolve()), "--out", str(output.resolve())],
                        stdout=log, stderr=subprocess.STDOUT, timeout=args.solve_timeout, check=True)
                except (subprocess.SubprocessError, OSError) as error:
                    raise RuntimeError(f"{key} execution failed; inspect {folder / (key+'.log')}") from error
            results[key] = json.loads(output.read_text())
            if results[key]["numerical_source_sha256"] != sources[key]:
                raise RuntimeError(f"{key} numerical source changed during solve")
        return results

    initial = run_pair(case, "initial")
    signature = forbidden_geometry(case, initial["baseline"])
    if signature is None or not evaluate(case, initial["fixed"])["passed"]:
        raise RuntimeError("Initial case must retain only the named regression on baseline and pass on fixed")
    print(f"Reducing {len(removable_points(case))} optional landmarks; protected failure {signature}", flush=True)

    def accepts(candidate, attempt):
        results = run_pair(candidate, f"attempt-{attempt:03d}")
        accepted = (forbidden_geometry(candidate, results["baseline"]) == signature
                    and evaluate(candidate, results["fixed"])["passed"])
        print(f"Attempt {attempt}: {'keep' if accepted else 'reject'} reduction, "
              f"{len(candidate['request']['observations'])} point picks", flush=True)
        return accepted

    reduced, summary = reduce_case(case, accepts, max_attempts=args.max_attempts)
    final = run_pair(reduced, "final")
    if forbidden_geometry(reduced, final["baseline"]) != signature or not evaluate(reduced, final["fixed"])["passed"]:
        raise RuntimeError("Final cold replay did not preserve the regression and passing control")
    write_case(reduced, args.out / "reduced.json")
    summary.update(signature=signature, elapsed_s=time.perf_counter()-started,
        original_points=len(case["request"]["points"]), reduced_points=len(reduced["request"]["points"]),
        original_picks=len(case["request"]["observations"]), reduced_picks=len(reduced["request"]["observations"]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Preserved regression: {summary['original_picks']} → {summary['reduced_picks']} picks; "
          f"budget exhausted={summary['budget_exhausted']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
