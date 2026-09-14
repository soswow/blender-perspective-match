"""Ledgered focused Refine regressions from the generic full-suite fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
if "match_perspective" not in sys.modules:
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    sys.modules["match_perspective"] = package

from .budget import ExperimentBudget
from match_perspective.core import focal_bundle, lens_refine
from match_perspective.core.sync.request import json_values


TESTS = {
    "weak_axis": "test_focal_constraint_reliability.FocalConstraintReliabilityTests.test_weak_axis_plane_recovers_through_public_point_fov",
    "four_view": "test_focal_bundle.PointFocalBundleTests.test_four_camera_fit_preserves_private_poses_and_returns_fitted_world",
    "candidate": "test_focal_candidate.FocalCandidateTests.test_public_result_preserves_candidate_and_refusal",
}
SESSION = Path("/tmp/pm-joint-fit-tests/fullsuite-fix")


def _archive() -> tuple[str, dict[str, str]]:
    paths = sorted((ROOT / "core").rglob("*.py"))
    paths += sorted((ROOT / "tools/synthetic_sync").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_focal*.py"))
    paths += [ROOT / "tests/test_joint_fit_features.py", Path(__file__),
              ROOT / "tools/synthetic_sync/cases/focal-constraint-reliability/weak-axis-hard.json",
              ROOT / "tools/synthetic_sync/cases/independent-focal-production/four-view.json",
              ROOT / "tools/synthetic_sync/cases/independent-focal-production/run-03/sync-ledger.jsonl"]
    paths = sorted(set(paths))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    archive = SESSION / "source" / key
    for path in paths:
        relative = path.relative_to(ROOT)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[str(relative)]
    return key, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=TESTS)
    args = parser.parse_args()
    source_key, hashes = _archive()
    metadata = {"session": "fullsuite-fix-20260914", "python": sys.version,
                "numpy": np.__version__, "platform": platform.platform()}
    ledger = SESSION / "attempts.jsonl"
    original_sync = lens_refine.sync_module.solve_landmark_sync
    original_bundle = lens_refine.fit_independent_focals
    with ExperimentBudget(ledger, metadata=metadata, max_calls=12,
                          wall_seconds=600.0, per_call_seconds=120.0) as budget:
        def measured(kind, original, *call_args, **call_kwargs):
            previous = sum(row.get("kind") == "started" and
                           str(row.get("label", "")).startswith(kind + ":")
                           for row in budget.rows)
            if previous >= (4 if kind == "sync" else 8):
                raise RuntimeError(f"{kind} sub-budget exhausted")
            exact = {"case": args.case, "source_key": source_key,
                     "sources": hashes,
                     "args": json_values(call_args),
                     "kwargs": json_values({key: value for key, value in call_kwargs.items()
                                            if key not in {"cancel_check", "progress_callback",
                                                           "diagnostic_callback"}})}
            with budget.attempt(f"{kind}:{args.case}", exact) as attempt:
                result = original(*call_args, **call_kwargs)
                attempt.complete({"result": json_values(result)})
            return result

        with mock.patch.object(lens_refine.sync_module, "solve_landmark_sync",
                               side_effect=lambda *a, **k: measured("sync", original_sync, *a, **k)), \
             mock.patch.object(lens_refine, "fit_independent_focals",
                               side_effect=lambda *a, **k: measured("bundle", original_bundle, *a, **k)):
            suite = unittest.TestLoader().loadTestsFromName(TESTS[args.case])
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"case": args.case, "source_key": source_key,
                      "ledger": str(ledger), "passed": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
