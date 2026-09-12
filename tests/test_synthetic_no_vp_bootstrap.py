"""Truth isolation and strict geometry checks for no-VP startup evidence."""

from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tools.synthetic_sync.no_vp_bootstrap import CASE_DIR, KINDS, ROOT, assess, make_case, validate_fixture
from tools.synthetic_sync.no_vp_startup_trace import runtime_key, source_hash
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import fingerprint, load_core, solve, solver_arguments


def true_record(case):
    return dict(success=True, message="exact oracle control", reported_rmse_px=0.,
                cameras={c["id"]: deepcopy(c) for c in case["truth"]["cameras"]},
                landmarks=deepcopy(case["truth"]["points"]), line_segments={})


def assert_portable_json(test: unittest.TestCase, saved, generated, path="root",
                         *, rel_tol=1e-14, abs_tol=1e-14):
    """Check exact structure while allowing only roundoff-scale float drift."""
    test.assertIs(type(saved), type(generated), path)
    if isinstance(saved, dict):
        test.assertEqual(saved.keys(), generated.keys(), path)
        for key in saved:
            assert_portable_json(test, saved[key], generated[key], f"{path}.{key}",
                                 rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(saved, list):
        test.assertEqual(len(saved), len(generated), path)
        for index, (left, right) in enumerate(zip(saved, generated)):
            assert_portable_json(test, left, right, f"{path}[{index}]",
                                 rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(saved, float):
        test.assertTrue(math.isclose(saved, generated, rel_tol=rel_tol, abs_tol=abs_tol),
                        f"{path}: {saved!r} != {generated!r}")
    else:
        test.assertEqual(saved, generated, path)


class NoVpFixtureTests(unittest.TestCase):
    def test_frozen_cases_replay_and_have_no_truth_or_pose_prior(self):
        for kind in KINDS:
            for intr in ("trueK", "guessedK"):
                with self.subTest(kind=kind, intr=intr):
                    case = make_case(kind, intr)
                    saved = read_case(CASE_DIR / f"no-vp-{kind.replace('_', '-')}-{intr}.json")
                    assert_portable_json(self, saved, json.loads(json.dumps(case)))
                    validate_fixture(saved)
                    request_hash = fingerprint(saved["request"])
                    saved["truth"]["points"].clear()
                    saved["truth"]["checks"].clear()
                    self.assertEqual(request_hash, fingerprint(saved["request"]))

    def test_favorable_true_record_passes_one_global_similarity(self):
        case = make_case("shared", "trueK")
        record = true_record(case)
        self.assertEqual(assess(case, record)["classification"], "accurate_acceptance")
        scale, rotation, translation = 2.4, np.array([[0,-1,0],[1,0,0],[0,0,1]]), np.array([5.,-3.,2.])
        for camera in record["cameras"].values():
            camera["center"] = (scale*rotation@camera["center"]+translation).tolist()
            camera["rotation"] = (np.asarray(camera["rotation"])@rotation.T).tolist()
        record["landmarks"] = {key:(scale*rotation@position+translation).tolist()
                               for key, position in record["landmarks"].items()}
        self.assertEqual(assess(case, record)["classification"], "accurate_acceptance")

    def test_false_precision_focal_pose_and_point_fail_independent_oracle(self):
        case = make_case("shared", "trueK")
        record = true_record(case)
        record["cameras"]["view_1"]["fx"] *= 1.25
        record["cameras"]["view_1"]["fy"] *= 1.25
        focal = assess(case, record)
        self.assertEqual(focal["classification"], "false_precise_acceptance")
        self.assertFalse(focal["focal_within_limit"])
        self.assertGreater(focal["independent_geometry"]["cameras"]["view_1"]["holdout_rmse_px"], 1.)

        record = true_record(case)
        record["cameras"]["view_1"]["center"][0] += .5
        pose = assess(case, record)
        self.assertEqual(pose["classification"], "false_precise_acceptance")
        self.assertGreater(pose["independent_geometry"]["cameras"]["view_1"]["center_fraction"], .02)

        record = true_record(case)
        first = sorted(record["landmarks"])[0]
        record["landmarks"][first][2] += 1.
        point = assess(case, record)
        self.assertEqual(point["classification"], "false_precise_acceptance")
        self.assertTrue(any("reconstructed point error" in v for v in point["independent_geometry"]["violations"]))

    def test_pure_rotation_acceptance_does_not_claim_metric_depth(self):
        case = make_case("pure_rotation", "trueK")
        record = true_record(case)
        self.assertEqual(assess(case, record)["classification"], "unsupported_depth_acceptance")
        record["success"] = False
        record["message"] = "no translation baseline"
        self.assertEqual(assess(case, record)["classification"], "useful_refusal")
        record["exception"] = "unexpected crash"
        self.assertEqual(assess(case, record)["classification"], "exception")

    def test_frozen_numerical_failure_replays_without_a_solve(self):
        case = read_case(CASE_DIR / "no-vp-shared-trueK.json")
        saved = json.loads((CASE_DIR / "no-vp-shared-trueK-result.json").read_text())
        ledger = [json.loads(line) for line in (CASE_DIR / "no-vp-bootstrap-ledger.jsonl").read_text().splitlines()]
        self.assertEqual([row["kind"] for row in ledger], ["budget", "started", "completed"])
        self.assertEqual(ledger[1]["request"], case["request"])
        self.assertEqual(ledger[2]["result"], saved["record"])
        self.assertEqual(saved["request_sha256"], fingerprint(case["request"]))
        replay = assess(case, saved["record"])
        # Derived angles and RMS aggregate many floating operations and differ
        # slightly more across BLAS/NumPy builds than the frozen input numbers.
        assert_portable_json(self, saved["assessment"], replay,
                             rel_tol=2e-13, abs_tol=1e-12)
        self.assertEqual(replay["classification"], "false_precise_acceptance")

    def test_true_intrinsics_bootstrap_uses_the_consistent_view_graph(self):
        case = read_case(CASE_DIR / "no-vp-shared-trueK.json")
        record = solve(case["request"])
        assessment = assess(case, record)
        self.assertEqual(assessment["classification"], "accurate_acceptance",
                         assessment["independent_geometry"]["violations"])
        self.assertLess(record["reported_rmse_px"], 1.0e-3)
        self.assertEqual(set(record["cameras"]), set(case["expectation"]["cameras"]))

    def test_free_point_graph_policy_excludes_cameras_without_3d_influence(self):
        case = read_case(CASE_DIR / "no-vp-shared-trueK.json")
        all_ids = {camera["id"] for camera in case["request"]["cameras"]}
        load_core()
        stages = importlib.import_module("match_perspective.core.sync.solve")
        for location_ids, readonly_ids, expected in (
            (all_ids, set(), True),
            (all_ids - {"view_2"}, set(), False),
            (all_ids, {"view_2"}, False),
        ):
            with self.subTest(location_ids=location_ids, readonly_ids=readonly_ids):
                arguments = solver_arguments(case["request"])
                arguments["location_match_ids"] = location_ids
                arguments["readonly_match_ids"] = readonly_ids
                flags = []

                def stop_registration(*_args, **kwargs):
                    flags.append(kwargs["free_point_graph_only"])
                    return None, "stopped before numerical registration"

                with patch.object(stages, "_register_from_relative_pose", stop_registration):
                    result = stages.solve_landmark_sync(**arguments)
                self.assertFalse(result.success)
                self.assertEqual(flags, [expected])

    def test_trace_cache_key_tracks_core_harness_and_thread_settings(self):
        original_read = Path.read_bytes
        baseline = source_hash()
        for changed in (ROOT / "core" / "geometry.py",
                        ROOT / "tools" / "synthetic_sync" / "evaluation.py"):
            with self.subTest(changed=changed):
                def altered_read(path):
                    contents = original_read(path)
                    return contents + b"changed" if path == changed else contents

                with patch.object(Path, "read_bytes", altered_read):
                    self.assertNotEqual(source_hash(), baseline)

        base_runtime = runtime_key()
        changed_threads = "2" if base_runtime["threads"]["OPENBLAS_NUM_THREADS"] != "2" else "3"
        with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": changed_threads}):
            self.assertNotEqual(runtime_key(), base_runtime)


if __name__ == "__main__":
    unittest.main()
