"""Checks for the independent oracle and a bounded synthetic Sync corpus."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import look_at, object_mesh, project, visible
from tools.synthetic_sync.scenarios import FAMILIES, generate, read_case, write_case
from tools.synthetic_sync.solver import fingerprint, solve, solver_arguments


def true_record(case):
    return dict(success=True, message="oracle", reported_rmse_px=0.0,
                cameras={c["id"]: deepcopy(c) for c in case["truth"]["cameras"]},
                landmarks=deepcopy(case["truth"]["points"]), line_segments={})


class SyntheticOracleTests(unittest.TestCase):
    def test_projection_has_analytical_pixel_and_depth_conventions(self):
        camera = dict(center=[1, 2, 3], rotation=np.eye(3).tolist(), fx=100, fy=200, cx=40, cy=30)
        uv, depth = project([[2, 4, 7], [1, 2, 1]], camera)
        np.testing.assert_allclose(uv, [[65, 130], [40, 30]])
        np.testing.assert_allclose(depth, [4, -2])

    def test_occluded_surface_is_not_a_pick(self):
        camera = dict(center=[0, -6, 1], rotation=look_at([0, -6, 1], [0, 0, 1]),
                      width=960, height=720, fx=700, fy=700, cx=480, cy=360)
        mesh = object_mesh()
        self.assertTrue(visible([0, -0.9, 0.7], camera, mesh))
        self.assertFalse(visible([0, 0.9, 0.7], camera, mesh))
        self.assertFalse(visible([0, -7, 0.7], camera, mesh))

    def test_exact_case_roundtrip_and_seed_replay(self):
        case = generate("mixed_lines", seed=7, noise_px=0.3)
        self.assertEqual(json.dumps(case), json.dumps(generate("mixed_lines", 7, 0.3)))
        self.assertNotEqual(fingerprint(case["request"]), fingerprint(generate("mixed_lines", 8, 0.3)["request"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            write_case(case, path)
            self.assertEqual(json.dumps(case), json.dumps(read_case(path)))

    def test_withheld_geometry_is_disjoint_and_not_solver_evidence(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                case = generate(family)
                train = np.asarray(list(case["truth"]["points"].values()))
                checks = np.asarray([p["position"] for p in case["truth"]["checks"]])
                self.assertGreater(float(np.linalg.norm(train[:, None] - checks, axis=2).min()), 1e-6)
                before = repr(solver_arguments(case["request"]))
                case["truth"]["points"].clear()
                case["truth"]["checks"].clear()
                self.assertEqual(before, repr(solver_arguments(case["request"])))

    def test_perfect_planar_fit_can_still_fail_withheld_object(self):
        case = generate("ground")
        camera = dict(id="view_0", center=[0, 0, -5], rotation=np.eye(3).tolist(),
                      width=960, height=720, fx=500, fy=500, cx=480, cy=360)
        wrong = dict(camera, center=[0, 0, -10], fx=1000, fy=1000)
        plane_picks = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
        np.testing.assert_allclose(project(plane_picks, camera)[0], project(plane_picks, wrong)[0])
        case["expectation"]["cameras"] = ["view_0"]
        case["truth"]["cameras"] = [camera]
        case["truth"]["checks"] = [dict(id=str(i), position=[x, y, z], views=["view_0"])
            for i, (x, y, z) in enumerate([[-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1], [-1,0,2], [1,0,2]])]
        record = true_record(case)
        record["cameras"]["view_0"] = wrong
        assessment = evaluate(case, record)
        self.assertFalse(assessment["passed"])
        self.assertGreater(assessment["cameras"]["view_0"]["holdout_rmse_px"], 10)

    def test_one_global_similarity_is_allowed_but_camera_drift_is_not(self):
        case = generate("free_scale")
        record = true_record(case)
        rotation = np.array([[0,-1,0], [1,0,0], [0,0,1]])
        translation, scale = np.array([5, -3, 2]), 2.3
        for camera in record["cameras"].values():
            camera["center"] = (scale * rotation @ camera["center"] + translation).tolist()
            camera["rotation"] = (np.asarray(camera["rotation"]) @ rotation.T).tolist()
        record["landmarks"] = {k:(scale * rotation @ v + translation).tolist() for k,v in record["landmarks"].items()}
        self.assertTrue(evaluate(case, record)["passed"])
        record["cameras"]["view_1"]["center"][0] += 0.5
        self.assertFalse(evaluate(case, record)["passed"])

    def test_missing_camera_and_false_acceptance_fail(self):
        case = generate("ground")
        record = true_record(case)
        del record["cameras"]["view_2"]
        self.assertFalse(evaluate(case, record)["passed"])
        ambiguous = generate("collinear")
        self.assertFalse(evaluate(ambiguous, true_record(ambiguous))["passed"])

    def test_cached_line_constraints_use_the_production_tuple_contract(self):
        args = solver_arguments(generate("mixed_lines")["request"])
        self.assertTrue(args["parallel_pairs"])
        hash(tuple(args["parallel_pairs"]))

    def test_focused_core_test_runs_alone_from_an_unrelated_directory(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_unittests.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script),
                "test_core.CoreGeometryTests.test_vanishing_point_intersection"],
                cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_exception_is_not_a_valid_refusal(self):
        case = generate("collinear")
        record = true_record(case)
        record.update(success=False, exception="unexpected crash")
        self.assertFalse(evaluate(case, record)["passed"])

    def test_locked_pose_cannot_drift_within_the_general_accuracy_budget(self):
        case = generate("locked_bridge")
        record = true_record(case)
        record["cameras"]["view_1"]["center"][0] += 0.0002
        assessment = evaluate(case, record)
        self.assertIn("view_1: explicitly locked camera moved", assessment["violations"])


class SyntheticSyncCorpusTests(unittest.TestCase):
    def test_seed_zero_corpus(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                case = generate(family)
                result = solve(case["request"])
                assessment = evaluate(case, result)
                self.assertTrue(assessment["passed"], assessment["violations"])


if __name__ == "__main__":
    unittest.main()
