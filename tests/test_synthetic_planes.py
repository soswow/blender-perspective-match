"""Shared-plane constraints must preserve independent geometry and floor pins."""

import unittest
from pathlib import Path

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.planes import plane_case, remove_planes
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import solve
from test_synthetic_sync import true_record


class SyntheticPlaneTests(unittest.TestCase):
    def test_free_plane_cannot_lift_hard_ground_during_initialization(self):
        case = read_case(Path(__file__).resolve().parents[1]/"tools/synthetic_sync/cases/free-plane-hard-ground.json")
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])

    def test_floor_oracle_rejects_drift_even_with_true_cameras(self):
        case = plane_case("free_ground")
        result = true_record(case)
        self.assertTrue(evaluate(case, result)["passed"])
        result["landmarks"]["ground_0"][2] += .0004
        assessment = evaluate(case, result)
        self.assertFalse(assessment["passed"])
        self.assertTrue(any("ground distance" in violation for violation in assessment["violations"]))

    def test_ground_slack_still_allows_plane_members_to_refine(self):
        case = plane_case("free_ground", noise_px=.3)
        case["request"]["ground_slack"] = .02
        case["expectation"].pop("ground_max_distance")
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])
        self.assertGreater(max(abs(result["landmarks"][key][2]) for key in ("ground_0", "ground_2")), 1e-5)

    def test_plane_removal_retains_all_other_evidence_and_truth(self):
        case = plane_case("free_ground", noise_px=.3)
        removed = remove_planes(case)
        self.assertEqual(removed["truth"], case["truth"])
        self.assertEqual(removed["expectation"], case["expectation"])
        for key in case["request"]:
            if key != "plane_groups":
                self.assertEqual(removed["request"][key], case["request"][key])

    def test_added_plane_points_are_visible_and_separate_from_withheld_checks(self):
        for family in ("free_ground", "free_tilted"):
            case = plane_case(family)
            cameras = {c["id"]: c for c in case["truth"]["cameras"]}
            checks = np.asarray([p["position"] for p in case["truth"]["checks"]])
            for pick in case["request"]["observations"]:
                if not pick["landmark_id"].startswith("tilted_"):
                    continue
                position = case["truth"]["points"][pick["landmark_id"]]
                camera = cameras[pick["match_id"]]
                self.assertTrue(visible(position, camera, case["truth"]["mesh"]))
                np.testing.assert_allclose(project([position], camera)[0][0], [pick["u"], pick["v"]], atol=1e-10)
                self.assertGreater(np.min(np.linalg.norm(checks-position, axis=1)), 1e-5)


if __name__ == "__main__":
    unittest.main()
