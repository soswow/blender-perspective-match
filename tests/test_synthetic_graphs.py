"""Sparse overlap must propagate physical ground scale and honor camera roles."""

from pathlib import Path
import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.graphs import graph_case
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import solve


def frozen_chain():
    return read_case(Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases/ground-chain.json")


class SyntheticGraphTests(unittest.TestCase):
    def test_exact_chain_retains_ground_scale_beyond_anchor_overlap(self):
        case = frozen_chain()
        result = solve(case["request"])
        assessment = evaluate(case,result)
        self.assertTrue(assessment["passed"], assessment["violations"])
        for point in case["request"]["points"]:
            if point["ground"]:
                np.testing.assert_allclose(result["landmarks"][point["id"]], case["truth"]["points"][point["id"]], atol=1e-4)

    def test_fit_only_poses_from_previous_views_ground_without_bridging_3d(self):
        case = frozen_chain()
        case["request"].update(location_match_ids=["view_0","view_1","view_3","view_4"],readonly_match_ids=["view_2"])
        case["expectation"].update(cameras=["view_0","view_1","view_2"],excluded_cameras=["view_3","view_4"])
        result = solve(case["request"])
        assessment = evaluate(case,result)
        self.assertTrue(assessment["passed"], assessment["violations"])
        self.assertFalse(set(case["graph"]["edges"][2]["landmarks"]) & set(result["landmarks"]))

    def test_truth_and_overlap_are_independent_and_physically_possible(self):
        case = frozen_chain()
        truth, request = case["truth"], case["request"]
        by_point = {}
        cameras = {c["id"]:c for c in truth["cameras"]}
        checks = np.asarray([p["position"] for p in truth["checks"]])
        for observation in request["observations"]:
            key, camera_id = observation["landmark_id"],observation["match_id"]
            by_point.setdefault(key,set()).add(camera_id)
            position, camera = truth["points"][key], cameras[camera_id]
            self.assertTrue(visible(position,camera,truth["mesh"]))
            np.testing.assert_allclose(project([position],camera)[0][0],[observation["u"],observation["v"]],atol=1e-10)
            self.assertGreater(np.linalg.norm(checks-position,axis=1).min(),1e-5)
        for edge in case["graph"]["edges"]:
            ground = [truth["points"][p["id"]] for p in request["points"] if p["ground"] and p["id"] in edge["landmarks"]]
            self.assertEqual(len(ground),4)
            self.assertEqual(np.linalg.matrix_rank(np.asarray(ground)-np.mean(ground,axis=0)),2)
            for key in edge["landmarks"]:
                self.assertEqual(by_point[key],set(edge["cameras"]))
        self.assertNotEqual(request["cameras"][2]["center"],truth["cameras"][2]["center"])

    def test_bridge_role_changes_only_participation_and_expected_coverage(self):
        chain, changed = graph_case(), graph_case("fit_only_bridge")
        self.assertEqual(chain["truth"],changed["truth"])
        for key in chain["request"]:
            if key not in {"location_match_ids","readonly_match_ids"}:
                self.assertEqual(chain["request"][key],changed["request"][key])


if __name__ == "__main__":
    unittest.main()
