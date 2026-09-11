"""A recovered camera must not erase ground constraints or spoil the solved views."""

from pathlib import Path
import threading
import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.recovery import observe_recovery
from tools.synthetic_sync.scenarios import generate, read_case
from tools.synthetic_sync.solver import load_core, solver_arguments


class SyntheticRecoveryTests(unittest.TestCase):
    def test_recovery_can_copy_candidate_with_background_cancellation_callback(self):
        case = read_case(Path(__file__).resolve().parents[1]/"tools/synthetic_sync/cases/recovered-ground-conflict.json")
        _core,sync = load_core()
        cancel = threading.Event()
        result = sync.solve_landmark_sync(**solver_arguments(case["request"]),cancel_check=cancel.is_set)
        self.assertTrue(result.success,result.message)
        self.assertIn("view_2",result.similarities)

    def test_final_update_preserves_hard_ground_and_supported_camera_accuracy(self):
        case = read_case(Path(__file__).resolve().parents[1]/"tools/synthetic_sync/cases/recovered-ground-conflict.json")
        result,trace = observe_recovery(case)
        self.assertTrue(result["success"],result["message"])
        self.assertIn("view_2",result["cameras"])
        self.assertEqual(len(trace),1)
        self.assertEqual(trace[0]["eligible"],["view_2"])
        self.assertLess(max(map(abs,trace[0]["before"]["ground_z"].values())),1e-6)
        self.assertEqual(set(trace[0]["before"]["ground_z"]),set(trace[0]["after"]["ground_z"]))
        self.assertLess(max(map(abs,trace[0]["after"]["ground_z"].values())),1e-3)
        assessment = evaluate(case,result)
        # Deliberately contradictory recovered-camera picks do not warrant a
        # strict accuracy promise for that camera; the healthy view still does.
        self.assertLess(assessment["cameras"]["view_1"]["holdout_rmse_px"],case["expectation"]["holdout_rmse_px"])

    def test_useful_stage_update_keeps_soft_known_3d_and_plane_constraints(self):
        case = generate("ground")
        case["request"].update(ground_slack=.02,known_3d_slack=.08,plane_slack=0.,
            plane_groups=[(p["id"],"Y",1) for p in case["request"]["points"][:4]])
        for point in case["request"]["points"][:3]:
            point["known"] = list(case["truth"]["points"][point["id"]])
            point["known"][2] += .03
        # A stage-only control: the pose is already valid, so explicitly mark
        # it recovered to test this update independently of registration routing.
        result,trace = observe_recovery(case,stage_control_recovered=["view_2"])
        self.assertFalse(trace[0]["kept_joint_geometry"])
        self.assertTrue(trace[0]["after"]["plane_spread"]["Y#1"]["active"])
        self.assertLess(trace[0]["after"]["plane_spread"]["Y#1"]["max_distance"],1e-5)
        assessment = evaluate(case,result)
        self.assertTrue(assessment["passed"],assessment["violations"])
        for key in trace[0]["before"]["known_gap"]:
            before = np.asarray(trace[0]["before"]["landmarks"][key])
            after = np.asarray(trace[0]["after"]["landmarks"][key])
            truth = np.asarray(case["truth"]["points"][key])
            self.assertGreater(trace[0]["after"]["known_gap"][key],.005)
            self.assertLess(np.linalg.norm(after-truth),np.linalg.norm(before-truth))


if __name__ == "__main__":
    unittest.main()
