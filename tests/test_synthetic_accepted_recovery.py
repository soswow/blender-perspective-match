"""Hard line relations survive an accepted recovered-camera update and rebuild."""

from pathlib import Path
import unittest

from tools.synthetic_sync.accepted_recovery import (
    _plane_member_spread,
    observe,
)
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.scenarios import read_case


class AcceptedRecoveryTests(unittest.TestCase):
    def test_accepted_update_preserves_plane_and_axis_at_final_rebuild(self):
        case = read_case(Path(__file__).resolve().parents[1] /
                         "tools/synthetic_sync/cases/accepted-recovery.json")
        production, live = observe(case, freeze=False)
        frozen, control = observe(case, freeze=True)

        self.assertTrue(production["success"], production["message"])
        self.assertTrue(frozen["success"], frozen["message"])
        self.assertTrue(live["candidate_accepted"])
        self.assertFalse(live["after_guard"]["kept_joint_geometry"])
        self.assertGreater(live["stage_delta"]["landmark"], 1e-3)
        self.assertEqual(production["request_sha256"], frozen["request_sha256"])
        self.assertTrue(evaluate(case, production)["passed"])
        self.assertTrue(evaluate(case, frozen)["passed"])

        for trace in (live, control):
            self.assertLess(
                _plane_member_spread(trace["after_final_line_rebuild"], case), 1e-5
            )
            for label in ("before", "after_final_line_rebuild"):
                self.assertLess(
                    trace[label]["independent"]["declared_axis_direction_sine"],
                    1e-8,
                    (trace["routing"], label),
                )
        self.assertLess(
            live["candidate"]["independent"]["declared_axis_direction_sine"],
            1e-8,
        )
        final_z = live["after_final_line_rebuild"]["independent"]["line_endpoint_z_range"]
        for value, expected in zip(final_z, (0.1, 1.3)):
            self.assertAlmostEqual(value, expected, delta=0.05)

    def test_signed_plane_spread_detects_opposite_sides(self):
        case = {
            "truth": {"planes": [{"axis": "Y", "bucket": 1,
                                  "normal": [0, 1, 0], "origin": [0, -0.9, 0]}]},
            "request": {"plane_groups": [("left", "Y", 1), ("right", "Y", 1),
                                          ("line", "Y", 1)]},
        }
        snapshot = {"record": {
            "landmarks": {"left": [0, -0.89, 0], "right": [0, -0.91, 0]},
            "line_segments": {"line": [[1, -0.89, 0], [1, -0.91, 1]]},
        }}
        self.assertAlmostEqual(_plane_member_spread(snapshot, case), 0.02)


if __name__ == "__main__":
    unittest.main()
