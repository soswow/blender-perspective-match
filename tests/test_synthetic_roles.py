"""Fit Only cameras cannot provide reconstruction evidence through constraints."""

import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.roles import one_sided_fit_only, parallel_line_with_fit_only_stroke, weak_line_with_fit_only_stroke
from tools.synthetic_sync.solver import solve


class SyntheticRoleTests(unittest.TestCase):
    def test_fit_only_cannot_seed_one_sided_mirror_features(self):
        for family in ("mirror_points", "mirror_lines"):
            with self.subTest(family=family):
                case = one_sided_fit_only(family)
                result = solve(case["request"])
                assessment = evaluate(case, result)
                self.assertTrue(assessment["passed"], assessment["violations"])

    def test_known_partner_still_supplies_geometry_seen_only_by_fit_only(self):
        case = one_sided_fit_only("mirror_points")
        left, right = case["request"]["mirror_pairs"][0]
        next(point for point in case["request"]["points"] if point["id"] == left)["known"] = case["truth"]["points"][left]
        case["expectation"]["required_points"] = [left, right]
        case["expectation"]["excluded_points"] = [key for key in case["expectation"]["excluded_points"] if key not in {left, right}]
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])

    def test_fit_only_stroke_cannot_repair_weak_mirror_geometry_or_clear_warning(self):
        base, extra = weak_line_with_fit_only_stroke()
        before, after = solve(base["request"]), solve(extra["request"])
        self.assertTrue(before["success"], before["message"])
        self.assertTrue(after["success"], after["message"])
        for key in base["expectation"]["required_lines"]:
            np.testing.assert_allclose(before["line_segments"][key], after["line_segments"][key], atol=1e-7)
            self.assertIn(key, after["weak_line_ids"])
            self.assertAlmostEqual(before["line_support_angles_deg"][key], after["line_support_angles_deg"][key], places=6)

    def test_fit_only_stroke_cannot_move_parallel_line(self):
        base, extra = parallel_line_with_fit_only_stroke()
        before, after = solve(base["request"]), solve(extra["request"])
        self.assertTrue(before["success"], before["message"])
        self.assertTrue(after["success"], after["message"])
        for case, result in ((base, before), (extra, after)):
            assessment = evaluate(case, result)
            self.assertTrue(assessment["passed"], assessment["violations"])
        np.testing.assert_allclose(before["line_segments"]["edge_1"], after["line_segments"]["edge_1"], atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
