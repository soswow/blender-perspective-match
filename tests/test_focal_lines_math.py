"""Infinite-line chart and stroke residual checks."""

from unittest import TestCase
from unittest import mock

import numpy as np

from match_perspective.core.focal_lines import LineChart, endpoint_distances
from match_perspective.core.focal_bundle import fit_independent_focals
from match_perspective.core import sync
try:
    from .test_focal_lines_integration import _inputs
except ImportError:
    from test_focal_lines_integration import _inputs
from tools.synthetic_sync import focal_lines


class FocalLineMathTests(TestCase):
    def test_malformed_duplicate_and_self_mirror_links_refuse_before_line_seed(self):
        case = focal_lines.generate()
        calibrations, points, initial = _inputs(case)
        lines = [sync.SyncLineObservation(**item)
                 for item in case["request"]["line_observations"]]
        valid = case["request"]["mirror_pairs"]
        for bad in (["mirror_edge_a"], ["mirror_edge_a", "mirror_edge_a"],
                    ["mirror_edge_a", "missing"], "ab"):
            pairs = [*valid, bad]
            with self.subTest(bad=bad), mock.patch(
                    "match_perspective.core.focal_bundle._reconstruct_line_from_observations") as seed:
                result = fit_independent_focals(
                    calibrations, points, initial, anchor_id=case["request"]["anchor_id"],
                    line_observations=lines, mirror_pairs=pairs,
                    mirror_plane=case["request"]["mirror_plane"],
                    mirror_landmark_id=case["request"]["mirror_landmark_id"])
                self.assertFalse(result.accepted)
                seed.assert_not_called()
        with mock.patch("match_perspective.core.focal_bundle._reconstruct_line_from_observations") as seed:
            result = fit_independent_focals(
                calibrations, points, initial, anchor_id=case["request"]["anchor_id"],
                line_observations=lines,
                mirror_pairs=[*valid, ["mirror_edge_b", "mirror_edge_a"]],
                mirror_plane=case["request"]["mirror_plane"],
                mirror_landmark_id=case["request"]["mirror_landmark_id"])
            self.assertFalse(result.accepted)
            seed.assert_not_called()

    def test_chart_has_no_along_line_position_gauge(self):
        direction = np.array((0.3, -0.4, 0.8660254))
        direction /= np.linalg.norm(direction)
        first = LineChart(np.array((0.2, 0.9, 3.0)), direction)
        second = LineChart(np.array((0.2, 0.9, 3.0)) + 20.0 * direction, direction)
        self.assertTrue(np.allclose(first.initial, second.initial, atol=1e-13))
        point, recovered = first.decode(first.initial)
        self.assertAlmostEqual(float(point @ recovered), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(recovered)), 1.0, places=12)

    def test_stroke_endpoints_are_distances_to_infinite_projected_line(self):
        rotation = np.eye(3)
        center = np.zeros(3)
        principal = np.array((320.0, 240.0))
        point = np.array((1.0, 0.2, 4.0))
        direction = np.array((0.2, 0.7, 0.4))
        focal = 700.0
        sample = np.asarray([point + t * direction for t in (-0.5, 3.0)])
        uv = sample[:, :2] * focal / sample[:, 2, None] + principal
        self.assertLess(np.max(abs(endpoint_distances(
            point, direction, rotation, center, focal, principal, uv))), 1e-11)
        shifted = uv + np.array((0.0, 5.0))
        measured = endpoint_distances(point, direction, rotation, center,
                                      focal, principal, shifted)
        self.assertAlmostEqual(abs(measured[0]), abs(measured[1]), places=10)
        self.assertGreater(abs(measured[0]), 0.1)
