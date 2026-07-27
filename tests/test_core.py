"""Pure geometry regressions ported from the desktop Python core."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core


class CoreGeometryTests(unittest.TestCase):
    """Protect the camera solve and surface geometry invariants."""

    def test_vanishing_point_intersection(self) -> None:
        vanishing = core.vanishing_point_from_lines(
            [
                core.LineSegment(0.0, 0.0, 100.0, 100.0),
                core.LineSegment(0.0, 100.0, 100.0, 0.0),
            ]
        )
        self.assertIsNotNone(vanishing)
        self.assertTrue(np.allclose(vanishing[:2], (50.0, 50.0), atol=1.0e-6))

    def test_length_weighting_prefers_long_inliers(self) -> None:
        vanishing = core.vanishing_point_from_lines(
            [
                core.LineSegment(100.0, 100.0, 700.0, 500.0),
                core.LineSegment(100.0, 500.0, 700.0, 100.0),
                core.LineSegment(390.0, 10.0, 410.0, 30.0),
            ]
        )
        self.assertIsNotNone(vanishing)
        self.assertLess(abs(float(vanishing[0]) - 400.0), 25.0)
        self.assertLess(abs(float(vanishing[1]) - 300.0), 25.0)

    def test_distortion_round_trip(self) -> None:
        points = np.array(
            ((100.0, 80.0), (640.0, 360.0), (1100.0, 600.0), (200.0, 500.0))
        )
        ideal = core.undistort_points(points, 800.0, 800.0, 640.0, 360.0, 0.12)
        observed = core.distort_points(ideal, 800.0, 800.0, 640.0, 360.0, 0.12)
        self.assertTrue(np.allclose(points, observed, atol=0.05))

    def test_three_vp_principal_point(self) -> None:
        points = {
            "x": np.array((5.0, 5.0, 1.0)),
            "y": np.array((9.0, 5.0, 1.0)),
            "z": np.array((5.0, 9.0, 1.0)),
        }
        principal = core.principal_point_from_three_vps(points, 10, 10)
        self.assertIsNotNone(principal)
        self.assertTrue(np.allclose(principal, (5.0, 5.0)))

    def test_perspective_rectangle_rejects_degenerate_quad(self) -> None:
        corners = core.perspective_rectangle_corners(
            (10.0, 10.0),
            (10.5, 10.5),
            np.array((1.0, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
        )
        self.assertIsNone(corners)

    def test_surface_grid_has_two_lines_per_division(self) -> None:
        grid = core.surface_grid(
            [(0.0, 0.0), (100.0, 10.0), (90.0, 80.0), (5.0, 70.0)],
            4,
        )
        self.assertEqual(len(grid), 6)


if __name__ == "__main__":
    unittest.main()
