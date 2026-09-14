"""Joint-fitter point and stroke projection against Sync camera geometry."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package

from match_perspective.core import geometry
from match_perspective.core.focal_projection import (
    INVALID_LINE_RESIDUAL_PX,
    line_endpoint_distances,
    project_stroke_line,
    project_camera_points,
)
from match_perspective.core.sync.projection import (
    _project_world_line_to_image,
    project_private_point,
)
from match_perspective.core.sync.types import SimilarityTransform


class FocalProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.k = geometry.CameraIntrinsics(870.0, 740.0, 490.0, 350.0, 1000, 700)
        self.points = np.array((
            (0.2, -0.3, 2.0), (-0.5, 0.1, 3.2), (0.8, 0.6, 4.1),
        ))

    def _distortion_cases(self):
        return (
            (0.0, ()),
            (0.015, ()),
            (0.0, (0.11, -0.04, 0.003, -0.002, 0.008)),
            (0.0, (0.11, -0.04, 0.003, -0.002, 0.008, 0.002, -0.001, 0.0005)),
            # Brown takes precedence over division in imported camera data.
            (0.015, (0.11, -0.04, 0.003, -0.002, 0.008)),
        )

    def test_point_pixels_match_sync_with_non_square_focals_and_distortion(self):
        for division_lambda, brown_conrady in self._distortion_cases():
            with self.subTest(division_lambda=division_lambda, brown_conrady=brown_conrady):
                camera = geometry.Calibration(
                    self.k, np.eye(3), division_lambda=division_lambda,
                    brown_conrady=brown_conrady,
                )
                expected = np.asarray([
                    project_private_point(q, camera) for q in self.points
                ])
                actual, _, _ = project_camera_points(
                    self.points, self.k, division_lambda=division_lambda,
                    brown_conrady=brown_conrady, jacobian=False,
                )
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_point_jacobians_match_full_projection_differences(self):
        for division_lambda, brown_conrady in self._distortion_cases():
            with self.subTest(division_lambda=division_lambda, brown_conrady=brown_conrady):
                pixels, dq, dlog = project_camera_points(
                    self.points, self.k, division_lambda=division_lambda,
                    brown_conrady=brown_conrady,
                )
                assert dq is not None and dlog is not None
                for axis in range(3):
                    shifted = self.points.copy()
                    step = 1e-6
                    shifted[:, axis] += step
                    changed, _, _ = project_camera_points(
                        shifted, self.k, division_lambda=division_lambda,
                        brown_conrady=brown_conrady, jacobian=False,
                    )
                    np.testing.assert_allclose(
                        dq[:, :, axis], (changed - pixels) / step,
                        rtol=2e-4, atol=3e-3,
                    )
                scale = np.exp(1e-6)
                scaled_k = geometry.CameraIntrinsics(
                    self.k.fx * scale, self.k.fy * scale,
                    self.k.cx, self.k.cy, self.k.image_width, self.k.image_height,
                )
                changed, _, _ = project_camera_points(
                    self.points, scaled_k, division_lambda=division_lambda,
                    brown_conrady=brown_conrady, jacobian=False,
                )
                np.testing.assert_allclose(
                    dlog, (changed - pixels) / 1e-6, rtol=2e-5, atol=2e-4,
                )

    def test_clipped_depth_derivative_retains_penalty(self):
        q = np.asarray(((0.2, -0.3, -0.1),))
        pixels, dq, dlog = project_camera_points(q, self.k)
        assert dq is not None and dlog is not None
        self.assertEqual(float(dq[0, 0, 2]), -1000.0)
        self.assertEqual(float(dq[0, 1, 2]), -1000.0)
        self.assertGreater(float(pixels[0, 0]), 1e6)
        np.testing.assert_allclose(dlog[0], pixels[0] - (self.k.cx, self.k.cy) - 100.001)

    def test_stroke_residual_matches_sync_projected_line(self):
        rotation = np.array(((0.965925826, 0.0, -0.258819045),
                             (0.0, 1.0, 0.0),
                             (0.258819045, 0.0, 0.965925826)))
        center = np.array((0.2, -0.1, 0.3))
        point = np.array((0.6, -0.2, 4.0))
        direction = np.array((0.3, 0.7, 0.4))
        endpoints = np.array(((350.0, 280.0), (650.0, 490.0)))
        for division_lambda, brown_conrady in self._distortion_cases():
            with self.subTest(division_lambda=division_lambda, brown_conrady=brown_conrady):
                camera = geometry.Calibration(
                    self.k, rotation, center, division_lambda=division_lambda,
                    brown_conrady=brown_conrady,
                )
                line = project_stroke_line(
                    point, direction, rotation, center, self.k, endpoints,
                    division_lambda=division_lambda, brown_conrady=brown_conrady)
                self.assertIsNotNone(line)
                assert line is not None
                actual = line_endpoint_distances(
                    point, direction, rotation, center, self.k, endpoints,
                    division_lambda=division_lambda, brown_conrady=brown_conrady,
                )
                np.testing.assert_allclose(actual, endpoints @ line[:2] + line[2], atol=1e-10)
                self.assertAlmostEqual(float(np.linalg.norm(line[:2])), 1.0, places=10)

    def test_distorted_stroke_is_invariant_to_line_slide_scale_and_world_frame(self):
        rotation = np.array(((0.965925826, 0.0, -0.258819045),
                             (0.0, 1.0, 0.0),
                             (0.258819045, 0.0, 0.965925826)))
        center = np.array((0.2, -0.1, 0.3))
        point = np.array((0.6, -0.2, 4.0))
        direction = np.array((0.3, 0.7, 0.4))
        endpoints = np.array(((350.0, 280.0), (650.0, 490.0)))
        brown = (0.11, -0.04, 0.003, -0.002, 0.008)
        base = line_endpoint_distances(point, direction, rotation, center,
                                       self.k, endpoints, brown_conrady=brown)
        slid = line_endpoint_distances(point + 100.0 * direction, direction,
                                       rotation, center, self.k, endpoints,
                                       brown_conrady=brown)
        scaled = line_endpoint_distances(4.3 * point, 4.3 * direction,
                                         rotation, 4.3 * center, self.k, endpoints,
                                         brown_conrady=brown)
        angle = 0.37
        world = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                          (np.sin(angle), np.cos(angle), 0.0),
                          (0.0, 0.0, 1.0)))
        origin = np.array((7.0, -3.0, 2.0))
        framed = line_endpoint_distances(
            world @ point + origin, world @ direction, rotation @ world.T,
            world @ center + origin, self.k, endpoints, brown_conrady=brown)
        for trial in (slid, scaled, framed):
            np.testing.assert_allclose(trial, base, rtol=0, atol=1e-9)

    def test_line_through_camera_has_finite_invalid_residual(self):
        actual = line_endpoint_distances(
            np.zeros(3), np.array((0.2, 0.1, 1.0)),
            np.eye(3), np.zeros(3), self.k,
            np.array(((300.0, 200.0), (600.0, 400.0))),
        )
        np.testing.assert_array_equal(actual, np.full(2, INVALID_LINE_RESIDUAL_PX))


if __name__ == "__main__":
    unittest.main()
