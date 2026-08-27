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

    def test_brown_conrady_round_trip(self) -> None:
        # From ROS plumb_bob YAML (k1, k2, p1, p2, k3).
        brown = (-0.0014045245854740534, 0.0004527363803767526, 0.0, 0.0, 0.0)
        points = np.array(
            ((100.0, 80.0), (1505.0, 2004.0), (2900.0, 3900.0), (200.0, 3500.0))
        )
        fx, fy, cx, cy = 1587.08, 1593.07, 1505.33, 2003.91
        ideal = core.undistort_points(points, fx, fy, cx, cy, 0.0, brown)
        observed = core.distort_points(ideal, fx, fy, cx, cy, 0.0, brown)
        self.assertTrue(np.allclose(points, observed, atol=0.05))

    def test_brown_conrady_with_tangential_round_trip(self) -> None:
        brown = (0.04, -0.12, 0.0015, -0.0008, 0.18)
        points = np.array(((80.0, 60.0), (640.0, 360.0), (1200.0, 700.0)))
        ideal = core.undistort_points(points, 900.0, 900.0, 640.0, 360.0, 0.0, brown)
        observed = core.distort_points(ideal, 900.0, 900.0, 640.0, 360.0, 0.0, brown)
        self.assertTrue(np.allclose(points, observed, atol=0.05))

    def test_rational_polynomial_round_trip(self) -> None:
        brown = (0.05, -0.02, 0.001, -0.0005, 0.01, 0.002, -0.001, 0.0003)
        points = np.array(((80.0, 60.0), (640.0, 360.0), (1200.0, 700.0)))
        ideal = core.undistort_points(points, 900.0, 900.0, 640.0, 360.0, 0.0, brown)
        observed = core.distort_points(ideal, 900.0, 900.0, 640.0, 360.0, 0.0, brown)
        self.assertTrue(np.allclose(points, observed, atol=0.05))

    def test_brown_conrady_strong_lens_round_trip(self) -> None:
        # Vintage Helios-style plumb_bob with large k2/k3.
        brown = (
            0.03949001840032624,
            -0.8131490107263357,
            -0.0017174455546758443,
            -0.00017175299763625847,
            6.289058039435576,
        )
        points = np.array(
            ((200.0, 150.0), (1727.0, 2675.0), (3000.0, 4800.0), (400.0, 5000.0))
        )
        fx, fy, cx, cy = 9641.58, 9683.23, 1726.89, 2675.15
        ideal = core.undistort_points(points, fx, fy, cx, cy, 0.0, brown)
        observed = core.distort_points(ideal, fx, fy, cx, cy, 0.0, brown)
        self.assertTrue(np.allclose(points, observed, atol=0.2))

    def test_brown_conrady_wins_over_lambda(self) -> None:
        points = np.array(((200.0, 150.0), (900.0, 500.0)))
        brown = (0.05, 0.0, 0.0, 0.0, 0.0)
        mixed = core.undistort_points(points, 800.0, 800.0, 640.0, 360.0, 0.12, brown)
        brown_only = core.undistort_points(
            points, 800.0, 800.0, 640.0, 360.0, 0.0, brown
        )
        lambda_only = core.undistort_points(points, 800.0, 800.0, 640.0, 360.0, 0.12)
        self.assertTrue(np.allclose(mixed, brown_only, atol=1.0e-9))
        self.assertFalse(np.allclose(mixed, lambda_only, atol=0.5))

    def test_brown_conrady_matches_opencv_when_present(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        brown = (-0.0014045245854740534, 0.0004527363803767526, 0.001, -0.0005, 0.0)
        fx, fy, cx, cy = 1587.08, 1593.07, 1505.33, 2003.91
        points = np.array(
            ((120.0, 90.0), (1505.0, 2004.0), (2800.0, 3700.0)),
            dtype=np.float64,
        )
        ours = core.undistort_points(points, fx, fy, cx, cy, 0.0, brown)
        camera_matrix = np.array(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        dist = np.array(brown, dtype=np.float64)
        opencv = cv2.undistortPoints(
            points.reshape(-1, 1, 2).astype(np.float32),
            camera_matrix,
            dist,
            P=camera_matrix,
        ).reshape(-1, 2)
        self.assertTrue(np.allclose(ours, opencv, atol=0.05))
        xyz = np.column_stack(
            (
                (points[:, 0] - cx) / fx,
                (points[:, 1] - cy) / fy,
                np.ones(len(points)),
            )
        )
        projected, _jacobian = cv2.projectPoints(
            xyz.astype(np.float64),
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            dist,
        )
        distorted = core.distort_points(points, fx, fy, cx, cy, 0.0, brown)
        self.assertTrue(np.allclose(distorted, projected.reshape(-1, 2), atol=0.05))

    def test_three_vp_principal_point(self) -> None:
        points = {
            "x": np.array((5.0, 5.0, 1.0)),
            "y": np.array((9.0, 5.0, 1.0)),
            "z": np.array((5.0, 9.0, 1.0)),
        }
        principal = core.principal_point_from_three_vps(points, 10, 10)
        self.assertIsNotNone(principal)
        self.assertTrue(np.allclose(principal, (5.0, 5.0)))

    def test_complete_vanishing_points_fills_missing_horizontal(self) -> None:
        """X + upright (Z) should imply the second ground-plane VP for the horizon."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=640.0,
            cy=360.0,
            image_width=1280,
            image_height=720,
        )
        # Orthogonal pair ZX around the principal point with known focal.
        vanishing_x = np.array((640.0 + 800.0, 360.0, 1.0))
        vanishing_y = np.array((640.0, 360.0 + 800.0, 1.0))
        completed = core.complete_vanishing_points(
            {"x": vanishing_x, "y": vanishing_y},
            intrinsics,
        )
        self.assertIn("z", completed)
        implied = completed["z"]
        self.assertGreater(abs(float(implied[2])), 1.0e-10)
        implied_xy = implied[:2] / implied[2]
        # World-Y VP should be left of center for this right-handed completion.
        self.assertLess(float(implied_xy[0]), 640.0)

    def test_vanishing_direction_round_trip(self) -> None:
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=500.0,
            cy=400.0,
            image_width=1000,
            image_height=800,
        )
        direction = np.array([0.3, -0.2, 1.0])
        vanishing = core.vanishing_from_camera_direction(direction, intrinsics)
        recovered = core._normalized_direction(vanishing, intrinsics)
        expected = direction / np.linalg.norm(direction)
        self.assertTrue(np.allclose(recovered, expected, atol=1.0e-9))

    def test_refine_camera_can_flip_only_horizontal_axis_polarity(self) -> None:
        """The user polarity choice reverses X/Y together and preserves Z."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=640.0,
            cy=360.0,
            image_width=1280,
            image_height=720,
        )
        bundles = {
            "x": [
                core.LineSegment(100.0, 250.0, 800.0, 300.0),
                core.LineSegment(100.0, 450.0, 800.0, 400.0),
            ],
            "z": [
                core.LineSegment(300.0, 100.0, 350.0, 650.0),
                core.LineSegment(900.0, 100.0, 850.0, 650.0),
            ],
            "y": [],
        }
        normal = core.refine_camera(
            bundles,
            intrinsics,
            lock_focal=True,
            estimate_principal_point=False,
        )
        flipped = core.refine_camera(
            bundles,
            intrinsics,
            lock_focal=True,
            estimate_principal_point=False,
            initial_rotation=normal.rotation_w2c,
            flip_xy_polarity=True,
        )

        self.assertTrue(
            np.allclose(flipped.rotation_w2c[:, :2], -normal.rotation_w2c[:, :2])
        )
        self.assertTrue(
            np.allclose(flipped.rotation_w2c[:, 2], normal.rotation_w2c[:, 2])
        )
        self.assertAlmostEqual(
            float(np.linalg.det(flipped.rotation_w2c)),
            1.0,
            places=9,
        )

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

    def test_can_solve_orientation_known_k_single_lines(self) -> None:
        bundles = {
            "x": [core.LineSegment(10.0, 10.0, 80.0, 20.0)],
            "y": [core.LineSegment(20.0, 90.0, 25.0, 10.0)],
            "z": [core.LineSegment(90.0, 40.0, 30.0, 45.0)],
        }
        self.assertTrue(
            core.can_solve_orientation(bundles, lock_focal=True, vp_mode="3")
        )
        self.assertFalse(
            core.can_solve_orientation(bundles, lock_focal=False, vp_mode="3")
        )
        self.assertFalse(
            core.can_solve_orientation(
                {"x": bundles["x"], "y": bundles["y"], "z": []},
                lock_focal=True,
                vp_mode="3",
            )
        )

    def test_rotation_from_one_orthogonal_line_per_axis(self) -> None:
        """Known K + one edge per world axis recovers orientation."""
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=640.0,
            cy=360.0,
            image_width=1280,
            image_height=720,
        )
        # Mild camera tilt / yaw in OpenCV w2c convention.
        yaw = np.radians(25.0)
        pitch = np.radians(-12.0)
        cosine_y, sine_y = np.cos(yaw), np.sin(yaw)
        cosine_p, sine_p = np.cos(pitch), np.sin(pitch)
        rotation_yaw = np.array(
            [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]]
        )
        rotation_pitch = np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine_p, -sine_p], [0.0, sine_p, cosine_p]]
        )
        rotation = rotation_pitch @ rotation_yaw

        def project(point: np.ndarray) -> tuple[float, float]:
            pixel = np.array(
                [
                    intrinsics.fx * point[0] / point[2] + intrinsics.cx,
                    intrinsics.fy * point[1] / point[2] + intrinsics.cy,
                ]
            )
            return float(pixel[0]), float(pixel[1])

        def axis_segment(column: int, origin: np.ndarray) -> core.LineSegment:
            direction = rotation[:, column]
            point_a = origin
            point_b = origin + 1.4 * direction
            image_a = project(point_a)
            image_b = project(point_b)
            return core.LineSegment(image_a[0], image_a[1], image_b[0], image_b[1])

        # Non-meeting edges (different origins) still determine attitude.
        bundles = {
            "x": [axis_segment(0, np.array([-0.8, 0.3, 4.0]))],
            "z": [axis_segment(1, np.array([0.5, -0.4, 3.5]))],
            "y": [axis_segment(2, np.array([-0.2, 0.6, 5.0]))],
        }
        recovered = core.rotation_from_orthogonal_lines(bundles, intrinsics)
        self.assertIsNotNone(recovered)
        # Compare absolute column alignment (sign flips are convention).
        for column in range(3):
            cosine = abs(float(np.dot(recovered[:, column], rotation[:, column])))
            self.assertGreater(cosine, 0.999)

        calibration = core.refine_camera(bundles, intrinsics, lock_focal=True)
        for column in range(3):
            cosine = abs(
                float(np.dot(calibration.rotation_w2c[:, column], rotation[:, column]))
            )
            self.assertGreater(cosine, 0.999)


    def test_extra_line_on_one_axis_improves_known_k_solve(self) -> None:
        """2+1+1 should pull toward the doubled axis, not ignore the second line."""
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=640.0,
            cy=360.0,
            image_width=1280,
            image_height=720,
        )
        yaw = np.radians(18.0)
        pitch = np.radians(-8.0)
        cosine_y, sine_y = np.cos(yaw), np.sin(yaw)
        cosine_p, sine_p = np.cos(pitch), np.sin(pitch)
        rotation = (
            np.array(
                [[1.0, 0.0, 0.0], [0.0, cosine_p, -sine_p], [0.0, sine_p, cosine_p]]
            )
            @ np.array(
                [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]]
            )
        )

        def project(point: np.ndarray) -> tuple[float, float]:
            return (
                float(intrinsics.fx * point[0] / point[2] + intrinsics.cx),
                float(intrinsics.fy * point[1] / point[2] + intrinsics.cy),
            )

        def axis_segment(
            direction: np.ndarray, origin: np.ndarray
        ) -> core.LineSegment:
            point_b = origin + 1.5 * direction
            image_a = project(origin)
            image_b = project(point_b)
            return core.LineSegment(image_a[0], image_a[1], image_b[0], image_b[1])

        # Deliberately bias the first X line; second X line is exact.
        skewed_x = rotation[:, 0] + np.array([0.0, 0.18, 0.0])
        skewed_x = skewed_x / np.linalg.norm(skewed_x)
        single = {
            "x": [axis_segment(skewed_x, np.array([-0.6, 0.2, 4.0]))],
            "z": [axis_segment(rotation[:, 1], np.array([0.4, -0.3, 3.6]))],
            "y": [axis_segment(rotation[:, 2], np.array([-0.1, 0.5, 4.8]))],
        }
        doubled = {
            "x": [
                single["x"][0],
                axis_segment(rotation[:, 0], np.array([0.7, -0.2, 4.2])),
            ],
            "z": single["z"],
            "y": single["y"],
        }
        from_single = core.rotation_from_orthogonal_lines(single, intrinsics)
        from_doubled = core.rotation_from_orthogonal_lines(doubled, intrinsics)
        self.assertIsNotNone(from_single)
        self.assertIsNotNone(from_doubled)
        single_align = sum(
            abs(float(np.dot(from_single[:, column], rotation[:, column])))
            for column in range(3)
        )
        doubled_align = sum(
            abs(float(np.dot(from_doubled[:, column], rotation[:, column])))
            for column in range(3)
        )
        self.assertGreater(doubled_align, single_align)
        self.assertGreater(
            abs(float(np.dot(from_doubled[:, 0], rotation[:, 0]))),
            abs(float(np.dot(from_single[:, 0], rotation[:, 0]))),
        )

    def test_known_k_three_fixed_axes_tries_svd_sign_patterns(self) -> None:
        """2+ lines/axis must not lock onto a single bad SVD sign pattern.

        Fixture: auto-detected bundles from a real still where the default
        (+,+,+) Procrustes seed polished to ~87° residual; other sign patterns
        recover ~0°. Regression for Detect VP Lines → crazy orientation.
        """
        intrinsics = core.CameraIntrinsics(
            fx=1583.8,
            fy=1583.8,
            cx=1500.0,
            cy=2000.0,
            image_width=3000,
            image_height=4000,
        )
        bundles = {
            "x": [
                core.LineSegment(2940.2264, 3473.6702, 1875.6214, 2597.3209),
                core.LineSegment(112.2659, 3322.0963, 167.7556, 2349.4858),
                core.LineSegment(2994.7651, 669.0202, 1657.4890, 946.9968),
                core.LineSegment(1367.8397, 3447.8317, 1479.4957, 3665.5128),
                core.LineSegment(2080.1317, 2005.7297, 2262.3186, 2081.0768),
                core.LineSegment(1080.5143, 2224.7454, 1381.6463, 2571.5824),
            ],
            "y": [
                core.LineSegment(-3.6395, 1164.2616, 314.2606, 1990.3084),
                core.LineSegment(2835.0988, 2317.1093, 2715.1560, 2584.9987),
                core.LineSegment(1708.2521, 986.1470, 1679.7289, 1716.7223),
                core.LineSegment(1230.0238, 3541.4795, 1269.1778, 3743.8313),
            ],
            "z": [
                core.LineSegment(1.7521, 765.5633, 1700.8734, 946.1683),
                core.LineSegment(988.6050, 3386.4552, 1334.8075, 3131.9792),
                core.LineSegment(1460.3174, 1971.0899, 1638.5843, 1915.7333),
                core.LineSegment(1005.5682, 1475.5297, 1232.6378, 1453.3864),
            ],
        }
        rotation = core.rotation_from_orthogonal_lines(bundles, intrinsics)
        self.assertIsNotNone(rotation)
        calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=rotation,
            camera_center=core.default_camera_center(rotation),
            division_lambda=0.0,
        )
        # Pre-fix default sign pattern scored ~2000 px / ~87° on this fixture.
        self.assertLess(core.vp_line_residual_rms(calibration, bundles), 20.0)
        self.assertLess(core.vp_angular_residual_degrees(calibration, bundles), 1.0)


if __name__ == "__main__":
    unittest.main()
