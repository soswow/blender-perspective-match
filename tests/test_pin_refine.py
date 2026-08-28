"""Known 3D pin refine: source-image residuals with distortion locked."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import pin_refine
from match_perspective.core.sync.projection import project_private_point
from sync_fixtures import _look_at_rotation


def _project(point, calibration: core.Calibration) -> tuple[float, float]:
    projected = project_private_point(point, calibration)
    assert projected is not None, "test point is behind the camera"
    return float(projected[0]), float(projected[1])


def _true_pinhole() -> core.Calibration:
    return core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=1003.65,
            fy=1003.65,
            cx=616.82,
            cy=563.09,
            image_width=1200,
            image_height=1000,
        ),
        rotation_w2c=_look_at_rotation(
            np.array((-2.5, -7.0, 2.0), dtype=np.float64),
            np.array((0.6, 0.4, 0.7), dtype=np.float64),
        ),
        camera_center=np.array((-2.5, -7.0, 2.0), dtype=np.float64),
    )


_WORLD_POINTS = (
    np.array((0.0, 0.0, 0.0), dtype=np.float64),
    np.array((2.0, 0.0, 0.0), dtype=np.float64),
    np.array((0.0, 2.2, 0.0), dtype=np.float64),
    np.array((2.0, 2.2, 0.0), dtype=np.float64),
    np.array((0.0, 0.0, 1.6), dtype=np.float64),
    np.array((2.0, 0.0, 1.6), dtype=np.float64),
    np.array((0.4, 1.1, 0.5), dtype=np.float64),
    np.array((1.6, 0.5, 1.2), dtype=np.float64),
    np.array((1.1, 1.9, 0.8), dtype=np.float64),
)


def _pins_from_camera(calibration: core.Calibration) -> list[pin_refine.KnownPin]:
    pins: list[pin_refine.KnownPin] = []
    for index, point in enumerate(_WORLD_POINTS):
        u_coord, v_coord = _project(point, calibration)
        pins.append(
            pin_refine.KnownPin(
                landmark_id=f"p{index}",
                point_private=point,
                u=u_coord,
                v=v_coord,
            )
        )
    return pins


def _axis_segment(
    calibration: core.Calibration,
    start: np.ndarray,
    end: np.ndarray,
) -> core.LineSegment:
    u1_coord, v1_coord = _project(start, calibration)
    u2_coord, v2_coord = _project(end, calibration)
    return core.LineSegment(u1_coord, v1_coord, u2_coord, v2_coord)


def _vp_lines(calibration: core.Calibration) -> dict[core.AxisId, list[core.LineSegment]]:
    """World-axis box edges projected into source pixels."""
    return {
        "x": [
            _axis_segment(
                calibration,
                np.array((0.0, 0.0, 0.0)),
                np.array((2.0, 0.0, 0.0)),
            ),
            _axis_segment(
                calibration,
                np.array((0.0, 2.2, 0.0)),
                np.array((2.0, 2.2, 0.0)),
            ),
            _axis_segment(
                calibration,
                np.array((0.0, 0.0, 1.6)),
                np.array((2.0, 0.0, 1.6)),
            ),
        ],
        "z": [
            _axis_segment(
                calibration,
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 2.2, 0.0)),
            ),
            _axis_segment(
                calibration,
                np.array((2.0, 0.0, 0.0)),
                np.array((2.0, 2.2, 0.0)),
            ),
        ],
        "y": [
            _axis_segment(
                calibration,
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 0.0, 1.6)),
            ),
            _axis_segment(
                calibration,
                np.array((2.0, 0.0, 0.0)),
                np.array((2.0, 0.0, 1.6)),
            ),
            _axis_segment(
                calibration,
                np.array((2.0, 2.2, 0.0)),
                np.array((2.0, 2.2, 1.6)),
            ),
        ],
    }


class PinRefineTests(unittest.TestCase):
    def test_requires_four_pins(self) -> None:
        true = _true_pinhole()
        result = pin_refine.refine_from_known_pins(true, _pins_from_camera(true)[:3])
        self.assertFalse(result.success)
        self.assertIn("at least 4", result.message)

    def test_identity_at_lambda_zero(self) -> None:
        true = _true_pinhole()
        result = pin_refine.refine_from_known_pins(
            true,
            _pins_from_camera(true),
            _vp_lines(true),
        )
        self.assertTrue(result.success)
        self.assertLess(result.pin_rms_px, 0.05)
        self.assertAlmostEqual(result.calibration.intrinsics.fx, true.intrinsics.fx, delta=2.0)
        self.assertAlmostEqual(result.calibration.intrinsics.cx, true.intrinsics.cx, delta=2.0)
        self.assertAlmostEqual(result.calibration.intrinsics.cy, true.intrinsics.cy, delta=2.0)
        self.assertAlmostEqual(result.calibration.division_lambda, 0.0, places=6)

    def test_recovers_focal_and_principal_point(self) -> None:
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        start.intrinsics.fx = 1276.36
        start.intrinsics.fy = 1276.36
        start.intrinsics.cx = 636.34
        start.intrinsics.cy = 946.57
        start.camera_center = true.camera_center + np.array((0.15, -0.2, 0.08))
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
        )
        self.assertTrue(result.success)
        self.assertLess(result.pin_rms_px, 2.0)
        self.assertAlmostEqual(result.calibration.intrinsics.fx, true.intrinsics.fx, delta=55.0)
        self.assertAlmostEqual(result.calibration.intrinsics.cx, true.intrinsics.cx, delta=40.0)
        self.assertAlmostEqual(result.calibration.intrinsics.cy, true.intrinsics.cy, delta=80.0)
        self.assertLess(
            float(np.linalg.norm(result.calibration.camera_center - true.camera_center)),
            0.4,
        )

    def test_pinhole_beats_stored_overfit_lambda(self) -> None:
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        start.intrinsics.fx = 1276.36
        start.intrinsics.fy = 1276.36
        start.intrinsics.cx = 636.34
        start.intrinsics.cy = 946.57
        start.division_lambda = -0.292208
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
        )
        self.assertLess(result.vp_line_rms_px, 10.0)
        if result.success:
            self.assertLess(result.pin_rms_px, 25.0)

    def test_lambda_zero_does_not_recenter_principal_point(self) -> None:
        """Off-center PP is not a pinhole trial when λ is already 0."""
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        start.intrinsics.cx = 622.0
        start.intrinsics.cy = 742.0
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.hypothesis, pin_refine.HYPOTHESIS_STORED)
        self.assertLess(result.vp_line_rms_px, result.initial_vp_line_rms_px + 2.0)
        self.assertLess(result.pin_rms_px, 2.0)

    def test_locked_lambda_recovers_distorted_pins(self) -> None:
        true = _true_pinhole()
        true.division_lambda = -0.15
        start = pin_refine.copy_calibration(true)
        start.intrinsics.fx = 900.0
        start.intrinsics.fy = 900.0
        start.intrinsics.cx = 600.0
        start.intrinsics.cy = 500.0
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.hypothesis, pin_refine.HYPOTHESIS_STORED)
        self.assertAlmostEqual(result.calibration.division_lambda, -0.15, places=6)
        self.assertLess(result.pin_rms_px, 0.5)
        self.assertAlmostEqual(result.calibration.intrinsics.fx, true.intrinsics.fx, delta=20.0)

    def test_lock_rotation_keeps_seed_orientation(self) -> None:
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        optical = np.array(true.rotation_w2c[2], dtype=np.float64)
        optical /= float(np.linalg.norm(optical))
        angle = np.radians(3.0)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        skew = np.array(
            (
                (0.0, -optical[2], optical[1]),
                (optical[2], 0.0, -optical[0]),
                (-optical[1], optical[0], 0.0),
            ),
            dtype=np.float64,
        )
        roll = (
            np.eye(3, dtype=np.float64) * cosine
            + sine * skew
            + (1.0 - cosine) * np.outer(optical, optical)
        )
        start.rotation_w2c = roll @ true.rotation_w2c
        start.camera_center = true.camera_center + np.array((0.12, -0.08, 0.04))
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
            lock_rotation=True,
        )
        np.testing.assert_allclose(
            result.calibration.rotation_w2c,
            start.rotation_w2c,
            atol=1.0e-9,
        )
        self.assertLess(
            float(np.linalg.norm(result.calibration.camera_center - true.camera_center)),
            0.4,
        )

    def test_lock_focal_keeps_seed_focal(self) -> None:
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        start.intrinsics.fx = 1200.0
        start.intrinsics.fy = 1200.0
        start.camera_center = true.camera_center + np.array((0.1, -0.1, 0.05))
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            _vp_lines(true),
            lock_rotation=True,
            lock_focal=True,
        )
        self.assertAlmostEqual(result.calibration.intrinsics.fx, 1200.0, delta=1.0e-6)

    def test_orient_from_vp_keeps_uprights_when_principal_point_moves(self) -> None:
        true = _true_pinhole()
        start = pin_refine.copy_calibration(true)
        start.intrinsics.cy = true.intrinsics.cy + 80.0
        start.camera_center = true.camera_center + np.array((0.08, -0.05, 0.03))
        lines = _vp_lines(true)
        result = pin_refine.refine_from_known_pins(
            start,
            _pins_from_camera(true),
            lines,
            orient_from_vp=True,
        )
        self.assertTrue(result.success)
        self.assertLess(result.pin_rms_px, 2.0)
        self.assertLess(max(result.axis_rms_px), 2.0)

    def test_joint_score_penalizes_a_wrecked_vp_axis(self) -> None:
        true = _true_pinhole()
        stored = pin_refine.PinRefineResult(
            True,
            true,
            pin_rms_px=3.0,
            pin_max_px=4.0,
            axis_rms_px=(1.0, 14.0, 1.2),
            vp_line_rms_px=2.0,
            hypothesis=pin_refine.HYPOTHESIS_STORED,
        )
        pinhole = pin_refine.PinRefineResult(
            True,
            true,
            pin_rms_px=4.2,
            pin_max_px=5.0,
            axis_rms_px=(2.0, 3.5, 2.1),
            vp_line_rms_px=2.4,
            hypothesis=pin_refine.HYPOTHESIS_PINHOLE,
        )
        self.assertLess(pin_refine._joint_score(pinhole), pin_refine._joint_score(stored))

    def test_vp_guardrail_rejects_pose_that_wrecks_lines(self) -> None:
        true = _true_pinhole()
        rolled = pin_refine.copy_calibration(true)
        optical = true.rotation_w2c[2]
        angle = np.radians(35.0)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        skew = np.array(
            (
                (0.0, -optical[2], optical[1]),
                (optical[2], 0.0, -optical[0]),
                (-optical[1], optical[0], 0.0),
            ),
            dtype=np.float64,
        )
        roll = (
            np.eye(3, dtype=np.float64) * cosine
            + sine * skew
            + (1.0 - cosine) * np.outer(optical, optical)
        )
        rolled.rotation_w2c = roll @ true.rotation_w2c
        result = pin_refine.refine_from_known_pins(
            true,
            _pins_from_camera(rolled),
            _vp_lines(true),
        )
        self.assertFalse(result.success)
        self.assertIn("VP", result.message)
