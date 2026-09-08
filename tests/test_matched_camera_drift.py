"""MATCHED live Camera vs stored calibration drift."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core


def _calibration(
    *,
    camera_center=(0.0, 0.0, 1.7),
    fx: float = 900.0,
    cx: float = 600.0,
    cy: float = 450.0,
    yaw_deg: float = 0.0,
) -> core.Calibration:
    yaw = np.radians(yaw_deg)
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    rotation = np.array(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )
    return core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=fx,
            fy=fx,
            cx=cx,
            cy=cy,
            image_width=1200,
            image_height=900,
        ),
        rotation_w2c=rotation,
        camera_center=np.array(camera_center, dtype=np.float64),
    )


class MatchedCameraDriftTests(unittest.TestCase):
    def test_identical_pose_is_not_drifted(self) -> None:
        calibration = _calibration()
        self.assertFalse(core.private_pose_is_drifted(calibration, calibration))

    def test_tiny_roundtrip_noise_is_not_drifted(self) -> None:
        stored = _calibration()
        live = _calibration(camera_center=(1.0e-4, -2.0e-4, 1.7001))
        self.assertFalse(core.private_pose_is_drifted(stored, live, live_scale=1.0001))

    def test_18cm_center_shift_is_drifted(self) -> None:
        stored = _calibration()
        live = _calibration(camera_center=(0.016, -0.124, 1.57))
        self.assertTrue(core.private_pose_is_drifted(stored, live))

    def test_scale_away_from_one_is_drifted(self) -> None:
        calibration = _calibration()
        self.assertTrue(
            core.private_pose_is_drifted(calibration, calibration, live_scale=0.92)
        )

    def test_yaw_and_focal_and_pp_count_as_drift(self) -> None:
        stored = _calibration()
        self.assertTrue(core.private_pose_is_drifted(stored, _calibration(yaw_deg=1.0)))
        self.assertTrue(core.private_pose_is_drifted(stored, _calibration(fx=980.0)))
        self.assertTrue(
            core.private_pose_is_drifted(stored, _calibration(cx=620.0, cy=470.0))
        )
