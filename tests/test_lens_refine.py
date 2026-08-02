"""Tests for VP residual helper and lens refine cost plumbing."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective import lens_refine


class VpResidualTests(unittest.TestCase):
    def test_residual_finite_after_locked_refine(self) -> None:
        # Two orthogonal vanishing directions via simple line pairs.
        bundles = {
            "x": [
                core.LineSegment(100.0, 200.0, 700.0, 220.0),
                core.LineSegment(100.0, 400.0, 700.0, 380.0),
            ],
            "y": [
                core.LineSegment(300.0, 50.0, 310.0, 550.0),
                core.LineSegment(500.0, 50.0, 490.0, 550.0),
            ],
            "z": [],
        }
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        calibration = core.refine_camera(
            bundles,
            intrinsics,
            lock_focal=True,
            estimate_principal_point=False,
            estimate_distortion=False,
        )
        residual = core.vp_angular_residual_degrees(calibration, bundles)
        self.assertGreaterEqual(residual, 0.0)
        self.assertLess(residual, 45.0)


class LensRefinePlumbingTests(unittest.TestCase):
    def test_evaluation_count_excludes_already_scored_grid_centers(self) -> None:
        self.assertEqual(
            lens_refine.estimate_refine_evaluation_count(1),
            29,
        )
        self.assertEqual(
            lens_refine.estimate_refine_evaluation_count(4),
            113,
        )

    def test_calibration_at_focal_changes_fx(self) -> None:
        bundles = {
            "x": [
                core.LineSegment(100.0, 200.0, 700.0, 220.0),
                core.LineSegment(100.0, 400.0, 700.0, 380.0),
            ],
            "z": [
                core.LineSegment(200.0, 100.0, 220.0, 500.0),
                core.LineSegment(600.0, 100.0, 580.0, 500.0),
            ],
            "y": [],
        }
        match = lens_refine.MatchLensInput(
            match_id="A",
            line_bundles=bundles,
            intrinsics=core.CameraIntrinsics(
                fx=900.0,
                fy=900.0,
                cx=400.0,
                cy=300.0,
                image_width=800,
                image_height=600,
            ),
        )
        calibration = lens_refine.calibration_at_focal(match, 750.0)
        self.assertAlmostEqual(calibration.intrinsics.fx, 750.0, places=3)
        self.assertAlmostEqual(calibration.intrinsics.fy, 750.0, places=3)
        self.assertTrue(np.isfinite(calibration.rotation_w2c).all())

    def test_cancel_check_stops_early(self) -> None:
        bundles = {
            "x": [
                core.LineSegment(100.0, 200.0, 700.0, 220.0),
                core.LineSegment(100.0, 400.0, 700.0, 380.0),
            ],
            "z": [
                core.LineSegment(200.0, 100.0, 220.0, 500.0),
                core.LineSegment(600.0, 100.0, 580.0, 500.0),
            ],
            "y": [],
        }
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        matches = [
            lens_refine.MatchLensInput(
                match_id="A",
                line_bundles=bundles,
                intrinsics=intrinsics,
            ),
            lens_refine.MatchLensInput(
                match_id="B",
                line_bundles=bundles,
                intrinsics=intrinsics,
            ),
        ]
        calls = {"n": 0}

        def cancel_after_two() -> bool:
            calls["n"] += 1
            return calls["n"] > 2

        result = lens_refine.refine_lenses_from_landmarks(
            matches,
            [],
            anchor_id="A",
            cancel_check=cancel_after_two,
        )
        self.assertTrue(result.cancelled)
        self.assertIn("cancelled", result.message.lower())


if __name__ == "__main__":
    unittest.main()
