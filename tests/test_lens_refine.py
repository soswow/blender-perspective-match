"""Tests for VP residual helper and lens refine cost plumbing."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import lens_refine, sync


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
        line_rms = core.vp_line_residual_rms(calibration, bundles)
        self.assertTrue(np.isfinite(line_rms))
        self.assertGreaterEqual(line_rms, 0.0)


class LensRefinePlumbingTests(unittest.TestCase):
    def test_joint_cost_rejects_bad_vp_line_rms(self) -> None:
        """Hard VP guardrails must fail trials that wreck line agreement."""
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
        calibration = core.refine_camera(bundles, intrinsics, lock_focal=True)
        match = lens_refine.MatchLensInput(
            match_id="A",
            line_bundles=bundles,
            intrinsics=intrinsics,
        )
        sync_ok = sync.SyncSolveResult(
            similarities={},
            landmarks={},
            mean_reprojection_px=2.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="ok",
            success=True,
        )
        # Force an impossible line-RMS ceiling so any real residual fails.
        cost = lens_refine._joint_cost(
            {"A": calibration},
            {"A": match},
            sync_ok,
            vp_weight=4.0,
            max_vp_line_rms=1.0e-9,
        )
        self.assertGreaterEqual(cost, lens_refine._FAILURE_COST * 0.5)

    def test_joint_cost_allows_noisy_baseline_vp(self) -> None:
        """Relative VP guardrails must not freeze refine on a messy start."""
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
        calibration = core.refine_camera(bundles, intrinsics, lock_focal=True)
        match = lens_refine.MatchLensInput(
            match_id="A",
            line_bundles=bundles,
            intrinsics=intrinsics,
        )
        sync_ok = sync.SyncSolveResult(
            similarities={},
            landmarks={},
            mean_reprojection_px=12.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="ok",
            success=True,
        )
        _vp_term, line_rms, angle = lens_refine._vp_terms(
            {"A": calibration}, {"A": match}
        )
        # Absolute ceiling alone would reject; baseline must keep this alive.
        cost = lens_refine._joint_cost(
            {"A": calibration},
            {"A": match},
            sync_ok,
            vp_weight=4.0,
            max_vp_line_rms=1.0e-9,
            max_vp_angle_deg=1.0e-9,
            baseline_max_line_rms=line_rms,
            baseline_max_angle=angle,
        )
        self.assertLess(cost, lens_refine._FAILURE_COST * 0.5)
        self.assertGreater(cost, 0.0)

    def test_evaluation_count_excludes_already_scored_grid_centers(self) -> None:
        self.assertEqual(
            lens_refine.estimate_refine_evaluation_count(1),
            29,
        )
        # 4 free cameras: coordinate descent (113) + 3 coupled pairs × 8 + 2 scale.
        self.assertEqual(
            lens_refine.estimate_refine_evaluation_count(4),
            139,
        )
        self.assertEqual(
            lens_refine.estimate_refine_evaluation_count(4, share_lens=True),
            15,
        )

    def test_ranked_couple_pairs_prefer_shared_landmarks(self) -> None:
        observations = [
            sync.SyncObservation("A", "p1", 0.0, 0.0),
            sync.SyncObservation("B", "p1", 1.0, 1.0),
            sync.SyncObservation("B", "p2", 2.0, 2.0),
            sync.SyncObservation("C", "p2", 3.0, 3.0),
            sync.SyncObservation("C", "p3", 4.0, 4.0),
            sync.SyncObservation("D", "p3", 5.0, 5.0),
        ]
        pairs = lens_refine._ranked_couple_pairs(
            ["A", "B", "C", "D"],
            observations,
            limit=2,
        )
        self.assertEqual(len(pairs), 2)
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("B", "C"), pairs)

    def test_scaled_calibration_keeps_pose_and_aspect(self) -> None:
        base = core.Calibration(
            core.CameraIntrinsics(
                fx=1000.0,
                fy=980.0,
                cx=400.0,
                cy=300.0,
                image_width=800,
                image_height=600,
            ),
            rotation_w2c=np.eye(3, dtype=np.float64),
            camera_center=np.array((1.0, 2.0, 3.0), dtype=np.float64),
        )
        scaled = lens_refine.calibration_scaled_keep_pose(base, 1.1)
        self.assertAlmostEqual(scaled.intrinsics.fx, 1100.0, places=3)
        self.assertAlmostEqual(scaled.intrinsics.fy, 1078.0, places=3)
        self.assertTrue(np.allclose(scaled.rotation_w2c, base.rotation_w2c))
        self.assertTrue(np.allclose(scaled.camera_center, base.camera_center))

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
