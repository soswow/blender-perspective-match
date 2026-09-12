"""Tests for VP residual helper and lens refine cost plumbing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

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


class LensEvidenceTests(unittest.TestCase):
    def test_real_search_keeps_recovered_camera_in_score(self):
        from tools.synthetic_sync.lens_support import run_search
        from tools.synthetic_sync.scenarios import read_case
        path = Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases/lens-recovered-score.json"
        report = run_search(read_case(path))
        self.assertEqual(report["lost_point_picks"], [])
        initial, selected = report["initial"], report["selected"]
        self.assertLessEqual(selected["all_points"]["rmse_px"], initial["all_points"]["rmse_px"] + 1e-6)
        for camera in ("view_0", "view_1"):
            self.assertLess(selected["assessment"]["cameras"][camera]["holdout_rmse_px"], 1e-4)
        self.assertFalse(report["improved"])
        # The harmful trial genuinely still has a lower Sync headline: changing
        # the outer comparison must not silently change ordinary Sync reporting.
        misleading = [trial for trial in report["trials"]
                      if trial["record"]["success"]
                      and trial["record"]["reported_rmse_px"] < initial["record"]["reported_rmse_px"] - 1.
                      and trial["all_points"]["rmse_px"] > initial["all_points"]["rmse_px"] + 1.]
        self.assertTrue(misleading)

    def test_real_search_recovers_biased_lenses_and_refused_start(self):
        from tools.synthetic_sync.lens_support import lens_case, run_search
        for kind in ("recovery", "refused_start"):
            with self.subTest(kind=kind):
                report = run_search(lens_case(kind))
                self.assertTrue(report["improved"])
                self.assertEqual(report["initial"]["record"]["success"], kind == "recovery")
                self.assertTrue(report["selected"]["assessment"]["passed"],
                                report["selected"]["assessment"]["violations"])
                self.assertLess(report["selected"]["all_points"]["rmse_px"], 1e-4)

    def test_real_search_keeps_useful_numerical_refusal(self):
        from tools.synthetic_sync.lens_support import lens_case, run_search
        report = run_search(lens_case("refused_improving"))
        self.assertFalse(report["initial"]["record"]["success"])
        self.assertFalse(report["selected"]["record"]["success"])
        self.assertTrue(report["improved"])
        self.assertLess(report["final_cost"], report["initial_cost"] - 20.)
        self.assertTrue(all(delta < 0 for delta in report["fx_deltas"].values()))

    def test_point_score_uses_geometry_independently_of_display_diagnostics(self):
        calibration = core.Calibration(core.CameraIntrinsics(800., 800., 400., 300., 800, 600),
                                       np.eye(3), np.zeros(3))
        calibrations = {key: calibration for key in ("A", "B")}
        result = sync.SyncSolveResult(
            similarities={key: sync.SimilarityTransform() for key in ("A", "B")},
            landmarks={key: np.array([0., 0., 2.]) for key in ("p", "q", "r")},
            mean_reprojection_px=1., per_match_rmse_px={"A": 3., "B": 9.},
            per_landmark_rmse_px={}, message="fixture")
        observations = [sync.SyncObservation("A", "p", 403., 300.),
                        sync.SyncObservation("B", "q", 409., 300.),
                        sync.SyncObservation("B", "r", 409., 300.),
                        sync.SyncObservation("B", "missing", 0., 0.),
                        sync.SyncObservation("disconnected", "p", 0., 0.)]
        for error in (None, 1., float("nan"), float("inf"), -1.):
            with self.subTest(display_error=error):
                if error is None:
                    result.per_match_rmse_px.pop("B", None)
                else:
                    result.per_match_rmse_px["B"] = error
                self.assertAlmostEqual(lens_refine._sync_rmse(result, observations, calibrations), np.sqrt(57.))
        for position in ([0., 0., -2.], [float("nan"), 0., 2.], [0., 0., float("inf")]):
            with self.subTest(position=position), np.errstate(invalid="ignore"):
                bad = deepcopy(result)
                bad.landmarks["q"] = np.asarray(position)
                self.assertTrue(np.isinf(lens_refine._joint_cost(
                    calibrations, {}, bad, vp_weight=0., observations=observations)))
        # Line-only input keeps its previous scalar policy.
        self.assertEqual(lens_refine._sync_rmse(result, []), 1.)

    def test_every_search_branch_preserves_incumbent_support(self):
        base = core.Calibration(core.CameraIntrinsics(800., 800., 400., 300., 800, 600),
                                np.eye(3), np.zeros(3))
        matches = [lens_refine.MatchLensInput(key, {}, base.intrinsics, base_calibration=base)
                   for key in ("A", "B")]
        observations = [sync.SyncObservation("A", "p", 410., 300.),
                        sync.SyncObservation("B", "q", 410., 300.)]
        baseline = sync.SyncSolveResult(
            similarities={key: sync.SimilarityTransform() for key in ("A", "B")},
            landmarks={key: np.array([0., 0., 2.]) for key in ("p", "q")},
            mean_reprojection_px=10., per_match_rmse_px={"A": 10., "B": 10.},
            per_landmark_rmse_px={}, message="fixture",
            line_segments={"edge": (np.zeros(3), np.ones(3))})
        phases = [(True, "Same lens ·"), (True, "Same lens refine ·"),
                  (False, "Pass 1/1 · A ·"), (False, "Pass 1/1 · A refine"),
                  (False, "Coupled · A/B"), (False, "Coupled · global")]
        for share_lens, phase in phases:
            for loss in ("camera", "point", "line", "success"):
                with self.subTest(phase=phase, loss=loss):
                    candidate = deepcopy(baseline)
                    candidate.mean_reprojection_px = 1.
                    for point in candidate.landmarks.values():
                        point[0] = .025
                    candidate.per_match_rmse_px = {"A": 1., "B": 1.}
                    if loss == "camera":
                        del candidate.similarities["B"]
                    elif loss == "point":
                        del candidate.landmarks["q"]
                    elif loss == "line":
                        del candidate.line_segments["edge"]
                    else:
                        candidate.success = False
                    progress = [""]
                    offered = []
                    def numerical(*args, **kwargs):
                        harmful = progress[0].startswith(phase)
                        if harmful:
                            offered.append(progress[0])
                        return candidate if harmful else baseline
                    def at_focal(match, focal):
                        return lens_refine.calibration_scaled_keep_pose(base, focal/base.intrinsics.fx)
                    with patch.object(lens_refine, "_run_sync", numerical), patch.object(
                        lens_refine, "calibration_at_focal", at_focal):
                        result = lens_refine.refine_lenses_from_landmarks(
                            matches, observations, anchor_id="A", share_lens=share_lens,
                            passes=1, coarse_samples=3, refine_samples=3, couple_samples=3,
                            progress_callback=lambda step, total, label: progress.__setitem__(0, label))
                    self.assertTrue(offered)
                    self.assertIs(result.sync_result, baseline)
                    self.assertFalse(result.improved)

    def test_support_can_grow_from_partial_success(self):
        partial = sync.SyncSolveResult(
            similarities={"A": sync.SimilarityTransform()}, landmarks={"p": np.zeros(3)},
            mean_reprojection_px=3., per_match_rmse_px={"A": 3.},
            per_landmark_rmse_px={}, message="partial")
        expanded = deepcopy(partial)
        expanded.similarities["B"] = sync.SimilarityTransform()
        expanded.landmarks["q"] = np.ones(3)
        self.assertTrue(lens_refine._retains_sync_support(expanded, partial))
        self.assertFalse(lens_refine._retains_sync_support(partial, expanded))


if __name__ == "__main__":
    unittest.main()
