"""Focused controls for the frozen-geometry distortion postpass."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from match_perspective import core
from match_perspective.core import distortion_refine, lens_refine, sync
from match_perspective.core.focal_bundle import FocalBundleOutcome
from match_perspective.core.joint_fit_score import JointFitScore


class DistortionRefineTests(unittest.TestCase):
    @staticmethod
    def _oracle_distort(ideal, intrinsics, division_lambda):
        """Independent scalar inversion of ideal r = observed r / (1 + λr²)."""
        normalized = np.array((
            (ideal[0] - intrinsics.cx) / intrinsics.fx,
            (ideal[1] - intrinsics.cy) / intrinsics.fy,
        ), dtype=float)
        ideal_radius = float(np.linalg.norm(normalized))
        if ideal_radius < 1.0e-12 or abs(division_lambda) < 1.0e-12:
            return np.asarray(ideal, dtype=float)
        observed_radius = ideal_radius
        for _ in range(30):
            denominator = 1.0 + division_lambda * observed_radius * observed_radius
            residual = observed_radius / denominator - ideal_radius
            derivative = ((1.0 - division_lambda * observed_radius * observed_radius) /
                          (denominator * denominator))
            observed_radius -= residual / derivative
        distorted = normalized * (observed_radius / ideal_radius)
        return np.array((
            distorted[0] * intrinsics.fx + intrinsics.cx,
            distorted[1] * intrinsics.fy + intrinsics.cy,
        ))

    def _case(self, *, truth_lambda: float, noise_px: float = 0.0):
        intrinsics = core.CameraIntrinsics(800.0, 800.0, 400.0, 300.0, 800, 600)
        start = core.Calibration(intrinsics, np.eye(3), np.zeros(3))
        truth = core.Calibration(
            intrinsics, np.eye(3), np.zeros(3), division_lambda=truth_lambda)
        rng = np.random.default_rng(317)
        landmarks = {}
        observations = []
        held_out = []
        xs = (-0.46, -0.32, -0.18, -0.06, 0.06, 0.18, 0.32, 0.46)
        ys = (-0.33, -0.20, -0.07, 0.07, 0.20, 0.33)
        for y_index, y_value in enumerate(ys):
            for x_index, x_value in enumerate(xs):
                ideal = np.array((
                    intrinsics.cx + intrinsics.fx * x_value,
                    intrinsics.cy + intrinsics.fy * y_value,
                ))
                point = np.array((
                    (ideal[0] - intrinsics.cx) / intrinsics.fx * 2.0,
                    (ideal[1] - intrinsics.cy) / intrinsics.fy * 2.0,
                    2.0,
                ))
                observed = self._oracle_distort(ideal, intrinsics, truth_lambda)
                if noise_px:
                    observed = observed + rng.normal(0.0, noise_px, 2)
                self.assertGreaterEqual(float(observed[0]), 0.0)
                self.assertLessEqual(float(observed[0]), intrinsics.image_width)
                self.assertGreaterEqual(float(observed[1]), 0.0)
                self.assertLessEqual(float(observed[1]), intrinsics.image_height)
                point_id = f"p{y_index}_{x_index}"
                if (x_index + y_index) % 4:
                    landmarks[point_id] = point
                    observations.append(sync.SyncObservation(
                        "camera", point_id, float(observed[0]), float(observed[1])))
                else:
                    held_out.append((point, observed))
        result = sync.SyncSolveResult(
            similarities={"camera": sync.SimilarityTransform()},
            landmarks=landmarks,
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="fixture",
            success=True,
        )
        return start, truth, observations, result, held_out

    @staticmethod
    def _held_out_rmse(calibration, held_out):
        errors = []
        for point, observed in held_out:
            camera = calibration.rotation_w2c @ (point - calibration.camera_center)
            ideal = np.array((
                calibration.intrinsics.fx * camera[0] / camera[2] + calibration.intrinsics.cx,
                calibration.intrinsics.fy * camera[1] / camera[2] + calibration.intrinsics.cy,
            ))
            projected = DistortionRefineTests._oracle_distort(
                ideal, calibration.intrinsics, calibration.division_lambda)
            errors.append(np.asarray(projected) - observed)
        return float(np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=1))))

    def test_recovers_barrel_distortion_and_improves_withheld_geometry(self):
        start, _truth, observations, result, held_out = self._case(
            truth_lambda=-0.12)
        outcome = distortion_refine.refine_division_distortion(
            {"camera": start}, observations, result, pick_sigma_px=0.5)
        self.assertEqual(outcome.accepted_match_ids, ["camera"])
        fitted = outcome.calibrations["camera"]
        self.assertAlmostEqual(fitted.division_lambda, -0.12, delta=0.005)
        self.assertLess(
            self._held_out_rmse(fitted, held_out),
            self._held_out_rmse(start, held_out) * 0.05,
        )
        self.assertEqual(fitted.intrinsics, start.intrinsics)
        self.assertTrue(np.array_equal(fitted.rotation_w2c, start.rotation_w2c))
        self.assertTrue(np.array_equal(fitted.camera_center, start.camera_center))

    def test_no_distortion_with_noise_keeps_original_calibration(self):
        start, _truth, observations, result, _held_out = self._case(
            truth_lambda=0.0, noise_px=0.35)
        outcome = distortion_refine.refine_division_distortion(
            {"camera": start}, observations, result, pick_sigma_px=0.35)
        self.assertEqual(outcome.accepted_match_ids, [])
        self.assertIs(outcome.calibrations["camera"], start)
        self.assertIn("validation", outcome.skipped_reasons["camera"])

    def test_brown_conrady_is_preserved(self):
        start, _truth, observations, result, _held_out = self._case(
            truth_lambda=-0.12)
        start.brown_conrady = (-0.1, 0.02, 0.0, 0.0, 0.0)
        outcome = distortion_refine.refine_division_distortion(
            {"camera": start}, observations, result, pick_sigma_px=0.5)
        self.assertEqual(outcome.accepted_match_ids, [])
        self.assertIs(outcome.calibrations["camera"], start)
        self.assertIn("Brown", outcome.skipped_reasons["camera"])

    def test_cancellation_discards_partial_postpass(self):
        start, _truth, observations, result, _held_out = self._case(
            truth_lambda=-0.12)
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls > 5

        outcome = distortion_refine.refine_division_distortion(
            {"camera": start}, observations, result,
            pick_sigma_px=0.5, cancel_check=cancel)
        self.assertTrue(outcome.cancelled)
        self.assertEqual(outcome.accepted_match_ids, [])
        self.assertIs(outcome.calibrations["camera"], start)

    def test_full_image_invertibility_rejects_unsafe_positive_lambda(self):
        wide = core.CameraIntrinsics(350.0, 350.0, 400.0, 300.0, 800, 600)
        self.assertFalse(distortion_refine._division_valid_over_image(wide, 0.12))
        self.assertTrue(distortion_refine._division_valid_over_image(wide, 0.02))

    def test_existing_lambda_outside_bound_is_preserved(self):
        start, _truth, observations, result, _held_out = self._case(
            truth_lambda=-0.12)
        start.division_lambda = -0.5
        outcome = distortion_refine.refine_division_distortion(
            {"camera": start}, observations, result, pick_sigma_px=0.5)
        self.assertEqual(outcome.accepted_match_ids, [])
        self.assertIs(outcome.calibrations["camera"], start)
        self.assertIn("outside", outcome.skipped_reasons["camera"])

    def _wrapper_case(self):
        intrinsics = core.CameraIntrinsics(800.0, 800.0, 400.0, 300.0, 800, 600)
        base = {
            key: core.Calibration(intrinsics, np.eye(3), np.zeros(3))
            for key in ("A", "B")
        }
        matches = [lens_refine.MatchLensInput(
            key, {}, intrinsics, base_calibration=base[key]) for key in ("A", "B")]
        observations = [sync.SyncObservation(key, "p", 400.0, 300.0)
                        for key in ("A", "B")]
        initial = sync.SyncSolveResult(
            similarities={key: sync.SimilarityTransform() for key in ("A", "B")},
            landmarks={"p": np.array((0.0, 0.0, 2.0))},
            mean_reprojection_px=0.0, per_match_rmse_px={},
            per_landmark_rmse_px={}, message="initial", success=True)
        fitted_calibrations = {
            key: lens_refine.calibration_scaled_keep_pose(base[key], 1.05)
            for key in base
        }
        fitted = sync.SyncSolveResult(
            similarities=dict(initial.similarities), landmarks=dict(initial.landmarks),
            mean_reprojection_px=0.0, per_match_rmse_px={},
            per_landmark_rmse_px={}, message="fitted", success=True)
        distorted_calibrations = {
            key: distortion_refine._calibration_at_lambda(calibration, -0.1)
            for key, calibration in fitted_calibrations.items()
        }

        bundle = FocalBundleOutcome(
            True, "accepted", calibrations=fitted_calibrations,
            sync_result=fitted, intervals_px={key: (780.0, 900.0) for key in base})
        postpass = distortion_refine.DistortionRefineOutcome(
            distorted_calibrations, ["A", "B"], {})
        return matches, observations, initial, bundle, postpass

    def test_wrapper_bypasses_postpass_by_default(self):
        matches, observations, initial, bundle, postpass = self._wrapper_case()
        with mock.patch.object(lens_refine, "_run_sync", return_value=initial), \
             mock.patch.object(lens_refine, "JointFitScorer", self._fake_scorer(4.0, observations)), \
             mock.patch.object(lens_refine, "fit_independent_focals", return_value=bundle), \
             mock.patch.object(lens_refine, "refine_division_distortion",
                               return_value=postpass) as refine_distortion, \
             mock.patch.object(lens_refine, "joint_line_support_diagnostics",
                               return_value=({}, [])):
            result = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="A", known_world={"p": (0.0, 0.0, 2.0)},
                estimate_focal_from_points=True, estimate_distortion=False)
        refine_distortion.assert_not_called()
        self.assertIs(result.calibrations, bundle.calibrations)
        self.assertNotIn("distortion", result.message)

    @staticmethod
    def _fake_scorer(distortion_objective, observations):
        class FakeScorer:
            def __init__(self, _request, _initial, **_kwargs):
                self.point_observations = observations
                self.effective_point_weights = [(item.match_id, item.landmark_id, 1.0)
                                                for item in observations]
                self.downweighted_landmark_ids = []

            def score(self, _result, *, calibrations=None):
                if any(abs(item.division_lambda) > 1.0e-8
                       for item in calibrations.values()):
                    objective = distortion_objective
                elif any(item.intrinsics.fx > 810.0 for item in calibrations.values()):
                    objective = 5.0
                else:
                    objective = 10.0
                return JointFitScore(
                    objective=objective, point_rmse_px=objective,
                    line_rmse_px=0.0, valid=True,
                    supported_observations=len(observations))
        return FakeScorer

    def test_wrapper_publishes_only_globally_improving_postpass(self):
        for distortion_objective, accepted in ((4.0, True), (6.0, False)):
            with self.subTest(distortion_objective=distortion_objective):
                matches, observations, initial, bundle, postpass = self._wrapper_case()
                with mock.patch.object(lens_refine, "_run_sync", return_value=initial), \
                     mock.patch.object(lens_refine, "JointFitScorer",
                                       self._fake_scorer(distortion_objective, observations)), \
                     mock.patch.object(lens_refine, "fit_independent_focals",
                                       return_value=bundle), \
                     mock.patch.object(lens_refine, "refine_division_distortion",
                                       return_value=postpass), \
                     mock.patch.object(lens_refine, "joint_line_support_diagnostics",
                                       return_value=({}, [])):
                    result = lens_refine.refine_lenses_from_landmarks(
                        matches, observations, anchor_id="A",
                        known_world={"p": (0.0, 0.0, 2.0)},
                        estimate_focal_from_points=True, estimate_distortion=True)
                if accepted:
                    self.assertIs(result.calibrations, postpass.calibrations)
                    self.assertEqual(result.final_cost, 4.0)
                    self.assertEqual(result.sync_result.mean_reprojection_px, 4.0)
                    self.assertIn("distortion refined", result.message)
                else:
                    self.assertIs(result.calibrations, bundle.calibrations)
                    self.assertEqual(result.final_cost, 5.0)
                    self.assertEqual(result.sync_result.mean_reprojection_px, 5.0)
                    self.assertIn("distortion unchanged", result.message)


if __name__ == "__main__":
    unittest.main()
