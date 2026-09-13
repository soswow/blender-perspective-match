"""A better constrained fit can have a slightly worse point-only start score."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core import geometry, sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_orientation
from tools.synthetic_sync.geometry import project


class FocalCandidateTradeoffTests(TestCase):
    def test_physical_candidate_survives_point_rmse_tradeoff(self):
        case = focal_orientation.generate()
        request, truth = case["request"], case["truth"]
        start_points, start_cameras = focal_orientation.anchor_gauge_start(case)
        true_cameras = {item["id"]: item for item in truth["cameras"]}
        calibrations, similarities = {}, {}
        for stored in request["cameras"]:
            camera_id = stored["id"]
            true = true_cameras[camera_id]
            # Give the image-perfect start its true K. It remains in the
            # tilted anchor chart, where the declared world priors are false.
            k = geometry.CameraIntrinsics(
                true["fx"], true["fy"], stored["cx"], stored["cy"],
                stored["width"], stored["height"])
            calibration = geometry.Calibration(
                k, np.asarray(stored["rotation"]), np.asarray(stored["center"]))
            calibrations[camera_id] = calibration
            target = start_cameras[camera_id]
            rotation = np.asarray(target["rotation"]).T @ calibration.rotation_w2c
            similarities[camera_id] = sync.SimilarityTransform(
                1., rotation,
                np.asarray(target["center"]) - rotation @ calibration.camera_center)
        similarities[request["anchor_id"]] = sync.SimilarityTransform()
        observations = [sync.SyncObservation(**item) for item in request["observations"]]
        start_errors = []
        for item in request["observations"]:
            uv = project([start_points[item["landmark_id"]]],
                         start_cameras[item["match_id"]])[0][0]
            start_errors.append(np.linalg.norm(uv - (item["u"], item["v"])))
        start_rmse = float(np.sqrt(np.mean(np.square(start_errors))))
        self.assertLess(start_rmse, 1e-9)
        group = request["plane_groups"]
        self.assertGreater(np.ptp([start_points[key][0] for key, _, _ in group]), .05)
        mirror_origin, mirror_normal = (np.asarray(value) for value in request["mirror_plane"])
        initial_mirror_gaps = []
        for left, right in request["mirror_pairs"]:
            a, b = np.asarray(start_points[left]), np.asarray(start_points[right])
            reflected = a - 2 * mirror_normal * float(mirror_normal @ (a - mirror_origin))
            initial_mirror_gaps.append(np.linalg.norm(reflected - b))
        self.assertGreater(max(initial_mirror_gaps), .05)
        initial = sync.SyncSolveResult(
            similarities=similarities,
            landmarks={key: np.asarray(value) for key, value in start_points.items()},
            mean_reprojection_px=start_rmse, per_match_rmse_px={},
            per_landmark_rmse_px={}, message="Exact pixels; wrong world frame",
            success=True)
        # Isolate a completed numerical endpoint that the normal calibration
        # gate refuses. The fit, constraints and camera/point checks remain real.
        with patch("match_perspective.core.focal_bundle._fits_constrained_noise_model",
                   return_value=False):
            outcome = fit_independent_focals(
                calibrations, observations, initial,
                anchor_id=request["anchor_id"], pick_sigma_px=case["pick_sigma_px"],
                fx_span=.25, plane_groups=request["plane_groups"], plane_slack=0.,
                mirror_pairs=request["mirror_pairs"],
                mirror_plane=request["mirror_plane"], mirror_slack=0.)
        self.assertFalse(outcome.accepted)
        self.assertIn("stated pick noise", outcome.reason)
        candidate = outcome.candidate
        self.assertIsNotNone(candidate, outcome.reason)
        self.assertGreater(candidate.fitted_rmse_px, start_rmse)
        self.assertLess(candidate.fitted_rmse_px, .01)
        self.assertIsNotNone(candidate.initial_objective)
        self.assertIsNotNone(candidate.fitted_objective)
        self.assertLess(candidate.fitted_objective, candidate.initial_objective)
        self.assertEqual(candidate.reason, outcome.reason)
        points = candidate.sync_result.landmarks
        self.assertLess(np.ptp([points[key][0] for key, _, _ in group]), 1e-4)
        origin, normal = (np.asarray(value) for value in request["mirror_plane"])
        for left, right in request["mirror_pairs"]:
            reflected = points[left] - 2 * normal * float(normal @ (points[left] - origin))
            self.assertLess(np.linalg.norm(reflected - points[right]), 1e-4)
        self.assertLess(np.linalg.norm(
            candidate.calibrations[request["anchor_id"]].rotation_w2c -
            np.asarray(true_cameras[request["anchor_id"]]["rotation"])), .001)
        # A disjoint point set checks the applied world frame and camera
        # projection, without aligning on the withheld geometry itself.
        holdout_errors = []
        for camera_id, true in true_cameras.items():
            calibration = candidate.calibrations[camera_id]
            similarity = candidate.sync_result.similarities[camera_id]
            recovered = dict(true,
                fx=calibration.intrinsics.fx, fy=calibration.intrinsics.fy,
                rotation=calibration.rotation_w2c @ similarity.rotation.T,
                center=(similarity.scale * similarity.rotation @ calibration.camera_center +
                        similarity.translation))
            for key, point in truth["holdouts"].items():
                pixel = project([point], recovered)[0][0]
                holdout_errors.append(np.linalg.norm(
                    pixel - truth["holdout_pixels"][key][camera_id]))
        self.assertLess(max(holdout_errors), .01)
