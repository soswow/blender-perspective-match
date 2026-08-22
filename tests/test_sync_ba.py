"""Joint bundle adjustment and Diagnose leave-one-out."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from sync_fixtures import _look_at_rotation, _project, _rodrigues_z, _synthetic_scene


class BundleAdjustSyncTests(unittest.TestCase):
    """Joint bundle adjustment and Diagnose leave-one-out."""

    def test_leave_one_out_flags_outlier_landmark(self) -> None:
        """Diagnose helper should show removing a bad pick improves RMSE."""
        matches, observations, _true_sim, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        for observation in observations:
            if (
                observation.match_id == "other"
                and observation.landmark_id == "p4"
            ):
                observation.u += 80.0
                break
        baseline = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        report = sync.leave_one_out_landmark_report(
            matches,
            observations,
            anchor_id="anchor",
            baseline=baseline,
            top_k=3,
        )
        self.assertTrue(report)
        # The intentionally broken pick should appear and help when removed.
        names = [item[0] for item in report]
        self.assertTrue(any("p4" in name for name in names) or report[0][2] < report[0][1])
        best = min(report, key=lambda item: item[2])
        self.assertLess(best[2], baseline.mean_reprojection_px + 1.0)


    def test_auto_downweight_marks_severe_outlier(self) -> None:
        """Severe landmark RMSE should soft-downweight that landmark for BA."""
        observations = [
            sync.SyncObservation(
                match_id="anchor",
                landmark_id="good_a",
                u=100.0,
                v=100.0,
                weight=1.0,
            ),
            sync.SyncObservation(
                match_id="anchor",
                landmark_id="good_b",
                u=120.0,
                v=120.0,
                weight=1.0,
            ),
            sync.SyncObservation(
                match_id="anchor",
                landmark_id="good_c",
                u=140.0,
                v=140.0,
                weight=1.0,
            ),
            sync.SyncObservation(
                match_id="anchor",
                landmark_id="bad",
                u=200.0,
                v=200.0,
                weight=1.0,
            ),
        ]
        adjusted, ids = sync._auto_downweight_outlier_observations(
            observations,
            {"good_a": 2.0, "good_b": 3.0, "good_c": 2.5, "bad": 80.0},
        )
        self.assertEqual(ids, ["bad"])
        by_id = {item.landmark_id: item.weight for item in adjusted}
        self.assertLess(by_id["bad"], by_id["good_a"])
        self.assertAlmostEqual(by_id["good_a"], 1.0)


    def test_ba_jacobian_matches_finite_differences(self) -> None:
        """Block-analytic BA Jacobian should track dense finite differences."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        match_map = {item.match_id: item for item in matches}
        solved = sync.solve_landmark_sync(matches, observations, anchor_id="anchor")
        self.assertTrue(solved.success, solved.message)
        similarities = dict(solved.similarities)
        similarities["other"] = sync.SimilarityTransform(
            scale=solved.similarities["other"].scale,
            rotation=sync._rodrigues(
                sync._log_rodrigues(solved.similarities["other"].rotation)
                + np.array((0.04, -0.02, 0.03))
            ),
            translation=solved.similarities["other"].translation
            + np.array((0.15, -0.08, 0.04)),
        )
        landmarks = {
            landmark_id: point.copy()
            for landmark_id, point in solved.landmarks.items()
        }
        free_match_ids = ["other"]
        free_landmark_ids = [
            landmark_id
            for landmark_id in landmarks
            if not any(
                item.landmark_id == landmark_id and item.on_ground
                for item in observations
                if item.match_id == "anchor"
            )
        ][:3]
        fixed_landmarks = {
            landmark_id: landmarks[landmark_id].copy()
            for landmark_id in landmarks
            if landmark_id not in free_landmark_ids
        }
        params = sync._pack_ba_params(
            free_match_ids,
            free_landmark_ids,
            similarities,
            landmarks,
            lock_scale=True,
        )
        residual_kwargs = {
            "free_match_ids": free_match_ids,
            "free_landmark_ids": free_landmark_ids,
            "fixed_landmarks": fixed_landmarks,
            "anchor_id": "anchor",
            "matches": match_map,
            "observations": observations,
            "line_constraints": [],
            "lock_scale": True,
            "fixed_scales": {"other": 1.0},
            "huber_delta": 1.0e6,  # disable Huber so FD matches IRLS weights
        }
        analytic = sync._jacobian_ba(params, residual_kwargs)
        step = 1.0e-5
        base = sync._ba_residual_vector(params, **residual_kwargs)
        numeric = np.zeros_like(analytic)
        for index in range(params.size):
            perturbed = params.copy()
            delta = step if abs(float(params[index])) < 1.0 else step * abs(
                float(params[index])
            )
            perturbed[index] += delta
            sample = sync._ba_residual_vector(perturbed, **residual_kwargs)
            numeric[:, index] = (sample - base) / delta
        self.assertEqual(analytic.shape, numeric.shape)
        self.assertTrue(
            np.allclose(analytic, numeric, rtol=2.0e-2, atol=5.0e-2),
            msg=f"max abs diff {np.max(np.abs(analytic - numeric)):.4g}",
        )

