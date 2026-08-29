"""Joint bundle adjustment and Diagnose leave-one-out."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from match_perspective.core.sync import solve as solve_module
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

    def test_leave_one_out_computes_baseline_when_omitted(self) -> None:
        """Diagnose must import the solver before using it with no baseline."""
        matches, observations, _true_sim, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        report = sync.leave_one_out_landmark_report(
            matches,
            observations,
            anchor_id="anchor",
            top_k=1,
        )
        self.assertTrue(report)
        name, with_rmse, without_rmse = report[0]
        self.assertTrue(name)
        self.assertGreater(with_rmse, 0.0)
        self.assertGreaterEqual(without_rmse, 0.0)

    def test_leave_one_out_keeps_only_baseline_registered_matches(self) -> None:
        """Rejected cameras must not be globally re-registered per candidate."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        matches = list(matches) + [matches[1].__class__("rejected", matches[1].calibration)]
        observations = list(observations) + [
            sync.SyncObservation("rejected", "p4", 50.0, 60.0),
        ]
        baseline = sync.SyncSolveResult(
            similarities={
                "anchor": sync.SimilarityTransform(),
                "other": sync.SimilarityTransform(),
            },
            landmarks={"p4": np.zeros(3)},
            mean_reprojection_px=25.0,
            per_match_rmse_px={"anchor": 10.0, "other": 30.0},
            per_landmark_rmse_px={"p4": 50.0},
            message="partial",
            success=True,
        )
        solved = sync.SyncSolveResult(
            similarities=baseline.similarities,
            landmarks=baseline.landmarks,
            mean_reprojection_px=8.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="fast",
            success=True,
        )
        with mock.patch.object(
            solve_module,
            "solve_landmark_sync",
            return_value=solved,
        ) as solve_mock:
            report = sync.leave_one_out_landmark_report(
                matches,
                observations,
                anchor_id="anchor",
                baseline=baseline,
                top_k=1,
            )

        self.assertTrue(report)
        called_matches, called_observations = solve_mock.call_args.args[:2]
        self.assertEqual(
            {match.match_id for match in called_matches},
            {"anchor", "other"},
        )
        self.assertNotIn(
            "rejected",
            {observation.match_id for observation in called_observations},
        )
        self.assertTrue(solve_mock.call_args.kwargs["use_pose_cache"])

    def test_leave_one_out_honours_cancellation(self) -> None:
        """A queued Diagnose cancellation must stop before another re-solve."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        baseline = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        with self.assertRaises(sync.SyncCancelled):
            sync.leave_one_out_landmark_report(
                matches,
                observations,
                anchor_id="anchor",
                baseline=baseline,
                cancel_check=lambda: True,
            )

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

    def test_auto_downweight_skips_protected_outlier(self) -> None:
        """Raised Sync Weight must keep full pull even when RMSE is high."""
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
                landmark_id="keep",
                u=200.0,
                v=200.0,
                weight=8.0,
                protect_outlier=True,
            ),
        ]
        adjusted, ids = sync._auto_downweight_outlier_observations(
            observations,
            {"good_a": 2.0, "good_b": 3.0, "good_c": 2.5, "keep": 80.0},
        )
        self.assertEqual(ids, [])
        by_id = {item.landmark_id: item.weight for item in adjusted}
        self.assertEqual(by_id["keep"], 8.0)


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


    def test_spatial_balance_upweights_isolated_cell(self) -> None:
        """A lone corner pick should outweigh each tag in a central cluster."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=np.eye(3, dtype=np.float64),
            camera_center=np.zeros(3, dtype=np.float64),
        )
        matches = {"cam": sync.SyncMatchInput("cam", calibration)}
        observations = [
            sync.SyncObservation("cam", f"c{index}", 400.0 + index, 300.0, weight=1.0)
            for index in range(8)
        ]
        observations.append(
            sync.SyncObservation("cam", "edge", 40.0, 30.0, weight=1.0)
        )
        balanced = sync._balance_observation_weights(
            observations, matches, radial_gain=0.0
        )
        by_id = {item.landmark_id: item.weight for item in balanced}
        self.assertGreater(by_id["edge"] / by_id["c0"], 3.0)
        self.assertAlmostEqual(
            float(np.mean([item.weight for item in balanced])),
            1.0,
            places=6,
        )


    def test_spatial_balance_preserves_confidence_ratio(self) -> None:
        """High vs Normal in the same cell should keep their 4:1 ratio."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=np.eye(3, dtype=np.float64),
            camera_center=np.zeros(3, dtype=np.float64),
        )
        matches = {"cam": sync.SyncMatchInput("cam", calibration)}
        observations = [
            sync.SyncObservation("cam", "high", 410.0, 305.0, weight=4.0),
            sync.SyncObservation("cam", "normal", 390.0, 295.0, weight=1.0),
        ]
        balanced = sync._balance_observation_weights(observations, matches)
        by_id = {item.landmark_id: item.weight for item in balanced}
        self.assertAlmostEqual(by_id["high"] / by_id["normal"], 4.0, places=6)

    def test_spatial_balance_preserves_protect_outlier(self) -> None:
        """Spatial reweight must not drop the Sync Weight protect flag."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=np.eye(3, dtype=np.float64),
            camera_center=np.zeros(3, dtype=np.float64),
        )
        matches = {"cam": sync.SyncMatchInput("cam", calibration)}
        observations = [
            sync.SyncObservation(
                "cam", "keep", 40.0, 30.0, weight=4.0, protect_outlier=True
            ),
            sync.SyncObservation("cam", "plain", 400.0, 300.0, weight=1.0),
        ]
        balanced = sync._balance_observation_weights(observations, matches)
        by_id = {item.landmark_id: item for item in balanced}
        self.assertTrue(by_id["keep"].protect_outlier)
        self.assertFalse(by_id["plain"].protect_outlier)


    def test_edge_landmarks_keep_camera_depth_with_cluster(self) -> None:
        """Many central tags plus a few edge picks should still recover depth."""
        matches, observations, true_sim, _center, shared_center = (
            _cluster_and_edge_scene()
        )
        result = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(result.success, result.message)
        recovered = result.similarities["other"].transform_point(
            matches[1].calibration.camera_center
        )
        true_depth = float(np.linalg.norm(shared_center))
        recovered_depth = float(np.linalg.norm(recovered))
        self.assertLess(
            abs(recovered_depth - true_depth) / true_depth,
            0.12,
            msg=(
                f"depth {recovered_depth:.3f} vs true {true_depth:.3f} "
                f"({result.message})"
            ),
        )
        other_cal = matches[1].calibration
        inner: list[float] = []
        outer: list[float] = []
        similarity = result.similarities["other"]
        for observation in observations:
            if observation.match_id != "other":
                continue
            point = result.landmarks.get(observation.landmark_id)
            if point is None:
                continue
            projected = sync.project_private_point(
                similarity.inverse_point(point), other_cal
            )
            if projected is None:
                continue
            error = float(
                np.hypot(projected[0] - observation.u, projected[1] - observation.v)
            )
            radius = sync._image_radius_norm(
                observation.u, observation.v, other_cal
            )
            if radius < 0.35:
                inner.append(error)
            elif radius >= 0.55:
                outer.append(error)
        self.assertTrue(inner and outer, "need both inner and outer picks")
        inner_rmse = float(np.sqrt(np.mean(np.square(inner))))
        outer_rmse = float(np.sqrt(np.mean(np.square(outer))))
        self.assertLess(
            outer_rmse,
            inner_rmse * 3.0 + 4.0,
            msg=f"outer {outer_rmse:.1f}px vs inner {inner_rmse:.1f}px",
        )

    def test_ground_slack_does_not_pin_off_plane_on_ground(self) -> None:
        """Positive slack leaves On Ground free instead of pinning the Z=0 raycast."""
        metric = {"g": np.array((1.0, 2.0, 0.0), dtype=np.float64)}
        triangulated = {"g": np.array((1.0, 2.0, 0.08), dtype=np.float64)}
        pinned = sync._consistent_metric_landmarks(
            metric, triangulated, set(), ground_slack=0.0
        )
        self.assertIn("g", pinned)
        free = sync._consistent_metric_landmarks(
            metric, triangulated, set(), ground_slack=0.02
        )
        self.assertNotIn("g", free)

    def _known_3d_two_view_scene(self):
        """Two cameras seeing four Known 3D points; picks come from true XYZ."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0, fy=800.0, cx=400.0, cy=300.0, image_width=800, image_height=600
        )
        true_sim = sync.SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues_z(0.35),
            translation=np.array((3.0, -2.0, 0.5), dtype=np.float64),
        )
        true_landmarks = {
            "k0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "k1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "k2": np.array((0.0, 2.0, 0.0), dtype=np.float64),
            "k3": np.array((1.0, 1.0, 1.5), dtype=np.float64),
        }
        anchor_center = np.array((-3.0, -4.0, 2.0), dtype=np.float64)
        anchor_calibration = core.Calibration(
            intrinsics,
            _look_at_rotation(anchor_center, np.array((0.5, 0.5, 0.0))),
            anchor_center,
        )
        shared_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
        shared_rotation = _look_at_rotation(shared_center, np.array((0.5, 0.5, 0.5)))
        rotation_private = shared_rotation @ true_sim.rotation
        center_private = true_sim.rotation.T @ (shared_center - true_sim.translation)
        other_calibration = core.Calibration(
            intrinsics, rotation_private, center_private
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor_calibration),
            sync.SyncMatchInput("other", other_calibration),
        ]
        observations = []
        for landmark_id, point in true_landmarks.items():
            observations.append(
                sync.SyncObservation(
                    "anchor", landmark_id, *_project(point, anchor_calibration)
                )
            )
            observations.append(
                sync.SyncObservation(
                    "other",
                    landmark_id,
                    *_project(true_sim.inverse_point(point), other_calibration),
                )
            )
        return matches, observations, true_landmarks, true_sim

    def test_known_3d_slack_zero_keeps_empty_pin(self) -> None:
        """Slack 0 leaves a biased Known 3D point on its Empty."""
        matches, observations, true_landmarks, _true_sim = self._known_3d_two_view_scene()
        known_world = {key: point.copy() for key, point in true_landmarks.items()}
        known_world["k3"] = true_landmarks["k3"] + np.array(
            (0.12, 0.0, 0.0), dtype=np.float64
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            ground_slack=0.0,
            known_3d_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(
            np.allclose(result.landmarks["k3"], known_world["k3"], atol=1.0e-9)
        )

    def test_known_3d_slack_eases_biased_pin_toward_picks(self) -> None:
        """Positive slack lets 2D picks pull a biased Known 3D point toward truth."""
        matches, observations, true_landmarks, _true_sim = self._known_3d_two_view_scene()
        known_world = {key: point.copy() for key, point in true_landmarks.items()}
        known_world["k3"] = true_landmarks["k3"] + np.array(
            (0.12, 0.0, 0.0), dtype=np.float64
        )
        frozen = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            ground_slack=0.0,
            known_3d_slack=0.0,
        )
        eased = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            ground_slack=0.0,
            known_3d_slack=0.15,
        )
        self.assertTrue(frozen.success, frozen.message)
        self.assertTrue(eased.success, eased.message)
        dist_frozen = float(
            np.linalg.norm(frozen.landmarks["k3"] - true_landmarks["k3"])
        )
        dist_eased = float(
            np.linalg.norm(eased.landmarks["k3"] - true_landmarks["k3"])
        )
        self.assertLess(dist_eased, dist_frozen - 0.02)
        dist_from_empty = float(
            np.linalg.norm(eased.landmarks["k3"] - known_world["k3"])
        )
        self.assertLess(dist_from_empty, 0.4)
        self.assertLess(eased.mean_reprojection_px, frozen.mean_reprojection_px)

    def test_effective_ground_z_slack_uses_tighter_of_two(self) -> None:
        """On Ground Known 3D Z slack is min(ground, known); others keep ground."""
        known = {"k0"}
        self.assertEqual(
            sync._effective_ground_z_slack(
                "k0", ground_slack=0.01, known_3d_slack=0.05, known_ids=known
            ),
            0.01,
        )
        self.assertEqual(
            sync._effective_ground_z_slack(
                "k0", ground_slack=0.05, known_3d_slack=0.01, known_ids=known
            ),
            0.01,
        )
        self.assertEqual(
            sync._effective_ground_z_slack(
                "g", ground_slack=0.02, known_3d_slack=0.05, known_ids=known
            ),
            0.02,
        )

    def test_on_ground_known_3d_z_follows_tighter_ground_slack(self) -> None:
        """A floor Known 3D pick must not sink by the looser Known 3D slack."""
        matches, observations, true_landmarks, true_sim = (
            self._known_3d_two_view_scene()
        )
        for observation in observations:
            if observation.landmark_id in {"k0", "k1", "k2"}:
                observation.on_ground = True
        low = true_landmarks["k0"] + np.array((0.0, 0.0, -0.05), dtype=np.float64)
        anchor_calibration = matches[0].calibration
        other_calibration = matches[1].calibration
        for observation in observations:
            if observation.landmark_id != "k0":
                continue
            if observation.match_id == "anchor":
                observation.u, observation.v = _project(low, anchor_calibration)
            else:
                observation.u, observation.v = _project(
                    true_sim.inverse_point(low), other_calibration
                )
        known_world = {key: point.copy() for key, point in true_landmarks.items()}
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            ground_slack=0.01,
            known_3d_slack=0.05,
        )
        self.assertTrue(result.success, result.message)
        height = abs(float(result.landmarks["k0"][2]))
        self.assertLess(height, 0.02, msg=f"Z={result.landmarks['k0'][2]:.4f}")

    def test_thaw_pass_still_converges_when_structure_starts_frozen(self) -> None:
        """Pass B (thaw 3D) must still lock a well-observed synthetic scene."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        original = sync.BA_FREE_LANDMARK_LIMIT
        import match_perspective.core.sync.solve as solve_module

        try:
            sync.BA_FREE_LANDMARK_LIMIT = 0
            solve_module.BA_FREE_LANDMARK_LIMIT = 0
            result = sync.solve_landmark_sync(
                matches,
                observations,
                anchor_id="anchor",
                ground_slack=0.02,
            )
        finally:
            sync.BA_FREE_LANDMARK_LIMIT = original
            solve_module.BA_FREE_LANDMARK_LIMIT = original
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 2.0)
        self.assertIn("thaw 3D", result.message)


    def test_triangulation_downweights_nearly_parallel_view(self) -> None:
        """A grazing extra ray must not pull the point off the strong stereo pair."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0, fy=800.0, cx=400.0, cy=300.0, image_width=800, image_height=600
        )
        point = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        left = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((-2.0, 0.0, 2.0)), point),
            np.array((-2.0, 0.0, 2.0), dtype=np.float64),
        )
        right = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((2.0, 0.0, 2.0)), point),
            np.array((2.0, 0.0, 2.0), dtype=np.float64),
        )
        grazing = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((-1.95, 0.02, 2.02)), point),
            np.array((-1.95, 0.02, 2.02), dtype=np.float64),
        )
        origins = []
        directions = []
        for calibration in (left, right, grazing):
            projected = sync.project_private_point(point, calibration)
            self.assertIsNotNone(projected)
            origin, direction = sync.camera_ray_private(
                float(projected[0]), float(projected[1]), calibration
            )
            origins.append(origin)
            directions.append(direction)
        lateral = np.cross(directions[2], np.array((0.0, 0.0, 1.0)))
        lateral = lateral / max(float(np.linalg.norm(lateral)), 1.0e-12)
        origins[2] = origins[2] + 0.35 * lateral
        units = [item / max(float(np.linalg.norm(item)), 1.0e-12) for item in directions]
        naive = sync._linear_ray_midpoint(origins, units, [1.0, 1.0, 1.0])
        stereo = sync.triangulate_midpoint(origins[:2], directions[:2])
        with_grazing = sync.triangulate_midpoint(origins, directions)
        self.assertIsNotNone(naive)
        self.assertIsNotNone(stereo)
        self.assertIsNotNone(with_grazing)
        naive_error = float(np.linalg.norm(naive - stereo))
        weighted_error = float(np.linalg.norm(with_grazing - stereo))
        self.assertGreater(naive_error, 1.0e-4)
        self.assertLess(weighted_error, 0.85 * naive_error)


    def test_triangulation_drops_behind_camera_view(self) -> None:
        """A view that places the point behind the camera is ignored."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0, fy=800.0, cx=400.0, cy=300.0, image_width=800, image_height=600
        )
        point = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        left = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((-2.0, 0.0, 2.0)), point),
            np.array((-2.0, 0.0, 2.0), dtype=np.float64),
        )
        right = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((2.0, 0.0, 2.0)), point),
            np.array((2.0, 0.0, 2.0), dtype=np.float64),
        )
        behind = core.Calibration(
            intrinsics,
            _look_at_rotation(np.array((0.0, 0.0, -2.0)), np.array((0.0, 0.0, -4.0))),
            np.array((0.0, 0.0, -2.0), dtype=np.float64),
        )
        origins = []
        directions = []
        for calibration in (left, right):
            projected = sync.project_private_point(point, calibration)
            origin, direction = sync.camera_ray_private(
                float(projected[0]), float(projected[1]), calibration
            )
            origins.append(origin)
            directions.append(direction)
        origins.append(behind.camera_center.copy())
        directions.append(np.array((0.0, 0.0, -1.0), dtype=np.float64))
        stereo = sync.triangulate_midpoint(origins[:2], directions[:2])
        with_behind = sync.triangulate_midpoint(origins, directions)
        self.assertIsNotNone(stereo)
        self.assertIsNotNone(with_behind)
        self.assertTrue(np.allclose(with_behind, stereo, atol=0.05))


    def test_refine_triangulation_drops_outlier_observation(self) -> None:
        """Gauss–Newton must ignore a view whose pick is tens of pixels off."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0, fy=800.0, cx=400.0, cy=300.0, image_width=800, image_height=600
        )
        point = np.array((0.0, 0.0, 0.5), dtype=np.float64)
        cameras = {
            "left": core.Calibration(
                intrinsics,
                _look_at_rotation(np.array((-2.0, 0.0, 2.0)), point),
                np.array((-2.0, 0.0, 2.0), dtype=np.float64),
            ),
            "right": core.Calibration(
                intrinsics,
                _look_at_rotation(np.array((2.0, 0.0, 2.0)), point),
                np.array((2.0, 0.0, 2.0), dtype=np.float64),
            ),
            "front": core.Calibration(
                intrinsics,
                _look_at_rotation(np.array((0.0, -2.0, 2.0)), point),
                np.array((0.0, -2.0, 2.0), dtype=np.float64),
            ),
        }
        similarities = {
            match_id: sync.SimilarityTransform() for match_id in cameras
        }
        matches = {
            match_id: sync.SyncMatchInput(match_id, calibration)
            for match_id, calibration in cameras.items()
        }
        observations = []
        origins = []
        directions = []
        for match_id, calibration in cameras.items():
            projected = sync.project_private_point(point, calibration)
            u_coord, v_coord = float(projected[0]), float(projected[1])
            if match_id == "front":
                u_coord += 80.0
            observations.append(
                sync.SyncObservation(match_id, "p0", u_coord, v_coord)
            )
            origin, direction = sync.camera_ray_private(u_coord, v_coord, calibration)
            origins.append(origin)
            directions.append(direction)
        seed = sync.triangulate_midpoint(origins[:2], directions[:2])
        self.assertIsNotNone(seed)
        refined = sync.refine_triangulated_point(
            seed, observations, similarities, matches
        )
        self.assertIsNotNone(refined)
        self.assertLess(float(np.linalg.norm(refined - point)), 0.08)


def _cluster_and_edge_scene() -> tuple:
    """Many central ground tags, few elevated edge tags, slightly long fx."""
    true_fx = 900.0
    stored_fx = 900.0 * 1.06
    true_intrinsics = core.CameraIntrinsics(
        fx=true_fx,
        fy=true_fx,
        cx=400.0,
        cy=300.0,
        image_width=800,
        image_height=600,
    )
    stored_intrinsics = core.CameraIntrinsics(
        fx=stored_fx,
        fy=stored_fx,
        cx=400.0,
        cy=300.0,
        image_width=800,
        image_height=600,
    )
    true_landmarks: dict[str, np.ndarray] = {}
    ground_ids: set[str] = set()
    rng = np.random.default_rng(0)
    for index in range(12):
        landmark_id = f"c{index}"
        true_landmarks[landmark_id] = np.array(
            (
                rng.uniform(-0.22, 0.22),
                rng.uniform(-0.16, 0.16),
                0.0,
            ),
            dtype=np.float64,
        )
        ground_ids.add(landmark_id)
    for index, (x_coord, y_coord, z_coord) in enumerate(
        (
            (-1.6, -1.1, 0.0),
            (1.6, -1.1, 0.0),
            (-1.6, 1.1, 1.05),
            (1.6, 1.1, 1.05),
        )
    ):
        landmark_id = f"e{index}"
        true_landmarks[landmark_id] = np.array(
            (x_coord, y_coord, z_coord), dtype=np.float64
        )
        if z_coord <= 1.0e-6:
            ground_ids.add(landmark_id)

    true_sim = sync.SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues_z(0.28),
        translation=np.array((2.4, -1.1, 0.35), dtype=np.float64),
    )
    anchor_center = np.array((-3.2, -3.6, 2.4), dtype=np.float64)
    anchor_rotation = _look_at_rotation(
        anchor_center, np.array((0.0, 0.0, 0.4), dtype=np.float64)
    )
    anchor_calibration = core.Calibration(
        intrinsics=true_intrinsics,
        rotation_w2c=anchor_rotation,
        camera_center=anchor_center,
    )
    shared_center = np.array((3.4, -0.4, 1.55), dtype=np.float64)
    shared_rotation = _look_at_rotation(
        shared_center, np.array((0.0, 0.0, 0.45), dtype=np.float64)
    )
    rotation_private = shared_rotation @ true_sim.rotation
    center_private = true_sim.rotation.T @ (shared_center - true_sim.translation)
    other_true = core.Calibration(
        intrinsics=true_intrinsics,
        rotation_w2c=rotation_private,
        camera_center=center_private,
    )
    other_calibration = core.Calibration(
        intrinsics=stored_intrinsics,
        rotation_w2c=rotation_private,
        camera_center=center_private,
    )
    private_landmarks = {
        key: true_sim.inverse_point(point) for key, point in true_landmarks.items()
    }
    matches = [
        sync.SyncMatchInput("anchor", anchor_calibration),
        sync.SyncMatchInput("other", other_calibration),
    ]
    observations: list[sync.SyncObservation] = []
    for landmark_id, shared_point in true_landmarks.items():
        observations.append(
            sync.SyncObservation(
                "anchor",
                landmark_id,
                *_project(shared_point, anchor_calibration),
                on_ground=landmark_id in ground_ids,
            )
        )
        observations.append(
            sync.SyncObservation(
                "other",
                landmark_id,
                *_project(private_landmarks[landmark_id], other_true),
                on_ground=landmark_id in ground_ids,
            )
        )
    return matches, observations, true_sim, center_private, shared_center

