"""Landmark-graph sync solver regressions."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync


def _look_at_rotation(camera_center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build an OpenCV world-to-camera rotation looking toward target (Z forward)."""
    forward = target - camera_center
    forward = forward / np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.9:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def _rodrigues_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _project(point, calibration: core.Calibration) -> tuple[float, float]:
    projected = sync.project_private_point(point, calibration)
    assert projected is not None
    return float(projected[0]), float(projected[1])


def _synthetic_scene(*, with_ground: bool, yaw: float = 0.35) -> tuple:
    """Build two calibrated matches + landmark observations for sync tests."""
    intrinsics = core.CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=400.0,
        cy=300.0,
        image_width=800,
        image_height=600,
    )
    true_landmarks = {
        "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
        "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
        "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
        "p3": np.array((1.5, 1.0, 0.0), dtype=np.float64),
        "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
        "p5": np.array((-0.5, 1.2, 1.5), dtype=np.float64),
        "p6": np.array((0.8, -0.4, 0.9), dtype=np.float64),
    }
    ground_ids = {"p0", "p1", "p2", "p3"} if with_ground else set()

    true_sim = sync.SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues_z(yaw),
        translation=np.array((3.0, -2.0, 0.5), dtype=np.float64),
    )

    anchor_center = np.array((-3.0, -4.0, 2.0), dtype=np.float64)
    anchor_rotation = _look_at_rotation(anchor_center, np.array((0.5, 0.5, 0.0)))
    anchor_calibration = core.Calibration(
        intrinsics=intrinsics,
        rotation_w2c=anchor_rotation,
        camera_center=anchor_center,
    )

    shared_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
    shared_rotation = _look_at_rotation(shared_center, np.array((0.5, 0.5, 0.5)))
    rotation_private = shared_rotation @ true_sim.rotation
    center_private = true_sim.rotation.T @ (shared_center - true_sim.translation)
    other_calibration = core.Calibration(
        intrinsics=intrinsics,
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
    observations = []
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
                *_project(private_landmarks[landmark_id], other_calibration),
                on_ground=landmark_id in ground_ids,
            )
        )
    return matches, observations, true_sim, center_private, shared_center


class LandmarkSyncTests(unittest.TestCase):
    """Protect SfM-style 2D↔2D registration (optional ground for scale)."""

    def test_similarity_round_trip(self) -> None:
        transform = sync.SimilarityTransform(
            scale=1.0,
            rotation=_look_at_rotation(
                np.array((1.0, 2.0, 3.0)),
                np.array((0.0, 0.0, 0.0)),
            ),
            translation=np.array((4.0, -1.0, 2.0)),
        )
        point = np.array((1.5, -0.5, 3.0), dtype=np.float64)
        shared = transform.transform_point(point)
        recovered = transform.inverse_point(shared)
        self.assertTrue(np.allclose(point, recovered, atol=1.0e-9))

    def test_batched_shared_projection_matches_scalar_projection(self) -> None:
        """The optimized projection path must preserve scalar solver geometry."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        calibration = matches[1].calibration
        points = np.array(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.5, 0.2),
                (0.5, 1.5, 1.8),
                (-0.4, 0.8, 0.7),
            ),
            dtype=np.float64,
        )
        projected, valid = sync._project_shared_points(
            points,
            calibration,
            true_sim,
        )
        self.assertTrue(valid.all())
        for index, point in enumerate(points):
            expected = sync.project_private_point(
                true_sim.inverse_point(point),
                calibration,
            )
            self.assertIsNotNone(expected)
            self.assertTrue(np.allclose(projected[index], expected, atol=1.0e-9))

    def test_correspondences_recover_relative_pose(self) -> None:
        """Enough 2D↔2D picks solve orientation + baseline direction (no ground)."""
        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=False
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 1.0)

        recovered = result.similarities["other"]
        self.assertAlmostEqual(recovered.scale, 1.0, places=6)
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.08))

        # Absolute baseline scale is free without ground — only the direction of
        # C_b - C_a must match the truth.
        recovered_center = recovered.transform_point(center_private)
        true_offset = shared_center - matches[0].calibration.camera_center
        recovered_offset = recovered_center - matches[0].calibration.camera_center
        true_dir = true_offset / np.linalg.norm(true_offset)
        recovered_dir = recovered_offset / np.linalg.norm(recovered_offset)
        self.assertTrue(np.allclose(recovered_dir, true_dir, atol=0.08))

    def test_ground_fixes_absolute_baseline_scale(self) -> None:
        """Optional On Ground landmarks pin baseline length into the anchor world."""
        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=True
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 1.0)
        self.assertTrue(result.bundle_adjusted)

        recovered = result.similarities["other"]
        self.assertAlmostEqual(recovered.scale, 1.0, places=6)
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.05))
        self.assertTrue(np.allclose(recovered.translation, true_sim.translation, atol=0.2))

        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.2))

    def test_lock_rotation_and_translation_keeps_identity_empty(self) -> None:
        """Both locks: cameras stay put; BA only adjusts shared landmarks."""
        # Cameras already share a Blender frame (identity Empty is correct).
        matches, _observations, _true_sim, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        shared_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
        other_shared = core.Calibration(
            intrinsics=matches[1].calibration.intrinsics,
            rotation_w2c=_look_at_rotation(
                shared_center,
                np.array((0.5, 0.5, 0.5), dtype=np.float64),
            ),
            camera_center=shared_center,
        )
        true_landmarks = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
            "p3": np.array((1.5, 1.0, 0.0), dtype=np.float64),
            "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
            "p5": np.array((-0.5, 1.2, 1.5), dtype=np.float64),
            "p6": np.array((0.8, -0.4, 0.9), dtype=np.float64),
        }
        ground_ids = {"p0", "p1", "p2", "p3"}
        matches = [
            matches[0],
            sync.SyncMatchInput("other", other_shared),
        ]
        observations = []
        for landmark_id, point in true_landmarks.items():
            for match_id, calibration in (
                ("anchor", matches[0].calibration),
                ("other", other_shared),
            ):
                u_coord, v_coord = _project(point, calibration)
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        landmark_id,
                        u_coord,
                        v_coord,
                        on_ground=landmark_id in ground_ids and match_id == "anchor",
                        landmark_name=landmark_id,
                    )
                )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            lock_rotation=True,
            lock_translation=True,
        )
        self.assertTrue(result.success, result.message)
        other = result.similarities["other"]
        self.assertTrue(np.allclose(other.rotation, np.eye(3), atol=1.0e-9))
        self.assertTrue(np.allclose(other.translation, np.zeros(3), atol=1.0e-9))
        self.assertAlmostEqual(other.scale, 1.0, places=6)
        self.assertGreaterEqual(len(result.landmarks), 5)
        self.assertLess(result.mean_reprojection_px, 2.0)

    def test_lock_rotation_allows_90_degree_axis_jumps(self) -> None:
        """Lock Rotation may permute world axes by 90°, but not finer angles."""
        rotations = sync._axis_aligned_rotations()
        self.assertEqual(len(rotations), 24)
        for matrix in rotations:
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=9)
            self.assertTrue(np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-12))
        noisy = _rodrigues_z(0.5 * np.pi + 0.08)
        snapped = sync._snap_to_axis_aligned_rotation(noisy)
        self.assertTrue(np.allclose(snapped, _rodrigues_z(0.5 * np.pi), atol=1.0e-12))

        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=True,
            yaw=0.5 * np.pi,
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            lock_rotation=True,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 2.0)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=1.0e-8))
        self.assertTrue(np.allclose(recovered.translation, true_sim.translation, atol=0.2))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.2))

        identity_matches, identity_observations, identity_sim, _, _ = _synthetic_scene(
            with_ground=True,
            yaw=0.0,
        )
        identity_result = sync.solve_landmark_sync(
            identity_matches,
            identity_observations,
            anchor_id="anchor",
            lock_rotation=True,
        )
        self.assertTrue(identity_result.success, identity_result.message)
        identity_recovered = identity_result.similarities["other"]
        self.assertTrue(np.allclose(identity_recovered.rotation, np.eye(3), atol=1.0e-8))
        self.assertTrue(
            np.allclose(identity_recovered.translation, identity_sim.translation, atol=0.2)
        )

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

    def test_metric_sync_can_warm_start_from_previous_pose(self) -> None:
        """Nearby lens trials can reuse a solved metric pose without global search."""
        matches, observations, _true_sim, _center_private, _shared_center = (
            _synthetic_scene(with_ground=True)
        )
        initial = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(initial.success, initial.message)
        repeated = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            initial_similarities=initial.similarities,
        )
        self.assertTrue(repeated.success, repeated.message)
        self.assertLess(repeated.mean_reprojection_px, 1.0)
        self.assertTrue(
            np.allclose(
                repeated.similarities["other"].matrix(),
                initial.similarities["other"].matrix(),
                atol=1.0e-4,
            )
        )

    def test_collinear_landmarks_fail_with_clear_message(self) -> None:
        """Nearly collinear 2D picks must not KeyError — report geometry instead."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        anchor_center = np.array((-3.0, -4.0, 2.0), dtype=np.float64)
        anchor_rotation = _look_at_rotation(anchor_center, np.array((0.5, 0.5, 0.0)))
        anchor_calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=anchor_rotation,
            camera_center=anchor_center,
        )
        other_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
        other_rotation = _look_at_rotation(other_center, np.array((0.5, 0.5, 0.5)))
        other_calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=other_rotation,
            camera_center=other_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor_calibration),
            sync.SyncMatchInput("PM_mpv-shot0004_Origin", other_calibration),
        ]
        # Force collinear image picks (not via 3D projection — perspective can
        # add tiny numeric width that passes the spread check).
        observations = []
        for index, u_coordinate in enumerate((100.0, 200.0, 300.0, 400.0, 500.0)):
            landmark_id = f"line{index}"
            observations.append(
                sync.SyncObservation(
                    "anchor",
                    landmark_id,
                    u_coordinate,
                    250.0,
                )
            )
            observations.append(
                sync.SyncObservation(
                    "PM_mpv-shot0004_Origin",
                    landmark_id,
                    u_coordinate + 40.0,
                    280.0,
                )
            )

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertFalse(result.success)
        self.assertNotEqual(result.message, "'PM_mpv-shot0004_Origin'")
        self.assertIn("collinear", result.message.lower())

    def test_known_world_line_plus_offline_correspondences(self) -> None:
        """Known 3D verts on a line + off-line 2D↔2D should register metrically."""
        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=False
        )
        # Treat the four ground-plane corners as Known 3D (a degenerate line-ish
        # set alone would be weak; together with off-plane 2D↔2D it pins scale).
        known_world = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
            "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
        }
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 1.0)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.05))
        self.assertTrue(np.allclose(recovered.translation, true_sim.translation, atol=0.25))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.25))
        self.assertIn("known 3D", result.message)

    def test_collinear_known_3d_uses_offline_2d_pairs(self) -> None:
        """Sparse/collinear Known 3D must not cold-start PnP; 2D↔2D carries pose."""
        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=False
        )
        # Only two Known 3D on a line — scale cue only, not a full PnP set.
        known_world = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
        }
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 10.0)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.08))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.35))

    def test_known_world_pnp_without_five_shared_2d(self) -> None:
        """≥3 Known 3D + 2D in the other still is enough without five 2D↔2D pairs."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        true_sim = sync.SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues_z(0.35),
            translation=np.array((3.0, -2.0, 0.5), dtype=np.float64),
        )
        anchor_center = np.array((-3.0, -4.0, 2.0), dtype=np.float64)
        anchor_rotation = _look_at_rotation(anchor_center, np.array((0.5, 0.5, 0.0)))
        anchor_calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=anchor_rotation,
            camera_center=anchor_center,
        )
        shared_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
        shared_rotation = _look_at_rotation(shared_center, np.array((0.5, 0.5, 0.5)))
        rotation_private = shared_rotation @ true_sim.rotation
        center_private = true_sim.rotation.T @ (shared_center - true_sim.translation)
        other_calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=rotation_private,
            camera_center=center_private,
        )
        known_world = {
            "k0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "k1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "k2": np.array((0.0, 2.0, 0.0), dtype=np.float64),
            "k3": np.array((1.0, 1.0, 1.5), dtype=np.float64),
        }
        matches = [
            sync.SyncMatchInput("anchor", anchor_calibration),
            sync.SyncMatchInput("other", other_calibration),
        ]
        observations = []
        for landmark_id, point in known_world.items():
            private_point = true_sim.inverse_point(point)
            # Only pick in the other still — Known 3D supplies shared XYZ.
            observations.append(
                sync.SyncObservation(
                    "other",
                    landmark_id,
                    *_project(private_point, other_calibration),
                )
            )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
        )
        self.assertTrue(result.success, result.message)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.08))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.3))

    def test_known_3d_lines_register_pose(self) -> None:
        """≥3 Known 3D edges with 2D segments in the other still register pose."""
        matches, _observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        # Three non-parallel edges in shared world.
        known_lines = {
            "edge0": (
                np.array((0.0, 0.0, 0.0), dtype=np.float64),
                np.array((2.0, 0.0, 0.0), dtype=np.float64),
            ),
            "edge1": (
                np.array((0.0, 0.0, 0.0), dtype=np.float64),
                np.array((0.0, 2.0, 0.0), dtype=np.float64),
            ),
            "edge2": (
                np.array((0.0, 0.0, 1.0), dtype=np.float64),
                np.array((1.0, 1.0, 2.0), dtype=np.float64),
            ),
        }
        line_observations: list[sync.SyncLineObservation] = []
        for landmark_id, (point_a, point_b) in known_lines.items():
            private_a = true_sim.inverse_point(point_a)
            private_b = true_sim.inverse_point(point_b)
            ua, va = _project(private_a, other)
            ub, vb = _project(private_b, other)
            line_observations.append(
                sync.SyncLineObservation(
                    match_id="other",
                    landmark_id=landmark_id,
                    u1=ua,
                    v1=va,
                    u2=ub,
                    v2=vb,
                    landmark_name=landmark_id,
                )
            )
        result = sync.solve_landmark_sync(
            matches,
            [],
            anchor_id="anchor",
            line_observations=line_observations,
            known_lines=known_lines,
        )
        self.assertTrue(result.success, result.message)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.12))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.5))

    def test_parallel_line_pair_prefers_consistent_rotation(self) -> None:
        """Parallel-tagged lines should penalize a wrong relative yaw."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        # Two parallel edges along shared X.
        edge_a = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((2.0, 0.0, 0.0), dtype=np.float64),
        )
        edge_b = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((2.0, 1.0, 0.5), dtype=np.float64),
        )
        line_observations: list[sync.SyncLineObservation] = []
        for landmark_id, (point_a, point_b) in (
            ("edge_a", edge_a),
            ("edge_b", edge_b),
        ):
            for match_id, calibration, sim in (
                ("anchor", matches[0].calibration, sync.SimilarityTransform()),
                ("other", other, true_sim),
            ):
                if match_id == "anchor":
                    private_a, private_b = point_a, point_b
                else:
                    private_a = sim.inverse_point(point_a)
                    private_b = sim.inverse_point(point_b)
                ua, va = _project(private_a, calibration)
                ub, vb = _project(private_b, calibration)
                line_observations.append(
                    sync.SyncLineObservation(
                        match_id=match_id,
                        landmark_id=landmark_id,
                        u1=ua,
                        v1=va,
                        u2=ub,
                        v2=vb,
                        landmark_name=landmark_id,
                    )
                )
        good = sync._parallel_pair_rotation_error(
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_b"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_b"
            ),
            matches[0].calibration,
            other,
            true_sim,
        )
        bad_sim = sync.SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues_z(1.2),
            translation=true_sim.translation.copy(),
        )
        bad = sync._parallel_pair_rotation_error(
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_b"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_b"
            ),
            matches[0].calibration,
            other,
            bad_sim,
        )
        self.assertIsNotNone(good)
        self.assertIsNotNone(bad)
        self.assertLess(good, 0.05)
        self.assertGreater(bad, good + 0.2)

    def test_parallel_enforcement_locks_free_line_directions(self) -> None:
        """Is-Parallel-To should force free-line meshes to share a direction."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        # Two truly parallel edges along shared X; second will be seeded skewed.
        edge_good = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((2.0, 0.0, 0.0), dtype=np.float64),
        )
        edge_true = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((2.0, 1.0, 0.5), dtype=np.float64),
        )
        # Skewed seed (~30°) for the second free line mesh.
        edge_skewed = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((1.7, 1.0, 1.5), dtype=np.float64),
        )
        line_observations: dict[str, list[sync.SyncLineObservation]] = {
            "edge_good": [],
            "edge_bad": [],
        }
        for landmark_id, (point_a, point_b) in (
            ("edge_good", edge_good),
            ("edge_bad", edge_true),
        ):
            for match_id, calibration, sim in (
                ("anchor", matches[0].calibration, sync.SimilarityTransform()),
                ("other", matches[1].calibration, true_sim),
            ):
                if match_id == "anchor":
                    private_a, private_b = point_a, point_b
                else:
                    private_a = sim.inverse_point(point_a)
                    private_b = sim.inverse_point(point_b)
                ua, va = _project(private_a, calibration)
                ub, vb = _project(private_b, calibration)
                line_observations[landmark_id].append(
                    sync.SyncLineObservation(
                        match_id=match_id,
                        landmark_id=landmark_id,
                        u1=ua,
                        v1=va,
                        u2=ub,
                        v2=vb,
                        landmark_name=landmark_id,
                    )
                )
        match_map = {item.match_id: item for item in matches}
        similarities = {
            "anchor": sync.SimilarityTransform(),
            "other": true_sim,
        }
        line_segments = {
            "edge_good": edge_good,
            "edge_bad": edge_skewed,
        }
        landmarks = {
            "edge_good": 0.5 * (edge_good[0] + edge_good[1]),
            "edge_bad": 0.5 * (edge_skewed[0] + edge_skewed[1]),
        }
        before = sync._parallel_direction_error(
            edge_skewed[1] - edge_skewed[0],
            edge_good[1] - edge_good[0],
        )
        self.assertGreater(before, 0.3)
        sync._enforce_parallel_line_segments(
            line_segments,
            landmarks,
            [("edge_good", "edge_bad")],
            line_observations,
            similarities,
            match_map,
            known_lines={},
        )
        after = sync._parallel_direction_error(
            line_segments["edge_bad"][1] - line_segments["edge_bad"][0],
            line_segments["edge_good"][1] - line_segments["edge_good"][0],
        )
        self.assertLess(after, 0.05)

    def test_confidence_weight_mapping(self) -> None:
        self.assertEqual(sync.confidence_weight("HIGH"), 4.0)
        self.assertEqual(sync.confidence_weight("NORMAL"), 1.0)
        self.assertEqual(sync.confidence_weight("LOW"), 0.25)
        self.assertEqual(sync.confidence_weight("unknown"), 1.0)

    def test_low_confidence_outlier_softens_pose_pull(self) -> None:
        """A badly placed Low pick should bias pose less than the same High pick."""
        matches, observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=True
        )

        def with_outlier(weight: float) -> list[sync.SyncObservation]:
            cloned: list[sync.SyncObservation] = []
            for observation in observations:
                cloned.append(
                    sync.SyncObservation(
                        match_id=observation.match_id,
                        landmark_id=observation.landmark_id,
                        u=observation.u,
                        v=observation.v,
                        on_ground=observation.on_ground,
                        landmark_name=observation.landmark_name,
                        weight=1.0,
                    )
                )
            for observation in cloned:
                if (
                    observation.match_id == "other"
                    and observation.landmark_id == "p4"
                ):
                    observation.u += 55.0
                    observation.weight = weight
                    break
            return cloned

        result_low = sync.solve_landmark_sync(
            matches,
            with_outlier(0.25),
            anchor_id="anchor",
        )
        result_high = sync.solve_landmark_sync(
            matches,
            with_outlier(4.0),
            anchor_id="anchor",
        )
        self.assertTrue(result_low.success, result_low.message)

        def rotation_error(result: sync.SyncSolveResult) -> float:
            recovered = result.similarities["other"].rotation
            delta = recovered.T @ true_sim.rotation
            return float(np.linalg.norm(delta - np.eye(3)))

        # High-confidence outlier should pull the pose farther from truth.
        self.assertLessEqual(
            rotation_error(result_low),
            rotation_error(result_high) + 1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
