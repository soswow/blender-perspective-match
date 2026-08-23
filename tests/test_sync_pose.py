"""Pairwise pose, PnP, IPPE, and registration bridges."""

from __future__ import annotations

import math
import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from sync_fixtures import _look_at_rotation, _project, _rodrigues_z, _synthetic_scene


class PoseSyncTests(unittest.TestCase):
    """Pairwise pose, PnP, IPPE, and registration bridges."""

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


    def test_graph_bridge_recovers_camera_below_ground(self) -> None:
        """A pending camera may bridge through a solved non-anchor view and flip."""
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        local_center = np.array((-3.0, -4.0, 2.4), dtype=np.float64)
        local_rotation = _look_at_rotation(
            local_center,
            np.array((0.7, 0.7, 0.4), dtype=np.float64),
        )

        def local_calibration() -> core.Calibration:
            return core.Calibration(
                intrinsics,
                local_rotation.copy(),
                local_center.copy(),
            )

        def similarity_for_camera(
            center_shared: np.ndarray,
            target_shared: np.ndarray,
        ) -> sync.SimilarityTransform:
            shared_rotation = _look_at_rotation(center_shared, target_shared)
            rotation = shared_rotation.T @ local_rotation
            return sync.SimilarityTransform(
                scale=1.0,
                rotation=rotation,
                translation=center_shared - rotation @ local_center,
            )

        anchor_similarity = sync.SimilarityTransform()
        back_similarity = similarity_for_camera(
            np.array((4.0, -3.0, 2.2), dtype=np.float64),
            np.array((0.7, 0.7, 0.4), dtype=np.float64),
        )
        bottom_center = np.array((0.0, 2.5, -2.0), dtype=np.float64)
        bottom_similarity = similarity_for_camera(
            bottom_center,
            np.array((0.7, 0.7, 0.6), dtype=np.float64),
        )
        similarities = {
            "anchor": anchor_similarity,
            "back": back_similarity,
            "bottom": bottom_similarity,
        }
        match_map = {
            match_id: sync.SyncMatchInput(match_id, local_calibration())
            for match_id in similarities
        }
        matches = list(match_map.values())
        ground_points = {
            "g0": np.array((-1.2, -0.8, 0.0)),
            "g1": np.array((1.8, -0.7, 0.0)),
            "g2": np.array((1.7, 1.8, 0.0)),
            "g3": np.array((-1.1, 1.6, 0.0)),
            "g4": np.array((0.2, 0.1, 0.0)),
            "g5": np.array((0.8, 1.0, 0.0)),
        }
        bridge_points = {
            "b0": np.array((-0.4, -0.2, 0.2)),
            "b1": np.array((1.4, -0.1, 1.1)),
            "b2": np.array((1.5, 1.2, 0.3)),
            "b3": np.array((-0.3, 1.3, 1.0)),
            "b4": np.array((0.5, 0.4, 0.7)),
            "b5": np.array((1.0, 0.8, 0.2)),
        }
        observations: list[sync.SyncObservation] = []

        def observe(
            match_id: str,
            landmark_id: str,
            point: np.ndarray,
            ground=False,
            pixel_offset=(0.0, 0.0),
        ):
            private_point = similarities[match_id].inverse_point(point)
            u_coord, v_coord = _project(
                private_point,
                match_map[match_id].calibration,
            )
            observations.append(
                sync.SyncObservation(
                    match_id,
                    landmark_id,
                    u_coord + pixel_offset[0],
                    v_coord + pixel_offset[1],
                    on_ground=ground,
                )
            )

        for landmark_id, point in ground_points.items():
            observe("anchor", landmark_id, point, True)
            observe("back", landmark_id, point, True)
        for index, (landmark_id, point) in enumerate(bridge_points.items()):
            if index < 2:
                # Sparse overlaps can be imprecise or accidentally disagree
                # with the stronger bridge; they must not choose its pose branch.
                observe(
                    "anchor",
                    landmark_id,
                    point,
                    pixel_offset=(80.0, -60.0),
                )
            observe("back", landmark_id, point)
            observe("bottom", landmark_id, point)

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        recovered = result.similarities["bottom"]
        recovered_center = recovered.transform_point(local_center)
        self.assertLess(float(recovered_center[2]), -0.5)
        recovered_forward = recovered.rotation @ local_rotation.T[:, 2]
        self.assertGreater(float(recovered_forward[2]), 0.05)
        self.assertFalse(sync._is_collapsed_scale(recovered.scale))
        self.assertAlmostEqual(float(recovered.scale), 1.0, delta=0.05)

    def test_collapsed_log_scale_rewrites_as_rigid_camera(self) -> None:
        """LM floor s=exp(-18) must not be applied as an Empty scale."""
        intrinsics = core.CameraIntrinsics(
            fx=900.0,
            fy=900.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        private_center = np.array((0.0, -5.0, 1.7), dtype=np.float64)
        private_rotation = _look_at_rotation(
            private_center,
            np.array((0.5, 0.5, 0.0), dtype=np.float64),
        )
        calibration = core.Calibration(
            intrinsics, private_rotation, private_center
        )
        true_center = np.array((0.4, -0.6, -0.5), dtype=np.float64)
        true_rotation = _look_at_rotation(
            true_center,
            np.array((0.4, -0.6, 0.7), dtype=np.float64),
        )
        rotation_sim = true_rotation.T @ private_rotation
        collapsed = sync.SimilarityTransform(
            scale=math.exp(-sync.LOG_SCALE_CLIP),
            rotation=rotation_sim,
            translation=true_center.copy(),
        )
        self.assertTrue(sync._is_collapsed_scale(collapsed.scale))
        rigid = sync._metric_scale_similarity(collapsed, calibration)
        self.assertFalse(sync._is_collapsed_scale(rigid.scale))
        self.assertAlmostEqual(float(rigid.scale), 1.0, places=6)
        self.assertAlmostEqual(
            abs(float(np.linalg.det(rigid.matrix()[:3, :3]))), 1.0, places=6
        )
        self.assertTrue(
            np.allclose(
                rigid.transform_point(private_center),
                collapsed.transform_point(private_center),
                atol=1.0e-6,
            )
        )
        axis_private = private_rotation.T[:, 2]
        self.assertTrue(
            np.allclose(
                rigid.rotation @ axis_private,
                collapsed.rotation @ axis_private,
                atol=1.0e-8,
            )
        )


    def test_registration_candidate_uses_graph_to_resolve_pair_scale(self) -> None:
        """Whole-graph fit should choose scale within one pairwise pose branch."""
        raw = sync.SimilarityTransform(
            translation=np.array((0.0, 0.0, -4.0), dtype=np.float64)
        )
        scaled = sync.SimilarityTransform(
            translation=np.array((0.0, 0.0, -1.0), dtype=np.float64)
        )
        refined = sync.SimilarityTransform(
            translation=np.array((0.0, 0.0, -0.75), dtype=np.float64)
        )
        wrong_branch = sync.SimilarityTransform(
            translation=np.array((2.0, 1.0, 1.5), dtype=np.float64)
        )
        selected = sync._select_registration_candidate(
            [
                (0.54, 56_759.0, raw),
                (0.54, 181.0, scaled),
                (3.10, 4.91, refined),
                (22.0, 1.0, wrong_branch),
            ]
        )
        self.assertIs(selected, refined)


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


    def test_seven_pairs_recover_180_degree_yaw_at_free_baseline_scale(self) -> None:
        """A valid seven-point pose must survive arbitrary baseline scaling."""
        intrinsics = core.CameraIntrinsics(
            fx=2865.96435546875,
            fy=2865.96435546875,
            cx=1496.944580078125,
            cy=2021.071044921875,
            image_width=3000,
            image_height=4000,
        )
        anchor = core.Calibration(
            intrinsics,
            np.array(
                (
                    (0.8189265132, -0.5736991763, 0.0151208136),
                    (-0.1400556564, -0.2253344953, -0.9641622305),
                    (0.5565462708, 0.7874602675, -0.2648822069),
                ),
                dtype=np.float64,
            ),
            np.array((-1.1623859406, -1.6919496059, 1.7000000477)),
        )
        other = core.Calibration(
            intrinsics,
            np.array(
                (
                    (-0.7696733475, 0.3569790721, 0.5293098092),
                    (0.0371971242, 0.8527356982, -0.5210165381),
                    (-0.6373533607, -0.3813237250, -0.6696065068),
                ),
                dtype=np.float64,
            ),
            np.array((0.7041849494, 0.0871355608, 1.7000000477)),
        )
        image_pairs = (
            ((1317.6358643, 2895.2255859), (848.9439087, 3226.3789062)),
            ((795.4300537, 1549.1529541), (934.4205322, 2043.3775635)),
            ((1906.5633545, 1321.6416016), (1874.6555176, 2783.4299316)),
            ((1371.7391357, 1901.9372559), (1198.5540771, 3067.3647461)),
            ((937.6423950, 2508.5764160), (706.5392456, 2635.0869141)),
            ((2092.9230957, 1596.9106445), (1866.3979492, 2992.4057617)),
            ((1756.7243652, 2410.5490723), (1288.6295166, 3252.9245605)),
        )
        pairs = [
            (
                sync.SyncObservation("anchor", f"p{index}", *anchor_uv),
                sync.SyncObservation("other", f"p{index}", *other_uv),
            )
            for index, (anchor_uv, other_uv) in enumerate(image_pairs)
        ]

        recovered = sync._solve_relative_from_pairs(pairs, anchor, other)

        self.assertIsNotNone(recovered)
        assert recovered is not None
        errors = sync._reprojection_errors_for_similarity(
            recovered, pairs, anchor, other, [], []
        )
        self.assertLess(float(np.sqrt(np.mean(np.square(errors)))), 1.0)
        self.assertTrue(
            np.allclose(
                recovered.rotation,
                np.diag((-1.0, -1.0, 1.0)),
                atol=0.08,
            )
        )


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


    def test_ippe_matches_homography_when_plane_origin_is_off_axis(self) -> None:
        """IPPE must recover H even when the plane origin is far from the PP."""
        intrinsics = core.CameraIntrinsics(3821.0, 3821.0, 2016.0, 1512.0, 4032, 3024)
        camera_center = np.array((0.2, 0.3, 1.7), dtype=np.float64)
        true_camera = core.Calibration(
            intrinsics,
            _look_at_rotation(camera_center, np.array((0.2, 0.3, 0.0))),
            camera_center,
        )
        leftover = core.Calibration(
            intrinsics,
            _look_at_rotation(
                np.array((-2.0, -3.0, 1.7)), np.array((0.5, 0.5, 0.0))
            ),
            np.array((-2.0, -3.0, 1.7)),
        )
        ground_points = np.array(
            (
                (-0.8, -0.9, 0.0),
                (0.9, -0.85, 0.0),
                (-0.7, 1.1, 0.0),
                (0.15, 1.05, 0.0),
                (0.85, 1.15, 0.0),
                (-0.2, 0.95, 0.0),
                (0.4, -0.8, 0.0),
            ),
            dtype=np.float64,
        )
        image_points = np.array(
            [_project(point, true_camera) for point in ground_points],
            dtype=np.float64,
        )
        poses = sync._planar_homography_similarities(
            ground_points, image_points, leftover
        )
        self.assertTrue(poses)
        best_rmse = float("inf")
        best_center = None
        for pose in poses:
            errors = sync._reprojection_errors_for_similarity(
                pose,
                [],
                leftover,
                leftover,
                list(ground_points),
                list(image_points),
            )
            if not errors:
                continue
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            if rmse < best_rmse:
                best_rmse = rmse
                best_center = pose.transform_point(leftover.camera_center)
        self.assertLess(best_rmse, 2.0, f"IPPE reprojection {best_rmse:.1f}px")
        self.assertIsNotNone(best_center)
        self.assertTrue(np.allclose(best_center, camera_center, atol=0.08))

