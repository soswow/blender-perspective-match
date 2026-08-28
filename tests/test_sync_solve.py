"""Top-level solve: peel, resect, nadir, partial sync."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from match_perspective.core.sync import solve as solve_module
from sync_fixtures import _look_at_rotation, _project, _rodrigues_z, _synthetic_scene


class SolveSyncTests(unittest.TestCase):
    """Top-level solve: peel, resect, nadir, partial sync."""

    def test_confidence_weight_mapping(self) -> None:
        self.assertEqual(sync.confidence_weight("HIGH"), 4.0)
        self.assertEqual(sync.confidence_weight("NORMAL"), 1.0)
        self.assertEqual(sync.confidence_weight("LOW"), 0.25)
        self.assertEqual(sync.confidence_weight("unknown"), 1.0)

    def test_resect_mismatch_checks_only_worst_picks_with_local_refits(self) -> None:
        """Mismatch diagnosis must not repeat the global PnP seed grid."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        other_observations = [
            observation
            for observation in observations
            if observation.match_id == "other"
        ]
        observations_by_landmark = {
            observation.landmark_id: [observation]
            for observation in other_observations
        }
        cloud = {
            observation.landmark_id: np.array((index, index % 2, 0.5))
            for index, observation in enumerate(other_observations)
        }
        seed = sync.SimilarityTransform()
        retry_kwargs = {
            "observations_by_landmark": observations_by_landmark,
            "matches": {match.match_id: match for match in matches},
            "anchor_id": "anchor",
            "known_lines": None,
            "line_observations_by_landmark": None,
            "parallel_pairs": None,
            "lock_rotation": False,
            "lock_translation": False,
            "use_pose_cache": False,
            "cancel_check": lambda: False,
        }
        with mock.patch.object(
            solve_module,
            "_try_register_against_cloud",
            return_value=None,
        ) as register_mock:
            report = solve_module._resect_mismatch_picks(
                "other",
                cloud,
                retry_kwargs,
                seed,
            )

        self.assertEqual(report, [])
        self.assertEqual(
            register_mock.call_count,
            sync.RESECT_MISMATCH_CANDIDATE_LIMIT,
        )
        for call in register_mock.call_args_list:
            self.assertTrue(call.kwargs["initial_only"])
            self.assertIs(call.kwargs["initial_similarity"], seed)

    def test_solve_honours_cancellation_before_registration(self) -> None:
        """Background Diagnose cancellation must abort the core solve."""
        matches, observations, _true, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        with self.assertRaises(sync.SyncCancelled):
            sync.solve_landmark_sync(
                matches,
                observations,
                anchor_id="anchor",
                cancel_check=lambda: True,
            )


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


    def test_wrong_on_ground_falls_back_to_2d_pairs(self) -> None:
        """Elevated tags marked On Ground must not veto a good 2D↔2D pose."""
        matches, observations, true_sim, center_private, shared_center = (
            _synthetic_scene(with_ground=False)
        )
        poisoned = [
            sync.SyncObservation(
                match_id=observation.match_id,
                landmark_id=observation.landmark_id,
                u=observation.u,
                v=observation.v,
                on_ground=True,
                landmark_name=observation.landmark_name,
                weight=observation.weight,
            )
            for observation in observations
        ]
        result = sync.solve_landmark_sync(
            matches,
            poisoned,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 15.0)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.08))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.6))


    def test_partial_sync_keeps_registered_matches(self) -> None:
        """One un-registerable camera must not abort cameras that already locked."""
        matches, observations, true_sim, center_private, shared_center = (
            _synthetic_scene(with_ground=False)
        )
        isolated_calibration = core.Calibration(
            matches[1].calibration.intrinsics,
            matches[1].calibration.rotation_w2c.copy(),
            matches[1].calibration.camera_center + np.array((0.4, -0.2, 0.1)),
        )
        matches = list(matches) + [
            sync.SyncMatchInput("isolated", isolated_calibration)
        ]
        observations = list(observations) + [
            sync.SyncObservation("anchor", "only_a", 120.0, 140.0),
            sync.SyncObservation("isolated", "only_a", 180.0, 160.0),
            sync.SyncObservation("anchor", "only_b", 520.0, 180.0),
            sync.SyncObservation("isolated", "only_b", 480.0, 220.0),
        ]
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("other", result.similarities)
        self.assertNotIn("isolated", result.similarities)
        self.assertIn("isolated", result.message)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.08))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.6))


    def test_nadir_camera_registers_from_ground_plane(self) -> None:
        """A downward still with leftover private pose must lock from On Ground."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
            np.array((0.1, -0.2, 0.0)),
            np.array((0.8, 0.6, 0.0)),
            np.array((-0.4, 0.3, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
            np.array((-0.5, 0.4, 0.8)),
            np.array((0.0, 0.0, 0.8)),
        )
        anchor_center = np.array((-3.0, -4.0, 2.2), dtype=np.float64)
        anchor = core.Calibration(
            intrinsics,
            _look_at_rotation(anchor_center, np.array((0.2, 0.1, 0.0))),
            anchor_center,
        )
        nadir_center = np.array((0.15, 0.25, 2.4), dtype=np.float64)
        nadir_true = core.Calibration(
            intrinsics,
            _look_at_rotation(nadir_center, np.array((0.15, 0.25, 0.0))),
            nadir_center,
        )
        leftover_center = np.array((-2.0, -3.0, 1.7), dtype=np.float64)
        leftover = core.Calibration(
            intrinsics,
            _look_at_rotation(leftover_center, np.array((0.5, 0.5, 0.0))),
            leftover_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("nadir", leftover),
        ]
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            observations.append(
                sync.SyncObservation(
                    "anchor", f"g{index}", *_project(point, anchor), True
                )
            )
            observations.append(
                sync.SyncObservation(
                    "nadir", f"g{index}", *_project(point, nadir_true), True
                )
            )
        for index, point in enumerate(elevated_points):
            observations.append(
                sync.SyncObservation(
                    "anchor", f"e{index}", *_project(point, anchor), False
                )
            )
            observations.append(
                sync.SyncObservation(
                    "nadir", f"e{index}", *_project(point, nadir_true), False
                )
            )

        planar = sync._planar_homography_similarities(
            np.stack(ground_points),
            np.array(
                [_project(point, nadir_true) for point in ground_points],
                dtype=np.float64,
            ),
            leftover,
        )
        self.assertTrue(planar)
        recovered_seed = planar[0].transform_point(leftover.camera_center)
        self.assertTrue(np.allclose(recovered_seed, nadir_center, atol=0.2))

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("nadir", result.similarities)
        self.assertLess(result.mean_reprojection_px, 2.0)
        recovered = result.similarities["nadir"]
        recovered_center = recovered.transform_point(leftover.camera_center)
        self.assertTrue(np.allclose(recovered_center, nadir_center, atol=0.15))
        axis_private = leftover.rotation_w2c.T @ np.array((0.0, 0.0, 1.0))
        axis_shared = recovered.rotation @ axis_private
        axis_shared = axis_shared / np.linalg.norm(axis_shared)
        self.assertGreater(abs(float(axis_shared[2])), 0.95)


    def test_nadir_camera_with_stretched_fy_still_registers(self) -> None:
        """Portrait K stretched onto a landscape still (fy≪fx) must still lock."""
        square = core.CameraIntrinsics(3821.0, 3821.0, 2016.0, 1512.0, 4032, 3024)
        stretched = core.CameraIntrinsics(3821.0, 2150.0, 2016.0, 1512.0, 4032, 3024)
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
            np.array((0.1, -0.2, 0.0)),
            np.array((0.8, 0.6, 0.0)),
            np.array((-0.4, 0.3, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
            np.array((-0.5, 0.4, 0.8)),
            np.array((0.0, 0.0, 0.8)),
        )
        anchor_center = np.array((-3.0, -4.0, 2.2), dtype=np.float64)
        anchor = core.Calibration(
            core.CameraIntrinsics(800.0, 800.0, 400.0, 300.0, 800, 600),
            _look_at_rotation(anchor_center, np.array((0.2, 0.1, 0.0))),
            anchor_center,
        )
        nadir_center = np.array((0.15, 0.25, 2.4), dtype=np.float64)
        nadir_true = core.Calibration(
            square,
            _look_at_rotation(nadir_center, np.array((0.15, 0.25, 0.0))),
            nadir_center,
        )
        leftover_center = np.array((-2.0, -3.0, 1.7), dtype=np.float64)
        leftover = core.Calibration(
            stretched,
            _look_at_rotation(leftover_center, np.array((0.5, 0.5, 0.0))),
            leftover_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("nadir", leftover),
        ]
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            observations.append(
                sync.SyncObservation(
                    "anchor", f"g{index}", *_project(point, anchor), True
                )
            )
            observations.append(
                sync.SyncObservation(
                    "nadir", f"g{index}", *_project(point, nadir_true), True
                )
            )
        for index, point in enumerate(elevated_points):
            observations.append(
                sync.SyncObservation(
                    "anchor", f"e{index}", *_project(point, anchor), False
                )
            )
            observations.append(
                sync.SyncObservation(
                    "nadir", f"e{index}", *_project(point, nadir_true), False
                )
            )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("nadir", result.similarities)
        self.assertLess(result.mean_reprojection_px, 3.0)
        self.assertAlmostEqual(leftover.intrinsics.fy, leftover.intrinsics.fx, delta=1.0e-6)


    def test_nadir_camera_retries_after_joint_peel(self) -> None:
        """A downward still peeled from BA still locks against the joint 3D."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
            np.array((0.1, -0.2, 0.0)),
            np.array((0.8, 0.6, 0.0)),
            np.array((-0.4, 0.3, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
            np.array((-0.5, 0.4, 0.8)),
            np.array((0.0, 0.0, 0.8)),
        )
        anchor_center = np.array((-3.0, -4.0, 2.2), dtype=np.float64)
        anchor = core.Calibration(
            intrinsics,
            _look_at_rotation(anchor_center, np.array((0.2, 0.1, 0.0))),
            anchor_center,
        )
        side_center = np.array((3.4, -3.1, 2.3), dtype=np.float64)
        side = core.Calibration(
            intrinsics,
            _look_at_rotation(side_center, np.array((0.1, 0.2, 0.0))),
            side_center,
        )
        nadir_center = np.array((0.15, 0.25, 2.4), dtype=np.float64)
        nadir_true = core.Calibration(
            intrinsics,
            _look_at_rotation(nadir_center, np.array((0.15, 0.25, 0.0))),
            nadir_center,
        )
        leftover_center = np.array((-2.0, -3.0, 1.7), dtype=np.float64)
        leftover = core.Calibration(
            intrinsics,
            _look_at_rotation(leftover_center, np.array((0.5, 0.5, 0.0))),
            leftover_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("side", side),
            sync.SyncMatchInput("nadir", leftover),
        ]
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            for match_id, calibration in (
                ("anchor", anchor),
                ("side", side),
                ("nadir", nadir_true),
            ):
                observations.append(
                    sync.SyncObservation(
                        match_id, f"g{index}", *_project(point, calibration), True
                    )
                )
        for index, point in enumerate(elevated_points):
            for match_id, calibration in (("anchor", anchor), ("side", side)):
                observations.append(
                    sync.SyncObservation(
                        match_id, f"e{index}", *_project(point, calibration), False
                    )
                )
            u, v = _project(point, nadir_true)
            # Off-plane picks that disagree with the side views, so joint BA
            # peels the nadir camera; floor tags still place it on retry.
            observations.append(
                sync.SyncObservation("nadir", f"e{index}", u + 180.0, v, False)
            )

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("side", result.similarities)
        self.assertIn("nadir", result.similarities)
        self.assertIn("recovered 'nadir'", result.message)
        self.assertNotIn("skipped 'nadir'", result.message)
        recovered = result.similarities["nadir"]
        recovered_center = recovered.transform_point(leftover.camera_center)
        self.assertTrue(np.allclose(recovered_center, nadir_center, atol=0.2))
        axis_private = leftover.rotation_w2c.T @ np.array((0.0, 0.0, 1.0))
        axis_shared = recovered.rotation @ axis_private
        axis_shared = axis_shared / np.linalg.norm(axis_shared)
        self.assertGreater(abs(float(axis_shared[2])), 0.95)
        self.assertLess(result.per_match_rmse_px.get("side", 99.0), 5.0)
        self.assertAlmostEqual(leftover.intrinsics.fx, 800.0, delta=1.0)


    def test_mismatched_ground_pick_is_named_when_still_skipped(self) -> None:
        """A wrong On Ground correspondence on one still is named in the skip."""
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
            np.array((0.1, -0.2, 0.0)),
            np.array((0.8, 0.6, 0.0)),
            np.array((-0.4, 0.3, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
            np.array((-0.5, 0.4, 0.8)),
        )
        bad_point = np.array((1.2, 1.3, 0.0), dtype=np.float64)

        def _camera(center: np.ndarray, target: np.ndarray) -> core.Calibration:
            return core.Calibration(
                intrinsics,
                _look_at_rotation(center, target),
                center,
            )

        anchor = _camera(
            np.array((-3.0, -4.0, 2.2)), np.array((0.2, 0.1, 0.0))
        )
        wing = _camera(
            np.array((3.4, -3.1, 2.3)), np.array((0.1, 0.2, 0.0))
        )
        top = _camera(
            np.array((0.2, 0.3, 3.2)), np.array((0.2, 0.3, 0.0))
        )
        face_true = _camera(
            np.array((-0.2, 4.0, 1.4)), np.array((0.1, 0.2, 0.4))
        )
        leftover = _camera(
            np.array((-2.0, -3.0, 1.7)), np.array((0.5, 0.5, 0.0))
        )
        cameras = {
            "anchor": anchor,
            "wing": wing,
            "top": top,
            "face": face_true,
        }
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("wing", wing),
            sync.SyncMatchInput("top", top),
            sync.SyncMatchInput("face", leftover),
        ]
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            for match_id, calibration in cameras.items():
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"g{index}",
                        *_project(point, calibration),
                        True,
                        landmark_name=f"g{index}",
                    )
                )
        for index, point in enumerate(elevated_points):
            for match_id, calibration in cameras.items():
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"e{index}",
                        *_project(point, calibration),
                        False,
                        landmark_name=f"e{index}",
                    )
                )
        for match_id, calibration in (("wing", wing), ("top", top), ("face", face_true)):
            u, v = _project(bad_point, calibration)
            if match_id == "face":
                u += 400.0
            observations.append(
                sync.SyncObservation(
                    match_id,
                    "bad",
                    u,
                    v,
                    True,
                    landmark_name="id21-25h9",
                )
            )

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("wing", result.similarities)
        self.assertIn("top", result.similarities)
        self.assertNotIn("face", result.similarities)
        self.assertIn("skipped 'face'", result.message)
        self.assertIn("id21-25h9", result.message)
        self.assertIn("mismatched pick", result.message)
        self.assertTrue(
            any(item[1] == "id21-25h9" for item in result.inconsistent_picks),
            result.inconsistent_picks,
        )

    def test_resected_views_triangulate_exclusive_landmarks(self) -> None:
        """Bottom-only tags still get 3D after their bridging cameras peel.

        Two below-ground stills share floor tags with the top graph (so they
        recover after peel) and exclusive underside tags with a hanging still
        that never sees the joint cloud. That hanging still must be PnP'd from
        the exclusive 3D instead of keeping a leftover pose and 0 px RMSE.
        """
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
            np.array((0.1, -0.2, 0.0)),
            np.array((0.8, 0.6, 0.0)),
            np.array((-0.4, 0.3, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
            np.array((-0.5, 0.4, 0.8)),
            np.array((0.0, 0.0, 0.8)),
        )
        bottom_points = (
            np.array((-0.5, -0.4, 0.0)),
            np.array((0.6, -0.3, 0.0)),
            np.array((0.5, 0.5, 0.0)),
            np.array((-0.4, 0.6, 0.0)),
            np.array((0.1, 0.1, 0.0)),
            np.array((0.2, -0.2, 0.0)),
        )
        leftover_center = np.array((0.0, -5.0, 1.7), dtype=np.float64)
        leftover = core.Calibration(
            intrinsics,
            _look_at_rotation(leftover_center, np.array((0.5, 0.5, 0.0))),
            leftover_center,
        )
        anchor_center = np.array((-3.0, -4.0, 2.2), dtype=np.float64)
        anchor = core.Calibration(
            intrinsics,
            _look_at_rotation(anchor_center, np.array((0.2, 0.1, 0.0))),
            anchor_center,
        )
        side_center = np.array((3.4, -3.1, 2.3), dtype=np.float64)
        side = core.Calibration(
            intrinsics,
            _look_at_rotation(side_center, np.array((0.1, 0.2, 0.0))),
            side_center,
        )
        bridge_a_center = np.array((0.55, -0.45, -1.6), dtype=np.float64)
        bridge_a_true = core.Calibration(
            intrinsics,
            _look_at_rotation(bridge_a_center, np.array((0.2, 0.2, 0.0))),
            bridge_a_center,
        )
        bridge_b_center = np.array((-0.5, 0.55, -1.55), dtype=np.float64)
        bridge_b_true = core.Calibration(
            intrinsics,
            _look_at_rotation(bridge_b_center, np.array((0.2, 0.2, 0.0))),
            bridge_b_center,
        )
        underside_center = np.array((0.2, 0.15, -1.4), dtype=np.float64)
        underside_true = core.Calibration(
            intrinsics,
            _look_at_rotation(underside_center, np.array((0.2, 0.15, 0.0))),
            underside_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("side", side),
            sync.SyncMatchInput("bridge_a", leftover),
            sync.SyncMatchInput("bridge_b", leftover),
            sync.SyncMatchInput("underside", leftover),
        ]
        observations: list[sync.SyncObservation] = []
        true_by_id = {
            "anchor": anchor,
            "side": side,
            "bridge_a": bridge_a_true,
            "bridge_b": bridge_b_true,
            "underside": underside_true,
        }
        for index, point in enumerate(ground_points):
            for match_id in ("anchor", "side", "bridge_a", "bridge_b"):
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"g{index}",
                        *_project(point, true_by_id[match_id]),
                        True,
                        landmark_name=f"g{index}",
                    )
                )
        for index, point in enumerate(elevated_points):
            for match_id in ("anchor", "side"):
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"e{index}",
                        *_project(point, true_by_id[match_id]),
                        False,
                        landmark_name=f"e{index}",
                    )
                )
            for match_id in ("bridge_a", "bridge_b"):
                u_coord, v_coord = _project(point, true_by_id[match_id])
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"e{index}",
                        u_coord + 180.0,
                        v_coord,
                        False,
                        landmark_name=f"e{index}",
                    )
                )
        for index, point in enumerate(bottom_points):
            for match_id in ("bridge_a", "bridge_b", "underside"):
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"b{index}",
                        *_project(point, true_by_id[match_id]),
                        False,
                        landmark_name=f"b{index}",
                    )
                )

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("underside", result.similarities)
        for index in range(len(bottom_points)):
            self.assertIn(f"b{index}", result.landmarks, result.message)
        self.assertGreater(
            result.per_match_rmse_px.get("underside", -1.0),
            0.0,
            result.per_match_rmse_px,
        )
        self.assertLess(
            result.per_match_rmse_px.get("underside", 99.0),
            8.0,
            result.message,
        )
        recovered = result.similarities["underside"]
        recovered_center = recovered.transform_point(leftover.camera_center)
        self.assertLess(float(recovered_center[2]), -0.2)
        self.assertTrue(
            np.allclose(recovered_center, underside_center, atol=0.35),
            recovered_center,
        )

