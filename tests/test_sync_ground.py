"""Calibrated ground-plane initialization."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from sync_fixtures import _look_at_rotation, _project, _rodrigues_z, _synthetic_scene


class GroundPlaneSyncTests(unittest.TestCase):
    """Calibrated ground-plane initialization."""

    def test_calibrated_ground_points_infer_anchor_vertical(self) -> None:
        intrinsics = core.CameraIntrinsics(
            fx=920.0,
            fy=900.0,
            cx=410.0,
            cy=295.0,
            image_width=820,
            image_height=590,
        )
        ground_points = (
            np.array((-1.5, -1.0, 0.0)),
            np.array((1.8, -0.8, 0.0)),
            np.array((2.0, 1.4, 0.0)),
            np.array((-1.2, 1.7, 0.0)),
            np.array((0.2, -0.1, 0.0)),
            np.array((0.9, 0.7, 0.0)),
        )
        true_calibrations = {
            "anchor": core.Calibration(
                intrinsics,
                _look_at_rotation(
                    np.array((-3.0, -4.0, 2.4)), np.array((0.2, 0.1, 0.0))
                ),
                np.array((-3.0, -4.0, 2.4)),
            ),
            "other_a": core.Calibration(
                intrinsics,
                _look_at_rotation(
                    np.array((3.5, -2.5, 2.0)), np.array((0.1, 0.3, 0.0))
                ),
                np.array((3.5, -2.5, 2.0)),
            ),
            "other_b": core.Calibration(
                intrinsics,
                _look_at_rotation(
                    np.array((2.8, 3.2, 2.8)), np.array((0.0, 0.2, 0.0))
                ),
                np.array((2.8, 3.2, 2.8)),
            ),
        }
        # Match inputs deliberately have no useful VP orientation; only K is
        # valid. Image observations come from the true cameras above.
        matches = [
            sync.SyncMatchInput(
                match_id,
                core.Calibration(
                    intrinsics,
                    np.eye(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                ),
            )
            for match_id in true_calibrations
        ]
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            for match_id, calibration in true_calibrations.items():
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"g{index}",
                        *_project(point, calibration),
                        on_ground=True,
                    )
                )

        two_view_initialization = sync.estimate_anchor_ground_plane(
            matches[:2],
            observations,
            anchor_id="anchor",
        )
        self.assertIsNone(two_view_initialization)

        outlier = next(
            item
            for item in observations
            if item.match_id == "other_b" and item.landmark_id == "g5"
        )
        original_u = outlier.u
        outlier.u += 40.0
        robust_initialization = sync.estimate_anchor_ground_plane(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertIsNotNone(robust_initialization)
        outlier.u = original_u

        initialization = sync.estimate_anchor_ground_plane(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertIsNotNone(initialization)
        assert initialization is not None
        expected_plane_facing = -true_calibrations["anchor"].rotation_w2c[:, 2]
        self.assertGreater(
            float(np.dot(initialization.plane_normal_camera, expected_plane_facing)),
            0.999,
        )
        self.assertEqual(
            set(initialization.supporting_match_ids), {"other_a", "other_b"}
        )
        for match_id in initialization.supporting_match_ids:
            expected_relative = (
                true_calibrations[match_id].rotation_w2c
                @ true_calibrations["anchor"].rotation_w2c.T
            )
            self.assertTrue(
                np.allclose(
                    initialization.relative_rotations[match_id],
                    expected_relative,
                    atol=1.0e-6,
                )
            )
        inferred_rotation = sync.rotation_from_ground_normal(
            -initialization.plane_normal_camera
        )
        self.assertTrue(
            np.allclose(
                inferred_rotation[:, 2],
                true_calibrations["anchor"].rotation_w2c[:, 2],
                atol=1.0e-6,
            )
        )
        self.assertAlmostEqual(float(np.linalg.det(inferred_rotation)), 1.0, places=9)

        match_map = {item.match_id: item for item in matches}
        match_map["anchor"].calibration.rotation_w2c = inferred_rotation
        match_map["anchor"].calibration.camera_center = core.default_camera_center(
            inferred_rotation
        )
        first_by_match = {
            match_id: next(
                item for item in observations if item.match_id == match_id
            )
            for match_id in match_map
        }
        anchor_first = first_by_match["anchor"]
        match_map["anchor"].calibration.camera_center, _scale = (
            core.apply_origin_and_scale(
                match_map["anchor"].calibration,
                (anchor_first.u, anchor_first.v),
            )
        )
        for match_id in initialization.supporting_match_ids:
            calibration = match_map[match_id].calibration
            calibration.rotation_w2c = (
                initialization.relative_rotations[match_id] @ inferred_rotation
            )
            calibration.camera_center = core.default_camera_center(
                calibration.rotation_w2c,
                height=1.7 * initialization.plane_distance_ratios[match_id],
            )
            first = first_by_match[match_id]
            calibration.camera_center, _scale = core.apply_origin_and_scale(
                calibration,
                (first.u, first.v),
            )

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.mean_reprojection_px, 1.0)
        for point in result.landmarks.values():
            self.assertAlmostEqual(float(point[2]), 0.0, delta=1.0e-6)


    def test_pure_rotation_does_not_claim_ground_normal(self) -> None:
        intrinsics = core.CameraIntrinsics(800.0, 800.0, 400.0, 300.0, 800, 600)
        center = np.array((0.0, -4.0, 2.0), dtype=np.float64)
        anchor_true = core.Calibration(
            intrinsics,
            _look_at_rotation(center, np.zeros(3, dtype=np.float64)),
            center,
        )
        other_true = core.Calibration(
            intrinsics,
            _look_at_rotation(center, np.array((0.5, 0.0, 0.0))),
            center,
        )
        matches = [
            sync.SyncMatchInput(
                match_id,
                core.Calibration(intrinsics, np.eye(3), np.zeros(3)),
            )
            for match_id in ("anchor", "other")
        ]
        observations = []
        for index, point in enumerate(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (0.2, 0.3, 0.0),
            )
        ):
            world = np.asarray(point, dtype=np.float64)
            observations.extend(
                (
                    sync.SyncObservation(
                        "anchor", f"g{index}", *_project(world, anchor_true), True
                    ),
                    sync.SyncObservation(
                        "other", f"g{index}", *_project(world, other_true), True
                    ),
                )
            )
        self.assertIsNone(
            sync.estimate_anchor_ground_plane(
                matches,
                observations,
                anchor_id="anchor",
            )
        )


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

