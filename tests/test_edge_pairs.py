"""Pairwise covering: non-default axes together, oracle vs stored state.

See ``tests/edge_pairs.md``. Generate from a true camera; solve with stored K/pose.
"""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective.core import ros_camera_info, sync
from pair_fixtures import (
    YAML_CX,
    YAML_CY,
    YAML_FX,
    YAML_FY,
    YAML_HEIGHT,
    YAML_WIDTH,
    ViewSpec,
    build_views,
    k_for_plate,
    optical_axis_z,
)


class KRemapPairTests(unittest.TestCase):
    """Plate size vs calibrated size (axis A × how K arrives)."""

    def test_swap_landscape_yaml_onto_portrait(self) -> None:
        """A swap × landscape YAML → portrait still."""
        fx, fy, cx, cy, kind = ros_camera_info.remap_intrinsics_to_size(
            YAML_FY,
            YAML_FX,
            YAML_CY,
            YAML_CX,
            YAML_HEIGHT,
            YAML_WIDTH,
            YAML_WIDTH,
            YAML_HEIGHT,
        )
        self.assertEqual(kind, "rotated")
        self.assertAlmostEqual(fx, YAML_FX)
        self.assertAlmostEqual(fy, YAML_FY)
        self.assertAlmostEqual(cx, YAML_CX)
        self.assertAlmostEqual(cy, YAML_CY)

    def test_crop_is_scaled_not_rotated(self) -> None:
        """A crop × YAML: 4000×2250 is not a transpose of 3000×4000."""
        fx, fy, cx, cy, kind = ros_camera_info.remap_intrinsics_to_size(
            YAML_FX,
            YAML_FY,
            YAML_CX,
            YAML_CY,
            YAML_WIDTH,
            YAML_HEIGHT,
            4000,
            2250,
        )
        self.assertEqual(kind, "scaled")
        scale = 4000 / YAML_WIDTH
        self.assertAlmostEqual(fx, YAML_FX * scale)
        self.assertAlmostEqual(fy, YAML_FY * scale)
        self.assertAlmostEqual(cx, YAML_CX * scale)
        self.assertAlmostEqual(cy, YAML_CY * (2250 / YAML_HEIGHT))


class SyncPairTests(unittest.TestCase):
    """Tilt × structure × leftover pose × remapped K."""

    def test_nadir_landscape_after_portrait_yaml(self) -> None:
        """A swap × C nadir × D raised × E leftover."""
        leftover = np.array((-2.0, -3.0, 1.7), dtype=np.float64)
        matches, observations, true_by_id = build_views(
            [
                ViewSpec(
                    "anchor",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((-3.0, -4.0, 2.2)),
                    np.array((0.2, 0.1, 0.0)),
                ),
                ViewSpec(
                    "side",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((3.4, -3.1, 2.3)),
                    np.array((0.1, 0.2, 0.0)),
                ),
                ViewSpec(
                    "nadir",
                    YAML_HEIGHT,
                    YAML_WIDTH,
                    np.array((0.15, 0.25, 2.4)),
                    np.array((0.15, 0.25, 0.0)),
                    leftover_center=leftover,
                ),
            ]
        )
        landscape_k = k_for_plate(YAML_HEIGHT, YAML_WIDTH)
        self.assertAlmostEqual(landscape_k.fx, YAML_FY)
        self.assertAlmostEqual(landscape_k.fy, YAML_FX)
        result = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("nadir", result.similarities)
        self.assertLess(result.per_match_rmse_px.get("nadir", 99.0), 8.0)
        self.assertLess(result.per_match_rmse_px.get("side", 99.0), 8.0)
        stored = next(item.calibration for item in matches if item.match_id == "nadir")
        recovered_center = result.similarities["nadir"].transform_point(
            stored.camera_center
        )
        self.assertTrue(
            np.allclose(
                recovered_center, true_by_id["nadir"].camera_center, atol=0.25
            )
        )
        self.assertGreater(
            optical_axis_z(stored, result.similarities["nadir"]), 0.95
        )
        self.assertAlmostEqual(stored.intrinsics.fx, YAML_FY, delta=1.0)

    def test_nadir_raised_only_does_not_crash(self) -> None:
        """A swap × D raised-only on the nadir still (no floor tags there)."""
        leftover = np.array((-2.0, -3.0, 1.7), dtype=np.float64)
        matches, observations, _true = build_views(
            [
                ViewSpec(
                    "anchor",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((-3.0, -4.0, 2.2)),
                    np.array((0.2, 0.1, 0.0)),
                ),
                ViewSpec(
                    "side",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((3.4, -3.1, 2.3)),
                    np.array((0.1, 0.2, 0.0)),
                ),
                ViewSpec(
                    "nadir",
                    YAML_HEIGHT,
                    YAML_WIDTH,
                    np.array((0.15, 0.25, 2.4)),
                    np.array((0.15, 0.25, 0.0)),
                    leftover_center=leftover,
                ),
            ],
            nadir_ground=False,
            nadir_raised=True,
        )
        result = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("side", result.similarities)
        self.assertNotIn("Could not", result.message)

    def test_from_below_raised_only_keeps_metric_scale(self) -> None:
        """C from-below × D raised-only × E leftover look-at."""
        leftover = np.array((0.0, -5.0, 1.7), dtype=np.float64)
        matches, observations, true_by_id = build_views(
            [
                ViewSpec(
                    "anchor",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((-3.0, -4.0, 2.2)),
                    np.array((0.2, 0.1, 0.0)),
                ),
                ViewSpec(
                    "side",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((3.4, -3.1, 2.3)),
                    np.array((0.1, 0.2, 0.0)),
                ),
                ViewSpec(
                    "below",
                    YAML_WIDTH,
                    YAML_HEIGHT,
                    np.array((0.2, 0.3, -1.6)),
                    np.array((0.2, 0.3, 0.68)),
                    leftover_center=leftover,
                    leftover_target=np.array((0.5, 0.5, 0.0)),
                    skip_ground=True,
                ),
            ]
        )
        result = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("below", result.similarities)
        recovered = result.similarities["below"]
        stored = next(item.calibration for item in matches if item.match_id == "below")
        self.assertFalse(sync._is_collapsed_scale(recovered.scale))
        self.assertAlmostEqual(float(recovered.scale), 1.0, delta=0.05)
        recovered_center = recovered.transform_point(stored.camera_center)
        self.assertLess(float(recovered_center[2]), -0.3)
        axis_private = stored.rotation_w2c.T[:, 2]
        axis_shared = recovered.rotation @ axis_private
        self.assertGreater(float(axis_shared[2]), 0.2)
        self.assertLess(result.per_match_rmse_px.get("below", 99.0), 8.0)
        self.assertTrue(
            np.allclose(
                recovered_center,
                true_by_id["below"].camera_center,
                atol=0.35,
            )
        )
