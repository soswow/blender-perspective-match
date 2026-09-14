"""Constrained focal uncertainty uses only free columns of the joint chart."""

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_bundle import (_active_pixel_jacobian,
                                                  _estimated_dense_jacobian_bytes,
                                                  _fits_constrained_noise_model,
                                                  MAX_DENSE_JACOBIAN_BYTES)


class ActiveFocalColumnsTests(TestCase):
    def test_constrained_noise_chart_matches_active_covariance_columns(self):
        # A metric plane can freeze most 3D coordinates while leaving two
        # focal scales and a side camera pose active.
        full = np.arange(36 * 35, dtype=float).reshape(36, 35) / 500.0
        active = np.array((0, 1, 2, 3, 4, 5, 6, 7))
        weights = np.linspace(0.8, 1.3, 36)
        reduced = _active_pixel_jacobian(full, 36, active, weights)
        self.assertEqual(reduced.shape, (36, 8))
        np.testing.assert_allclose(reduced,
                                   full[:, active] / weights[:, None])
        influence = np.zeros((8, 36))
        self.assertTrue(_fits_constrained_noise_model(
            np.zeros(36), reduced, influence, 1.0))

    def test_dense_graph_memory_estimate_bounds_full_chart_before_allocation(self):
        modest = _estimated_dense_jacobian_bytes(260, 249, 0, 83, 0, 0)
        oversized = _estimated_dense_jacobian_bytes(15000, 15000, 0, 5000, 0, 0)
        self.assertLess(modest, MAX_DENSE_JACOBIAN_BYTES)
        self.assertGreater(oversized, MAX_DENSE_JACOBIAN_BYTES)
