"""Bounded LM step controls independent of Blender and SciPy."""
from __future__ import annotations

import unittest

import numpy as np

from tools.synthetic_sync.solver import load_core

load_core()
from match_perspective.core.focal_optimizer import bounded_lm_step


class BoundedLMStepTests(unittest.TestCase):
    def test_active_bound_resolves_coupled_free_variable(self):
        # Unconstrained optimum is (2, 1); the first coordinate is capped at 1.
        # Clipping leaves the second coordinate at 1; re-solving moves it to 2.
        jac = np.array([[1., 1.], [1., 0.]])
        residual = np.array([-3., -2.])
        x = np.zeros(2)
        lower = np.array([-np.inf, -np.inf])
        upper = np.array([1., np.inf])
        step = bounded_lm_step(jac, residual, x, lower, upper, 1e-8)
        np.testing.assert_allclose(step, [1., 2.], atol=1e-6)
        naive = np.linalg.lstsq(jac, -residual, rcond=None)[0]
        naive[0] = min(naive[0], 1.)
        self.assertLess(np.linalg.norm(jac @ step + residual),
                        np.linalg.norm(jac @ naive + residual))

    def test_outward_gradient_freezes_upper_bound(self):
        jac = np.eye(2)
        residual = np.array([-2., -3.])
        x = np.array([1., 0.])
        step = bounded_lm_step(jac, residual, x, np.array([-np.inf, -np.inf]),
                               np.array([1., np.inf]), 1e-6)
        self.assertEqual(step[0], 0.)
        self.assertAlmostEqual(step[1], 3., places=5)

    def test_lower_bound_releases_for_inward_gradient(self):
        step = bounded_lm_step(np.eye(2), np.array([-2., -1.]), np.zeros(2),
                               np.array([0., -np.inf]), np.array([np.inf, np.inf]), 1e-6)
        self.assertGreater(step[0], 1.99)
        self.assertGreater(step[1], 0.99)

    def test_unbounded_agrees_with_augmented_least_squares(self):
        jac = np.array([[1., 2.], [3., -1.], [0.5, 1.]])
        residual = np.array([2., -1., 3.])
        damping = 0.05
        scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-8)
        scaled = jac / scale
        expected_scaled = np.linalg.lstsq(
            np.vstack((scaled, np.sqrt(damping)*np.eye(2))),
            np.concatenate((-residual, np.zeros(2))), rcond=None)[0]
        step = bounded_lm_step(jac, residual, np.zeros(2),
                               np.full(2, -np.inf), np.full(2, np.inf), damping)
        np.testing.assert_allclose(step, expected_scaled/scale, atol=1e-12)

    def test_cancellation_checked_before_resolve(self):
        jac = np.array([[1., 1.], [1., 0.]])
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls == 2
        with self.assertRaises(InterruptedError):
            bounded_lm_step(jac, np.array([-3., -2.]), np.zeros(2),
                            np.full(2, -np.inf), np.array([1., np.inf]),
                            1e-8, cancel_check=cancel)
        self.assertEqual(calls, 2)
