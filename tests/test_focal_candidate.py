"""Retained best fits are explicit provisional results, never accepted calibration."""
from unittest import TestCase, mock

import numpy as np

from match_perspective.core import focal_bundle, lens_refine
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.solver import result_record
from test_focal_bundle import _inputs, _saved_initial


class FocalCandidateTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case, cls.matches, cls.observations = _inputs('four-view')
        cls.initial = _saved_initial(cls.case, cls.matches, 'four-view')
        cls.calibrations = {item.match_id: item.base_calibration for item in cls.matches}

    def fit(self, **kwargs):
        return focal_bundle.fit_independent_focals(
            self.calibrations, self.observations, self.initial, anchor_id='view_0', **kwargs)

    def check_candidate(self, result):
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)
        self.assertFalse(result.calibrations)
        self.assertFalse(result.intervals_px)
        candidate = result.candidate
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.reason, result.reason)
        self.assertLess(candidate.fitted_rmse_px, candidate.initial_rmse_px)
        record = result_record(candidate.sync_result, self.case['request']['cameras'],
                               calibrations=candidate.calibrations)
        residuals = []
        for pick in self.observations:
            uv, depth = project([record['landmarks'][pick.landmark_id]],
                                 record['cameras'][pick.match_id])
            self.assertGreater(depth[0], 0)
            residuals.append(uv[0] - [pick.u, pick.v])
        self.assertAlmostEqual(np.sqrt(np.mean(np.sum(np.asarray(residuals)**2, axis=1))),
                               candidate.fitted_rmse_px, places=7)
        return candidate

    def test_incomplete_fit_retains_improved_geometry_without_accepting_it(self):
        with mock.patch.object(focal_bundle, 'MAX_ITERATIONS', 2):
            result = self.fit()
        self.assertIn('did not converge', result.reason)
        self.assertIn('reached the 2-iteration limit', result.reason)
        self.check_candidate(result)

    def test_stalled_step_is_distinct_from_iteration_limit(self):
        diagnostics = []
        with mock.patch.object(focal_bundle, 'bounded_lm_step',
                               side_effect=lambda jac, residual, x, *a, **k: np.zeros_like(x)):
            result = self.fit(diagnostic_callback=diagnostics.append)
        self.assertIn('no improving step found at iteration 1', result.reason)
        self.assertEqual(diagnostics[0]['stop_reason'], 'no_improving_step')
        self.assertEqual(diagnostics[0]['recent_relative_improvements'], [])
        self.assertIsNone(result.candidate)

    def test_focal_bound_preserves_the_explicit_use_option(self):
        result = self.fit(fx_span=.01)
        self.assertIn('search bound', result.reason)
        self.check_candidate(result)

    def test_zero_work_and_cancellation_never_offer_a_candidate(self):
        with mock.patch.object(focal_bundle, 'MAX_ITERATIONS', 0):
            self.assertIsNone(self.fit().candidate)
        result = self.fit(cancel_check=lambda: True)
        self.assertEqual(result.reason, 'Cancelled')
        self.assertIsNone(result.candidate)

    def test_hard_geometry_failure_never_offers_a_candidate(self):
        from test_focal_orientation import OrientationFixtureTests
        # A post-fit hard-prior rejection is distinct from calibration refusal.
        with mock.patch('match_perspective.core.focal_constraints.PointFocalConstraints.world_gaps',
                        return_value=(float('inf'), 0.)):
            _, result = OrientationFixtureTests()._fit(constrained=True)
        self.assertIn('plane', result.reason.lower())
        self.assertFalse(result.accepted)
        self.assertIsNone(result.candidate)

    def test_public_result_preserves_candidate_and_refusal(self):
        with mock.patch.object(lens_refine, '_run_sync', return_value=self.initial), \
             mock.patch.object(focal_bundle, 'MAX_ITERATIONS', 2):
            result = lens_refine.refine_lenses_from_landmarks(
                self.matches, self.observations, anchor_id='view_0',
                estimate_focal_from_points=True)
        self.assertFalse(result.improved)
        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.reason, result.refusal_reason)
        self.assertFalse(result.focal_intervals)
