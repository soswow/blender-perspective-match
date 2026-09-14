"""Fixed-focal publication selects the represented, fully scored endpoint."""

from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core import focal_bundle
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core.sync.types import SyncLineObservation
from test_joint_fit_features import _fixture, _pixel


class FixedEndpointPublicScoreTests(TestCase):
    def _case(self):
        request, seed, points, centers = _fixture(known_line=True)
        represented = deepcopy(seed)
        represented.landmarks = deepcopy(points)
        for match in request.matches:
            represented.similarities[match.match_id].translation = (
                centers[match.match_id] - match.calibration.camera_center)
        incumbent = deepcopy(represented)
        incumbent.similarities["side"].translation[0] += 0.01
        calibrations = {match.match_id: match.calibration for match in request.matches}
        scorer = JointFitScorer(request, incumbent)
        self.assertTrue(scorer.score(incumbent).valid)
        self.assertTrue(scorer.score(represented).valid)
        self.assertLess(scorer.score(represented).objective,
                        scorer.score(incumbent).objective)
        return request, incumbent, represented, calibrations, scorer

    def _publish(self, request, incumbent, represented, calibrations,
                 internal_objective):
        bundle = focal_bundle.FocalBundleOutcome(
            True, "", calibrations=calibrations, sync_result=represented,
            initial_rmse_px=1.0, fitted_rmse_px=0.1,
            fitted_objective=internal_objective)
        with patch.object(focal_bundle, "fit_independent_focals", return_value=bundle):
            return focal_bundle.refine_fixed_focals(request, incumbent)

    def test_represented_improvement_survives_internal_chart_roundoff(self):
        request, incumbent, represented, calibrations, scorer = self._case()
        public = scorer.score(represented).objective
        outcome = self._publish(request, incumbent, represented, calibrations,
                                public + 2.0e-6 * max(public, 1.0))
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertAlmostEqual(outcome.fitted_objective, public, places=9)
        self.assertEqual(outcome.sync_result.joint_final_objective,
                         outcome.fitted_objective)
        self.assertEqual(outcome.sync_result.joint_point_weights,
                         scorer.effective_point_weights)

    def test_represented_worsening_is_refused_even_if_internal_cost_improves(self):
        request, incumbent, represented, calibrations, scorer = self._case()
        represented.similarities["side"].translation[0] += 0.03
        self.assertGreater(scorer.score(represented).objective,
                           scorer.score(incumbent).objective)
        outcome = self._publish(request, incumbent, represented, calibrations,
                                0.0)
        self.assertFalse(outcome.accepted)
        self.assertIn("worsened", outcome.reason)

    def test_missing_represented_line_is_refused(self):
        request, incumbent, represented, calibrations, _scorer = self._case()
        del represented.line_segments["edge"]
        outcome = self._publish(request, incumbent, represented, calibrations,
                                0.0)
        self.assertFalse(outcome.accepted)
        self.assertIn("edge", outcome.reason)

    def test_mirrored_infinite_line_score_ignores_helper_segment_slide(self):
        request, _initial, points, centers = _fixture(known_line=True)
        first, second = request.known_lines["edge"]
        mirrored = lambda point: np.asarray((-point[0], point[1], point[2]))
        mirror_first, mirror_second = mirrored(first), mirrored(second)
        mirror_second = mirror_second + np.asarray((0.0, 0.0005, 0.0))
        request.known_lines = None
        request.mirror_pairs = [("edge", "reflected_edge")]
        request.mirror_plane = (np.zeros(3), np.asarray((1.0, 0.0, 0.0)))
        represented = deepcopy(_initial)
        represented.landmarks = deepcopy(points)
        represented.line_segments["reflected_edge"] = (mirror_first, mirror_second)
        for match in request.matches:
            camera_id = match.match_id
            represented.similarities[camera_id].translation = (
                centers[camera_id] - match.calibration.camera_center)
            left = 0.8 * mirror_first + 0.2 * mirror_second
            right = 0.2 * mirror_first + 0.8 * mirror_second
            uv0 = _pixel(left, match.calibration, represented.similarities[camera_id])
            uv1 = _pixel(right, match.calibration, represented.similarities[camera_id])
            request.line_observations.append(SyncLineObservation(
                camera_id, "reflected_edge", float(uv0[0]), float(uv0[1]),
                float(uv1[0]), float(uv1[1])))
        scorer = JointFitScorer(request, represented)
        before = scorer.score(represented)
        self.assertTrue(before.valid, before.reason)
        shifted = deepcopy(represented)
        for key, slide in (("edge", 3.0), ("reflected_edge", -4.0)):
            left, right = shifted.line_segments[key]
            direction = (right - left) / np.linalg.norm(right - left)
            shifted.line_segments[key] = (left + slide * direction,
                                          right + slide * direction)
        after = scorer.score(shifted)
        self.assertTrue(after.valid, after.reason)
        self.assertAlmostEqual(after.objective, before.objective, places=8)
