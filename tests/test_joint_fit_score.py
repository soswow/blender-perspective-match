"""Current-evidence scoring checks for common fixed/free joint selection."""

from __future__ import annotations

from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core.joint_fit_score import JointFitScorer, supported_joint_request
from match_perspective.core.sync.projection import _project_world_line_to_image
from test_joint_fit_features import _fixture


class JointFitScoreTests(TestCase):
    def _truth(self, request, initial, points, centers):
        result = deepcopy(initial)
        result.landmarks = deepcopy(points)
        for match in request.matches:
            result.similarities[match.match_id].translation = (
                centers[match.match_id] - match.calibration.camera_center)
        return result

    def test_same_weights_compare_full_current_evidence(self):
        request, initial, points, centers = _fixture()
        scorer = JointFitScorer(request, initial)
        start = scorer.score(initial)
        truth = scorer.score(self._truth(request, initial, points, centers))
        self.assertFalse(start.valid)
        self.assertIn("Hard world relation", start.reason)
        self.assertTrue(truth.valid, truth.reason)
        self.assertEqual(truth.supported_observations, len(request.observations))
        self.assertLess(truth.objective, 1e-15)
        self.assertLess(truth.constraint_gaps["ground_m"], 1e-12)

    def test_float32_persisted_anchor_rotation_keeps_certified_score(self):
        request, initial, points, centers = _fixture()
        truth = self._truth(request, initial, points, centers)
        frozen_weights = JointFitScorer(request, truth).effective_point_weights
        anchor = next(item for item in request.matches if item.match_id == request.anchor_id)
        anchor.calibration.rotation_w2c = np.asarray(
            anchor.calibration.rotation_w2c, dtype=np.float32,
        ).astype(np.float64)
        scorer = JointFitScorer(
            request, truth, frozen_point_weights=frozen_weights,
        )
        score = scorer.score(truth)
        self.assertTrue(score.valid, score.reason)
        self.assertEqual(score.supported_observations, len(request.observations))
        self.assertEqual(scorer.effective_point_weights, frozen_weights)

    def test_missing_geometry_and_line_are_explicit(self):
        request, initial, points, centers = _fixture(known_line=True)
        scorer = JointFitScorer(request, initial)
        truth = self._truth(request, initial, points, centers)
        self.assertTrue(scorer.score(truth).valid)
        del truth.landmarks["p4"]
        missing = scorer.score(truth)
        self.assertFalse(missing.valid)
        self.assertIn("p4", missing.reason)
        truth = self._truth(request, initial, points, centers)
        del truth.line_segments["edge"]
        missing = scorer.score(truth)
        self.assertFalse(missing.valid)
        self.assertIn("edge", missing.reason)

    def test_known_line_and_aspect_distortion_are_scored(self):
        request, initial, points, centers = _fixture(
            distortion=True, known_line=True)
        scorer = JointFitScorer(request, initial)
        truth = scorer.score(self._truth(request, initial, points, centers))
        self.assertTrue(truth.valid, truth.reason)
        self.assertGreater(truth.line_rmse_px, 0.0)
        self.assertLess(truth.line_rmse_px, 1.0)
        self.assertEqual(truth.supported_observations,
                         len(request.observations) + len(request.line_observations))

    def test_line_rms_uses_scalar_endpoint_distances(self):
        request, initial, points, centers = _fixture(known_line=True)
        truth_result = self._truth(request, initial, points, centers)
        stroke = request.line_observations[0]
        first, second = request.known_lines["edge"]
        image_line = _project_world_line_to_image(
            (first + second) * 0.5, second - first,
            request.matches[0].calibration,
            truth_result.similarities[stroke.match_id])
        stroke.u1 += 3.0 * image_line[0]
        stroke.v1 += 3.0 * image_line[1]
        stroke.u2 += 4.0 * image_line[0]
        stroke.v2 += 4.0 * image_line[1]
        score = JointFitScorer(request, truth_result).score(truth_result)
        self.assertTrue(score.valid, score.reason)
        self.assertAlmostEqual(score.line_rmse_px, np.sqrt(25.0 / 6.0), places=8)
        self.assertAlmostEqual(score.per_match_line_rmse_px[stroke.match_id],
                               np.sqrt(25.0 / 2.0), places=8)

    def test_partial_graph_reports_all_excluded_evidence_and_relations(self):
        request, initial, _points, _centers = _fixture(known_line=True)
        request.plane_groups = [("p1", "Z", 1), ("p4", "Z", 1)]
        request.parallel_pairs = [("edge", "WORLD_AXIS_X")]
        del initial.similarities["detail"]
        del initial.landmarks["p4"]
        restricted, coverage = supported_joint_request(request, initial)
        self.assertEqual(coverage["requested_point_picks"], 21)
        self.assertEqual(coverage["fitted_point_picks"], 12)
        self.assertEqual(coverage["requested_line_strokes"], 3)
        self.assertEqual(coverage["fitted_line_strokes"], 2)
        self.assertIn("detail", coverage["skipped_camera_ids"])
        self.assertIn("p4", coverage["skipped_point_ids"])
        self.assertIn("plane:Z:1", coverage["skipped_relation_ids"])
        self.assertFalse(restricted.plane_groups)
        self.assertEqual(restricted.parallel_pairs, [("edge", "WORLD_AXIS_X")])
        self.assertEqual(set(restricted.known_world), {"p0", "p5"})

    def test_root_translation_lock_is_scored_as_root_property(self):
        request, initial, points, centers = _fixture()
        truth = self._truth(request, initial, points, centers)
        request.lock_translation = True
        score = JointFitScorer(request, truth).score(truth)
        self.assertFalse(score.valid)
        self.assertIn("Root translation lock", score.reason)

    def test_unpicked_known_line_remains_parallel_reference(self):
        request, initial, points, centers = _fixture(known_line=True)
        for stroke in request.line_observations:
            stroke.landmark_id = "free_edge"
        initial.line_segments["free_edge"] = deepcopy(initial.line_segments["edge"])
        request.parallel_pairs = [("free_edge", "edge")]
        truth = self._truth(request, initial, points, centers)
        restricted, coverage = supported_joint_request(request, truth)
        self.assertEqual(restricted.parallel_pairs, [("free_edge", "edge")])
        self.assertFalse(coverage["skipped_relation_ids"])
        score = JointFitScorer(restricted, truth).score(truth)
        self.assertTrue(score.valid, score.reason)
        self.assertLess(score.constraint_gaps["line_direction_sine"], 1e-12)

    def test_certified_weights_survive_outlier_threshold_crossing(self):
        request, initial, points, centers = _fixture()
        result = self._truth(request, initial, points, centers)
        for item in request.observations:
            if item.landmark_id == "p4":
                item.u += 3.0

        def scorer_at_residual(error):
            by_landmark = {key: [1.0] for key in points}
            by_landmark["p4"] = [error]
            with patch.object(JointFitScorer, "_pixel_errors",
                              return_value=({}, {}, by_landmark)):
                return JointFitScorer(request, result)

        first = scorer_at_residual(21.0)
        repeated = scorer_at_residual(19.0)
        self.assertIn("p4", first.downweighted_landmark_ids)
        self.assertNotIn("p4", repeated.downweighted_landmark_ids)
        frozen = JointFitScorer(
            request, result, frozen_point_weights=first.effective_point_weights)
        self.assertEqual(frozen.downweighted_landmark_ids,
                         first.downweighted_landmark_ids)
        self.assertLess(first.score(result).objective,
                        repeated.score(result).objective)
        self.assertAlmostEqual(first.score(result).objective,
                               frozen.score(result).objective, places=9)

        reordered = deepcopy(request)
        reordered.observations.reverse()
        reordered_frozen = JointFitScorer(
            reordered, result, frozen_point_weights=first.effective_point_weights)
        self.assertAlmostEqual(reordered_frozen.score(result).objective,
                               first.score(result).objective, places=9)
        malformed = first.effective_point_weights.copy()
        camera, point, _weight = malformed[0]
        malformed[0] = (camera, point, 9999.0)
        self.assertIn("Certified point weights",
                      JointFitScorer(request, result,
                                     frozen_point_weights=malformed).weight_refusal)
