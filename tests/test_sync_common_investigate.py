"""Zero-solve tests for bounded common-objective investigation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from match_perspective.core.sync.investigate import _omit_landmark, common_leave_one_out
from match_perspective.core.sync.request import SyncSolveRequest
from match_perspective.core.sync.types import (
    SimilarityTransform, SyncCancelled, SyncLineObservation,
    SyncObservation, SyncSolveResult,
)


def _case() -> tuple[SyncSolveRequest, SyncSolveResult]:
    request = SyncSolveRequest(
        matches=[], anchor_id="anchor",
        observations=[
            SyncObservation("anchor", "point", 10, 20, landmark_name="Point"),
            SyncObservation("anchor", "reference", 12, 22),
            SyncObservation("anchor", "known", 14, 24),
        ],
        line_observations=[
            SyncLineObservation("anchor", "line", 0, 0, 20, 0,
                                landmark_name="Line"),
            SyncLineObservation("anchor", "known_line", 0, 2, 20, 2),
        ],
        known_world={"known": np.ones(3)},
        known_lines={"known_line": (np.zeros(3), np.ones(3))},
        mirror_pairs=[("point", "reference")],
        mirror_landmark_id="reference",
        plane_groups=[("point", "X", 1), ("reference", "X", 1)],
        parallel_pairs=[("line", "X"), ("known_line", "Y")],
    )
    result = SyncSolveResult(
        similarities={"anchor": SimilarityTransform()},
        landmarks={key: np.ones(3) for key in ("point", "reference", "known")},
        line_segments={key: (np.zeros(3), np.ones(3))
                       for key in ("line", "known_line")},
        mean_reprojection_px=40.0,
        per_match_rmse_px={},
        per_landmark_rmse_px={"point": 30.0, "line": 20.0,
                              "reference": 50.0, "known": 45.0,
                              "known_line": 40.0},
        message="baseline",
    )
    return request, result


def _coverage(request, _result):
    return request, {
        "requested_point_picks": len(request.observations),
        "fitted_point_picks": len(request.observations),
        "requested_line_strokes": len(request.line_observations or ()),
        "fitted_line_strokes": len(request.line_observations or ()),
        "skipped_camera_ids": [], "skipped_point_ids": [],
        "skipped_line_ids": [], "skipped_relation_ids": [],
    }


class FakeScorer:
    instances = []

    def __init__(self, request, incumbent, *, calibrations):
        self.request = request
        self.incumbent = incumbent
        self.calls = []
        self.instances.append(self)

    def score(self, result, *, calibrations):
        self.calls.append(result)
        is_candidate = result.message == "candidate"
        return SimpleNamespace(
            objective=8.0 if is_candidate else 10.0,
            point_rmse_px=3.0 if is_candidate else 2.0,
            line_rmse_px=1.0 if is_candidate else 4.0,
            valid=True, reason="",
        )


class CommonInvestigationTests(unittest.TestCase):
    def test_omitting_endpoint_removes_derived_line_relations(self):
        request, _baseline = _case()
        request.observations.append(SyncObservation("anchor", "endpoint_b", 16, 26))
        request.derived_lines = [("derived", "point", "endpoint_b")]
        request.plane_groups = [("derived", "X", 2), ("point", "Y", 3)]
        request.parallel_pairs = [("derived", "WORLD_AXIS_Z")]
        filtered, removed = _omit_landmark(request, "point")
        self.assertFalse(filtered.derived_lines)
        self.assertFalse(filtered.plane_groups)
        self.assertFalse(filtered.parallel_pairs)
        self.assertIn("parallel:derived:WORLD_AXIS_Z", removed)

    def setUp(self):
        FakeScorer.instances.clear()

    def test_omits_only_supported_free_landmarks_and_scores_same_evidence(self):
        request, baseline = _case()
        calls = []

        def solve(filtered, reduced, *, cancel_check):
            self.assertFalse(cancel_check())
            calls.append((filtered, reduced))
            return replace(reduced, message="candidate")

        outcome = common_leave_one_out(
            request, baseline, solve, scorer_factory=FakeScorer,
            supported_request_fn=_coverage,
        )
        self.assertFalse(outcome.incomplete)
        self.assertEqual([item.landmark_id for item in outcome.items],
                         ["point", "line"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(FakeScorer.instances), 2)
        self.assertTrue(all(len(scorer.calls) == 2 for scorer in FakeScorer.instances))
        for scorer in FakeScorer.instances:
            self.assertIs(scorer.request, calls[FakeScorer.instances.index(scorer)][0])
        point_request, point_baseline = calls[0]
        self.assertNotIn("point", [item.landmark_id for item in point_request.observations])
        self.assertNotIn("point", point_baseline.landmarks)
        self.assertEqual(point_request.mirror_pairs, [])
        self.assertIsNone(point_request.mirror_landmark_id)
        self.assertEqual(point_request.plane_groups, [("reference", "X", 1)])
        self.assertEqual(outcome.items[0].removed_relations,
                         ["mirror:point:reference", "plane:X:1:point"])
        self.assertEqual(outcome.items[1].removed_relations, ["parallel:line:X"])
        self.assertEqual(outcome.items[0].baseline_objective, 10.0)
        self.assertEqual(outcome.items[0].candidate_objective, 8.0)
        self.assertEqual(outcome.items[0].objective_improvement, 2.0)
        self.assertGreater(outcome.items[0].candidate_point_rmse_px,
                           outcome.items[0].baseline_point_rmse_px)

    def test_refuses_counterfactual_support_loss(self):
        request, baseline = _case()

        def coverage(filtered, result):
            if result.message == "candidate":
                shrunk = replace(filtered, line_observations=[])
                return shrunk, {**_coverage(filtered, result)[1],
                                "fitted_line_strokes": 0,
                                "skipped_line_ids": ["line"]}
            return _coverage(filtered, result)

        outcome = common_leave_one_out(
            request, baseline,
            lambda filtered, reduced, *, cancel_check:
                replace(reduced, message="candidate"),
            scorer_factory=FakeScorer, supported_request_fn=coverage,
        )
        self.assertTrue(outcome.incomplete)
        self.assertEqual(len(outcome.items), 1)
        self.assertFalse(outcome.items[0].candidate_valid)
        self.assertIn("lost surviving evidence", outcome.reason)
        self.assertEqual(len(FakeScorer.instances[0].calls), 1)

    def test_deadline_retains_partial_result_without_another_callback(self):
        request, baseline = _case()
        clock = [0.0]
        calls = []

        def solve(filtered, reduced, *, cancel_check):
            calls.append(filtered)
            clock[0] = 61.0
            return replace(reduced, message="candidate")

        with mock.patch("match_perspective.core.sync.investigate.time.monotonic",
                        side_effect=lambda: clock[0]):
            outcome = common_leave_one_out(
                request, baseline, solve,
                scorer_factory=FakeScorer, supported_request_fn=_coverage,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outcome.items), 1)
        self.assertTrue(outcome.items[0].candidate_valid)
        self.assertTrue(outcome.incomplete)
        self.assertIn("time limit", outcome.reason)

    def test_user_cancellation_propagates(self):
        request, baseline = _case()
        with self.assertRaises(SyncCancelled):
            common_leave_one_out(
                request, baseline, lambda *_args, **_kwargs: self.fail("solved"),
                cancel_check=lambda: True,
                scorer_factory=FakeScorer, supported_request_fn=_coverage,
            )

    def test_callback_failure_preserves_completed_comparison(self):
        request, baseline = _case()
        calls = [0]

        def solve(filtered, reduced, *, cancel_check):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("fit refused this geometry")
            return replace(reduced, message="candidate")

        outcome = common_leave_one_out(
            request, baseline, solve, scorer_factory=FakeScorer,
            supported_request_fn=_coverage,
        )
        self.assertEqual(calls[0], 2)
        self.assertTrue(outcome.incomplete)
        self.assertEqual(len(outcome.items), 2)
        self.assertTrue(outcome.items[0].candidate_valid)
        self.assertFalse(outcome.items[1].candidate_valid)
        self.assertIn("fit refused this geometry", outcome.reason)

    def test_callback_count_is_capped_at_five(self):
        request, baseline = _case()
        extras = [f"extra_{index}" for index in range(6)]
        request.observations.extend(
            SyncObservation("anchor", key, 0, 0) for key in extras)
        baseline.landmarks.update({key: np.ones(3) for key in extras})
        baseline.per_landmark_rmse_px.update(
            {key: 100.0 + index for index, key in enumerate(extras)})
        calls = []

        def solve(filtered, reduced, *, cancel_check):
            calls.append(filtered)
            return replace(reduced, message="candidate")

        outcome = common_leave_one_out(
            request, baseline, solve, max_candidates=100,
            scorer_factory=FakeScorer, supported_request_fn=_coverage,
        )
        self.assertEqual(len(calls), 5)
        self.assertEqual(outcome.candidates_ranked, 5)


if __name__ == "__main__":
    unittest.main()
