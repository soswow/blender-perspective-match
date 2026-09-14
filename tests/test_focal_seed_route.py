"""Zero-solve checks for the public free-focal seed selector."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from match_perspective.core import geometry, lens_refine, sync


def _calibration(center_x: float) -> geometry.Calibration:
    return geometry.Calibration(
        geometry.CameraIntrinsics(1000.0, 1000.0, 400.0, 300.0, 800, 600),
        rotation_w2c=np.eye(3),
        camera_center=np.array((center_x, 0.0, 0.0)),
    )


def _result(camera_ids: tuple[str, ...]) -> sync.SyncSolveResult:
    return sync.SyncSolveResult(
        similarities={key: sync.SimilarityTransform() for key in camera_ids},
        landmarks={"p": np.array((0.0, 0.0, 5.0))},
        mean_reprojection_px=1.0,
        per_match_rmse_px={key: 1.0 for key in camera_ids},
        per_landmark_rmse_px={"p": 1.0},
        message="registered",
    )


def _score(objective: float, support: int, *, line_mode: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        valid=True,
        objective=objective,
        supported_observations=support,
        point_rmse_px=0.0 if line_mode else objective,
        line_rmse_px=objective if line_mode else 0.0,
        per_match_point_rmse_px={},
        per_match_line_rmse_px={},
        per_match_rmse_px={},
        per_landmark_rmse_px={},
        constraint_gaps={},
        reason="",
    )


class FreeFocalSeedRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibrations = {"A": _calibration(0.0), "B": _calibration(1.0)}
        self.matches = [
            lens_refine.MatchLensInput(key, {}, cal.intrinsics,
                                       base_calibration=cal)
            for key, cal in self.calibrations.items()
        ]
        self.observations = [
            sync.SyncObservation("A", "p", 400.0, 300.0),
            sync.SyncObservation("B", "p", 200.0, 300.0),
        ]

    def _run(self, *, seed, fingerprint: str, seed_result,
             core_result, trial_result, scores: dict[int, float],
             startup_result=None, line_observations=None):
        seen_incumbents = []
        fit_starts = []
        support = len(self.observations) + len(line_observations or ())

        class FakeScorer:
            def __init__(self, _request, incumbent, *, calibrations, frozen_point_weights=None):
                self.point_observations = self_observations
                self.effective_point_weights = [(item.match_id, item.landmark_id, item.weight)
                                                for item in self_observations]
                self.downweighted_landmark_ids = ["p"]
                seen_incumbents.append(incumbent)

            def score(self, result, *, calibrations):
                value = scores.get(id(result))
                return (_score(value, support, line_mode=bool(line_observations))
                        if value is not None
                        else SimpleNamespace(valid=False, reason="unsupported"))

        self_observations = self.observations

        def fake_fit(_cals, _observations, initial, **_kwargs):
            fit_starts.append(initial)
            return SimpleNamespace(
                accepted=trial_result is not None,
                reason="No accepted fit" if trial_result is None else "",
                sync_result=trial_result,
                calibrations=dict(self.calibrations),
                intervals_px={},
                candidate=None,
            )

        with (
            patch.object(lens_refine.SyncSolveRequest, "evidence_sha256",
                         return_value=fingerprint, create=True),
            patch("match_perspective.core.sync.solve.solution_result_from_seed",
                  return_value=seed_result, create=True),
            patch.object(lens_refine, "JointFitScorer", FakeScorer),
            patch.object(lens_refine, "_run_sync", return_value=core_result) as run_sync,
            patch.object(lens_refine, "provisional_poses",
                         return_value=startup_result) as startup,
            patch.object(lens_refine, "fit_independent_focals",
                         side_effect=fake_fit),
            patch.object(lens_refine, "joint_line_support_diagnostics",
                         return_value=({}, [])),
        ):
            result = lens_refine.refine_lenses_from_landmarks(
                self.matches, self.observations, anchor_id="A",
                known_world={"p": np.array((0.0, 0.0, 5.0))},
                known_lines=({"edge": (np.array((-1.0, 0.0, 5.0)),
                                       np.array((1.0, 0.0, 5.0)))}
                             if line_observations else None),
                line_observations=line_observations,
                estimate_focal_from_points=True,
                initial_solution=seed,
            )
        return result, run_sync, startup, seen_incumbents, fit_starts

    def test_certified_complete_seed_bypasses_sync_initializer(self) -> None:
        applied = _result(("A", "B"))
        fitted = _result(("A", "B"))
        seed = SimpleNamespace(diagnostics=SimpleNamespace(joint_point_weights=[
            (item.match_id, item.landmark_id, item.weight) for item in self.observations]), evidence_sha256="same")
        result, run_sync, startup, scorers, starts = self._run(
            seed=seed, fingerprint="same", seed_result=applied,
            core_result=None, trial_result=fitted,
            scores={id(applied): 3.0, id(fitted): 2.0},
        )
        run_sync.assert_not_called()
        startup.assert_not_called()
        self.assertTrue(starts, result.refusal_reason)
        self.assertIs(starts[0], applied)
        self.assertIs(scorers[0], applied)
        self.assertTrue(result.improved)
        self.assertEqual(result.sync_result.downweighted_landmark_ids, ["p"])

    def test_line_only_report_uses_common_endpoint_rms(self) -> None:
        self.observations = []
        strokes = [
            sync.SyncLineObservation(key, "edge", 300.0, 300.0, 500.0, 300.0)
            for key in ("A", "B")
        ]
        applied = _result(("A", "B"))
        fitted = _result(("A", "B"))
        seed = SimpleNamespace(diagnostics=SimpleNamespace(joint_point_weights=[
            (item.match_id, item.landmark_id, item.weight) for item in self.observations]), evidence_sha256="same")
        result, run_sync, _startup, _scorers, _starts = self._run(
            seed=seed, fingerprint="same", seed_result=applied,
            core_result=None, trial_result=fitted,
            scores={id(applied): 3.0, id(fitted): 2.0},
            line_observations=strokes,
        )
        run_sync.assert_not_called()
        self.assertTrue(result.improved, result.refusal_reason)
        self.assertEqual(result.sync_result.point_rmse_px, 0.0)
        self.assertEqual(result.sync_result.line_rmse_px, 2.0)
        self.assertEqual(result.sync_result.mean_reprojection_px, 2.0)
        self.assertEqual(result.final_sync_rmse, 2.0)

    def test_changed_evidence_retains_better_applied_baseline(self) -> None:
        applied = _result(("A", "B"))
        core = _result(("A", "B"))
        fitted = _result(("A", "B"))
        seed = SimpleNamespace(diagnostics=SimpleNamespace(joint_point_weights=[
            (item.match_id, item.landmark_id, item.weight) for item in self.observations]), evidence_sha256="applied")
        result, run_sync, startup, scorers, starts = self._run(
            seed=seed, fingerprint="changed", seed_result=applied,
            core_result=core, trial_result=fitted,
            scores={id(applied): 3.0, id(core): 8.0, id(fitted): 4.0},
        )
        self.assertEqual(run_sync.call_count, 1, result.refusal_reason)
        startup.assert_not_called()
        self.assertIs(run_sync.call_args.kwargs["initial_solution"], seed)
        self.assertIs(scorers[0], applied)
        self.assertIs(starts[0], applied)
        self.assertIs(result.sync_result, applied)
        self.assertFalse(result.improved)
        self.assertIn("did not improve", result.refusal_reason)

    def test_changed_evidence_keeps_complete_seed_if_partial_startup_fails(self) -> None:
        calibration = _calibration(2.0)
        self.calibrations["C"] = calibration
        self.matches.append(lens_refine.MatchLensInput(
            "C", {}, calibration.intrinsics, base_calibration=calibration))
        self.observations.append(sync.SyncObservation("C", "p", 0.0, 300.0))
        applied = _result(("A", "B", "C"))
        partial = _result(("A", "B"))
        fitted = _result(("A", "B", "C"))
        seed = SimpleNamespace(diagnostics=SimpleNamespace(joint_point_weights=[
            (item.match_id, item.landmark_id, item.weight) for item in self.observations]), evidence_sha256="applied")
        result, run_sync, startup, _scorers, starts = self._run(
            seed=seed, fingerprint="changed", seed_result=applied,
            core_result=partial, trial_result=fitted,
            scores={id(applied): 3.0, id(fitted): 2.0},
            startup_result=(partial, "No provisional camera pose"),
        )
        run_sync.assert_called_once()
        startup.assert_called_once()
        self.assertIs(starts[0], applied)
        self.assertTrue(result.improved)

    def test_partial_startup_completes_before_weight_freeze(self) -> None:
        calibration = _calibration(2.0)
        self.calibrations["C"] = calibration
        self.matches.append(lens_refine.MatchLensInput(
            "C", {}, calibration.intrinsics, base_calibration=calibration))
        self.observations.append(sync.SyncObservation("C", "p", 0.0, 300.0))
        partial = _result(("A", "B"))
        complete = replace(partial, similarities={
            **partial.similarities, "C": sync.SimilarityTransform()})
        events = []
        support = len(self.observations)

        class FakeScorer:
            def __init__(self, _request, incumbent, *, calibrations, frozen_point_weights=None):
                events.append(("weights", tuple(sorted(incumbent.similarities))))
                self.point_observations = self_observations
                self.effective_point_weights = [(item.match_id, item.landmark_id, item.weight)
                                                for item in self_observations]
                self.downweighted_landmark_ids = []

            def score(self, _result, *, calibrations):
                return _score(1.0, support)

        self_observations = self.observations

        def fake_startup(_cals, _observations, initial, **_kwargs):
            events.append(("startup", tuple(sorted(initial.similarities))))
            return complete, ""

        with (
            patch.object(lens_refine, "_run_sync", return_value=partial),
            patch.object(lens_refine, "provisional_poses", side_effect=fake_startup),
            patch.object(lens_refine, "JointFitScorer", FakeScorer),
            patch.object(lens_refine, "fit_independent_focals",
                         return_value=SimpleNamespace(
                             accepted=False, reason="No accepted fit",
                             sync_result=None, candidate=None)),
        ):
            result = lens_refine.refine_lenses_from_landmarks(
                self.matches, self.observations, anchor_id="A",
                known_world={"p": np.array((0.0, 0.0, 5.0))},
                estimate_focal_from_points=True,
            )
        self.assertFalse(result.improved)
        self.assertEqual(events, [
            ("startup", ("A", "B")),
            ("weights", ("A", "B", "C")),
        ], result.refusal_reason)


if __name__ == "__main__":
    unittest.main()
