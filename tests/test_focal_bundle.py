"""Focused point-focal acceptance, refusal and fitted-pose regression checks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from match_perspective.core import geometry, lens_refine, sync
from match_perspective.core.focal_bundle import (
    EPIPOLAR_HINT_MAX_FITS, FocalBundleOutcome, _epipolar_conflict_pairs, _fits_noise_model,
    _has_depth_evidence, _homography_forward_error,
    _project_with_depth_penalty,
    fit_independent_focals,
)


FROZEN = (Path(__file__).resolve().parents[1] / "tools" / "synthetic_sync" /
          "cases" / "independent-focal-production")


def _inputs(name):
    case = json.loads((FROZEN / f"{name}.json").read_text())
    cameras = case["request"]["cameras"]
    matches = []
    for camera in cameras:
        k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                      camera["cy"], camera["width"], camera["height"])
        calibration = geometry.Calibration(k, np.asarray(camera["rotation"], float),
                                           np.asarray(camera["center"], float))
        matches.append(lens_refine.MatchLensInput(
            camera["id"], {}, k, base_calibration=calibration, freeze_focal=True))
    observations = [sync.SyncObservation(**item) for item in case["request"]["observations"]]
    return case, matches, observations


def _saved_initial(case, matches, label):
    path = FROZEN / "run-03" / "sync-ledger.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(row["attempt"] for row in records
                   if row.get("kind") == "started" and row["label"] == label)
    record = next(row["result"] for row in records
                  if row.get("kind") == "completed" and row["attempt"] == attempt)
    similarities = {}
    for item in matches:
        source = item.base_calibration
        world = record["cameras"][item.match_id]
        world_r = np.asarray(world["rotation"], float)
        world_c = np.asarray(world["center"], float)
        rotation = world_r.T @ source.rotation_w2c
        similarities[item.match_id] = sync.SimilarityTransform(
            1.0, rotation, world_c - rotation @ source.camera_center)
    return sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value, float) for key, value in record["landmarks"].items()},
        mean_reprojection_px=record["reported_rmse_px"],
        per_match_rmse_px={}, per_landmark_rmse_px={}, message="Frozen fresh Sync start",
        success=record["success"])


class PointFocalBundleTests(TestCase):
    def test_sync_wrapper_forwards_registration_callback(self):
        _case, matches, observations = _inputs("four-view")
        received = []

        def solve(*args, **kwargs):
            kwargs["progress_callback"]("Preparing sync graph")
            return object()

        with mock.patch.object(lens_refine.sync_module, "solve_landmark_sync",
                               side_effect=solve):
            lens_refine._run_sync(
                {item.match_id: item.base_calibration for item in matches},
                [item.match_id for item in matches], observations, [],
                "view_0", {}, {}, [], progress_callback=received.append)
        self.assertEqual(received, ["Preparing sync graph"])

    def test_point_startup_reports_sync_activity_before_registration_refusal(self):
        _case, matches, observations = _inputs("four-view")
        failed = sync.SyncSolveResult(
            similarities={}, landmarks={}, mean_reprojection_px=0.0,
            per_match_rmse_px={}, per_landmark_rmse_px={},
            message="Could not register", success=False)
        updates = []

        def register(*args, **kwargs):
            kwargs["progress_callback"]("Registering cameras: testing anchor pair")
            return failed

        with mock.patch.object(lens_refine, "_run_sync", side_effect=register):
            result = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="view_0",
                estimate_focal_from_points=True,
                progress_callback=lambda step, total, label: updates.append(
                    (step, total, label)))
        self.assertFalse(result.improved)
        self.assertEqual(updates[0][:2], (0, 101))
        self.assertIn("Registering cameras", updates[1][2])
        self.assertTrue(all(step == 0 for step, _, _ in updates))

    def test_point_bundle_zero_step_remains_indeterminate(self):
        _case, matches, observations = _inputs("four-view")
        initial = sync.SyncSolveResult(
            similarities={item.match_id: sync.SimilarityTransform() for item in matches},
            landmarks={}, mean_reprojection_px=0.0,
            per_match_rmse_px={}, per_landmark_rmse_px={},
            message="Registered", success=True)
        updates = []

        def bundle(*args, **kwargs):
            kwargs["progress_callback"](0, 100, "Preparing provisional cameras")
            kwargs["progress_callback"](1, 100, "Estimating focal from points")
            return FocalBundleOutcome(False, "Controlled refusal")

        with mock.patch.object(lens_refine, "_run_sync", return_value=initial), \
             mock.patch.object(lens_refine, "fit_independent_focals", side_effect=bundle):
            result = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="view_0",
                estimate_focal_from_points=True,
                progress_callback=lambda step, total, label: updates.append(
                    (step, total, label)))
        self.assertFalse(result.improved)
        self.assertEqual([step for step, _, _ in updates], [0, 0, 2])

    def test_line_relations_refuse_before_expensive_registration(self):
        _case, matches, observations = _inputs("four-view")
        strokes = [sync.SyncLineObservation(key, "edge", 0., 0., 10., 10.)
                   for key in ("view_0", "view_1")]
        for extra, message in (
            ({"plane_groups": [("edge", "FREE", 1)]}, "line Is in Plane"),
            ({"mirror_pairs": [("edge", observations[0].landmark_id)],
              "mirror_plane": (np.zeros(3), np.array([1., 0., 0.]))}, "mixed point/line"),
        ):
            with self.subTest(message=message), mock.patch.object(lens_refine, "_run_sync") as run:
                result = lens_refine.refine_lenses_from_landmarks(
                    matches, observations, anchor_id="view_0", estimate_focal_from_points=True,
                    line_observations=strokes, **extra)
                self.assertIn(message, result.refusal_reason)
                run.assert_not_called()

    def test_mirrored_line_requires_picks_for_both_members_before_registration(self):
        _case, matches, observations = _inputs("four-view")
        with mock.patch.object(lens_refine, "_run_sync") as run:
            outcome = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="view_0", estimate_focal_from_points=True,
                line_observations=[sync.SyncLineObservation("view_0", "edge_a", 0., 0., 1., 1.)],
                mirror_pairs=[("edge_a", "edge_b")],
                mirror_plane=(np.zeros(3), np.array([1., 0., 0.])))
        self.assertFalse(outcome.improved)
        self.assertIn("without two-view picks", outcome.refusal_reason)
        run.assert_not_called()

    def test_bad_two_view_correspondence_adds_tentative_pair_hint_on_refusal(self):
        for name in ("noisy-mixed", "noisy-shared"):
            case, matches, observations = _inputs(name)
            ids = [item.match_id for item in matches]
            with self.subTest(name=name):
                self.assertEqual(_epipolar_conflict_pairs(ids, observations, 1.0), [])
                self.assertEqual(_epipolar_conflict_pairs(ids, observations, 0.5), [])
                # This point occurs only in views 1 and 2, so the wrong pick
                # can be absorbed into a free 3D point with a small fit RMSE.
                wrong = next(o for o in observations
                             if o.match_id == "view_2" and o.landmark_id == "point_08")
                wrong.u += 20.0
                wrong.v -= 15.0
                self.assertIn(("view_1", "view_2"),
                              _epipolar_conflict_pairs(ids, observations, 1.0))
                self.assertIn(("view_1", "view_2"),
                              _epipolar_conflict_pairs(ids, observations, 0.5))
                initial = _saved_initial(case, matches, name)
                result = fit_independent_focals(
                    {item.match_id: item.base_calibration for item in matches},
                    observations, initial, anchor_id="view_0", pick_sigma_px=1.0)
                self.assertFalse(result.accepted)
                self.assertIn("view_1 and view_2", result.reason)
                self.assertIn("may be inconsistent", result.reason)
                self.assertNotIn("point_08", result.reason)

    def test_pair_hint_is_lazy_and_bounded(self):
        case, matches, observations = _inputs("noisy-mixed")
        initial = _saved_initial(case, matches, "noisy-mixed")
        with mock.patch("match_perspective.core.focal_bundle._epipolar_conflict_pairs",
                        side_effect=AssertionError("successful fits should not run a hint")):
            result = fit_independent_focals(
                {item.match_id: item.base_calibration for item in matches},
                observations, initial, anchor_id="view_0", pick_sigma_px=1.0)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(_epipolar_conflict_pairs(
            [item.match_id for item in matches], observations, 1.0,
            cancel_check=lambda: True), [])
        # Eight cameras with all 80 points visible would otherwise require
        # 28 × 80 leave-one-out models on a refusal path.
        many = [sync.SyncObservation(f"camera_{camera}", f"point_{point}",
                                     float(point * 17 % 997), float(point * 41 % 719))
                for camera in range(8) for point in range(80)]
        import match_perspective.core.focal_bundle as bundle
        with mock.patch.object(bundle, "_fit_fundamental", wraps=bundle._fit_fundamental) as fit:
            self.assertEqual(_epipolar_conflict_pairs(
                [f"camera_{camera}" for camera in range(8)], many, 1.0), [])
        self.assertEqual(fit.call_count, EPIPOLAR_HINT_MAX_FITS)

    def test_insufficient_camera_support_names_camera_and_count(self):
        case, matches, observations = _inputs("four-view")
        initial = _saved_initial(case, matches, "four-view")
        kept = {item.landmark_id for item in observations if item.match_id == "view_0"}
        kept = set(sorted(kept)[:8])
        sparse = [item for item in observations if item.landmark_id in kept]
        initial.landmarks = {key: value for key, value in initial.landmarks.items() if key in kept}
        result = fit_independent_focals(
            {item.match_id: item.base_calibration for item in matches},
            sparse, initial, anchor_id="view_0", pick_sigma_px=1.0)
        self.assertFalse(result.accepted)
        self.assertRegex(result.reason, r"view_[123] has [0-7] point picks")
        with mock.patch.object(lens_refine, "_run_sync") as run:
            public = lens_refine.refine_lenses_from_landmarks(
                matches, sparse, anchor_id="view_0",
                estimate_focal_from_points=True, pick_sigma_px=1.0)
        self.assertFalse(public.improved)
        self.assertEqual(public.refusal_reason, result.reason)
        run.assert_not_called()

    def test_near_duplicate_first_view_does_not_set_the_scale_gauge(self):
        _case, matches, observations = _inputs("four-view")
        initial = _saved_initial(_case, matches, "four-view")
        anchor = matches[0]
        duplicate = lens_refine.MatchLensInput(
            "near_anchor", {}, anchor.intrinsics, base_calibration=anchor.base_calibration,
            freeze_focal=True)
        initial.similarities[duplicate.match_id] = sync.SimilarityTransform()
        duplicate_picks = [sync.SyncObservation(duplicate.match_id, item.landmark_id,
                                               item.u, item.v, weight=item.weight)
                           for item in observations if item.match_id == anchor.match_id]
        reordered = [anchor, duplicate, matches[3], matches[2], matches[1]]
        outcome = fit_independent_focals(
            {item.match_id: item.base_calibration for item in reordered},
            observations + duplicate_picks, initial, anchor_id=anchor.match_id,
            pick_sigma_px=1.0, fx_span=0.4)
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(set(outcome.calibrations), {item.match_id for item in reordered})
        self.assertAlmostEqual(outcome.calibrations[duplicate.match_id].intrinsics.fx,
                               outcome.calibrations[anchor.match_id].intrinsics.fx,
                               delta=1e-3)

    def test_projection_derivative_matches_finite_difference_on_both_depth_sides(self):
        for xyz in ((0.3, -0.4, 2.0), (0.3, -0.4, -0.2)):
            q = np.asarray([xyz], float)
            predicted, derivative = _project_with_depth_penalty(
                q, 900.0, np.array([480.0, 360.0]))
            numerical = np.empty((2, 3))
            for axis in range(3):
                step = 1.0e-4 if xyz[2] < 0 else 1.0e-6
                shifted = q.copy()
                shifted[0, axis] += step
                changed, _ = _project_with_depth_penalty(
                    shifted, 900.0, np.array([480.0, 360.0]))
                numerical[:, axis] = (changed - predicted).ravel() / step
            np.testing.assert_allclose(derivative[0], numerical, rtol=1e-6, atol=2e-3)

    def test_four_camera_fit_preserves_private_poses_and_returns_fitted_world(self):
        case, matches, observations = _inputs("four-view")
        initial = _saved_initial(case, matches, "four-view")
        with mock.patch.object(lens_refine, "_run_sync", return_value=initial):
            outcome = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="view_0", fx_span=0.4,
                estimate_focal_from_points=True, pick_sigma_px=1.0)
        self.assertTrue(outcome.point_focal_mode)
        self.assertTrue(outcome.improved, outcome.refusal_reason)
        self.assertFalse(outcome.refusal_reason)
        self.assertEqual(set(outcome.focal_intervals), {item.match_id for item in matches})
        self.assertEqual(set(outcome.sync_result.similarities), set(outcome.calibrations))
        self.assertEqual(set(outcome.sync_result.landmarks),
                         {item.landmark_id for item in observations})
        truth_focals = {item["id"]: item["fx"] for item in case["truth"]["cameras"]}
        for item in matches:
            camera_id = item.match_id
            fitted = outcome.calibrations[camera_id]
            self.assertTrue(np.array_equal(fitted.rotation_w2c,
                                           item.base_calibration.rotation_w2c))
            self.assertTrue(np.array_equal(fitted.camera_center,
                                           item.base_calibration.camera_center))
            self.assertAlmostEqual(fitted.intrinsics.fx, truth_focals[camera_id], delta=1e-3)
            self.assertAlmostEqual(fitted.intrinsics.fy, fitted.intrinsics.fx)
            low, high = outcome.focal_intervals[camera_id]
            self.assertLess(low, fitted.intrinsics.fx)
            self.assertGreater(high, fitted.intrinsics.fx)
            similarity = outcome.sync_result.similarities[camera_id]
            self.assertAlmostEqual(np.linalg.det(similarity.rotation), 1.0, delta=1e-8)
            for observation in [o for o in observations if o.match_id == camera_id]:
                point = similarity.inverse_point(
                    outcome.sync_result.landmarks[observation.landmark_id])
                xyz = fitted.rotation_w2c @ (point - fitted.camera_center)
                self.assertGreater(xyz[2], 0.0)
                uv = fitted.intrinsics.fx * xyz[:2] / xyz[2] + [
                    fitted.intrinsics.cx, fitted.intrinsics.cy]
                self.assertLess(np.linalg.norm(uv - [observation.u, observation.v]), 1e-4)

    def test_planar_weak_and_rotation_picks_have_no_depth_evidence(self):
        for name in ("planar", "no-vp-weak-baseline-guessedK",
                     "no-vp-pure-rotation-guessedK"):
            case, matches, observations = _inputs(name)
            with self.subTest(name=name):
                self.assertFalse(_has_depth_evidence(
                    [item.match_id for item in matches], observations, 1.0))
                self.assertEqual(_epipolar_conflict_pairs(
                    [item.match_id for item in matches], observations, 1.0), [])
        case, matches, observations = _inputs("four-view")
        self.assertTrue(_has_depth_evidence(
            [item.match_id for item in matches], observations, 1.0))
        collinear = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
        warped = np.column_stack((np.arange(8, dtype=float)**2, np.zeros(8)))
        with self.assertRaises(ValueError):
            _homography_forward_error(collinear, warped, 1.0)
        degenerate = [sync.SyncObservation(camera, f"point_{index}", float(value[0]),
                                               float(value[1]))
                      for index in range(8) for camera, value in
                      (("a", collinear[index]), ("b", warped[index]),
                       ("c", warped[index] + [3.0, 0.0]))]
        self.assertFalse(_has_depth_evidence(["a", "b", "c"], degenerate, 1.0))

    def test_empty_axis_bundles_do_not_block_point_mode(self):
        case, matches, observations = _inputs("four-view")
        for item in matches:
            item.line_bundles = {axis: [] for axis in ("x", "y", "z")}
        initial = _saved_initial(case, matches, "four-view")
        with mock.patch.object(lens_refine, "_run_sync", return_value=initial) as run, \
             mock.patch.object(lens_refine, "fit_independent_focals",
                               return_value=FocalBundleOutcome(False, "sentinel")) as fit:
            outcome = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id="view_0", fx_span=0.4,
                estimate_focal_from_points=True)
        run.assert_called_once()
        self.assertEqual(fit.call_args.kwargs["fx_span"], 0.4)
        self.assertEqual(outcome.refusal_reason, "sentinel")

    def test_understated_pick_noise_cannot_certify_noisy_rotation(self):
        noisy_dir = FROZEN.parent / "independent-focal-noise"
        case = json.loads((noisy_dir / "no-vp-pure-rotation-guessedK.json").read_text())
        record = json.loads((noisy_dir / "dense" /
            "no-vp-pure-rotation-guessedK-dense-exact-start-1.json").read_text())["record"]
        request = case["request"]
        ids = [camera["id"] for camera in request["cameras"]]
        observations = [sync.SyncObservation(**item) for item in request["observations"]]
        # A falsely tiny sigma makes even homography pick noise look like
        # parallax; the independently checked point-fit noise model still fails.
        self.assertTrue(_has_depth_evidence(ids, observations, 0.1))
        errors = []
        for item in observations:
            camera = record["cameras"][item.match_id]
            point = np.asarray(record["landmarks"][item.landmark_id])
            xyz = np.asarray(camera["rotation"]) @ (point - camera["center"])
            projected = camera["fx"] * xyz[:2] / xyz[2] + [camera["cx"], camera["cy"]]
            errors.append(projected - [item.u, item.v])
        self.assertFalse(_fits_noise_model(np.asarray(errors), 83, 0.1))
        self.assertTrue(_fits_noise_model(np.asarray(errors), 83, 1.0))

    def test_unsupported_constraints_and_same_lens_refuse_before_registration(self):
        _case, matches, observations = _inputs("four-view")
        for option in (dict(share_lens=True), dict(known_world={"point_00": np.ones(3)}),
                       dict(lock_rotation=True), dict(readonly_match_ids={"view_1"}),
                       dict(line_observations=[sync.SyncLineObservation(
                           "view_0", "edge_0", 0.0, 0.0, 1.0, 1.0)])):
            with self.subTest(option=option), mock.patch.object(lens_refine, "_run_sync") as run:
                outcome = lens_refine.refine_lenses_from_landmarks(
                    matches, observations, anchor_id="view_0",
                    estimate_focal_from_points=True, **option)
                self.assertTrue(outcome.point_focal_mode)
                self.assertFalse(outcome.improved)
                self.assertTrue(outcome.refusal_reason)
                run.assert_not_called()
