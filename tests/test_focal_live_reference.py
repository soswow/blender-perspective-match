"""Live point mirror reference controls with a solver-independent pixel oracle."""

from __future__ import annotations

from unittest import TestCase, mock

import numpy as np

from match_perspective.core import geometry, lens_refine, sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_constraints, focal_live_reference


def _case():
    return focal_live_reference.generate()


def _inputs(case):
    matches = []
    similarities = {}
    true_cameras = {item["id"]: item for item in case["truth"]["cameras"]}
    for camera in case["request"]["cameras"]:
        k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                      camera["cy"], camera["width"], camera["height"])
        calibration = geometry.Calibration(
            k, np.asarray(camera["rotation"]), np.asarray(camera["center"]))
        matches.append(lens_refine.MatchLensInput(
            camera["id"], {}, k, base_calibration=calibration))
        true = true_cameras[camera["id"]]
        rotation = np.asarray(true["rotation"]).T @ np.asarray(camera["rotation"])
        center = np.asarray(true["center"])
        similarities[camera["id"]] = sync.SimilarityTransform(
            1.0, rotation, center - rotation @ camera["center"])
    observations = [sync.SyncObservation(**item)
                    for item in case["request"]["observations"]]
    initial = sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value) for key, value in case["truth"]["points"].items()},
        mean_reprojection_px=0.0, per_match_rmse_px={}, per_landmark_rmse_px={},
        message="Independent truth start", success=True)
    return matches, observations, initial


class LiveReferenceFocalTests(TestCase):
    def test_fixture_reference_is_independent_picked_plane_point(self):
        case = _case()
        focal_constraints.validate(case)
        request = case["request"]
        reference_id = request["mirror_landmark_id"]
        reference = np.asarray(case["truth"]["points"][reference_id])
        normal = np.asarray(request["mirror_plane"][1])
        first_pair = request["mirror_pairs"][0]
        midpoint = (np.asarray(case["truth"]["points"][first_pair[0]]) +
                    np.asarray(case["truth"]["points"][first_pair[1]])) / 2
        np.testing.assert_allclose(reference, midpoint, rtol=0, atol=1e-12)
        self.assertGreater(abs(normal @ (reference - request["mirror_plane"][0])), 1.0)
        self.assertEqual(len([item for item in request["observations"]
                              if item["landmark_id"] == reference_id]), 4)

    def test_ordinary_lens_api_forwards_live_reference_into_sync(self):
        case = _case()
        matches, observations, initial = _inputs(case)
        for item in matches:
            item.freeze_focal = True
        with mock.patch.object(lens_refine, "_run_sync", return_value=initial) as run:
            lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id=case["request"]["anchor_id"],
                mirror_pairs=case["request"]["mirror_pairs"],
                mirror_plane=case["request"]["mirror_plane"],
                mirror_landmark_id="reference")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["mirror_landmark_id"], "reference")
        with mock.patch.object(sync, "solve_landmark_sync", return_value=initial) as solve:
            lens_refine._run_sync(
                {item.match_id: item.base_calibration for item in matches},
                [item.match_id for item in matches], observations, [],
                case["request"]["anchor_id"], {}, {}, [],
                mirror_landmark_id="reference")
        self.assertEqual(solve.call_args.kwargs["mirror_landmark_id"], "reference")

    def test_direct_bundle_updates_reference_from_picks_despite_stale_initial_point(self):
        case = _case()
        matches, observations, initial = _inputs(case)
        reference_id = case["request"]["mirror_landmark_id"]
        true_reference = initial.landmarks[reference_id].copy()
        clean_initial = sync.SyncSolveResult(
            similarities=initial.similarities,
            landmarks={key: value.copy() for key, value in initial.landmarks.items()},
            mean_reprojection_px=initial.mean_reprojection_px,
            per_match_rmse_px={}, per_landmark_rmse_px={},
            message="Independent truth start", success=True)
        initial.landmarks[reference_id] += 0.07 * np.asarray(case["request"]["mirror_plane"][1])
        def fit(start):
            return fit_independent_focals(
                {item.match_id: item.base_calibration for item in matches},
                observations, start, anchor_id=case["request"]["anchor_id"],
                pick_sigma_px=case["pick_sigma_px"],
                mirror_pairs=case["request"]["mirror_pairs"],
                mirror_plane=case["request"]["mirror_plane"],
                mirror_landmark_id=reference_id)
        result = fit(initial)
        control = fit(clean_initial)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(control.accepted, control.reason)
        reconstructed = result.sync_result.landmarks[reference_id]
        self.assertLess(np.linalg.norm(reconstructed - true_reference), 0.025)
        self.assertGreater(np.linalg.norm(reconstructed - initial.landmarks[reference_id]), 0.06)
        self.assertLess(np.linalg.norm(reconstructed - control.sync_result.landmarks[reference_id]),
                        0.01)
        normal = np.asarray(case["request"]["mirror_plane"][1])
        for left, right in case["request"]["mirror_pairs"]:
            a = result.sync_result.landmarks[left]
            b = result.sync_result.landmarks[right]
            gap = b - (a - 2 * (normal @ (a - reconstructed)) * normal)
            self.assertLess(np.linalg.norm(gap), 0.01)

    def test_public_point_fov_accepts_live_reference_without_mirror_empty_position(self):
        case = _case()
        matches, observations, _initial = _inputs(case)
        request = case["request"]
        result = lens_refine.refine_lenses_from_landmarks(
            matches, observations, anchor_id=request["anchor_id"],
            estimate_focal_from_points=True, pick_sigma_px=case["pick_sigma_px"],
            mirror_pairs=request["mirror_pairs"], mirror_plane=request["mirror_plane"],
            mirror_landmark_id=request["mirror_landmark_id"])
        self.assertTrue(result.improved, result.refusal_reason)
        reference = result.sync_result.landmarks[request["mirror_landmark_id"]]
        true_reference = np.asarray(case["truth"]["points"][request["mirror_landmark_id"]])
        anchor = np.asarray(case["truth"]["cameras"][0]["center"])
        training_ids = [key for key in case["truth"]["points"]
                        if key != request["mirror_landmark_id"]]
        fitted = np.asarray([result.sync_result.landmarks[key] - anchor for key in training_ids])
        truth = np.asarray([case["truth"]["points"][key] for key in training_ids]) - anchor
        scale = float(np.sum(fitted * truth) / np.sum(fitted * fitted))
        self.assertGreater(scale, 0)
        self.assertLess(np.linalg.norm(anchor + scale * (reference - anchor) - true_reference),
                        0.25)

    def test_public_point_fov_refuses_missing_or_paired_reference_before_sync(self):
        from unittest import mock
        case = _case()
        matches, observations, _initial = _inputs(case)
        request = case["request"]
        for bad_id, reason in (([], "ID is invalid"),
                               ("missing", "needs two-view picks"),
                               (request["mirror_pairs"][0][0], "pair member")):
            with self.subTest(reference=bad_id), mock.patch.object(lens_refine, "_run_sync") as run:
                result = lens_refine.refine_lenses_from_landmarks(
                    matches, observations, anchor_id=request["anchor_id"],
                    estimate_focal_from_points=True,
                    mirror_pairs=request["mirror_pairs"], mirror_plane=request["mirror_plane"],
                    mirror_landmark_id=bad_id)
                self.assertIn(reason, result.refusal_reason)
                run.assert_not_called()
