"""Synthetic off-center crop control for fixed-principal-point focal fitting."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from tools.synthetic_sync.solver import load_core

load_core()
from match_perspective.core import sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_crop
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.solver import calibration, solver_arguments


def _oracle_start(case):
    request = case["request"]
    stored = {camera["id"]: camera for camera in request["cameras"]}
    similarities = {}
    for true in case["truth"]["cameras"]:
        private = stored[true["id"]]
        rotation = np.asarray(true["rotation"]).T @ np.asarray(private["rotation"])
        translation = np.asarray(true["center"]) - rotation @ np.asarray(private["center"])
        similarities[true["id"]] = sync.SimilarityTransform(1.0, rotation, translation)
    return sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(point) for key, point in case["truth"]["points"].items()},
        mean_reprojection_px=0.0, per_match_rmse_px={}, per_landmark_rmse_px={},
        message="Independent oracle pose/point start", success=True)


def _fit(case):
    request = case["request"]
    arguments = solver_arguments(request)
    return fit_independent_focals(
        {camera["id"]: calibration(camera) for camera in request["cameras"]},
        arguments["observations"], _oracle_start(case),
        anchor_id=request["anchor_id"], pick_sigma_px=1.0, fx_span=0.4,
        mirror_pairs=arguments["mirror_pairs"], mirror_plane=arguments["mirror_plane"],
        mirror_slack=arguments["mirror_slack"],
        mirror_landmark_id=arguments["mirror_landmark_id"])


def _withheld_max_px(case, outcome):
    """Project unused 3D checks with fitted cameras in the fixed oracle frame."""
    truth = case["truth"]
    private = {camera["id"]: camera for camera in case["request"]["cameras"]}
    errors = []
    for true in truth["cameras"]:
        camera_id = true["id"]
        similarity = outcome.sync_result.similarities[camera_id]
        source = private[camera_id]
        fitted = outcome.calibrations[camera_id].intrinsics
        recovered = dict(true, fx=fitted.fx, fy=fitted.fy, cx=fitted.cx, cy=fitted.cy,
                         center=(similarity.scale * similarity.rotation @ source["center"] + similarity.translation).tolist(),
                         rotation=(np.asarray(source["rotation"]) @ similarity.rotation.T).tolist())
        for point_id, position in truth["holdouts"].items():
            actual = project([position], recovered)[0][0]
            expected = truth["holdout_pixels"][point_id][camera_id]
            errors.append(float(np.linalg.norm(actual - expected)))
    return max(errors)


class FocalCropTests(TestCase):
    def test_crop_geometry_and_principal_point_translation(self):
        shifted = focal_crop.generate()
        centered = focal_crop.generate(centered_principal_point=True)
        for original, cropped, wrong in zip(
            focal_crop.focal_live_reference.generate()["truth"]["cameras"],
            shifted["truth"]["cameras"], centered["request"]["cameras"]
        ):
            self.assertEqual(cropped["width"], original["width"] - 480)
            self.assertEqual(cropped["height"], original["height"] - 280)
            self.assertEqual(cropped["cx"], original["cx"] - 400)
            self.assertEqual(cropped["cy"], original["cy"] - 200)
            self.assertEqual(wrong["cx"], wrong["width"] / 2)
            self.assertEqual(wrong["cy"], wrong["height"] / 2)
        self.assertEqual(shifted["request"]["observations"], centered["request"]["observations"])

    def test_shifted_k_recovers_withheld_and_centered_k_does_not(self):
        shifted = focal_crop.generate()
        correct = _fit(shifted)
        self.assertTrue(correct.accepted, correct.reason)
        good_error = _withheld_max_px(shifted, correct)
        self.assertLess(good_error, 0.1)

        centered = focal_crop.generate(centered_principal_point=True)
        wrong = _fit(centered)
        if wrong.accepted:
            bad_error = _withheld_max_px(centered, wrong)
            self.assertGreater(bad_error, max(1.0, good_error * 10))
        else:
            self.assertIn("search bound", wrong.reason)
