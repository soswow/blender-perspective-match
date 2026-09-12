"""Larger independent-lens fits with a separate projection oracle."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core import geometry, lens_refine, sync
from tools.synthetic_sync.geometry import look_at, project


def scaffold(count, *, planar=False):
    rng = np.random.default_rng(260912)
    points = rng.uniform(-1.2, 1.2, (24, 3))
    if planar:
        points[:, 2] = 0
    holdouts = rng.uniform(-1.2, 1.2, (12, 3))
    matches, observations, cameras = [], [], []
    for i, angle in enumerate(np.linspace(-0.9, 0.9, count)):
        center = np.array([7 * np.sin(angle), -7 * np.cos(angle), 2 + .6 * np.sin(3 * angle)])
        focal = float(700 + (i * 137) % 500)
        camera = dict(id=f'view_{i}', fx=focal, fy=focal, cx=480., cy=360.,
                      center=center, rotation=look_at(center, [0, 0, 0]))
        cameras.append(camera)
        guessed = focal * (1.08 if i % 2 else .94)
        k = geometry.CameraIntrinsics(guessed, guessed, 480., 360., 960, 720)
        calibration = geometry.Calibration(k, np.array(camera['rotation']), center.copy())
        matches.append(lens_refine.MatchLensInput(camera['id'], {}, k,
                                                base_calibration=calibration))
        pixels, depths = project(points, camera)
        assert np.all(depths > 0) and np.all((pixels > 0) & (pixels < [960, 720]))
        for j, uv in enumerate(pixels):
            # Partial picks, while preserving broad support in every view.
            if (i + j) % 5:
                observations.append(sync.SyncObservation(camera['id'], f'p{j}', *uv))
    initial = sync.SyncSolveResult(
        similarities={m.match_id: sync.SimilarityTransform() for m in matches},
        landmarks={f'p{i}': p.copy() for i, p in enumerate(points)},
        mean_reprojection_px=0., per_match_rmse_px={}, per_landmark_rmse_px={},
        message='Oracle pose start: bundle isolation only', success=True)
    return matches, observations, cameras, points, holdouts, initial


class FocalCameraCountTests(TestCase):
    def check_fit(self, count, *, fresh):
        matches, observations, cameras, points, holdouts, initial = scaffold(count)
        def run():
            return lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id='view_0', estimate_focal_from_points=True)
        if fresh:
            result = run()
        else:
            with patch.object(lens_refine, '_run_sync', return_value=initial):
                result = run()
        self.assertTrue(result.improved, result.refusal_reason)
        self.assertEqual(len(result.focal_intervals), count)
        # One scale from training geometry, never from withheld projections.
        origin = cameras[0]['center']
        fitted = np.array([result.sync_result.landmarks[f'p{i}'] for i in range(len(points))])
        relative = points - origin
        scale = np.sum((fitted - origin) * relative) / np.sum(relative**2)
        self.assertGreater(scale, 0)
        np.testing.assert_allclose(fitted, origin + scale * relative, atol=1e-4)
        for camera in cameras:
            key = camera['id']
            cal, sim = result.calibrations[key], result.sync_result.similarities[key]
            self.assertAlmostEqual(cal.intrinsics.fx, camera['fx'], delta=.02)
            fitted_camera = dict(camera, fx=cal.intrinsics.fx, fy=cal.intrinsics.fy,
                center=sim.transform_point(cal.camera_center),
                rotation=cal.rotation_w2c @ sim.rotation.T)
            predicted, depth = project(origin + scale * (holdouts - origin), fitted_camera)
            expected, _ = project(holdouts, camera)
            self.assertTrue(np.all(depth > 0))
            np.testing.assert_allclose(predicted, expected, atol=.01)

    def test_nine_cameras_from_fresh_registration(self):
        self.check_fit(9, fresh=True)

    def test_sixteen_camera_bundle(self):
        self.check_fit(16, fresh=False)

    def test_thirty_two_camera_bundle(self):
        self.check_fit(32, fresh=False)

    def test_above_resource_limit_refuses(self):
        matches, observations, _, _, _, _ = scaffold(33)
        result = lens_refine.refine_lenses_from_landmarks(
            matches, observations, anchor_id='view_0', estimate_focal_from_points=True)
        self.assertFalse(result.improved)
        self.assertIn('up to 32 cameras', result.refusal_reason)

    def test_nine_planar_cameras_still_refuse(self):
        matches, observations, _, _, _, initial = scaffold(9, planar=True)
        with patch.object(lens_refine, '_run_sync', return_value=initial):
            result = lens_refine.refine_lenses_from_landmarks(
                matches, observations, anchor_id='view_0', estimate_focal_from_points=True)
        self.assertFalse(result.improved)
        self.assertIn('depth', result.refusal_reason.lower())
