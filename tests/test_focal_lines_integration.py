"""Independent oracle checks for mixed point and line focal fitting."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from match_perspective.core import geometry, sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_lines
from tools.synthetic_sync.geometry import project


def _inputs(case):
    request = case["request"]
    truth_cameras = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    calibrations, similarities = {}, {}
    for camera in request["cameras"]:
        camera_id = camera["id"]
        k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                      camera["cy"], camera["width"], camera["height"])
        calibrations[camera_id] = geometry.Calibration(
            k, np.asarray(camera["rotation"]), np.asarray(camera["center"]))
        true = truth_cameras[camera_id]
        rotation = np.asarray(true["rotation"]).T @ np.asarray(camera["rotation"])
        center = np.asarray(true["center"])
        similarities[camera_id] = sync.SimilarityTransform(
            1.0, rotation, center - rotation @ camera["center"])
    initial = sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value) for key, value in case["truth"]["points"].items()},
        mean_reprojection_px=0.0, per_match_rmse_px={}, per_landmark_rmse_px={},
        message="Truth pose and point start; lines absent", success=True)
    return calibrations, [sync.SyncObservation(**item) for item in request["observations"]], initial


def _fit(case):
    calibrations, observations, initial = _inputs(case)
    request = case["request"]
    return fit_independent_focals(
        calibrations, observations, initial, anchor_id=request["anchor_id"],
        pick_sigma_px=case["pick_sigma_px"],
        line_observations=[sync.SyncLineObservation(**item) for item in request["line_observations"]],
        mirror_pairs=request["mirror_pairs"], mirror_plane=request["mirror_plane"],
        mirror_landmark_id=request["mirror_landmark_id"])


def _infinite_line_error(segment, truth):
    a, b = np.asarray(truth, float)
    direction = b - a
    direction /= np.linalg.norm(direction)
    points = np.asarray(segment, float)
    return float(max(np.linalg.norm(np.cross(point - a, direction)) for point in points))


class FocalLineIntegrationTests(TestCase):
    def test_one_view_line_uses_its_multiview_mirror_partner(self):
        case = focal_lines.generate()
        picks = case['request']['line_observations']
        first_view = case['request']['anchor_id']
        case['request']['line_observations'] = [p for p in picks
            if p['landmark_id'] != 'mirror_edge_b' or p['match_id'] == first_view]
        result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        self.assertLess(_infinite_line_error(result.sync_result.line_segments['mirror_edge_b'],
                                            case['truth']['lines']['mirror_edge_b']), 1e-4)

    def test_supplied_plane_with_only_line_mirror_pairs(self):
        case = focal_lines.generate()
        request = case['request']
        reference = request.pop('mirror_landmark_id')
        request['mirror_plane'][0] = case['truth']['points'][reference]
        request['mirror_pairs'] = [['mirror_edge_a', 'mirror_edge_b']]
        calibrations, observations, initial = _inputs(case)
        for slack in (0., .03):
            with self.subTest(slack=slack):
                result = fit_independent_focals(
                    calibrations, observations, initial, anchor_id=request['anchor_id'],
                    line_observations=[sync.SyncLineObservation(**o) for o in request['line_observations']],
                    mirror_pairs=request['mirror_pairs'], mirror_plane=request['mirror_plane'],
                    mirror_slack=slack)
                self.assertTrue(result.accepted, result.reason)
                for camera in case['truth']['cameras']:
                    self.assertLess(abs(result.calibrations[camera['id']].intrinsics.fx / camera['fx'] - 1), 1e-4)

    def test_fixture_has_independent_line_pixels_and_reversed_strokes(self):
        case = focal_lines.generate()
        self.assertEqual(len(case["truth"]["cameras"]), 4)
        self.assertEqual(len(case["request"]["line_observations"]), 12)
        for camera in case["truth"]["cameras"]:
            self.assertGreaterEqual(sum(item["match_id"] == camera["id"]
                                        for item in case["request"]["observations"]), 8)
        strokes = [item for item in case["request"]["line_observations"]
                   if item["landmark_id"] == "free_edge"]
        self.assertGreater(strokes[0]["v2"] - strokes[0]["v1"], 0)
        self.assertLess(strokes[1]["v2"] - strokes[1]["v1"], 0)
        self.assertFalse(case["request"]["lines"][0].get("known"))

    def test_joint_fit_recovers_lines_from_strokes_and_cameras(self):
        case = focal_lines.generate()
        result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        solved = result.sync_result
        self.assertEqual(set(solved.line_segments), set(case["truth"]["lines"]))
        for key, truth_line in case["truth"]["lines"].items():
            self.assertLess(_infinite_line_error(solved.line_segments[key], truth_line), 1e-4, key)
        for camera in case["truth"]["cameras"]:
            k = result.calibrations[camera["id"]].intrinsics
            self.assertLess(abs(k.fx - camera["fx"]) / camera["fx"], 1e-4)
            for key, truth_line in case["truth"]["lines"].items():
                # Truth endpoints are withheld from the fit. Their projections
                # test the recovered infinite line beyond marked intervals.
                uv, _ = project(truth_line, camera)
                a, b = uv
                direction = b - a
                direction /= np.linalg.norm(direction)
                # Compare against independently projected fitted endpoints.
                cal = result.calibrations[camera["id"]]
                similarity = solved.similarities[camera["id"]]
                fitted_camera = dict(camera, fx=k.fx, fy=k.fy,
                                     center=similarity.transform_point(cal.camera_center),
                                     rotation=cal.rotation_w2c @ similarity.rotation.T)
                fitted_uv, _ = project(solved.line_segments[key], fitted_camera)
                distance = max(abs(direction[0] * (point[1] - a[1]) -
                                   direction[1] * (point[0] - a[0]))
                               for point in fitted_uv)
                self.assertLess(distance, 0.01, (camera["id"], key))
            for point_id, xyz in case["truth"]["holdouts"].items():
                predicted, _ = project([xyz], fitted_camera)
                expected = np.asarray(case["truth"]["holdout_pixels"][point_id][camera["id"]])
                self.assertLess(np.linalg.norm(predicted[0] - expected), 0.01,
                                (camera["id"], point_id))

    def test_contradictory_mirror_stroke_refuses(self):
        result = _fit(focal_lines.generate(inconsistent=True))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)

    def test_one_view_mirror_lines_refuse(self):
        result = _fit(focal_lines.generate(weak=True))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)
