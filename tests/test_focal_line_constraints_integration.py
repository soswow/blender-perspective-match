"""Independent projections for line plane and parallel FOV relations."""

from __future__ import annotations

from unittest import TestCase, mock

import numpy as np

from match_perspective.core import geometry, sync, focal_bundle
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_line_constraints
from tools.synthetic_sync.geometry import project


def _inputs(case):
    request = case["request"]
    true_cameras = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    calibrations, similarities = {}, {}
    for camera in request["cameras"]:
        camera_id = camera["id"]
        k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                      camera["cy"], camera["width"], camera["height"])
        calibrations[camera_id] = geometry.Calibration(
            k, np.asarray(camera["rotation"]), np.asarray(camera["center"]))
        true = true_cameras[camera_id]
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


def _infinite_line_error(segment, truth):
    a, b = np.asarray(truth, float)
    direction = (b-a)/np.linalg.norm(b-a)
    return max(np.linalg.norm(np.cross(np.asarray(point)-a, direction)) for point in segment)


def _fit(case):
    request = case["request"]
    calibrations, observations, initial = _inputs(case)
    return fit_independent_focals(
        calibrations, observations, initial, anchor_id=request["anchor_id"],
        pick_sigma_px=case["pick_sigma_px"],
        line_observations=[sync.SyncLineObservation(**item)
                           for item in request["line_observations"]],
        plane_groups=request["plane_groups"], plane_slack=request["plane_slack"],
        parallel_pairs=request["parallel_pairs"],
        mirror_pairs=request.get("mirror_pairs"), mirror_plane=request.get("mirror_plane"))


class FocalLineConstraintIntegrationTests(TestCase):
    def test_tilted_anchor_recovers_line_planes_and_world_parallel_direction(self):
        from tools.synthetic_sync.focal_orientation import rotation

        case = focal_line_constraints.generate(axis_parallel=True)
        calibrations, observations, initial = _inputs(case)
        request = case['request']
        anchor = request['anchor_id']
        center = calibrations[anchor].camera_center.copy()
        turn = rotation()
        calibrations[anchor].rotation_w2c = calibrations[anchor].rotation_w2c @ turn.T
        initial.landmarks = {key: center + turn @ (point - center)
                             for key, point in initial.landmarks.items()}
        for camera in case['truth']['cameras']:
            key = camera['id']
            target_rotation = np.asarray(camera['rotation']) @ turn.T
            target_center = center + turn @ (np.asarray(camera['center']) - center)
            sim_rotation = target_rotation.T @ calibrations[key].rotation_w2c
            initial.similarities[key] = sync.SimilarityTransform(
                1., sim_rotation, target_center - sim_rotation @ calibrations[key].camera_center)
        initial.similarities[anchor] = sync.SimilarityTransform()
        result = fit_independent_focals(
            calibrations, observations, initial, anchor_id=anchor,
            pick_sigma_px=case['pick_sigma_px'],
            line_observations=[sync.SyncLineObservation(**item)
                               for item in request['line_observations']],
            plane_groups=request['plane_groups'], parallel_pairs=request['parallel_pairs'])
        self.assertTrue(result.accepted, result.reason)
        for key, truth_line in case['truth']['lines'].items():
            self.assertLess(_infinite_line_error(result.sync_result.line_segments[key], truth_line), .002)
        for camera in case['truth']['cameras']:
            key = camera['id']
            cal, sim = result.calibrations[key], result.sync_result.similarities[key]
            fitted = dict(camera, fx=cal.intrinsics.fx, fy=cal.intrinsics.fy,
                          rotation=cal.rotation_w2c @ sim.rotation.T,
                          center=sim.transform_point(cal.camera_center))
            for point in case['truth']['holdouts'].values():
                self.assertLess(np.linalg.norm(project([point], camera)[0] -
                                                project([point], fitted)[0]), .01)

    def test_parallel_only_uses_constrained_noise_accounting(self):
        case = focal_line_constraints.generate()
        case['request']['plane_groups'] = []
        with mock.patch.object(focal_bundle, '_fits_noise_model',
                               side_effect=AssertionError('Constraints counted as image rows')), \
             mock.patch.object(focal_bundle, '_fits_constrained_noise_model',
                               wraps=focal_bundle._fits_constrained_noise_model) as check:
            result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        check.assert_called_once()

    def test_parallel_lines_keep_plane_and_mirror_relations_together(self):
        case = focal_line_constraints.generate()
        first = np.asarray(case['truth']['lines']['free_edge_a'])
        second = np.asarray(case['truth']['lines']['free_edge_b'])
        direction = first[1] - first[0]
        direction /= np.linalg.norm(direction)
        difference = second.mean(axis=0) - first.mean(axis=0)
        normal = difference - direction * (difference @ direction)
        normal /= np.linalg.norm(normal)
        case['request']['mirror_pairs'] = [['free_edge_a', 'free_edge_b']]
        case['request']['mirror_plane'] = [((first.mean(axis=0) + second.mean(axis=0))/2).tolist(),
                                            normal.tolist()]
        result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        for line_id in ('free_edge_a', 'free_edge_b'):
            self.assertLess(_infinite_line_error(result.sync_result.line_segments[line_id],
                                                 case['truth']['lines'][line_id]), .002)

    def test_oracle_lines_are_distinct_strokes_in_supported_planes(self):
        case = focal_line_constraints.generate()
        request, truth = case["request"], case["truth"]
        for camera in truth["cameras"]:
            self.assertGreaterEqual(sum(o["match_id"] == camera["id"]
                                        for o in request["observations"]), 8)
        self.assertEqual(len(request["line_observations"]), 12)
        self.assertEqual(set(map(tuple, request["parallel_pairs"])),
                         {("free_edge_a", "free_edge_b")})
        self.assertLess(np.linalg.norm(np.cross(
            np.subtract(*truth["lines"]["free_edge_a"]),
            np.subtract(*truth["lines"]["free_edge_b"]))), 1e-10)
        for line_id, endpoints in truth["lines"].items():
            for camera in truth["cameras"]:
                stored = np.asarray(truth["line_oracle_pixels"][line_id][camera["id"]])
                self.assertGreater(np.linalg.norm(stored[1] - stored[0]), 20)
                full = project(endpoints, camera)[0]
                self.assertGreater(np.linalg.norm(stored - full), 1)

    def test_joint_fit_preserves_line_relations_and_withheld_projection(self):
        case = focal_line_constraints.generate()
        result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        solved = result.sync_result
        for line_id, truth_line in case["truth"]["lines"].items():
            self.assertLess(_infinite_line_error(solved.line_segments[line_id], truth_line),
                            0.002, line_id)
        directions = []
        for line_id in ("free_edge_a", "free_edge_b"):
            a, b = np.asarray(solved.line_segments[line_id])
            directions.append((b-a)/np.linalg.norm(b-a))
        self.assertLess(np.linalg.norm(np.cross(*directions)), 0.002)
        for camera in case["truth"]["cameras"]:
            k = result.calibrations[camera["id"]].intrinsics
            self.assertLess(abs(k.fx/camera["fx"] - 1), 0.002)
            cal = result.calibrations[camera["id"]]
            similarity = solved.similarities[camera["id"]]
            fitted = dict(camera, fx=k.fx, fy=k.fy,
                          center=similarity.transform_point(cal.camera_center),
                          rotation=cal.rotation_w2c @ similarity.rotation.T)
            for xyz in case["truth"]["holdouts"].values():
                expected = project([xyz], camera)[0][0]
                observed = project([xyz], fitted)[0][0]
                self.assertLess(np.linalg.norm(observed-expected), 0.05)

    def test_contradictory_line_stroke_refuses(self):
        result = _fit(focal_line_constraints.generate(inconsistent="parallel"))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)

    def test_false_parallel_declaration_refuses(self):
        case = focal_line_constraints.generate()
        case["request"]["parallel_pairs"] = [["axis_edge", "free_edge_a"]]
        result = _fit(case)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)

    def test_false_plane_declaration_refuses(self):
        case = focal_line_constraints.generate()
        groups = case["request"]["plane_groups"]
        case["request"]["plane_groups"] = [
            [item[0], "FREE", 1] if item[0] == "axis_edge" else item
            for item in groups]
        result = _fit(case)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)

    def test_world_axis_parallel_line_recovers_direction_and_fov(self):
        case = focal_line_constraints.generate(axis_parallel=True)
        result = _fit(case)
        self.assertTrue(result.accepted, result.reason)
        a, b = np.asarray(result.sync_result.line_segments["axis_edge"])
        direction = (b-a)/np.linalg.norm(b-a)
        self.assertLess(np.linalg.norm(np.cross(direction, [0, 0, 1])), 0.002)
        self.assertLess(_infinite_line_error(
            result.sync_result.line_segments["axis_edge"],
            case["truth"]["lines"]["axis_edge"]), 0.002)
        for camera in case["truth"]["cameras"]:
            k = result.calibrations[camera["id"]].intrinsics
            self.assertLess(abs(k.fx/camera["fx"] - 1), 0.002)
            cal = result.calibrations[camera["id"]]
            similarity = result.sync_result.similarities[camera["id"]]
            fitted = dict(camera, fx=k.fx, fy=k.fy,
                          center=similarity.transform_point(cal.camera_center),
                          rotation=cal.rotation_w2c @ similarity.rotation.T)
            for xyz in case["truth"]["holdouts"].values():
                expected = project([xyz], camera)[0][0]
                observed = project([xyz], fitted)[0][0]
                self.assertLess(np.linalg.norm(observed-expected), 0.05)
