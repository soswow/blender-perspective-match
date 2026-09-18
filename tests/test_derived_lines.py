"""From Points lines stay attached while shared relations move their endpoints."""

from __future__ import annotations

from copy import deepcopy
from unittest import TestCase, mock

import numpy as np

from match_perspective.core import sync
from match_perspective.core.derived_lines import derived_line_geometry
from match_perspective.core.focal_bundle import refine_fixed_focals
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core import lens_refine
from match_perspective.core.sync.request import SyncSolveRequest
from tools.synthetic_sync import focal_constraints
from tools.synthetic_sync import focal_line_constraints
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.solver import result_record, solver_arguments


def _from_points_case(*, oracle_focals=False, endpoint_noise_scale=1.0):
    case = focal_line_constraints.generate(axis_parallel=True)
    request, truth = case["request"], case["truth"]
    if oracle_focals:
        truth_cameras = {item["id"]: item for item in truth["cameras"]}
        for camera in request["cameras"]:
            camera["fx"] = truth_cameras[camera["id"]]["fx"]
            camera["fy"] = truth_cameras[camera["id"]]["fy"]
    first, second = (np.asarray(point, float)
                     for point in truth["lines"]["axis_edge"])
    extra = {
        "endpoint_a": first,
        "endpoint_b": second,
        "endpoint_a_plane_support": first + np.array((0.65, 0.0, 0.25)),
        "endpoint_b_plane_support": second + np.array((-0.55, 0.35, 0.0)),
    }
    noise = {
        camera["id"]: endpoint_noise_scale * np.asarray(value, float)
        for camera, value in zip(
            truth["cameras"], ((0.8, -0.6), (-0.7, 0.9), (0.6, 0.7), (-0.8, -0.5)))
    }
    for key, point in extra.items():
        truth["points"][key] = point.tolist()
        request["points"].append(dict(id=key, ground=False, known=None))
        for camera in truth["cameras"]:
            uv, depth = project([point], camera)
            assert depth[0] > 0
            measured = uv[0].copy()
            if key == "endpoint_b":
                measured += noise[camera["id"]]
            request["observations"].append(dict(
                match_id=camera["id"], landmark_id=key,
                u=float(measured[0]), v=float(measured[1]), weight=1.0))
    request["line_observations"] = [
        item for item in request["line_observations"]
        if item["landmark_id"] != "axis_edge"
    ]
    request["derived_lines"] = [["axis_edge", "endpoint_a", "endpoint_b"]]
    request["plane_groups"] += [
        ["endpoint_a", "Y", 3], ["endpoint_a_plane_support", "Y", 3],
        ["endpoint_b", "Z", 4], ["endpoint_b_plane_support", "Z", 4],
    ]
    return case


def _direction(result):
    first, second = result.line_segments["axis_edge"]
    value = second - first
    return value / np.linalg.norm(value)


class DerivedLineIntegrationTests(TestCase):
    def test_cold_sync_then_common_fit_moves_endpoints_and_keeps_attachment(self):
        # Fixed-focal continuation is assessed with independently correct fixed
        # intrinsics. The free-focal regression below owns recovery from the
        # deliberately perturbed focal starts in the base fixture.
        case = _from_points_case(oracle_focals=True, endpoint_noise_scale=5.0)
        arguments = solver_arguments(case["request"])
        initial = sync.solve_landmark_sync(**arguments)
        self.assertTrue(initial.success, initial.message)
        request = SyncSolveRequest(**arguments)
        outcome = refine_fixed_focals(request, initial)
        self.assertTrue(outcome.accepted, outcome.reason)
        solved = outcome.sync_result
        assert solved is not None
        segment = solved.line_segments["axis_edge"]
        np.testing.assert_allclose(segment[0], solved.landmarks["endpoint_a"], atol=1e-12)
        np.testing.assert_allclose(segment[1], solved.landmarks["endpoint_b"], atol=1e-12)
        constrained_sine = np.linalg.norm(np.cross(_direction(solved), (0, 0, 1)))
        self.assertLess(constrained_sine, 0.01)
        support_x = np.mean([solved.landmarks[f"axis_{index}"][0]
                             for index in range(4)])
        self.assertLess(abs(float(segment[0][0] - support_x)), 1e-3)
        self.assertLess(abs(float(segment[1][0] - support_x)), 1e-3)
        self.assertLess(abs(float(solved.landmarks["endpoint_a"][1] -
                                  solved.landmarks["endpoint_a_plane_support"][1])), 1e-3)
        self.assertLess(abs(float(solved.landmarks["endpoint_b"][2] -
                                  solved.landmarks["endpoint_b_plane_support"][2])), 1e-3)
        self.assertEqual(len([item for item in request.line_observations
                              if item.landmark_id == "axis_edge"]), 0)
        self.assertGreater(max(
            np.linalg.norm(solved.landmarks[key] - initial.landmarks[key])
            for key in ("endpoint_a", "endpoint_b")), 1e-8)
        score = JointFitScorer(request, solved).score(solved)
        self.assertTrue(score.valid, score.reason)
        assessment = focal_constraints.assess(
            case,
            result_record(solved, case["request"]["cameras"],
                          calibrations=outcome.calibrations),
            validated=True,
        )
        self.assertTrue(np.isfinite(assessment["withheld_max_px"]))
        self.assertLess(assessment["withheld_max_px"], 10.0)

        control_request = deepcopy(request)
        control_request.parallel_pairs = [
            pair for pair in control_request.parallel_pairs
            if "axis_edge" not in pair]
        control_request.plane_groups = [
            item for item in control_request.plane_groups if item[0] != "axis_edge"]
        control = refine_fixed_focals(control_request, initial)
        self.assertTrue(control.accepted, control.reason)
        control_solved = control.sync_result
        self.assertLess(abs(float(control_solved.landmarks["endpoint_a"][1] -
                                  control_solved.landmarks[
                                      "endpoint_a_plane_support"][1])), 1e-3)
        self.assertLess(abs(float(control_solved.landmarks["endpoint_b"][2] -
                                  control_solved.landmarks[
                                      "endpoint_b_plane_support"][2])), 1e-3)
        control_sine = np.linalg.norm(np.cross(_direction(control.sync_result), (0, 0, 1)))
        self.assertGreater(control_sine, 0.005)
        self.assertGreater(control_sine, 5.0 * constrained_sine)

    def test_invalid_and_coincident_endpoint_definitions_are_diagnosed(self):
        case = _from_points_case()
        arguments = solver_arguments(case["request"])
        arguments["derived_lines"] = [("axis_edge", "endpoint_a", "missing")]
        with self.assertRaisesRegex(ValueError, "missing point"):
            sync.solve_landmark_sync(**arguments)
        arguments["derived_lines"] = [("axis_edge", "endpoint_a", "endpoint_a")]
        with self.assertRaisesRegex(ValueError, "two different"):
            sync.solve_landmark_sync(**arguments)
        with self.assertRaisesRegex(ValueError, "coincident"):
            derived_line_geometry(
                {"a": np.zeros(3), "b": np.zeros(3)},
                [("edge", "a", "b")],
            )

    def test_free_focal_route_uses_endpoint_geometry_without_line_parameters(self):
        case = _from_points_case()
        arguments = solver_arguments(case["request"])
        true_cameras = {item["id"]: item for item in case["truth"]["cameras"]}
        similarities = {}
        matches = []
        for item in arguments["matches"]:
            private = item.calibration
            true = true_cameras[item.match_id]
            rotation = np.asarray(true["rotation"]).T @ private.rotation_w2c
            center = np.asarray(true["center"])
            similarities[item.match_id] = sync.SimilarityTransform(
                1.0, rotation, center - rotation @ private.camera_center)
            matches.append(lens_refine.MatchLensInput(
                item.match_id, {}, private.intrinsics,
                base_calibration=private,
            ))
        truth_points = {key: np.asarray(value, float)
                        for key, value in case["truth"]["points"].items()}
        initial = sync.SyncSolveResult(
            similarities=similarities,
            landmarks=truth_points,
            line_segments={key: tuple(np.asarray(point, float) for point in segment)
                           for key, segment in case["truth"]["lines"].items()},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="oracle-start bundle control",
            success=True,
        )
        with mock.patch.object(lens_refine, "_run_sync", return_value=initial):
            outcome = lens_refine.refine_lenses_from_landmarks(
                matches, arguments["observations"],
                anchor_id=arguments["anchor_id"],
                known_world=arguments["known_world"],
                line_observations=arguments["line_observations"],
                known_lines=arguments["known_lines"],
                derived_lines=arguments["derived_lines"],
                parallel_pairs=arguments["parallel_pairs"],
                fixed_similarities=arguments["fixed_similarities"],
                mirror_pairs=arguments["mirror_pairs"],
                mirror_plane=arguments["mirror_plane"],
                mirror_slack=arguments["mirror_slack"],
                mirror_landmark_id=arguments["mirror_landmark_id"],
                plane_groups=arguments["plane_groups"],
                plane_slack=arguments["plane_slack"],
                location_match_ids=arguments["location_match_ids"],
                readonly_match_ids=arguments["readonly_match_ids"],
                estimate_focal_from_points=True,
                pick_sigma_px=case.get("pick_sigma_px", 1.0),
            )
        self.assertTrue(outcome.improved, outcome.refusal_reason)
        segment = outcome.sync_result.line_segments["axis_edge"]
        np.testing.assert_allclose(
            segment[0], outcome.sync_result.landmarks["endpoint_a"], atol=1e-12)
        np.testing.assert_allclose(
            segment[1], outcome.sync_result.landmarks["endpoint_b"], atol=1e-12)
        self.assertLess(np.linalg.norm(np.cross(_direction(outcome.sync_result),
                                                (0, 0, 1))), 0.01)
