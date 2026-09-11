"""Shared-plane constraints must preserve independent geometry and floor pins."""

import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.planes import mirror_plane_reference, mirrored_line_plane_cases, plane_case, plane_support_case, remove_planes
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, load_core, solve, solver_arguments
from test_synthetic_sync import true_record


class SyntheticPlaneTests(unittest.TestCase):
    def test_plane_oracle_rejects_small_drift_with_true_cameras(self):
        case, _control = mirrored_line_plane_cases()
        result = true_record(case)
        result["line_segments"] = deepcopy(case["truth"]["lines"])
        self.assertTrue(evaluate(case,result)["passed"])
        result["line_segments"]["side_edge_left"] = (np.asarray(result["line_segments"]["side_edge_left"])+[0,0,.002]).tolist()
        assessment = evaluate(case,result)
        self.assertFalse(assessment["passed"])
        self.assertTrue(any("distance to FREE#1 plane" in item for item in assessment["violations"]))

    def test_plane_warning_requires_independent_support_and_actual_plane_membership(self):
        _core, sync = load_core()
        case, control = mirrored_line_plane_cases()
        arguments = solver_arguments(control["request"])
        result = sync.solve_landmark_sync(**arguments)
        support = sync.supported_line_planes(result.landmarks,result.line_segments,
            case["request"]["plane_groups"],{})
        self.assertEqual(set(support),set(case["truth"]["lines"]))
        self.assertFalse(sync.supported_line_planes(result.landmarks,result.line_segments,
            case["request"]["plane_groups"],{},excluded_support_ids={"plane_reference_2"}))
        without_third_reference = [group for group in case["request"]["plane_groups"] if group[0] != "plane_reference_2"]
        self.assertFalse(sync.supported_line_planes(result.landmarks,result.line_segments,without_third_reference,{}))
        observations = {}
        for pick in arguments["line_observations"]:
            observations.setdefault(pick.landmark_id,[]).append(pick)
        angles = sync.line_support_angles(result.line_segments,observations,result.similarities,
            {match.match_id:match for match in arguments["matches"]},
            mirror_pairs=arguments["mirror_pairs"],mirror_normal=arguments["mirror_plane"][1],support_planes=support)
        # The old geometry misses the plane: naming that plane must not clear
        # its warning before reconstruction actually honors the constraint.
        self.assertTrue(all(angle < 2. for angle in angles.values()))

    def test_supported_plane_survives_mirror_line_reconstruction(self):
        case = read_case(Path(__file__).resolve().parents[1]/"tools/synthetic_sync/cases/mirror-lines-with-plane.json")
        control = remove_planes(case)
        for metrics in mirror_plane_reference(case).values():
            self.assertLess(metrics["angle_error_deg"], 1.)
            self.assertGreater(metrics["support_angle_deg"], 60.)
        for variant in (case, control):
            with self.subTest(case=variant["name"]):
                result = solve(variant["request"])
                assessment = evaluate(variant,result)
                self.assertTrue(assessment["passed"],assessment["violations"])
                if variant is case:
                    self.assertFalse(result["weak_line_ids"])
                    left = np.asarray(result["line_segments"]["side_edge_left"])
                    right = np.asarray(result["line_segments"]["side_edge_right"])
                    reflected = left*np.array([-1,1,1])
                    direction = right[1]-right[0]
                    direction /= np.linalg.norm(direction)
                    self.assertLess(np.max(np.linalg.norm(np.cross(reflected-right[0],direction),axis=1)),1e-8)

    def test_plane_seed_requires_independent_support_and_a_forward_well_separated_ray(self):
        _core, sync = load_core()
        camera = dict(width=100, height=80, fx=100, fy=100, cx=50, cy=40,
                      center=[0,0,0], rotation=np.eye(3).tolist())
        matches = {"view": sync.SyncMatchInput("view", calibration(camera))}
        poses = {"view": sync.SimilarityTransform()}
        observation = sync.SyncObservation("view", "target", 60., 50.)

        def seeds(landmarks, axis, *, pick=observation, slack=0., locations=None):
            return sync.seed_plane_points(
                {key:np.asarray(point,dtype=float) for key,point in landmarks.items()}, {},
                [(key,axis,1) for key in (*landmarks,"target")], {"target":[pick]},
                poses, matches, plane_slack=slack, location_match_ids=locations)

        np.testing.assert_allclose(seeds({"support":[0,0,3]},"Z")["target"], [.3,.3,3], atol=1e-12)
        self.assertFalse(seeds({},"Z"))
        self.assertFalse(seeds({"support":[0,0,-3]},"Z"))
        self.assertFalse(seeds({"support":[0,0,3]},"Z", locations=set()))
        self.assertFalse(seeds({"support":[0,0,3]},"Z", slack=.01))
        self.assertFalse(seeds({"support":[1,0,0]},"X",
            pick=sync.SyncObservation("view","target",50.01,40.)))
        self.assertFalse(seeds({"a":[0,0,3],"b":[1,0,3]},"FREE"))
        self.assertFalse(seeds({"a":[0,0,3],"b":[1,0,3],"c":[2,0,3]},"FREE"))
        np.testing.assert_allclose(seeds({"a":[0,0,3],"b":[1,0,3],"c":[0,1,3]},"FREE")["target"], [.3,.3,3], atol=1e-12)

    def test_supported_plane_recovers_one_permitted_pick_without_fit_only_leakage(self):
        for axis in ("X", "FREE"):
            case = plane_support_case(axis)
            for variant in (case, remove_planes(case), plane_support_case(axis, fit_only=True)):
                with self.subTest(case=variant["name"]):
                    result = solve(variant["request"])
                    assessment = evaluate(variant, result)
                    self.assertTrue(assessment["passed"], assessment["violations"])
                    target = variant["plane_support"]["point"]
                    self.assertEqual(target in result["plane_seeded_landmark_ids"], target in variant["expectation"]["required_points"])

    def test_single_view_depth_has_an_independent_linear_solution(self):
        for axis in ("X", "FREE"):
            case = plane_support_case(axis)
            key = case["plane_support"]["point"]
            picks = [p for p in case["request"]["observations"] if p["landmark_id"] == key]
            self.assertEqual(len(picks), 1)
            pick = picks[0]
            camera = next(c for c in case["truth"]["cameras"] if c["id"] == pick["match_id"])
            plane = next(p for p in case["truth"]["planes"] if p["axis"] == axis)
            intrinsics = np.array([[camera["fx"],0,camera["cx"]], [0,camera["fy"],camera["cy"]], [0,0,1]])
            ray = np.asarray(camera["rotation"]).T @ np.linalg.solve(intrinsics, [pick["u"],pick["v"],1])
            normal, origin, center = (np.asarray(v) for v in (plane["normal"],plane["origin"],camera["center"]))
            distance = (normal@(origin-center))/(normal@ray)
            self.assertGreater(distance, 0)
            np.testing.assert_allclose(center+distance*ray, case["truth"]["points"][key], atol=1e-10)

    def test_free_plane_cannot_lift_hard_ground_during_initialization(self):
        case = read_case(Path(__file__).resolve().parents[1]/"tools/synthetic_sync/cases/free-plane-hard-ground.json")
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])

    def test_floor_oracle_rejects_drift_even_with_true_cameras(self):
        case = plane_case("free_ground")
        result = true_record(case)
        self.assertTrue(evaluate(case, result)["passed"])
        result["landmarks"]["ground_0"][2] += .0004
        assessment = evaluate(case, result)
        self.assertFalse(assessment["passed"])
        self.assertTrue(any("ground distance" in violation for violation in assessment["violations"]))

    def test_ground_slack_still_allows_plane_members_to_refine(self):
        case = plane_case("free_ground", noise_px=.3)
        case["request"]["ground_slack"] = .02
        case["expectation"].pop("ground_max_distance")
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])
        self.assertGreater(max(abs(result["landmarks"][key][2]) for key in ("ground_0", "ground_2")), 1e-5)

    def test_plane_removal_retains_all_other_evidence_and_truth(self):
        case = plane_case("free_ground", noise_px=.3)
        removed = remove_planes(case)
        self.assertEqual(removed["truth"], case["truth"])
        self.assertEqual(removed["expectation"], case["expectation"])
        for key in case["request"]:
            if key != "plane_groups":
                self.assertEqual(removed["request"][key], case["request"][key])

    def test_added_plane_points_are_visible_and_separate_from_withheld_checks(self):
        for family in ("free_ground", "free_tilted"):
            case = plane_case(family)
            cameras = {c["id"]: c for c in case["truth"]["cameras"]}
            checks = np.asarray([p["position"] for p in case["truth"]["checks"]])
            for pick in case["request"]["observations"]:
                if not pick["landmark_id"].startswith("tilted_"):
                    continue
                position = case["truth"]["points"][pick["landmark_id"]]
                camera = cameras[pick["match_id"]]
                self.assertTrue(visible(position, camera, case["truth"]["mesh"]))
                np.testing.assert_allclose(project([position], camera)[0][0], [pick["u"], pick["v"]], atol=1e-10)
                self.assertGreater(np.min(np.linalg.norm(checks-position, axis=1)), 1e-5)


if __name__ == "__main__":
    unittest.main()
