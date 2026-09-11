"""Shared-plane constraints must preserve independent geometry and floor pins."""

import unittest
from pathlib import Path

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.planes import plane_case, plane_support_case, remove_planes
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, load_core, solve
from test_synthetic_sync import true_record


class SyntheticPlaneTests(unittest.TestCase):
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
