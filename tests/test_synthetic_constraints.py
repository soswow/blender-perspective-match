"""Constraint contribution and independent reconstructed-geometry checks."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from tools.synthetic_sync.constraints import FAMILIES, add_mirror_support_stroke, constraint_case, mirror_line_reference, remove_constraint
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, load_core, solve
from test_synthetic_sync import true_record


class ConstraintOracleTests(unittest.TestCase):
    def test_accurate_cameras_cannot_hide_wrong_required_points(self):
        case = constraint_case("mirror_points")
        result = true_record(case)
        self.assertTrue(evaluate(case, result)["passed"])
        key = case["expectation"]["required_points"][0]
        result["landmarks"][key][0] += 0.2
        self.assertFalse(evaluate(case, result)["passed"])

    def test_warning_contract_requires_recognition_and_does_not_relax_other_cases(self):
        case = read_case(Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases/mirror-lines-weak.json")
        result = true_record(case)
        result["line_segments"] = deepcopy(case["truth"]["lines"])
        self.assertFalse(evaluate(case,result)["passed"])
        result["weak_line_ids"] = case["expectation"]["weak_lines"]
        self.assertTrue(evaluate(case,result)["passed"])
        key = case["expectation"]["required_lines"][0]
        result["line_segments"][key] = (np.asarray(result["line_segments"][key])+[.5,0,0]).tolist()
        self.assertTrue(evaluate(case,result)["passed"])
        case["expectation"].pop("weak_lines")
        case["expectation"]["outcome"] = "solve"
        self.assertFalse(evaluate(case,result)["passed"])

    def test_line_oracle_allows_different_extents_but_rejects_wrong_depth(self):
        case = constraint_case("mirror_lines")
        result = true_record(case)
        for key, ends in case["truth"]["lines"].items():
            a,b = np.asarray(ends)
            # Reversed endpoints and a longer helper describe the same line.
            result["line_segments"][key] = [(b+2*(b-a)).tolist(), (a-3*(b-a)).tolist()]
        self.assertTrue(evaluate(case, result)["passed"])
        key = case["expectation"]["required_lines"][0]
        result["line_segments"][key] = (np.asarray(result["line_segments"][key])+[.2,0,0]).tolist()
        self.assertFalse(evaluate(case, result)["passed"])

    def test_line_oracle_rejects_wrong_direction_through_correct_midpoint(self):
        case = constraint_case("mirror_lines")
        result = true_record(case)
        result["line_segments"] = deepcopy(case["truth"]["lines"])
        key = case["expectation"]["required_lines"][0]
        midpoint = np.mean(result["line_segments"][key],axis=0)
        result["line_segments"][key] = [(midpoint-[.2,0,0]).tolist(),(midpoint+[.2,0,0]).tolist()]
        self.assertFalse(evaluate(case,result)["passed"])

    def test_removing_constraint_preserves_every_measurement(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                case = constraint_case(family)
                removed = remove_constraint(case)
                for key in ("observations", "line_observations", "cameras"):
                    self.assertEqual(case["request"][key], removed["request"][key])
                self.assertEqual(case["truth"], removed["truth"])

    def test_unrelated_outlier_cannot_make_weak_line_support_look_strong(self):
        _core, sync = load_core()
        from match_perspective.core.sync.lines import line_support_angles
        from tools.synthetic_sync.geometry import project
        case = constraint_case("mirror_lines")
        matches = {c["id"]:sync.SyncMatchInput(c["id"],calibration(c)) for c in case["truth"]["cameras"]}
        similarities = {key:sync.SimilarityTransform() for key in matches}
        segments = {key:tuple(np.asarray(p) for p in ends) for key,ends in case["truth"]["lines"].items()}
        observations = {key:[] for key in segments}
        for pick in case["request"]["line_observations"]:
            observations[pick["landmark_id"]].append(sync.SyncLineObservation(**pick))
        def angles(known=None):
            return line_support_angles(segments,observations,similarities,matches,
                known_lines=known,mirror_pairs=case["request"]["mirror_pairs"],mirror_normal=np.array([1,0,0]))
        before = angles()
        camera = next(c for c in case["truth"]["cameras"] if c["id"] == "view_2")
        uv = project(segments["side_edge_right"],camera)[0] + [0,300]
        observations["side_edge_right"].append(sync.SyncLineObservation("view_2","side_edge_right",*uv[0],*uv[1]))
        self.assertEqual(angles(),before)
        # Known geometry on either mirrored partner supplies depth directly.
        self.assertEqual(angles({"side_edge_right":segments["side_edge_right"]}),{})

    def test_ordinary_free_line_support_uses_distinct_inlier_planes(self):
        _core, sync = load_core()
        from match_perspective.core.sync.lines import line_support_angles
        from tools.synthetic_sync.geometry import project
        case = constraint_case("mirror_lines")
        camera = next(c for c in case["truth"]["cameras"] if c["id"] == "view_1")
        near = deepcopy(camera)
        near["id"] = "near"
        near["center"][0] += 0.005
        far = next(c for c in case["truth"]["cameras"] if c["id"] == "view_2")
        points = tuple(np.asarray(p) for p in case["truth"]["lines"]["side_edge_right"])
        matches, similarities, observations = {}, {}, []
        for c in (camera,near,far):
            matches[c["id"]] = sync.SyncMatchInput(c["id"],calibration(c))
            similarities[c["id"]] = sync.SimilarityTransform()
            uv = project(points,c)[0]
            observations.append(sync.SyncLineObservation(c["id"],"edge",*uv[0],*uv[1]))
        weak = line_support_angles({"edge":points},{"edge":observations[:2]},similarities,matches)
        strong = line_support_angles({"edge":points},{"edge":observations},similarities,matches)
        self.assertLess(weak["edge"],1)
        self.assertGreater(strong["edge"],10)


class ConstraintCorpusTests(unittest.TestCase):
    def test_noisy_mirror_line_reports_weak_geometry_despite_low_fit_error(self):
        case = read_case(Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases/mirror-lines-weak.json")
        result = solve(case["request"])
        self.assertTrue(result["success"], result["message"])
        self.assertLess(result["reported_rmse_px"], 1.0)
        self.assertEqual(set(result.get("weak_line_ids", [])), set(case["expectation"]["required_lines"]))
        for angle in result["line_support_angles_deg"].values():
            self.assertLess(angle, 3.0)
        reference = mirror_line_reference(case,list(result["cameras"].values()))
        self.assertGreater(reference["direction_error_deg"],20)
        self.assertAlmostEqual(result["line_support_angles_deg"]["side_edge_left"],reference["support_angle_deg"],places=5)

    def test_another_view_reduces_mirror_line_error_and_clears_warning(self):
        base = read_case(Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases/mirror-lines-weak.json")
        case = add_mirror_support_stroke(base)
        result = solve(case["request"])
        assessment = evaluate(case,result)
        self.assertTrue(assessment["passed"],assessment["violations"])
        self.assertFalse(result["weak_line_ids"])
        self.assertGreater(min(result["line_support_angles_deg"].values()),10)

    def test_known_3d_lines_do_not_need_multiview_depth_support(self):
        case = constraint_case("known_lines")
        result = solve(case["request"])
        self.assertTrue(result["success"],result["message"])
        self.assertFalse(result["weak_line_ids"])
        self.assertFalse(result["line_support_angles_deg"])

    def test_constraints_supply_information_and_removal_is_honest(self):
        for family in FAMILIES:
            original = constraint_case(family)
            for case in (original, remove_constraint(original)):
                with self.subTest(case=case["name"]):
                    result = solve(case["request"])
                    assessment = evaluate(case,result)
                    self.assertTrue(assessment["passed"], assessment["violations"])


if __name__ == "__main__":
    unittest.main()
