"""Independent contracts for compatible plane/mirror/parallel combinations."""

from copy import deepcopy
import json
import unittest

import numpy as np

from tools.synthetic_sync.constraint_interactions import (
    evaluate_interactions, independent_direction_reference, interaction_cases,
    quantize_request_float32,
)
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.solver import load_core, solve
from test_synthetic_sync import true_record


class SyntheticConstraintInteractionTests(unittest.TestCase):
    def test_controls_remove_only_parallel_links_and_retain_independent_evidence(self):
        cases = interaction_cases()
        self.assertEqual(len(cases), 8)
        for active, removed in zip(cases[::2], cases[1::2]):
            self.assertEqual(active["truth"], removed["truth"])
            self.assertEqual(active["expectation"], removed["expectation"])
            expected = deepcopy(active["request"])
            expected["parallel_pairs"] = []
            self.assertEqual(expected, removed["request"])
        self.assertEqual(json.dumps(cases), json.dumps(interaction_cases()))

    def test_constructed_plane_mirror_and_parallel_constraints_are_compatible(self):
        case = interaction_cases()[0]
        record = true_record(case)
        record["line_segments"] = deepcopy(case["truth"]["lines"])
        assessment = evaluate_interactions(case, record)
        self.assertTrue(assessment["passed"], assessment["violations"])
        plane = case["truth"]["planes"][0]
        normal = np.asarray(plane["normal"], dtype=float)
        references = [point["known"] for point in case["request"]["points"] if point["known"] is not None]
        self.assertEqual(len(references), 3)
        self.assertEqual(np.linalg.matrix_rank(np.asarray(references) - references[0]), 2)
        np.testing.assert_allclose((np.asarray(references) - plane["origin"]) @ normal, 0, atol=1e-12)
        checks = np.asarray([point["position"] for point in case["truth"]["checks"]])
        known_ends = next(line["known"] for line in case["request"]["lines"] if line["known"] is not None)
        evidence = np.asarray(references + known_ends)
        self.assertGreater(np.min(np.linalg.norm(evidence[:, None] - checks, axis=2)), 1e-6)
        cameras = {camera["id"]: camera for camera in case["truth"]["cameras"]}
        for stroke in case["interaction"]["construction_strokes"]:
            self.assertTrue(all(visible(point, cameras[stroke["match_id"]], case["truth"]["mesh"]) for point in stroke["ends"]))
            if stroke["landmark_id"] == "parallel_reference":
                pick = next(p for p in case["request"]["line_observations"] if p["landmark_id"] == stroke["landmark_id"])
                np.testing.assert_allclose(project(stroke["ends"], cameras[stroke["match_id"]])[0],
                                           [[pick["u1"], pick["v1"]], [pick["u2"], pick["v2"]]], atol=1e-12)
        reference = independent_direction_reference(case)
        self.assertLess(reference["offset"], .003)
        np.testing.assert_allclose(np.dot(reference["direction"], normal), 0, atol=1e-12)

    def test_parallel_oracle_catches_small_in_plane_direction_drift_with_true_cameras(self):
        case = interaction_cases()[0]
        record = true_record(case)
        record["line_segments"] = deepcopy(case["truth"]["lines"])
        left = np.asarray(record["line_segments"]["side_edge_left"])
        left[1, 0] += .002
        record["line_segments"]["side_edge_left"] = left.tolist()
        record["line_segments"]["side_edge_right"] = (left * [-1, 1, 1]).tolist()
        # Both lines still lie in the physical plane and form an exact mirror.
        self.assertTrue(evaluate(case, record)["passed"])
        assessment = evaluate_interactions(case, record)
        self.assertFalse(assessment["passed"])
        self.assertEqual({row["relation"] for row in assessment["interaction_failures"]}, {"parallel"})
        # An expected weak-support warning cannot waive a hard direction.
        case["expectation"].update(outcome="warn", weak_lines=["side_edge_left", "side_edge_right"])
        record["weak_line_ids"] = case["expectation"]["weak_lines"]
        self.assertFalse(evaluate_interactions(case, record)["passed"])

    def test_reflection_oracle_catches_small_translation_with_true_cameras(self):
        case = interaction_cases()[0]
        record = true_record(case)
        record["line_segments"] = deepcopy(case["truth"]["lines"])
        record["line_segments"]["side_edge_left"] = (np.asarray(record["line_segments"]["side_edge_left"]) + [.002, 0, 0]).tolist()
        self.assertTrue(evaluate(case, record)["passed"])
        assessment = evaluate_interactions(case, record)
        self.assertFalse(assessment["passed"])
        self.assertEqual({row["relation"] for row in assessment["interaction_failures"]}, {"mirror"})

    def test_fixed_parallel_direction_survives_plane_and_mirror_reconstruction(self):
        for case in interaction_cases():
            with self.subTest(case=case["name"]):
                record = solve(case["request"])
                assessment = evaluate_interactions(case, record)
                self.assertTrue(assessment["passed"], assessment["violations"])
                known = case["interaction"]["parallel_reference"]
                np.testing.assert_array_equal(record["line_segments"][known], case["truth"]["lines"][known])

    def test_fit_only_stroke_cannot_move_jointly_constrained_mirror_lines(self):
        for source in interaction_cases()[0:3:2]:
            base = deepcopy(source)
            request, truth = base["request"], base["truth"]
            for cameras in (request["cameras"], truth["cameras"]):
                cameras.append(dict(deepcopy(cameras[2]), id="fit_view"))
            request["observations"] += [dict(pick, match_id="fit_view")
                for pick in request["observations"] if pick["match_id"] == "view_2"]
            request.update(location_match_ids=["view_0", "view_1", "view_2"], readonly_match_ids=["fit_view"])
            for point in truth["checks"]:
                if "view_2" in point["views"]:
                    point["views"].append("fit_view")
            base["expectation"]["cameras"].append("fit_view")
            extra = deepcopy(base)
            a, b = np.asarray(truth["lines"]["side_edge_right"])
            stroke = [a + .18 * (b - a), b - .1 * (b - a)]
            camera = truth["cameras"][-1]
            self.assertTrue(all(visible(point, camera, truth["mesh"]) for point in stroke))
            uv = project(stroke, camera)[0] + [.3, 0]
            extra["request"]["line_observations"].append(dict(match_id="fit_view", landmark_id="side_edge_right",
                u1=float(uv[0, 0]), v1=float(uv[0, 1]), u2=float(uv[1, 0]), v2=float(uv[1, 1]), weight=1.))
            records = []
            for case in (base, extra):
                record = solve(case["request"])
                assessment = evaluate_interactions(case, record)
                self.assertTrue(assessment["passed"], assessment["violations"])
                records.append(record)
            for key in source["expectation"]["required_lines"]:
                np.testing.assert_allclose(records[0]["line_segments"][key], records[1]["line_segments"][key], atol=1e-7, rtol=0)
            self.assertEqual(records[0]["weak_line_ids"], records[1]["weak_line_ids"])

    def test_float32_inputs_retain_compatible_hard_constraints_with_unchanged_truth(self):
        exact = interaction_cases()[0]
        case = quantize_request_float32(exact)
        self.assertEqual(case["truth"], exact["truth"])
        self.assertEqual(case["expectation"], exact["expectation"])
        record = solve(case["request"])
        assessment = evaluate_interactions(case, record)
        self.assertTrue(assessment["passed"], assessment["violations"])

    def test_invalid_or_incompatible_references_do_not_supply_exact_directions(self):
        load_core()
        from match_perspective.core.sync.lines import _fixed_parallel_line_directions

        for vector in ([0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 1], [1, 0, 0]):
            with self.subTest(reference=vector):
                known = {"reference": (np.zeros(3), np.array(vector, dtype=float))}
                self.assertFalse(_fixed_parallel_line_directions(
                    [("edge", "reference"), ("edge", "WORLD_AXIS_Z")], known))
        self.assertFalse(_fixed_parallel_line_directions([("edge", "WORLD_AXIS_X"), ("edge", "WORLD_AXIS_Z")], {}))


if __name__ == "__main__":
    unittest.main()
