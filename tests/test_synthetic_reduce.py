"""Reduction must retain the failure, independent checks and protected evidence."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.planes import mirrored_line_plane_cases
from tools.synthetic_sync.reduce import drop_points, forbidden_geometry, reduce_case, removable_points
from tools.synthetic_sync.roles import one_sided_fit_only
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import legacy_plane_arguments, solve
from test_synthetic_sync import true_record


class SyntheticReductionTests(unittest.TestCase):
    def test_historical_replay_omits_only_inactive_plane_defaults(self):
        arguments = dict(plane_groups=[], plane_slack=None, observations=["unchanged"], future_option=True)
        wanted = dict(observations=["unchanged"], future_option=True)
        adapted, omitted = legacy_plane_arguments(arguments, {"observations"})
        self.assertEqual(adapted, wanted)
        self.assertEqual(omitted, ["plane_groups", "plane_slack"])
        self.assertEqual(arguments["plane_groups"], [])
        self.assertEqual(legacy_plane_arguments(arguments, set(arguments)), (arguments, []))
        for active in (dict(plane_groups=[("point", "Z", 1)]), dict(plane_slack=.02)):
            with self.subTest(active=active), self.assertRaisesRegex(ValueError, "active plane"):
                legacy_plane_arguments(dict(arguments, **active), {"observations"})

    def test_line_accuracy_signature_preserves_other_checks_and_failure_kind(self):
        from tools.synthetic_sync.reduce import line_accuracy

        case, _control = mirrored_line_plane_cases()
        target = "side_edge_left"
        record = true_record(case)
        record["line_segments"] = deepcopy(case["truth"]["lines"])
        self.assertTrue(evaluate(case, record)["passed"])
        record["line_segments"][target] = (np.asarray(record["line_segments"][target])+[0,0,.002]).tolist()
        self.assertEqual(line_accuracy(case, record, [target]), {target: ["plane:FREE#1"]})
        larger = deepcopy(record)
        larger["line_segments"][target] = (np.asarray(larger["line_segments"][target])+[0,0,1]).tolist()
        self.assertEqual(line_accuracy(case, larger, [target]), {target: ["offset", "plane:FREE#1"]})
        self.assertNotEqual(line_accuracy(case, larger, [target]), line_accuracy(case, record, [target]))
        for change in ("missing_camera", "wrong_camera", "exception", "refusal", "missing_line",
                       "invalid_line", "wrong_reference", "other_line", "fixed"):
            bad = deepcopy(record)
            if change == "missing_camera":
                del bad["cameras"]["view_2"]
            elif change == "wrong_camera":
                bad["cameras"]["view_2"]["center"][0] += 1
            elif change == "exception":
                bad["exception"] = "unrelated bug"
            elif change == "refusal":
                bad["success"] = False
            elif change == "missing_line":
                del bad["line_segments"][target]
            elif change == "invalid_line":
                bad["line_segments"][target][1] = bad["line_segments"][target][0]
            elif change == "wrong_reference":
                bad["landmarks"]["plane_reference_0"][2] += 1
            elif change == "other_line":
                bad["line_segments"]["side_edge_right"][0][2] += 1
            else:
                bad["line_segments"][target] = deepcopy(case["truth"]["lines"][target])
            with self.subTest(change=change):
                self.assertIsNone(line_accuracy(case, bad, [target]))

    def test_line_accuracy_requires_explicit_accuracy_contract(self):
        from tools.synthetic_sync.reduce import line_accuracy

        case, control = mirrored_line_plane_cases()
        record = true_record(case)
        record["line_segments"] = deepcopy(case["truth"]["lines"])
        record["line_segments"]["side_edge_left"][0][2] += 1
        self.assertIsNotNone(line_accuracy(case, record, ["side_edge_left"]))
        for targets in ([], ["unknown"], ["side_edge_left", "side_edge_right"]):
            self.assertIsNone(line_accuracy(case, record, targets))
        control["request"]["plane_groups"] = deepcopy(case["request"]["plane_groups"])
        self.assertIsNone(line_accuracy(control, record, ["side_edge_left"]))
        case["expectation"]["gauge"] = "similarity"
        self.assertIsNone(line_accuracy(case, record, ["side_edge_left"]))

    def test_only_optional_free_points_can_be_removed(self):
        case = one_sided_fit_only("mirror_points")
        optional = removable_points(case)
        request, expectation = case["request"], case["expectation"]
        next(p for p in request["points"] if p["id"] == optional[0])["known"] = [1, 2, 3]
        expectation["required_points"] = [optional[1]]
        request["plane_groups"] = [(optional[2], "Z", 1), (optional[3], "Z", 1)]
        before = deepcopy(case)
        removed = set(removable_points(case))
        self.assertNotIn(optional[0], removed)
        self.assertNotIn(optional[1], removed)
        self.assertNotIn(optional[2], removed)
        self.assertNotIn(optional[3], removed)
        result = drop_points(case, removed)
        self.assertEqual(case, before)
        for key in case:
            if key != "request":
                self.assertEqual(case[key], result[key])
        for key in request:
            if key not in {"points", "observations"}:
                self.assertEqual(request[key], result["request"][key])
        self.assertFalse(removed & {o["landmark_id"] for o in result["request"]["observations"]})
        for protected in (optional[0], optional[1], "ground_0", expectation["excluded_points"][0], "absent"):
            with self.subTest(protected=protected), self.assertRaises(ValueError):
                drop_points(case, [protected])

    def test_failure_signature_rejects_loss_of_camera_accuracy_and_exceptions(self):
        case = one_sided_fit_only("mirror_points")
        record = dict(success=True, message="synthetic control", reported_rmse_px=0.0,
            cameras={c["id"]: deepcopy(c) for c in case["truth"]["cameras"]},
            landmarks=deepcopy(case["truth"]["points"]), line_segments={})
        self.assertEqual(forbidden_geometry(case, record)["points"], sorted(case["expectation"]["excluded_points"]))
        for change in ("missing", "wrong", "exception", "refusal", "fixed"):
            bad = deepcopy(record)
            if change == "missing":
                del bad["cameras"]["view_2"]
            elif change == "wrong":
                bad["cameras"]["view_2"]["center"][0] += 1
            elif change == "exception":
                bad["exception"] = "unrelated bug"
            elif change == "refusal":
                bad["success"] = False
            else:
                for key in case["expectation"]["excluded_points"]:
                    del bad["landmarks"][key]
            with self.subTest(change=change):
                self.assertIsNone(forbidden_geometry(case, bad))

    def test_chunk_reduction_retains_necessary_points_and_is_repeatable(self):
        case = one_sided_fit_only("mirror_lines")
        needed = set(removable_points(case)[::17])

        def accepts(candidate, _attempt):
            return needed <= {point["id"] for point in candidate["request"]["points"]}

        reduced, summary = reduce_case(case, accepts, max_attempts=100)
        self.assertEqual(set(removable_points(reduced)), needed)
        self.assertTrue(summary["locally_irreducible"])
        self.assertFalse(summary["budget_exhausted"])
        self.assertEqual((reduced, summary), reduce_case(case, accepts, max_attempts=100))
        self.assertTrue(any(not attempt["accepted"] for attempt in summary["trace"]))

    def test_budget_does_not_claim_minimality_or_accept_failed_deletions(self):
        case = one_sided_fit_only("mirror_lines")
        reduced, summary = reduce_case(case, lambda *_: False, max_attempts=1)
        self.assertEqual(reduced, case)
        self.assertEqual(summary["attempts"], 1)
        self.assertTrue(summary["budget_exhausted"])
        self.assertFalse(summary["locally_irreducible"])
        with self.assertRaises(ValueError):
            reduce_case(case, lambda *_: True, max_attempts=0)

    def test_frozen_reductions_keep_role_contract_and_withheld_accuracy(self):
        folder = Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases"
        for kind, picks in (("points", 21), ("lines", 15)):
            with self.subTest(kind=kind):
                case = read_case(folder / f"fit-only-mirror-{kind}-reduced.json")
                self.assertEqual(len(case["request"]["observations"]), picks)
                self.assertEqual(removable_points(case), [])
                self.assertTrue(case["expectation"]["excluded_"+kind])
                assessment = evaluate(case, solve(case["request"]))
                self.assertTrue(assessment["passed"], assessment["violations"])

    def test_frozen_line_reduction_retains_support_and_accuracy(self):
        from tools.synthetic_sync.reduce import line_accuracy

        folder = Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases"
        original = read_case(folder / "mirror-lines-with-plane.json")
        case = read_case(folder / "mirror-lines-with-plane-reduced.json")
        self.assertEqual(len(case["request"]["observations"]), 19)
        self.assertEqual(removable_points(case), [])
        expected = drop_points(original, removable_points(original))
        for key in ("request", "truth", "expectation", "plane_removal_expectation"):
            self.assertEqual(case[key], expected[key])
        result = solve(case["request"])
        assessment = evaluate(case, result)
        self.assertTrue(assessment["passed"], assessment["violations"])
        self.assertIsNone(line_accuracy(case, result, case["expectation"]["required_lines"]))


if __name__ == "__main__":
    unittest.main()
