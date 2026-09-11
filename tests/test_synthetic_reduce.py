"""Reduction must retain the failure, independent checks and protected evidence."""

from copy import deepcopy
from pathlib import Path
import unittest

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.reduce import drop_points, forbidden_geometry, reduce_case, removable_points
from tools.synthetic_sync.roles import one_sided_fit_only
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import solve


class SyntheticReductionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
