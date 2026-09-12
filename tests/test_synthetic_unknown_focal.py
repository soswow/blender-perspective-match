"""Read-only contracts for the bounded no-VP unknown-focal evidence."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import fingerprint
from tools.synthetic_sync.unknown_focal import METADATA, trial_request


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tools" / "synthetic_sync" / "cases" / "unknown-focal-continuation"
FROZEN = ROOT / "tools" / "synthetic_sync" / "cases"
# acos(trace(R)) loses angular precision near identity. A few double-precision
# ulps in the trace can move a reported near-zero angle by sqrt(epsilon) rad.
ROTATION_ROUNDOFF_DEG = math.degrees(math.sqrt(16 * sys.float_info.epsilon))


class UnknownFocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        rows = [json.loads(line) for line in (ARTIFACTS / "ledger.jsonl").read_text().splitlines()]
        self.header = rows[0]
        self.starts = [row for row in rows[1:] if row["kind"] == "started"]
        self.completed = [row for row in rows[1:] if row["kind"] == "completed"]

    def assertAssessmentEqual(self, left, right, path="root"):
        """Match derived assessments with field-specific numerical roundoff."""
        self.assertIs(type(left), type(right), path)
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assertAssessmentEqual(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            self.assertEqual(len(left), len(right), path)
            for index, (item_a, item_b) in enumerate(zip(left, right)):
                self.assertAssessmentEqual(item_a, item_b, f"{path}[{index}]")
        elif isinstance(left, float):
            angular = path.startswith("root.independent_geometry.cameras.") and path.endswith(".rotation_deg")
            abs_tol = ROTATION_ROUNDOFF_DEG if angular else 1e-12
            self.assertTrue(math.isclose(left, right, rel_tol=2e-13, abs_tol=abs_tol),
                            f"{path}: {left!r} != {right!r}")
        else:
            self.assertEqual(left, right, path)

    def test_ledger_reservations_and_source_runtime_identity(self):
        self.assertEqual(self.header["kind"], "budget")
        self.assertEqual(self.header["metadata"], METADATA)
        self.assertEqual((self.header["max_calls"], self.header["per_call_seconds"],
                          self.header["wall_seconds"]), (16, 180, 900))
        self.assertEqual(len(self.starts), 15)
        self.assertEqual(len(self.completed), 15)
        self.assertEqual([row["attempt"] for row in self.starts], list(range(1, 16)))
        self.assertEqual([row["attempt"] for row in self.completed], list(range(1, 16)))
        self.assertEqual(len({row["key"] for row in self.starts}), 15)
        self.assertLess(sum(row["elapsed_s"] for row in self.completed), 900)
        for start, completed in zip(self.starts, self.completed):
            with self.subTest(attempt=start["attempt"]):
                self.assertEqual(start["key"], completed["key"])
                payload = dict(label=start["label"], request=start["request"],
                               metadata=self.header["metadata"])
                expected_key = hashlib.sha256(json.dumps(
                    payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
                self.assertEqual(start["key"], expected_key)
                source = start["request"]["source_runtime"]
                self.assertRegex(source["source_sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(source["executable"])
                self.assertTrue(source["python"])
                self.assertTrue(source["numpy"])
                self.assertIn("opencv", source)
                self.assertTrue(source["environment"]["revision"])
                self.assertEqual(source["threads"]["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(source["threads"]["OMP_NUM_THREADS"], "1")

    def test_individual_sync_assessments_match_reserved_results(self):
        attempts = {start["label"]: (start, completed)
                    for start, completed in zip(self.starts, self.completed)}
        paths = sorted(ARTIFACTS.glob("no-vp-*.json"))
        self.assertEqual(len(paths), 10)
        for path in paths:
            artifact = json.loads(path.read_text())
            with self.subTest(label=artifact["label"]):
                start, completed = attempts[artifact["label"]]
                request = start["request"]["request"]
                self.assertEqual(fingerprint(request), artifact["request_sha256"])
                self.assertEqual(artifact["record"], completed["result"])
                case = read_case(ROOT / artifact["case_file"])
                self.assertEqual(request, trial_request(case, artifact["scale"]))
                case["request"] = request
                self.assertAssessmentEqual(artifact["assessment"],
                                           assess(case, artifact["record"]))
        for intrinsics in ("trueK", "guessedK"):
            saved = json.loads((ARTIFACTS / f"no-vp-pure-rotation-{intrinsics}-scale-1.json").read_text())
            self.assertEqual(saved["assessment"]["classification"],
                             "unsupported_depth_acceptance")
            self.assertTrue(saved["record"]["success"])

    def test_actual_shared_search_selection_and_oracle(self):
        report = json.loads((ARTIFACTS / "actual-shared-search.json").read_text())
        case = read_case(FROZEN / "no-vp-shared-guessedK.json")
        trials = report["trials"]
        self.assertEqual(len(trials), 5)
        self.assertEqual(len(self.starts), 10 + len(trials))
        selected_index = report["selected_index"]
        self.assertIn(selected_index, range(len(trials)))
        self.assertEqual([row["index"] for row in trials], list(range(5)))
        for trial in trials:
            with self.subTest(index=trial["index"]):
                start = self.starts[10 + trial["index"]]
                completed = self.completed[10 + trial["index"]]
                self.assertEqual(start["label"], f"actual-shared-search-{trial['index']}")
                request = start["request"]["request"]
                self.assertEqual(fingerprint(request), trial["request_sha256"])
                self.assertEqual(trial["record"], completed["result"])
                self.assertEqual(trial["warm_start"], bool(start["request"]["initial_similarities"]))
                for camera in request["cameras"]:
                    recovered = trial["record"]["cameras"][camera["id"]]
                    self.assertEqual([camera["fx"], camera["fy"]],
                                     [recovered["fx"], recovered["fy"]])
                evaluation_case = deepcopy(case)
                evaluation_case["request"] = request
                self.assertAssessmentEqual(trial["assessment"],
                                           assess(evaluation_case, trial["record"]))
        selected = trials[selected_index]
        self.assertEqual(selected_index, min(range(len(trials)),
                         key=lambda i: trials[i]["record"]["reported_rmse_px"]))
        self.assertTrue(report["improved"])
        self.assertTrue(selected["record"]["success"])
        self.assertEqual(selected["assessment"]["classification"], "accurate_acceptance")
        self.assertEqual(set(selected["record"]["cameras"]),
                         set(case["expectation"]["cameras"]))
        self.assertEqual(set(selected["record"]["landmarks"]),
                         {point["id"] for point in case["request"]["points"]})
        # No VP lines are present, so their cost is the same constant for every trial.
        self.assertTrue(math.isclose(report["final_cost"] - report["initial_cost"],
            selected["record"]["reported_rmse_px"] - trials[0]["record"]["reported_rmse_px"],
            rel_tol=2e-10, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
