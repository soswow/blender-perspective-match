"""Read-only checks of the bounded independent-focal prototype corpus."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

from tools.synthetic_sync.independent_focal import CASES, LIMITS, input_only
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import fingerprint


ROOT = Path(__file__).resolve().parents[1]
CASEDIR = ROOT / "tools" / "synthetic_sync" / "cases"
ARTIFACTS = CASEDIR / "independent-focal-prototype"
ROTATION_ROUNDOFF_DEG = math.degrees(math.sqrt(16 * sys.float_info.epsilon))


class IndependentFocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads((ARTIFACTS / "summary.json").read_text())
        self.sensitivity = json.loads((ARTIFACTS / "sensitivity.json").read_text())
        self.optimizer_rows = [json.loads(line) for line in
            (ARTIFACTS / "optimizer-ledger.jsonl").read_text().splitlines()]
        self.sensitivity_rows = [json.loads(line) for line in
            (ARTIFACTS / "sensitivity-ledger.jsonl").read_text().splitlines()]

    def assertDerivedEqual(self, left, right, path="root"):
        self.assertIs(type(left), type(right), path)
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assertDerivedEqual(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assertDerivedEqual(a, b, f"{path}[{index}]")
        elif isinstance(left, float):
            angular = path.startswith("root.independent_geometry.cameras.") and path.endswith(".rotation_deg")
            projected = path.startswith("root.independent_geometry.cameras.") and path.endswith(
                (".holdout_rmse_px", ".holdout_max_px"))
            absolute = ROTATION_ROUNDOFF_DEG if angular else (1e-9 if projected else 1e-12)
            self.assertTrue(math.isclose(left, right, rel_tol=2e-13, abs_tol=absolute),
                            f"{path}: {left!r} != {right!r}")
        else:
            self.assertEqual(left, right, path)

    def test_optimizer_ledger_has_exact_inputs_results_and_caps(self):
        self.assertEqual(len(self.optimizer_rows), 2 * len(CASES))
        self.assertEqual(self.summary["limits"], LIMITS)
        self.assertEqual(self.summary["total_work"], dict(residual=3749, jacobian=298))
        for index, name in enumerate(CASES):
            start, finish = self.optimizer_rows[2*index:2*index+2]
            artifact = json.loads((ARTIFACTS / f"{name}.json").read_text())
            case = read_case(CASEDIR / f"{name}.json")
            baseline = json.loads((CASEDIR / "unknown-focal-continuation" /
                f"{name}-scale-1.json").read_text())["record"]
            with self.subTest(name=name):
                self.assertEqual((start["kind"], finish["kind"]), ("started", "completed"))
                trial = start["trial"]
                self.assertEqual(trial["name"], name)
                self.assertEqual(trial["request"], case["request"])
                self.assertEqual(trial["baseline"], baseline)
                self.assertEqual(trial["source_runtime"], artifact["source_runtime"])
                self.assertEqual(trial["limits"], LIMITS)
                self.assertRegex(trial["source_runtime"]["source_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(trial["source_runtime"]["threads"]["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(trial["source_runtime"]["threads"]["OMP_NUM_THREADS"], "1")
                self.assertEqual(finish["name"], name)
                self.assertEqual(finish["result"], {key: artifact[key] for key in
                    ("record", "input_selection", "optimizer")})
                self.assertEqual(artifact["request_sha256"], fingerprint(trial["request"]))
                self.assertEqual(artifact["baseline_sha256"], hashlib.sha256(json.dumps(
                    baseline, sort_keys=True, allow_nan=False).encode()).hexdigest())
                self.assertLessEqual(artifact["optimizer"]["meter"]["residual"], LIMITS["per_case_residual"])
                self.assertLessEqual(artifact["optimizer"]["meter"]["jacobian"], LIMITS["per_case_jacobian"])
                self.assertLess(artifact["optimizer"]["elapsed_s"], LIMITS["per_case_seconds"])
                self.assertEqual(set(artifact["record"]["cameras"]),
                                 {c["id"] for c in case["request"]["cameras"]})
                self.assertEqual(set(artifact["record"]["landmarks"]),
                                 {p["id"] for p in case["request"]["points"]})
                self.assertFalse(any(artifact["input_selection"]["focal_at_bound"]))
                self.assertDerivedEqual(artifact["assessment"], assess(case, artifact["record"]))

    def test_sensitivity_ledger_is_read_only_and_assumption_is_explicit(self):
        self.assertEqual(len(self.sensitivity_rows), 2 * len(CASES))
        self.assertEqual(self.sensitivity["counts"], dict(residual=50, jacobian=4))
        self.assertEqual(self.sensitivity["limits"], dict(residual=120, jacobian=4, seconds=20.))
        self.assertEqual([row["name"] for row in self.sensitivity["rows"]], list(CASES))
        for index, name in enumerate(CASES):
            start, finish = self.sensitivity_rows[2*index:2*index+2]
            artifact = json.loads((ARTIFACTS / f"{name}.json").read_text())
            case = read_case(CASEDIR / f"{name}.json")
            with self.subTest(name=name):
                self.assertEqual((start["kind"], finish["kind"]), ("started", "completed"))
                self.assertEqual(start["trial"]["request"], case["request"])
                self.assertEqual(start["trial"]["candidate"], artifact["record"])
                self.assertEqual(start["trial"]["coordinate_sigma_assumed_px"], 0.5)
                self.assertEqual(finish["result"], {key: value for key, value in
                    self.sensitivity["rows"][index].items() if key != "name"})
                self.assertEqual(finish["result"]["request_sha256"], artifact["request_sha256"])
                self.assertEqual(finish["result"]["parameter_count"], 83)
                self.assertEqual(finish["result"]["coordinate_sigma_assumed_px"], 0.5)
        widths = {row["name"]: row["focal_approx_95pct_relative_half_width"]
                  for row in self.sensitivity["rows"]}
        self.assertTrue(all(0.08 < width < 0.21 for width in widths["no-vp-mixed-guessedK"]))
        self.assertTrue(all(width is None for width in widths["no-vp-pure-rotation-guessedK"]))

    def test_frozen_outcomes_do_not_hide_ambiguity_or_cap(self):
        outcomes = {name: json.loads((ARTIFACTS / f"{name}.json").read_text()) for name in CASES}
        self.assertEqual(outcomes["no-vp-shared-guessedK"]["assessment"]["classification"],
                         "accurate_acceptance")
        self.assertTrue(outcomes["no-vp-shared-guessedK"]["input_selection"]["optimizer_converged"])
        mixed = outcomes["no-vp-mixed-guessedK"]
        self.assertFalse(mixed["input_selection"]["optimizer_converged"])
        self.assertEqual(mixed["assessment"]["classification"], "useful_refusal")
        self.assertLess(max(mixed["assessment"]["focal_relative_error"].values()), 0.001)
        self.assertEqual(outcomes["no-vp-weak-baseline-guessedK"]["assessment"]["classification"],
                         "useful_refusal")
        self.assertEqual(outcomes["no-vp-pure-rotation-guessedK"]["assessment"]["classification"],
                         "useful_refusal")
        for name in ("no-vp-weak-baseline-guessedK", "no-vp-pure-rotation-guessedK"):
            self.assertLess(outcomes[name]["record"]["reported_rmse_px"], 0.002)
            self.assertFalse(outcomes[name]["assessment"]["independent_geometry"]["passed"])

    def test_input_extraction_excludes_oracle_and_rejects_unsupported_models(self):
        case = read_case(CASEDIR / "no-vp-mixed-guessedK.json")
        validate_fixture(case)
        baseline = json.loads((CASEDIR / "unknown-focal-continuation" /
            "no-vp-mixed-guessedK-scale-1.json").read_text())["record"]
        request, initial = input_only(case["request"], baseline)
        self.assertEqual(request, case["request"])
        self.assertEqual(initial, baseline)
        self.assertNotIn("truth", request)
        case["truth"]["cameras"][0]["fx"] *= 5
        self.assertEqual(input_only(case["request"], baseline), (request, initial))
        for edit in (
            lambda q: q.update(anchor_id=q["cameras"][1]["id"]),
            lambda q: q["cameras"][0].update(fy=q["cameras"][0]["fy"]*1.01),
            lambda q: q["cameras"][0].update(division_lambda=0.01),
            lambda q: q.update(lock_rotation=True),
            lambda q: q["points"][0].update(known=[0, 0, 0]),
            lambda q: q.update(readonly_match_ids=[q["cameras"][1]["id"]]),
            lambda q: q.update(location_match_ids=[q["cameras"][0]["id"]]),
        ):
            changed = deepcopy(request)
            edit(changed)
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    input_only(changed, baseline)


if __name__ == "__main__":
    unittest.main()
