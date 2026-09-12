"""Generator and oracle controls for the bounded biased-reference experiment."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from tools.synthetic_sync.biased_references import (
    BIASED_IDS,
    BIAS_VECTOR,
    KNOWN_IDS,
    SOFT_SLACK,
    UNBIASED_IDS,
    biased_reference_case,
    interpretation,
    required_controls_pass,
)
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.scenarios import read_case


def truth_record(case):
    """A perfect record for testing only the independent assessment."""
    return dict(
        success=True,
        message="oracle",
        reported_rmse_px=0.0,
        cameras={camera["id"]: deepcopy(camera) for camera in case["truth"]["cameras"]},
        landmarks=deepcopy(case["truth"]["points"]),
        line_segments={},
    )


class BiasedReferenceGeneratorTests(unittest.TestCase):
    def test_conditions_share_truth_picks_and_hard_metric_anchor(self):
        cases = {
            name: biased_reference_case(name)
            for name in ("unbiased-hard", "unbiased-soft", "biased-hard", "biased-soft")
        }
        reference = cases["unbiased-hard"]
        for case in cases.values():
            self.assertEqual(case["truth"], reference["truth"])
            self.assertEqual(case["request"]["observations"], reference["request"]["observations"])
            self.assertEqual(case["expectation"]["gauge"], "anchor")
            self.assertEqual(case["request"]["ground_slack"], 0.0)
            self.assertEqual(len([point for point in case["request"]["points"] if point["ground"]]), 6)
            self.assertFalse(case["request"]["fixed_similarities"])
        self.assertEqual(cases["unbiased-soft"]["request"]["known_3d_slack"], SOFT_SLACK)
        self.assertEqual(cases["biased-soft"]["request"]["known_3d_slack"], SOFT_SLACK)

    def test_bias_is_local_and_unbiased_references_are_unchanged(self):
        unbiased = biased_reference_case("unbiased-hard")
        biased = biased_reference_case("biased-hard")
        unbiased_points = {point["id"]: point for point in unbiased["request"]["points"]}
        biased_points = {point["id"]: point for point in biased["request"]["points"]}
        changed = []
        for key in KNOWN_IDS:
            delta = np.asarray(biased_points[key]["known"]) - unbiased_points[key]["known"]
            if np.linalg.norm(delta) > 0:
                changed.append(key)
                np.testing.assert_allclose(delta, BIAS_VECTOR)
        self.assertEqual(tuple(changed), BIASED_IDS)
        for key in UNBIASED_IDS:
            np.testing.assert_allclose(biased_points[key]["known"], biased["truth"]["points"][key])

    def test_fixed_noisy_draw_changes_only_observations(self):
        exact = biased_reference_case("biased-soft", noise_px=0.0)
        noisy = biased_reference_case("biased-soft", noise_px=0.3)
        self.assertEqual(exact["truth"], noisy["truth"])
        self.assertEqual(exact["request"]["points"], noisy["request"]["points"])
        exact_uv = [(item["u"], item["v"]) for item in exact["request"]["observations"]]
        noisy_uv = [(item["u"], item["v"]) for item in noisy["request"]["observations"]]
        self.assertNotEqual(exact_uv, noisy_uv)
        self.assertEqual(noisy_uv, [
            (item["u"], item["v"])
            for item in biased_reference_case("biased-soft", noise_px=0.3)["request"]["observations"]
        ])

    def test_checked_exact_cases_match_the_generator(self):
        case_root = Path(__file__).resolve().parents[1] / "tools/synthetic_sync/cases"
        for condition in ("unbiased-hard", "unbiased-soft", "biased-hard", "biased-soft"):
            with self.subTest(condition=condition):
                saved = read_case(case_root / f"biased-reference-exact-{condition}.json")
                generated = biased_reference_case(condition)
                self.assertEqual(json.dumps(saved), json.dumps(generated))


class BiasedReferenceOracleTests(unittest.TestCase):
    def test_anchor_gauge_uses_no_truth_fitted_alignment(self):
        case = biased_reference_case("biased-hard")
        assessment = evaluate(case, truth_record(case))
        self.assertTrue(assessment["passed"], assessment["violations"])
        self.assertEqual(assessment["alignment"]["scale"], 1.0)
        np.testing.assert_allclose(assessment["alignment"]["rotation"], np.eye(3))
        np.testing.assert_allclose(assessment["alignment"]["translation"], np.zeros(3))

    def test_low_reported_fit_and_accurate_cameras_cannot_hide_biased_points(self):
        case = biased_reference_case("biased-hard")
        record = truth_record(case)
        priors = {point["id"]: point["known"] for point in case["request"]["points"]}
        for key in BIASED_IDS:
            record["landmarks"][key] = priors[key]
        record["reported_rmse_px"] = 0.0
        assessment = evaluate(case, record)
        self.assertFalse(assessment["passed"])
        for key in BIASED_IDS:
            self.assertTrue(any(issue.startswith(f"{key}: reconstructed point error") for issue in assessment["violations"]))

    def test_corrupted_unbiased_control_fails_required_exit_guard(self):
        unbiased = dict(condition="unbiased-hard", success=True, oracle_passed=True)
        softened = dict(condition="unbiased-soft", success=True, oracle_passed=True)
        conflicted = dict(condition="biased-hard", success=True, oracle_passed=False)
        self.assertTrue(required_controls_pass([unbiased, softened, conflicted]))
        self.assertFalse(required_controls_pass([unbiased, conflicted]))
        softened["oracle_passed"] = False
        self.assertFalse(required_controls_pass([unbiased, softened, conflicted]))
        softened["oracle_passed"] = True
        softened["success"] = False
        self.assertFalse(required_controls_pass([unbiased, softened, conflicted]))

    def test_interpretation_tracks_actual_flags(self):
        def row(condition, truth, camera, *, passed=True, issues=None):
            return dict(
                condition=condition, noise_px=0.0, success=True,
                oracle_passed=passed, flags=dict(accuracy_violations=issues or []),
                references=dict(biased_truth_rms=truth, points={
                    key: dict(truth_error=truth) for key in BIASED_IDS
                }),
                withheld={"view_0": {}, "view_1": dict(holdout_rmse_px=camera)},
            )
        rows = [
            row("unbiased-hard", 0, 0), row("unbiased-soft", .01, .2),
            row("biased-hard", .18, 6.9), row("biased-soft", .2, 8,
                passed=False, issues=[f"{BIASED_IDS[0]}: reconstructed point error exceeds limit"]),
        ]
        description = interpretation(rows)
        self.assertIn("improvement -0.020", description)
        self.assertIn("improvement -1.100", description)
        self.assertIn("Soft biased points flag", description)
        self.assertIn("case flags", description)
        self.assertIn("differs by 0.010000", description)


if __name__ == "__main__":
    unittest.main()
