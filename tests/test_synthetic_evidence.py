"""Keep evidence-placement comparisons paired, independent and honest about cost."""

from copy import deepcopy
import unittest

import numpy as np

from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.evidence import add_landmark, candidates, resample_picks, scale_pick_noise, summarize
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.solver import fingerprint, solve


class EvidenceExperimentTests(unittest.TestCase):
    def test_extra_overhead_pick_has_identical_support_and_no_truth_pin(self):
        base = generate("overhead", 0, 0.3)
        before = deepcopy(base)
        support, _ = add_landmark(base, "outer_surface", include_target=False)
        all_views, _ = add_landmark(base, "outer_surface", include_target=True)
        self.assertEqual(base, before)
        self.assertEqual(all_views["request"]["observations"][:-1], support["request"]["observations"])
        self.assertEqual(len(support["request"]["observations"]), len(base["request"]["observations"])+2)
        self.assertEqual(len(all_views["request"]["observations"]), len(base["request"]["observations"])+3)
        self.assertEqual(all_views["request"]["points"][-1], dict(id="extra_outer_surface", ground=False, known=None))
        self.assertEqual(base["truth"]["checks"], all_views["truth"]["checks"])
        self.assertEqual(base["request"]["cameras"], all_views["request"]["cameras"])
        checks = np.array([p["position"] for p in base["truth"]["checks"]])
        for point in candidates(base).values():
            self.assertGreater(float(np.min(np.linalg.norm(checks-point, axis=1))), 1e-6)

    def test_resampling_changes_only_measurement_noise(self):
        base = generate("overhead", 0, 0.3)
        a, b = resample_picks(base, 100), resample_picks(base, 101)
        self.assertEqual(a, resample_picks(base, 100))
        self.assertNotEqual(fingerprint(a["request"]), fingerprint(b["request"]))
        self.assertEqual(base["truth"], a["truth"])
        for key in base["request"]:
            if key != "observations":
                self.assertEqual(base["request"][key], a["request"][key])
        self.assertEqual([(o["match_id"], o["landmark_id"]) for o in base["request"]["observations"]],
                         [(o["match_id"], o["landmark_id"]) for o in a["request"]["observations"]])

    def test_zero_and_half_noise_use_the_same_underlying_measurements(self):
        base = generate("overhead", 0, 0.3)
        exact, half = scale_pick_noise(base, 0), scale_pick_noise(base, 0.5)
        cameras = {c["id"]: c for c in base["truth"]["cameras"]}
        for original, a, b in zip(base["request"]["observations"], exact["request"]["observations"], half["request"]["observations"]):
            ideal = project([base["truth"]["points"][a["landmark_id"]]], cameras[a["match_id"]])[0][0]
            np.testing.assert_allclose([a["u"],a["v"]], ideal)
            np.testing.assert_allclose(np.array([b["u"],b["v"]])-ideal,
                                       0.5*(np.array([original["u"],original["v"]])-ideal))

    def test_summary_retains_lost_cameras_and_excludes_handpicked_draws(self):
        def row(trial, variant, value, *, complete=True, group="resampled"):
            return dict(group=group, trial=trial, variant=variant,
                metrics=dict(holdout_rmse_px=value, complete=complete, passed=complete))
        result = summarize([
            row("a", "baseline", 2), row("a", "outer_surface_three_picks", 1),
            row("b", "baseline", 1), row("b", "outer_surface_three_picks", None, complete=False),
            row("selected_failure", "baseline", 100, group="original"),
        ])
        self.assertEqual(result["baseline"]["median_px"], 1.5)
        extra = result["outer_surface_three_picks"]
        self.assertEqual(extra["trials"], 2)
        self.assertEqual(extra["incomplete"], 1)
        self.assertEqual(extra["vs_baseline"]["pairs"], 1)
        self.assertEqual(extra["vs_baseline"]["median_ratio"], 0.5)

    def test_augmented_exact_geometry_keeps_the_oracle(self):
        base = generate("overhead", 0)
        for candidate in candidates(base):
            with self.subTest(candidate=candidate):
                case, _ = add_landmark(base, candidate, include_target=True)
                assessment = evaluate(case, solve(case["request"]))
                self.assertTrue(assessment["passed"], assessment["violations"])


if __name__ == "__main__":
    unittest.main()
