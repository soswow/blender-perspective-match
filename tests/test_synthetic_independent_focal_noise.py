"""Read-only integrity checks for noisy no-VP focal and Sync evidence."""

import hashlib
import json
import unittest

from tools.synthetic_sync.independent_focal import CASES
from tools.synthetic_sync.independent_focal_noise import (
    CORPUS, LIMITS, SEEDS, START_SCALES, frozen_case,
)
from tools.synthetic_sync.solver import fingerprint


class NoisyFocalEvidenceTests(unittest.TestCase):
    def test_noisy_requests_are_frozen_before_fits(self):
        manifest = json.loads((CORPUS / "manifest.json").read_text())
        self.assertEqual(manifest["names"], list(CASES))
        self.assertEqual(manifest["seeds"], SEEDS)
        self.assertEqual(manifest["sigma_px"], 0.5)
        self.assertEqual(manifest["start_scales"], list(START_SCALES))
        self.assertEqual(manifest["limits"], LIMITS)
        for name in CASES:
            with self.subTest(name=name):
                path = CORPUS / f"{name}.json"
                frozen = json.loads(path.read_text())
                self.assertEqual(frozen, frozen_case(name))
                self.assertEqual(manifest["cases"][name]["file_sha256"],
                                 hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(manifest["cases"][name]["noisy_request_sha256"],
                                 fingerprint(frozen["request"]))

    def test_all_budgeted_trials_retain_exact_noisy_inputs(self):
        sync_rows = [json.loads(line) for line in (CORPUS / "sync-ledger.jsonl").read_text().splitlines()]
        self.assertEqual([row["kind"] for row in sync_rows],
                         ["budget", *[kind for _ in CASES for kind in ("started", "completed")]])
        sparse_rows = [json.loads(line) for line in (CORPUS / "optimizer-ledger.jsonl").read_text().splitlines()]
        dense_rows = [json.loads(line) for line in (CORPUS / "dense" / "optimizer-ledger.jsonl").read_text().splitlines()]
        self.assertEqual(len(sparse_rows), 24)
        self.assertEqual(len(dense_rows), 8)
        self.assertEqual(json.loads((CORPUS / "summary.json").read_text())["total_work"],
                         dict(residual=43941, jacobian=3510))
        self.assertEqual(json.loads((CORPUS / "dense" / "summary.json").read_text())["total_work"],
                         dict(residual=51721, jacobian=607))
        for rows, scales in ((sparse_rows, START_SCALES), (dense_rows, (1.,))):
            for index in range(0, len(rows), 2):
                start, done = rows[index:index+2]
                self.assertEqual((start["kind"], done["kind"]), ("started", "completed"))
                trial = start["trial"]
                name = trial["name"] if "name" in trial else trial["label"].split("-dense-")[0]
                self.assertIn(name, CASES)
                self.assertEqual(trial["request"], json.loads((CORPUS / f"{name}.json").read_text())["request"])
                self.assertEqual(trial["baseline"], json.loads((CORPUS / f"{name}-initial.json").read_text())["record"])
                self.assertEqual(trial["source_runtime"]["threads"]["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(trial["source_runtime"]["threads"]["OMP_NUM_THREADS"], "1")
                self.assertIn(trial.get("scale", trial.get("start_scale")), scales)
                self.assertRegex(trial["source_runtime"]["source_sha256"], r"^[0-9a-f]{64}$")

    def test_dense_convergence_does_not_certify_weak_geometry(self):
        reports = {name: json.loads((CORPUS / "dense" /
            f"{name}-dense-exact-start-1.json").read_text()) for name in CASES}
        self.assertTrue(all(report["record"]["success"] for report in reports.values()))
        self.assertTrue(all(report["record"]["reported_rmse_px"] < 0.5 for report in reports.values()))
        weak = reports["no-vp-weak-baseline-guessedK"]
        rotation = reports["no-vp-pure-rotation-guessedK"]
        self.assertTrue(any(weak["input_selection"]["focal_at_bound"]))
        self.assertFalse(weak["assessment"]["independent_geometry"]["passed"])
        self.assertFalse(rotation["assessment"]["independent_geometry"]["passed"])
        sensitivity = json.loads((CORPUS / "dense" / "diagnostics.json").read_text())
        rows = {row["name"]: row for row in sensitivity["rows"]}
        self.assertEqual(set(rows), set(CASES))
        self.assertGreater(min(rows["no-vp-weak-baseline-guessedK"][
            "focal_approx_95pct_upper_relative_excursion"]), 1e10)
        self.assertGreater(min(rows["no-vp-pure-rotation-guessedK"][
            "focal_approx_95pct_upper_relative_excursion"]), 1.)
        observability = json.loads((CORPUS / "observability.json").read_text())
        motion = {row["name"]: row for row in observability["rows"]}
        self.assertEqual(set(motion), set(CASES))
        for name in CASES:
            self.assertTrue(motion[name]["all_fitted_points_in_front"])
            self.assertEqual(motion[name]["request_sha256"],
                fingerprint(json.loads((CORPUS / f"{name}.json").read_text())["request"]))
        self.assertGreater(motion["no-vp-pure-rotation-guessedK"]["median_max_pair_parallax_deg"], 4.)
        self.assertTrue(all(pair["homography_symmetric_transfer_rmse_px"] < 2.
            for pair in motion["no-vp-pure-rotation-guessedK"]["pairs"].values()))


if __name__ == "__main__":
    unittest.main()
