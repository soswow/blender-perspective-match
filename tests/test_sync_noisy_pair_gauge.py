"""Regression for noisy free-point registration without pair-baseline collapse."""

from pathlib import Path
import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, solve


CASES = (Path(__file__).resolve().parents[1] / "tools" / "synthetic_sync" /
         "cases" / "independent-focal-noise" / "true-k-controls")


class NoisyPairGaugeTests(unittest.TestCase):
    def test_pair_refinement_keeps_seed_baseline_under_world_rescaling(self):
        case = read_case(CASES / "no-vp-mixed-guessedK.json")
        cameras = {item["id"]: calibration(item) for item in case["request"]["cameras"]}
        observations = {}
        for item in case["request"]["observations"]:
            observations.setdefault(item["landmark_id"], {})[item["match_id"]] = item
        anchor, other = cameras["view_1"], cameras["view_2"]
        pairs = [(sync.SyncObservation(**by_camera["view_1"]),
                  sync.SyncObservation(**by_camera["view_2"]))
                 for by_camera in observations.values()
                 if "view_1" in by_camera and "view_2" in by_camera]
        self.assertGreaterEqual(len(pairs), 8)

        offset = np.array((5.0, 1.0, 2.0))
        seed = sync.SimilarityTransform(
            scale=1.0, rotation=np.eye(3),
            translation=anchor.camera_center + offset - other.camera_center)
        result = sync.pose._refine_rigid_from_rays(seed, pairs, anchor, other)
        recovered_offset = result.transform_point(other.camera_center) - anchor.camera_center
        self.assertAlmostEqual(np.linalg.norm(recovered_offset), np.linalg.norm(offset), places=8)

        factor = 10.0
        big_anchor = core.Calibration(anchor.intrinsics, anchor.rotation_w2c,
                                      factor * anchor.camera_center)
        big_other = core.Calibration(other.intrinsics, other.rotation_w2c,
                                     factor * other.camera_center)
        big_seed = sync.SimilarityTransform(1.0, seed.rotation, factor * seed.translation)
        big_result = sync.pose._refine_rigid_from_rays(
            big_seed, pairs, big_anchor, big_other)
        big_offset = big_result.transform_point(big_other.camera_center) - big_anchor.camera_center
        self.assertAlmostEqual(np.linalg.norm(big_offset), factor * np.linalg.norm(offset), places=7)
        self.assertTrue(np.allclose(big_result.rotation, result.rotation, atol=2e-4))
        self.assertTrue(np.allclose(big_offset / factor, recovered_offset, atol=2e-4))

        locked_translation = sync.pose._refine_rigid_from_rays(
            seed, pairs, anchor, other, lock_translation=True)
        self.assertTrue(np.array_equal(locked_translation.translation, np.zeros(3)))
        locked_rotation = sync.pose._refine_rigid_from_rays(
            seed, pairs, anchor, other, lock_rotation=True)
        self.assertTrue(np.array_equal(locked_rotation.rotation, seed.rotation))

    def test_calibrated_noisy_free_points_do_not_accept_collapsed_scene(self):
        for name in ("no-vp-mixed-guessedK", "no-vp-shared-guessedK"):
            with self.subTest(name=name):
                case = read_case(CASES / f"{name}.json")
                record = solve(case["request"])
                self.assertTrue(record["success"], record["message"])
                self.assertEqual(len(record["cameras"]), 3)
                self.assertEqual(len(record["landmarks"]), 23)
                self.assertLess(record["reported_rmse_px"], 1.)
                geometry = assess(case, record)["independent_geometry"]
                self.assertTrue(all(camera["all_in_front"] for camera in geometry["cameras"].values()))
                self.assertLess(max(camera["holdout_rmse_px"] for camera in
                                    geometry["cameras"].values()), 3.)
                self.assertLess(max(camera["center_fraction"] for camera in
                                    geometry["cameras"].values()), 0.1)


if __name__ == "__main__":
    unittest.main()
