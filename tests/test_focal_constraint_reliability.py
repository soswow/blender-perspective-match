"""Independent weak-evidence fixture and withheld-shape oracle controls."""

from __future__ import annotations

from copy import deepcopy
import json
from unittest import TestCase

import numpy as np

from match_perspective.core import geometry, lens_refine, sync
from tools.synthetic_sync import focal_constraint_reliability as fixtures
from tools.synthetic_sync import focal_constraints as original
from tools.synthetic_sync.solver import result_record


class FocalConstraintReliabilityTests(TestCase):
    def _fit_at_truth(self, case):
        return dict(cameras={item["id"]: deepcopy(item) for item in case["truth"]["cameras"]},
                    landmarks=deepcopy(case["truth"]["points"]))

    def test_frozen_requests_have_partial_overlap_and_same_pick_controls(self):
        groups = (("weak-axis-hard", "weak-axis-removed", "weak-axis-wrong-member"),
                  ("weak-free-hard", "weak-free-removed"),
                  ("weak-mirror-hard", "weak-mirror-removed", "weak-mirror-tilted",
                   "weak-mirror-anchor-rotated", "weak-mirror-anchor-rotated-removed"))
        for group in groups:
            cases = [json.loads((fixtures.ROOT / f"{name}.json").read_text()) for name in group]
            for case in cases:
                fixtures.validate(case)
                self.assertEqual(case["truth"], cases[0]["truth"])
                self.assertEqual(case["request"]["observations"],
                                 cases[0]["request"]["observations"])
                saved = case["request"]["observations"]
                generated = fixtures.generate(case["name"])["request"]["observations"]
                self.assertEqual(len(saved), len(generated))
                for left, right in zip(saved, generated):
                    self.assertEqual(left.keys(), right.keys())
                    for key in ("match_id", "landmark_id", "weight"):
                        self.assertEqual(left[key], right[key])
                    # Cross-NumPy projection roundoff is not a fixture change.
                    np.testing.assert_allclose([left["u"], left["v"]],
                                               [right["u"], right["v"]],
                                               rtol=1e-14, atol=1e-14)
            counts = {point_id: 0 for point_id in cases[0]["truth"]["points"]}
            for pick in cases[0]["request"]["observations"]:
                counts[pick["landmark_id"]] += 1
            self.assertTrue(all(2 <= count <= 3 for count in counts.values()))

    def test_true_relations_are_independently_exact_and_wrong_references_are_distinct(self):
        axis = fixtures.generate("weak-axis-hard")
        free = fixtures.generate("weak-free-hard")
        mirror = fixtures.generate("weak-mirror-hard")
        axis_x = [axis["truth"]["points"][point_id][0]
                  for point_id, _, _ in axis["request"]["plane_groups"]]
        self.assertLess(np.ptp(axis_x), 1e-12)
        free_xyz = np.asarray([free["truth"]["points"][point_id]
                               for point_id, _, _ in free["request"]["plane_groups"]])
        self.assertLess(np.linalg.svd(free_xyz - free_xyz.mean(axis=0))[1][-1], 1e-12)
        for left, right in mirror["request"]["mirror_pairs"]:
            self.assertLess(np.linalg.norm(original._reflect(
                mirror["truth"]["points"][left], mirror["request"]["mirror_plane"]) -
                mirror["truth"]["points"][right]), 1e-12)
        wrong_axis = fixtures.generate("weak-axis-wrong-member")
        self.assertAlmostEqual(wrong_axis["truth"]["points"]["axis_decoy"][0] - 0.38,
                               0.04, places=12)
        tilted = fixtures.generate("weak-mirror-tilted")
        self.assertEqual(tilted["truth"], mirror["truth"])
        self.assertGreater(np.linalg.norm(np.asarray(tilted["request"]["mirror_plane"][1]) -
                                          mirror["request"]["mirror_plane"][1]), 0.03)

    def test_bad_anchor_changes_only_stored_orientation_and_relation_switch(self):
        correct = fixtures.generate("weak-mirror-hard")
        rotated = fixtures.generate("weak-mirror-anchor-rotated")
        removed = fixtures.generate("weak-mirror-anchor-rotated-removed")
        self.assertEqual(correct["truth"], rotated["truth"])
        self.assertEqual(correct["request"]["observations"], rotated["request"]["observations"])
        old, new = deepcopy(correct["request"]), deepcopy(rotated["request"])
        self.assertNotEqual(old["cameras"][0]["rotation"], new["cameras"][0]["rotation"])
        new["cameras"][0]["rotation"] = old["cameras"][0]["rotation"]
        self.assertEqual(old, new)
        self.assertEqual(rotated["truth"], removed["truth"])
        self.assertEqual(rotated["request"]["observations"], removed["request"]["observations"])

    def test_one_global_similarity_has_zero_shape_error_but_camera_distortion_does_not(self):
        case = fixtures.generate("weak-mirror-removed")
        transformed = self._fit_at_truth(case)
        angle = np.radians(17.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        scale, translation = 1.23, np.array((0.4, -0.2, 0.1))
        for camera in transformed["cameras"].values():
            camera["center"] = (scale * rotation @ camera["center"] + translation).tolist()
            camera["rotation"] = (np.asarray(camera["rotation"]) @ rotation.T).tolist()
        transformed["landmarks"] = {
            key: (scale * rotation @ point + translation).tolist()
            for key, point in transformed["landmarks"].items()}
        assessment = fixtures.assess(case, transformed)
        self.assertLess(assessment["shape_withheld_rmse_px"], 1e-9)
        self.assertLess(assessment["shape_training_point_rmse_world"], 1e-10)
        self.assertGreater(assessment["raw_frame_withheld_rmse_px"], 10.0)
        np.testing.assert_allclose(assessment["shape_training_rotation_w2w"], rotation, atol=1e-12)
        np.testing.assert_allclose(assessment["shape_training_translation_world"],
                                   translation, atol=1e-12)
        self.assertAlmostEqual(assessment["shape_training_scale"], scale, places=12)
        distorted = self._fit_at_truth(case)
        distorted["cameras"]["view_1"]["fx"] *= 1.08
        distorted["cameras"]["view_1"]["fy"] *= 1.08
        distorted["cameras"]["view_1"]["center"][1] += 0.3
        damaged = fixtures.assess(case, distorted)
        self.assertGreater(damaged["shape_withheld_rmse_px"], 5.0)

    def test_weak_axis_plane_recovers_through_public_point_fov(self):
        # The archived pre-fix run registers all views with the hard plane at
        # 2.894 px but fails to converge. Registration from identical picks
        # without the plane must let the final joint fit enforce it.
        case = json.loads((fixtures.ROOT / "weak-axis-hard.json").read_text())
        fixtures.validate(case)
        request = case["request"]
        matches = []
        for camera in request["cameras"]:
            k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                          camera["cy"], camera["width"], camera["height"])
            calibration = geometry.Calibration(k, np.asarray(camera["rotation"], float),
                                               np.asarray(camera["center"], float))
            matches.append(lens_refine.MatchLensInput(
                camera["id"], {}, k, base_calibration=calibration))
        observations = [sync.SyncObservation(**pick) for pick in request["observations"]]
        outcome = lens_refine.refine_lenses_from_landmarks(
            matches, observations, anchor_id=request["anchor_id"],
            share_lens=False, estimate_focal_from_points=True,
            pick_sigma_px=case["pick_sigma_px"], fx_span=0.4,
            plane_groups=request["plane_groups"], plane_slack=request["plane_slack"])
        self.assertTrue(outcome.improved, outcome.refusal_reason)
        self.assertLess(outcome.final_cost, 0.5)
        fitted_cameras = []
        for camera in request["cameras"]:
            updated = dict(camera)
            fitted = outcome.calibrations[camera["id"]]
            updated["fx"] = fitted.intrinsics.fx
            updated["fy"] = fitted.intrinsics.fy
            updated["rotation"] = fitted.rotation_w2c.tolist()
            updated["center"] = fitted.camera_center.tolist()
            fitted_cameras.append(updated)
        record = result_record(outcome.sync_result, fitted_cameras)
        assessment = fixtures.assess(case, record)
        self.assertLess(assessment["plane_rms_world"], 1e-4)
        self.assertLess(assessment["shape_withheld_rmse_px"], 4.0)
