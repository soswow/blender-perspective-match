"""Solver-free checks for the composed sequence oracle and its assessment."""

from __future__ import annotations

from copy import deepcopy
from unittest import TestCase

import numpy as np

from tools.synthetic_sync.sync_continuation import (
    _assess_sequence_case, _bundle_record, _sequence_case,
    _validate_sequence_case,
)
from tools.synthetic_sync.continuation_accuracy_control import (
    _frame_sensitivity, _remove_noise, _start_at_truth,
)
from tools.synthetic_sync.solver import result_record, solver_arguments
from match_perspective.core.sync import SyncSolveResult
from match_perspective.core.sync.request import SyncSolveRequest


def _truth_record(case):
    return dict(
        cameras={item["id"]: item for item in case["truth"]["cameras"]},
        landmarks=deepcopy(case["truth"]["points"]),
        line_segments=deepcopy(case["truth"]["lines"]),
    )


class ContinuationAssessmentTests(TestCase):
    def test_truth_control_preserves_private_pose_and_oracle_pixels(self):
        case = _sequence_case(mirror_kind="fixed")
        request = SyncSolveRequest(**solver_arguments(case["request"]))
        initial = SyncSolveResult({}, {}, 0.0, {}, {}, "oracle control")
        _start_at_truth(request, initial, case)
        record = result_record(initial, case["truth"]["cameras"],
                               calibrations=initial.calibrations)
        self.assertLess(_assess_sequence_case(case, record)["withheld_rmse_px"],
                        1e-8)
        self.assertEqual(set(record["line_segments"]), set(case["truth"]["lines"]))
        _remove_noise(request, case)
        for item in request.observations:
            expected = case["truth"]["oracle_pixels"][item.landmark_id][item.match_id]
            np.testing.assert_allclose((item.u, item.v), expected, rtol=0, atol=0)
        for item in request.line_observations:
            expected = case["truth"]["line_oracle_pixels"][item.landmark_id][item.match_id]
            np.testing.assert_allclose(((item.u1, item.v1), (item.u2, item.v2)),
                                       expected, rtol=0, atol=0)
        frame = _frame_sensitivity(case, record)
        self.assertLess(frame["raw_withheld_rmse_px"], 1e-8)
        self.assertLess(frame["training_rotation_withheld_rmse_px"], 1e-8)

    def test_all_composed_truth_cases_have_independent_zero_error(self):
        for kind in ("fixed", "none", "live"):
            with self.subTest(kind=kind):
                case = _sequence_case(mirror_kind=kind)
                assessment = _assess_sequence_case(case, _truth_record(case))
                self.assertLess(assessment["withheld_rmse_px"], 1e-9)
                self.assertEqual(set(assessment["plane_buckets"]),
                                 {"FREE:1", "X:2"})
                self.assertLess(assessment["plane_rms_world"], 1e-9)
                self.assertTrue(assessment["accuracy_passed"])
                self.assertTrue(assessment["relation_passed"])
                if kind == "fixed":
                    self.assertEqual(case["expectation"]["scale_gauge"],
                                     "anchor-plane-offset")
                else:
                    self.assertEqual(case["expectation"]["scale_gauge"], "free")

    def test_plane_metrics_keep_separate_buckets(self):
        case = _sequence_case(mirror_kind="none")
        record = _truth_record(case)
        record["landmarks"]["axis_0"][0] += 0.02
        assessment = _assess_sequence_case(case, record)
        self.assertLess(assessment["plane_buckets"]["FREE:1"]["point_rms_world"],
                        1e-9)
        self.assertGreater(assessment["plane_buckets"]["X:2"]["point_rms_world"],
                           0.001)
        self.assertFalse(assessment["relation_passed"])

    def test_live_mirror_line_pair_is_assessed_as_infinite_lines(self):
        case = _sequence_case(mirror_kind="live")
        record = _truth_record(case)
        ends = record["line_segments"]["mirror_edge_b"]
        for end in ends:
            end[0] += 0.05
        assessment = _assess_sequence_case(case, record)
        self.assertGreater(assessment["mirror_line_max_gap_world"], 0.01)
        self.assertFalse(assessment["relation_passed"])

    def test_fixture_rejects_altered_projection_oracle(self):
        case = _sequence_case(mirror_kind="fixed")
        case["truth"]["holdout_pixels"]["check_00"]["view_0"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "Withheld point oracle"):
            _validate_sequence_case(case)

    def test_direct_bundle_uses_private_pose_and_root_together(self):
        case = _sequence_case(mirror_kind="fixed")
        angle = 0.2
        root = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                         (np.sin(angle), np.cos(angle), 0.0),
                         (0.0, 0.0, 1.0)))
        calibrations, similarities = {}, {}
        for camera in case["truth"]["cameras"]:
            key = camera["id"]
            rotation = root if key == "view_1" else np.eye(3)
            calibrations[key] = dict(
                intrinsics={field: camera[field] for field in ("fx", "fy", "cx", "cy")},
                camera_center=(rotation.T @ camera["center"]).tolist(),
                rotation_w2c=(np.asarray(camera["rotation"]) @ rotation).tolist())
            similarities[key] = dict(scale=1.0, rotation=rotation.tolist(),
                                     translation=[0.0, 0.0, 0.0])
        outcome = dict(calibrations=calibrations, sync_result=dict(
            success=True, message="synthetic endpoint", mean_reprojection_px=0.0,
            similarities=similarities, landmarks=case["truth"]["points"],
            line_segments=case["truth"]["lines"]))
        record = _bundle_record(case, outcome)
        assessment = _assess_sequence_case(case, record)
        self.assertLess(assessment["withheld_rmse_px"], 1e-9)
        self.assertLess(assessment["center_rmse_world"], 1e-9)
