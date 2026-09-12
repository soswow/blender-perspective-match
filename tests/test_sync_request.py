"""Capture preserves the complete solver contract, including easy-to-omit state."""

from copy import deepcopy
from dataclasses import fields
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from match_perspective.core import sync
from match_perspective.core.sync.request import SyncSolveRequest, request_fingerprint
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.solver import solver_arguments


class SyncRequestTests(unittest.TestCase):
    def request(self):
        request = SyncSolveRequest(**solver_arguments(generate("locked_bridge")["request"]))
        request.lock_rotation, request.lock_translation = True, True
        request.ground_slack, request.known_3d_slack, request.mirror_slack = 0.13, 0.27, 0.09
        calibration = request.matches[0].calibration
        calibration.division_lambda = 0.015
        calibration.lambda_saturated = True
        calibration.brown_conrady = (0.01, -0.02, 0.001, -0.003, 0.005)
        request.observations[0].weight = 4
        request.observations[0].protect_outlier = False  # Confidence != landmark protection.
        request.observations[1].protect_outlier = True
        request.observations[1].landmark_name = "Point <A>"
        request.line_observations = [sync.SyncLineObservation("view_1", "edge", 11, 12, 20, 42, "Edge", 0.25)]
        request.known_lines = {"edge": (np.array([1, 2, 3]), np.array([4, 5, 6]))}
        request.parallel_pairs = [("edge", "WORLD_Z")]
        request.mirror_pairs = [("a", "b")]
        request.mirror_plane = (np.array([0.1, 0.2, 0.3]), np.array([1, 0, 0]))
        request.mirror_landmark_id = "mirror_center"
        request.plane_groups = [("p0", "Z", 2), ("edge", "FREE", 1)]
        request.plane_slack = 0.07
        request.location_match_ids = {"view_0", "view_1"}
        request.readonly_match_ids = {"view_2"}
        return request

    def test_request_fields_cover_every_solver_input(self):
        runtime = {"use_pose_cache", "cancel_check", "progress_callback"}
        self.assertEqual({field.name for field in fields(SyncSolveRequest)},
                         set(inspect.signature(sync.solve_landmark_sync).parameters) - runtime)
        report_runtime = {"top_k", "baseline", "cancel_check", "progress_callback"}
        self.assertEqual(set(self.request().leave_one_out_kwargs()),
                         set(inspect.signature(sync.leave_one_out_landmark_report).parameters) - report_runtime)

    def test_json_roundtrip_preserves_calibration_locks_slack_names_and_confidence(self):
        record = self.request().to_record()
        restored = SyncSolveRequest.from_record(json.loads(json.dumps(record, allow_nan=False)))
        self.assertEqual(record, restored.to_record())
        self.assertIsInstance(restored.matches[0].calibration.rotation_w2c, np.ndarray)
        self.assertIsInstance(restored.mirror_pairs[0], tuple)
        self.assertIsInstance(restored.known_lines["edge"], tuple)
        self.assertFalse(restored.observations[0].protect_outlier)
        self.assertEqual(restored.location_match_ids, {"view_0", "view_1"})
        self.assertEqual(restored.readonly_match_ids, {"view_2"})
        self.assertIsInstance(restored.location_match_ids, set)
        self.assertEqual(restored.to_record()["version"], 4)
        self.assertEqual(restored.mirror_landmark_id, "mirror_center")
        self.assertEqual(restored.plane_groups, [("p0", "Z", 2), ("edge", "FREE", 1)])
        self.assertEqual(restored.plane_slack, 0.07)

    def test_snapshot_and_decoded_arrays_do_not_alias_live_request(self):
        request = self.request()
        record = request.to_record()
        decoded = SyncSolveRequest.from_record(record)
        request.matches[0].calibration.camera_center[:] += 7
        decoded.fixed_similarities["view_1"].translation[:] += 2
        self.assertNotEqual(record, request.to_record())
        self.assertNotEqual(record, decoded.to_record())
        self.assertEqual(record, SyncSolveRequest.from_record(record).to_record())

    def test_integer_lists_and_arrays_have_the_same_geometric_snapshot(self):
        request = self.request()
        request.mirror_plane = [[0, 0, 0], [1, 0, 0]]
        request.known_world = {"metric": [1, 2, 3]}
        record = request.to_record()
        request.mirror_plane = tuple(np.asarray(p, dtype=float) for p in request.mirror_plane)
        request.known_world = {"metric": np.array([1., 2., 3.])}
        self.assertEqual(record, request.to_record())
        self.assertEqual(record, SyncSolveRequest.from_record(record).to_record())

    def test_missing_state_unknown_schema_corruption_and_nonfinite_values_fail(self):
        record = self.request().to_record()
        for change in (lambda r: r.update(version=99),
                       lambda r: r["inputs"].pop("ground_slack"),
                       lambda r: r["inputs"].update(lock_rotation=False)):
            altered = deepcopy(record)
            change(altered)
            with self.assertRaises(ValueError):
                SyncSolveRequest.from_record(altered)
        request = self.request()
        request.matches[0].calibration.camera_center[0] = np.nan
        with self.assertRaises(ValueError):
            request.to_record()
        malformed = deepcopy(record)
        malformed["inputs"]["mirror_plane"] = [1, 2, 3]
        malformed["sha256"] = request_fingerprint(malformed["inputs"])
        with self.assertRaises(ValueError):
            SyncSolveRequest.from_record(malformed)
        bad_reference = deepcopy(record)
        bad_reference["inputs"]["mirror_landmark_id"] = 23
        bad_reference["sha256"] = request_fingerprint(bad_reference["inputs"])
        with self.assertRaisesRegex(ValueError, "mirror_landmark_id"):
            SyncSolveRequest.from_record(bad_reference)
        changed_reference = deepcopy(record)
        changed_reference["inputs"]["mirror_landmark_id"] = "another-point"
        self.assertNotEqual(record["sha256"], request_fingerprint(changed_reference["inputs"]))

    def test_legacy_snapshot_preserves_pre_role_semantics_and_checks_checksum(self):
        legacy = self.request().to_record()
        legacy["version"] = 1
        for key in ("location_match_ids", "readonly_match_ids", "plane_groups", "plane_slack", "mirror_landmark_id"):
            del legacy["inputs"][key]
        legacy["sha256"] = request_fingerprint(legacy["inputs"])
        restored = SyncSolveRequest.from_record(legacy)
        self.assertIsNone(restored.location_match_ids)
        self.assertIsNone(restored.readonly_match_ids)
        self.assertIsNone(restored.plane_groups)
        self.assertIsNone(restored.plane_slack)
        self.assertEqual(restored.to_record()["version"], 4)
        self.assertIsNone(restored.mirror_landmark_id)
        legacy["inputs"]["lock_rotation"] = False
        with self.assertRaisesRegex(ValueError, "checksum"):
            SyncSolveRequest.from_record(legacy)

    def test_version_two_snapshot_preserves_roles_and_injects_plane_defaults(self) -> None:
        legacy = self.request().to_record()
        legacy["version"] = 2
        for key in ("plane_groups", "plane_slack", "mirror_landmark_id"):
            del legacy["inputs"][key]
        legacy["sha256"] = request_fingerprint(legacy["inputs"])
        restored = SyncSolveRequest.from_record(legacy)
        self.assertEqual(restored.location_match_ids, {"view_0", "view_1"})
        self.assertIsNone(restored.plane_groups)
        self.assertIsNone(restored.plane_slack)
        self.assertEqual(restored.to_record()["version"], 4)
        self.assertIsNone(restored.mirror_landmark_id)

    def test_version_three_snapshot_injects_mirror_landmark_default(self) -> None:
        legacy = self.request().to_record()
        legacy["version"] = 3
        del legacy["inputs"]["mirror_landmark_id"]
        legacy["sha256"] = request_fingerprint(legacy["inputs"])
        restored = SyncSolveRequest.from_record(legacy)
        self.assertIsNone(restored.mirror_landmark_id)
        self.assertEqual(restored.to_record()["version"], 4)
        legacy["inputs"]["mirror_landmark_id"] = "point"
        legacy["sha256"] = request_fingerprint(legacy["inputs"])
        with self.assertRaisesRegex(ValueError, "version 4"):
            SyncSolveRequest.from_record(legacy)

    def test_empty_camera_role_sets_are_distinct_from_unspecified(self):
        request = self.request()
        request.location_match_ids, request.readonly_match_ids = set(), set()
        empty = request.to_record()
        self.assertEqual(SyncSolveRequest.from_record(empty).location_match_ids, set())
        request.location_match_ids = None
        self.assertNotEqual(empty["sha256"], request.to_record()["sha256"])

    def test_diagnose_replays_reference_and_skips_removing_its_defining_point(self):
        request = self.request()
        reference = request.observations[0].landmark_id
        other = next(item.landmark_id for item in request.observations if item.landmark_id != reference)
        request.mirror_landmark_id = reference
        baseline = sync.SyncSolveResult(
            similarities={item.match_id: sync.SimilarityTransform() for item in request.matches},
            landmarks={}, mean_reprojection_px=4.0, per_match_rmse_px={},
            per_landmark_rmse_px={reference: 9.0, other: 5.0},
            message="baseline",
        )
        replayed = sync.SyncSolveResult(
            similarities=baseline.similarities, landmarks={}, mean_reprojection_px=2.0,
            per_match_rmse_px={}, per_landmark_rmse_px={}, message="replay",
        )
        with patch("match_perspective.core.sync.solve.solve_landmark_sync", return_value=replayed) as run:
            report = sync.leave_one_out_landmark_report(
                **request.leave_one_out_kwargs(), baseline=baseline,
            )
        self.assertEqual(len(report), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["mirror_landmark_id"], reference)
        self.assertTrue(all(item.landmark_id != other for item in run.call_args.args[1]))
        self.assertTrue(any(item.landmark_id == reference for item in run.call_args.args[1]))

    def test_replayed_solve_matches_original_with_active_fixed_pose(self):
        request = SyncSolveRequest(**solver_arguments(generate("locked_bridge")["request"]))
        request.ground_slack, request.known_3d_slack, request.mirror_slack = 0.03, 0.06, 0.04
        replay = SyncSolveRequest.from_record(request.to_record())
        original = sync.solve_landmark_sync(**request.solver_kwargs())
        restored = sync.solve_landmark_sync(**replay.solver_kwargs())
        self.assertTrue(original.success, original.message)
        self.assertEqual(set(original.similarities), set(restored.similarities))
        for key in original.similarities:
            np.testing.assert_allclose(original.similarities[key].matrix(), restored.similarities[key].matrix(), atol=1e-10)
        np.testing.assert_allclose(restored.similarities["view_1"].matrix(), request.fixed_similarities["view_1"].matrix(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
