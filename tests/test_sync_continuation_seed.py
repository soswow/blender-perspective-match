"""A prior solution survives exact evidence but cannot create unsupported new 3D."""

import unittest

import numpy as np

from match_perspective.core import Calibration, CameraIntrinsics, sync
from match_perspective.core.sync import solve as solve_module
from match_perspective.core.sync.request import SyncSolveRequest


class SyncContinuationSeedTests(unittest.TestCase):
    def test_seeded_entry_reaches_cancellation_before_registration(self):
        calibration = Calibration(
            CameraIntrinsics(800.0, 800.0, 480.0, 360.0, 960, 720),
            np.eye(3), np.array((0.0, 0.0, -5.0)),
        )
        request = SyncSolveRequest(
            matches=[sync.SyncMatchInput("a", calibration)],
            observations=[], anchor_id="a",
        )
        request.initial_solution = sync.SyncSolutionSeed(
            calibrations={"a": calibration},
            similarities={"a": sync.SimilarityTransform()},
            landmarks={}, line_segments={},
            evidence_sha256=request.evidence_sha256(),
        )
        with self.assertRaises(sync.SyncCancelled):
            sync.solve_landmark_sync(
                **request.solver_kwargs(), cancel_check=lambda: True,
            )

    def state(self, *, unchanged: bool):
        calibration = Calibration(
            CameraIntrinsics(800.0, 800.0, 480.0, 360.0, 960, 720),
            np.eye(3), np.array((0.0, 0.0, -5.0)),
        )
        point_pick = sync.SyncObservation("a", "single_point", 480.0, 360.0)
        line_pick = sync.SyncLineObservation("a", "single_line", 400.0, 300.0, 500.0, 300.0)
        return solve_module._SolveState(
            match_map={"a": sync.SyncMatchInput("a", calibration)},
            anchor_id="a", known_world={}, known_lines={}, parallel_pairs=[],
            mirror_pairs=[], mirror_plane=None, mirror_slack=0.0,
            mirror_landmark_id=None, plane_groups=[], plane_slack=0.0,
            lock_rotation=False, lock_translation=False, use_pose_cache=False,
            cancel_check=None, ground_slack=0.0, known_3d_slack=0.0,
            identity_result={"a": sync.SimilarityTransform()},
            valid_observations=[point_pick],
            observations_by_landmark_all={"single_point": [point_pick]},
            line_observations_by_landmark_all={"single_line": [line_pick]},
            observations_by_landmark={"single_point": [point_pick]},
            line_observations_by_landmark={"single_line": [line_pick]},
            usable_observations=[point_pick], landmark_ids=["single_point"],
            free_match_ids=[], fixed_match_ids=set(),
            similarities={"a": sync.SimilarityTransform()},
            skipped_unregistered=[], failure_detail="", connected={"a"},
            initial_line_segments={"single_line": (
                np.array((-1.0, 0.0, 1.0)), np.array((1.0, 0.0, 1.0)),
            )},
            seed_unchanged=unchanged,
        )

    def test_exact_seed_keeps_one_view_point_and_line_for_constraints(self):
        state = self.state(unchanged=True)
        carried = np.array((0.0, 0.0, 1.0))
        state.rebuild_landmarks(retained_landmarks={"single_point": carried})
        np.testing.assert_allclose(state.landmarks["single_point"], carried)
        self.assertIn("single_line", state.line_segments)
        self.assertEqual(state.initial_line_segments, {})

    def test_changed_evidence_does_not_keep_unsupported_one_view_3d(self):
        state = self.state(unchanged=False)
        state.rebuild_landmarks(retained_landmarks={"single_point": np.array((0.0, 0.0, 1.0))})
        self.assertNotIn("single_point", state.landmarks)
        self.assertNotIn("single_line", state.line_segments)


if __name__ == "__main__":
    unittest.main()
