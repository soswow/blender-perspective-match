"""Known 3D lines and Is-Parallel-To."""

from __future__ import annotations

import unittest

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from sync_fixtures import _look_at_rotation, _project, _rodrigues_z, _synthetic_scene


class LineSyncTests(unittest.TestCase):
    """Known 3D lines and Is-Parallel-To."""

    def test_known_3d_lines_register_pose(self) -> None:
        """≥3 Known 3D edges with 2D segments in the other still register pose."""
        matches, _observations, true_sim, center_private, shared_center = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        # Three non-parallel edges in shared world.
        known_lines = {
            "edge0": (
                np.array((0.0, 0.0, 0.0), dtype=np.float64),
                np.array((2.0, 0.0, 0.0), dtype=np.float64),
            ),
            "edge1": (
                np.array((0.0, 0.0, 0.0), dtype=np.float64),
                np.array((0.0, 2.0, 0.0), dtype=np.float64),
            ),
            "edge2": (
                np.array((0.0, 0.0, 1.0), dtype=np.float64),
                np.array((1.0, 1.0, 2.0), dtype=np.float64),
            ),
        }
        line_observations: list[sync.SyncLineObservation] = []
        for landmark_id, (point_a, point_b) in known_lines.items():
            private_a = true_sim.inverse_point(point_a)
            private_b = true_sim.inverse_point(point_b)
            ua, va = _project(private_a, other)
            ub, vb = _project(private_b, other)
            line_observations.append(
                sync.SyncLineObservation(
                    match_id="other",
                    landmark_id=landmark_id,
                    u1=ua,
                    v1=va,
                    u2=ub,
                    v2=vb,
                    landmark_name=landmark_id,
                )
            )
        result = sync.solve_landmark_sync(
            matches,
            [],
            anchor_id="anchor",
            line_observations=line_observations,
            known_lines=known_lines,
        )
        self.assertTrue(result.success, result.message)
        recovered = result.similarities["other"]
        self.assertTrue(np.allclose(recovered.rotation, true_sim.rotation, atol=0.12))
        recovered_center = recovered.transform_point(center_private)
        self.assertTrue(np.allclose(recovered_center, shared_center, atol=0.5))


    def test_parallel_line_pair_prefers_consistent_rotation(self) -> None:
        """Parallel-tagged lines should penalize a wrong relative yaw."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        # Two parallel edges along shared X.
        edge_a = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((2.0, 0.0, 0.0), dtype=np.float64),
        )
        edge_b = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((2.0, 1.0, 0.5), dtype=np.float64),
        )
        line_observations: list[sync.SyncLineObservation] = []
        for landmark_id, (point_a, point_b) in (
            ("edge_a", edge_a),
            ("edge_b", edge_b),
        ):
            for match_id, calibration, sim in (
                ("anchor", matches[0].calibration, sync.SimilarityTransform()),
                ("other", other, true_sim),
            ):
                if match_id == "anchor":
                    private_a, private_b = point_a, point_b
                else:
                    private_a = sim.inverse_point(point_a)
                    private_b = sim.inverse_point(point_b)
                ua, va = _project(private_a, calibration)
                ub, vb = _project(private_b, calibration)
                line_observations.append(
                    sync.SyncLineObservation(
                        match_id=match_id,
                        landmark_id=landmark_id,
                        u1=ua,
                        v1=va,
                        u2=ub,
                        v2=vb,
                        landmark_name=landmark_id,
                    )
                )
        good = sync._parallel_pair_rotation_error(
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_b"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_b"
            ),
            matches[0].calibration,
            other,
            true_sim,
        )
        bad_sim = sync.SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues_z(1.2),
            translation=true_sim.translation.copy(),
        )
        bad = sync._parallel_pair_rotation_error(
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "anchor" and item.landmark_id == "edge_b"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_a"
            ),
            next(
                item
                for item in line_observations
                if item.match_id == "other" and item.landmark_id == "edge_b"
            ),
            matches[0].calibration,
            other,
            bad_sim,
        )
        self.assertIsNotNone(good)
        self.assertIsNotNone(bad)
        self.assertLess(good, 0.05)
        self.assertGreater(bad, good + 0.2)


    def test_parallel_enforcement_locks_free_line_directions(self) -> None:
        """Is-Parallel-To should force free-line meshes to share a direction."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        # Two truly parallel edges along shared X; second will be seeded skewed.
        edge_good = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((2.0, 0.0, 0.0), dtype=np.float64),
        )
        edge_true = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((2.0, 1.0, 0.5), dtype=np.float64),
        )
        # Skewed seed (~30°) for the second free line mesh.
        edge_skewed = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((1.7, 1.0, 1.5), dtype=np.float64),
        )
        line_observations: dict[str, list[sync.SyncLineObservation]] = {
            "edge_good": [],
            "edge_bad": [],
        }
        for landmark_id, (point_a, point_b) in (
            ("edge_good", edge_good),
            ("edge_bad", edge_true),
        ):
            for match_id, calibration, sim in (
                ("anchor", matches[0].calibration, sync.SimilarityTransform()),
                ("other", matches[1].calibration, true_sim),
            ):
                if match_id == "anchor":
                    private_a, private_b = point_a, point_b
                else:
                    private_a = sim.inverse_point(point_a)
                    private_b = sim.inverse_point(point_b)
                ua, va = _project(private_a, calibration)
                ub, vb = _project(private_b, calibration)
                line_observations[landmark_id].append(
                    sync.SyncLineObservation(
                        match_id=match_id,
                        landmark_id=landmark_id,
                        u1=ua,
                        v1=va,
                        u2=ub,
                        v2=vb,
                        landmark_name=landmark_id,
                    )
                )
        match_map = {item.match_id: item for item in matches}
        similarities = {
            "anchor": sync.SimilarityTransform(),
            "other": true_sim,
        }
        line_segments = {
            "edge_good": edge_good,
            "edge_bad": edge_skewed,
        }
        landmarks = {
            "edge_good": 0.5 * (edge_good[0] + edge_good[1]),
            "edge_bad": 0.5 * (edge_skewed[0] + edge_skewed[1]),
        }
        before = sync._parallel_direction_error(
            edge_skewed[1] - edge_skewed[0],
            edge_good[1] - edge_good[0],
        )
        self.assertGreater(before, 0.3)
        sync._enforce_parallel_line_segments(
            line_segments,
            landmarks,
            [("edge_good", "edge_bad")],
            line_observations,
            similarities,
            match_map,
            known_lines={},
        )
        after = sync._parallel_direction_error(
            line_segments["edge_bad"][1] - line_segments["edge_bad"][0],
            line_segments["edge_good"][1] - line_segments["edge_good"][0],
        )
        self.assertLess(after, 0.05)

    def test_pose_refine_with_parallel_pairs_runs(self) -> None:
        """Is-Parallel-To pose refine must import the rotation-error helper."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        edge_a = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((2.0, 0.0, 0.0), dtype=np.float64),
        )
        edge_b = (
            np.array((0.0, 1.0, 0.5), dtype=np.float64),
            np.array((2.0, 1.0, 0.5), dtype=np.float64),
        )
        line_observations: list[sync.SyncLineObservation] = []
        for landmark_id, (point_a, point_b) in (
            ("edge_a", edge_a),
            ("edge_b", edge_b),
        ):
            for match_id, calibration, sim in (
                ("anchor", matches[0].calibration, sync.SimilarityTransform()),
                ("other", other, true_sim),
            ):
                if match_id == "anchor":
                    private_a, private_b = point_a, point_b
                else:
                    private_a = sim.inverse_point(point_a)
                    private_b = sim.inverse_point(point_b)
                ua, va = _project(private_a, calibration)
                ub, vb = _project(private_b, calibration)
                line_observations.append(
                    sync.SyncLineObservation(
                        match_id=match_id,
                        landmark_id=landmark_id,
                        u1=ua,
                        v1=va,
                        u2=ub,
                        v2=vb,
                        landmark_name=landmark_id,
                    )
                )

        def _line(match_id: str, landmark_id: str) -> sync.SyncLineObservation:
            return next(
                item
                for item in line_observations
                if item.match_id == match_id and item.landmark_id == landmark_id
            )

        refined = sync._refine_rigid_mixed(
            true_sim,
            [],
            [],
            [],
            matches[0].calibration,
            other,
            parallel_vp_constraints=[
                (
                    _line("anchor", "edge_a"),
                    _line("anchor", "edge_b"),
                    _line("other", "edge_a"),
                    _line("other", "edge_b"),
                )
            ],
            parallel_weight=12.0,
        )
        self.assertEqual(tuple(refined.rotation.shape), (3, 3))

