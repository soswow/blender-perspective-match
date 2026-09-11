"""Known 3D lines and Is-Parallel-To."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Load the solver without executing the add-on's bpy-dependent __init__.
_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package

from match_perspective import core
from match_perspective.core import sync
from match_perspective.core.sync import solve as solve_module
from sync_fixtures import (
    _look_at_rotation,
    _project,
    _rodrigues_z,
    _synthetic_scene,
    _three_view_scene,
)


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

    def test_world_axis_enforcement_locks_single_free_line(self) -> None:
        """An axis target should fix one free line without a second landmark."""
        skewed = (
            np.array((1.0, 2.0, 3.0), dtype=np.float64),
            np.array((2.0, 3.0, 4.0), dtype=np.float64),
        )
        line_segments = {"edge": skewed}
        landmarks = {"edge": 0.5 * (skewed[0] + skewed[1])}

        sync._enforce_parallel_line_segments(
            line_segments,
            landmarks,
            [("edge", "WORLD_AXIS_Z")],
            {"edge": []},
            {},
            {},
            known_lines={},
        )

        direction = line_segments["edge"][1] - line_segments["edge"][0]
        self.assertLess(
            sync._parallel_direction_error(
                direction,
                np.array((0.0, 0.0, 1.0), dtype=np.float64),
            ),
            1.0e-9,
        )

    def test_world_axis_constraint_measures_pose_rotation(self) -> None:
        """An axis-linked image line should penalize a rotated match Empty."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        point_a = np.array((0.0, 1.0, 0.5), dtype=np.float64)
        point_b = np.array((2.0, 1.0, 0.5), dtype=np.float64)
        private_a = true_sim.inverse_point(point_a)
        private_b = true_sim.inverse_point(point_b)
        ua, va = _project(private_a, matches[1].calibration)
        ub, vb = _project(private_b, matches[1].calibration)
        observation = sync.SyncLineObservation(
            match_id="other",
            landmark_id="edge",
            u1=ua,
            v1=va,
            u2=ub,
            v2=vb,
        )
        constraints = sync._axis_line_constraints_for_match(
            "other",
            [("edge", "WORLD_AXIS_X")],
            {"edge": [observation]},
        )
        self.assertEqual(len(constraints), 1)
        good = sync._axis_line_rotation_error(
            constraints[0][0], observation, matches[1].calibration, true_sim
        )
        bad_sim = sync.SimilarityTransform(
            scale=true_sim.scale,
            rotation=_rodrigues_z(1.0) @ true_sim.rotation,
            translation=true_sim.translation.copy(),
        )
        bad = sync._axis_line_rotation_error(
            constraints[0][0], observation, matches[1].calibration, bad_sim
        )
        self.assertIsNotNone(good)
        self.assertIsNotNone(bad)
        self.assertLess(good, 1.0e-6)
        self.assertGreater(bad, 0.2)

        refined = sync._refine_rigid_mixed(
            bad_sim,
            [],
            [],
            [],
            matches[0].calibration,
            matches[1].calibration,
            axis_line_constraints=constraints,
            parallel_weight=12.0,
        )
        refined_error = sync._axis_line_rotation_error(
            constraints[0][0], observation, matches[1].calibration, refined
        )
        self.assertIsNotNone(refined_error)
        self.assertLess(refined_error, bad)

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

    def test_pose_locked_cameras_anchor_free_line_geometry(self) -> None:
        """≥2 pose-locked line picks seed 3D; a bad free still adjusts instead."""
        matches, observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        # One horizontal edge in shared world.
        point_a = np.array((-0.8, 0.4, 0.6), dtype=np.float64)
        point_b = np.array((0.9, 0.4, 0.6), dtype=np.float64)
        ua1, va1 = _project(point_a, anchor_cal)
        ua2, va2 = _project(point_b, anchor_cal)
        private_a = true_sim.inverse_point(point_a)
        private_b = true_sim.inverse_point(point_b)
        uo1, vo1 = _project(private_a, other_cal)
        uo2, vo2 = _project(private_b, other_cal)
        line_observations = [
            sync.SyncLineObservation(
                "anchor",
                "edge",
                ua1,
                va1,
                ua2,
                va2,
                "edge",
            ),
            sync.SyncLineObservation(
                "other",
                "edge",
                uo1,
                vo1,
                uo2,
                vo2,
                "edge",
            ),
        ]
        locked = sync.SimilarityTransform(
            scale=true_sim.scale,
            rotation=true_sim.rotation.copy(),
            translation=true_sim.translation.copy(),
        )
        # Deliberately wrong free pose: would drag a jointly fit line if unlocked.
        bad_other = sync.SimilarityTransform(
            scale=true_sim.scale,
            rotation=true_sim.rotation.copy(),
            translation=true_sim.translation + np.array((0.0, 0.0, 2.5)),
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            line_observations=line_observations,
            fixed_similarities={"other": locked},
            initial_similarities={"other": bad_other},
        )
        self.assertTrue(result.success, result.message)
        segment = result.line_segments["edge"]
        span = float(np.linalg.norm(segment[1] - segment[0]))
        self.assertLess(span, 2.5, "pose-locked line should not stretch across the scene")
        mid = 0.5 * (segment[0] + segment[1])
        self.assertLess(float(np.linalg.norm(mid - 0.5 * (point_a + point_b))), 0.35)
        recovered = result.similarities["other"]
        self.assertTrue(np.array_equal(recovered.rotation, locked.rotation))
        self.assertTrue(np.array_equal(recovered.translation, locked.translation))

    def test_line_strokes_without_influence_location_do_not_move_3d(self) -> None:
        """A biased line stroke only pulls 3D when that camera may move 3D."""
        matches, observations, true_sim, _center, _shared, _true_landmarks = (
            _three_view_scene()
        )
        point_a = np.array((-0.8, 0.4, 0.6), dtype=np.float64)
        point_b = np.array((0.9, 0.4, 0.6), dtype=np.float64)
        true_mid = 0.5 * (point_a + point_b)
        other_cal = matches[1].calibration
        third_cal = matches[2].calibration
        private_a = true_sim.inverse_point(point_a)
        private_b = true_sim.inverse_point(point_b)
        ua1, va1 = _project(point_a, matches[0].calibration)
        ua2, va2 = _project(point_b, matches[0].calibration)
        uo1, vo1 = _project(private_a, other_cal)
        uo2, vo2 = _project(private_b, other_cal)
        ut1, vt1 = _project(point_a, third_cal)
        ut2, vt2 = _project(point_b, third_cal)
        line_observations = [
            sync.SyncLineObservation("anchor", "edge", ua1, va1, ua2, va2, "edge"),
            sync.SyncLineObservation("other", "edge", uo1, vo1, uo2, vo2, "edge"),
            sync.SyncLineObservation(
                "third", "edge", ut1 + 35.0, vt1, ut2 + 35.0, vt2, "edge"
            ),
        ]
        frozen = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            line_observations=line_observations,
            location_match_ids={"anchor", "other"},
        )
        pulled = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            line_observations=line_observations,
            location_match_ids={"anchor", "other", "third"},
        )
        self.assertTrue(frozen.success, frozen.message)
        self.assertTrue(pulled.success, pulled.message)
        frozen_mid = 0.5 * (
            frozen.line_segments["edge"][0] + frozen.line_segments["edge"][1]
        )
        pulled_mid = 0.5 * (
            pulled.line_segments["edge"][0] + pulled.line_segments["edge"][1]
        )
        frozen_err = float(np.linalg.norm(frozen_mid - true_mid))
        pulled_err = float(np.linalg.norm(pulled_mid - true_mid))
        self.assertLess(frozen_err, 0.35)
        self.assertGreater(pulled_err, frozen_err + 0.05)

    def test_line_reconstruction_skips_near_parallel_locked_pair(self) -> None:
        """A locked near-duplicate view must not pin line depth; use a baseline pair."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        point_a = np.array((0.0, -0.8, 0.7), dtype=np.float64)
        point_b = np.array((0.0, 0.8, 0.7), dtype=np.float64)
        midpoint = 0.5 * (point_a + point_b)
        direction = point_b - point_a
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        near_sim = sync.SimilarityTransform(
            translation=np.array((0.04, 0.01, 0.02), dtype=np.float64)
        )
        matches = list(matches) + [sync.SyncMatchInput("near", anchor_cal)]
        match_map = {item.match_id: item for item in matches}

        def _stroke(match_id, calibration, similarity):
            private_a = similarity.inverse_point(point_a)
            private_b = similarity.inverse_point(point_b)
            u1, v1 = _project(private_a, calibration)
            u2, v2 = _project(private_b, calibration)
            return sync.SyncLineObservation(
                match_id, "edge", u1, v1, u2, v2, "edge"
            )

        identity = sync.SimilarityTransform()
        observations = [
            _stroke("near", anchor_cal, near_sim),
            _stroke("anchor", anchor_cal, identity),
            _stroke("other", other_cal, true_sim),
        ]
        similarities = {
            "anchor": identity,
            "other": true_sim,
            "near": near_sim,
        }
        reconstructed = sync._reconstruct_line_from_observations(
            observations,
            similarities,
            match_map,
            prefer_match_ids={"near", "anchor"},
        )
        self.assertIsNotNone(reconstructed)
        point, found_direction = reconstructed
        unit = direction / float(np.linalg.norm(direction))
        found_unit = found_direction / float(np.linalg.norm(found_direction))
        self.assertLess(
            min(
                float(np.linalg.norm(np.cross(unit, found_unit))),
                float(np.linalg.norm(np.cross(unit, -found_unit))),
            ),
            0.08,
        )
        offset = point - midpoint
        distance = float(np.linalg.norm(offset - float(np.dot(offset, unit)) * unit))
        self.assertLess(distance, 0.25)

    def test_recovered_view_pins_free_line_depth(self) -> None:
        """A recovered still must re-intersect a free line instead of keeping locked depth."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        point_a = np.array((0.0, -0.8, 0.7), dtype=np.float64)
        point_b = np.array((0.0, 0.8, 0.7), dtype=np.float64)
        # Near-duplicate locked view of a different depth so the weak pair is offset.
        offset_a = np.array((0.0, -0.8, 0.15), dtype=np.float64)
        offset_b = np.array((0.0, 0.8, 0.15), dtype=np.float64)
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        near_sim = sync.SimilarityTransform(
            translation=np.array((0.04, 0.01, 0.02), dtype=np.float64)
        )
        matches = list(matches) + [sync.SyncMatchInput("near", anchor_cal)]
        match_map = {item.match_id: item for item in matches}

        def _stroke(match_id, calibration, similarity, start, end):
            private_a = similarity.inverse_point(start)
            private_b = similarity.inverse_point(end)
            u1, v1 = _project(private_a, calibration)
            u2, v2 = _project(private_b, calibration)
            return sync.SyncLineObservation(
                match_id, "edge", u1, v1, u2, v2, "edge"
            )

        identity = sync.SimilarityTransform()
        observations = [
            _stroke("near", anchor_cal, near_sim, offset_a, offset_b),
            _stroke("anchor", anchor_cal, identity, point_a, point_b),
            _stroke("other", other_cal, true_sim, point_a, point_b),
        ]
        other_obs = observations[2]

        def _rms(segment, similarity) -> float:
            midpoint = 0.5 * (segment[0] + segment[1])
            direction = segment[1] - segment[0]
            errors = sync._line_observation_reprojection_errors(
                midpoint, direction, other_obs, other_cal, similarity
            )
            return float(np.sqrt(np.mean(np.square(errors))))

        locked_state = SimpleNamespace(
            known_lines={},
            landmarks={},
            line_observations_by_landmark={"edge": observations},
            similarities={"anchor": identity, "near": near_sim},
            match_map=match_map,
            fixed_match_ids={"anchor", "near"},
            parallel_pairs=None,
            line_segments={},
        )
        solve_module._rebuild_free_line_segments(locked_state)
        self.assertIn("edge", locked_state.line_segments)
        locked_rms = _rms(locked_state.line_segments["edge"], true_sim)
        self.assertGreater(locked_rms, 15.0)

        recovered_state = SimpleNamespace(
            known_lines={},
            landmarks={},
            line_observations_by_landmark={"edge": observations},
            similarities={
                "anchor": identity,
                "near": near_sim,
                "other": true_sim,
            },
            match_map=match_map,
            fixed_match_ids={"anchor", "near"},
            parallel_pairs=None,
            line_segments=dict(locked_state.line_segments),
        )
        solve_module._rebuild_free_line_segments(recovered_state)
        recovered_rms = _rms(recovered_state.line_segments["edge"], true_sim)
        self.assertLess(
            recovered_rms,
            4.0,
            f"recovered stroke stayed at {recovered_rms:.1f} px "
            f"(locked-only was {locked_rms:.1f})",
        )
        point_a, point_b = recovered_state.line_segments["edge"]
        fitted = sync._fit_line_fixed_direction(
            point_b - point_a,
            observations,
            recovered_state.similarities,
            match_map,
            prefer_match_ids={"anchor", "near"},
        )
        self.assertIsNotNone(fitted)
        fit_point, fit_direction = fitted
        fit_errors = sync._line_observation_reprojection_errors(
            fit_point, fit_direction, other_obs, other_cal, true_sim
        )
        fit_rms = float(np.sqrt(np.mean(np.square(fit_errors))))
        self.assertLess(
            fit_rms,
            12.0,
            f"locked-prefer fit dropped the recovered stroke ({fit_rms:.1f} px)",
        )
        recovered_state.line_segments["edge_r"] = (
            np.array((point_a[0], -point_a[1], point_a[2])),
            np.array((point_b[0], -point_b[1], point_b[2])),
        )
        recovered_state.landmarks["edge_r"] = 0.5 * (
            recovered_state.line_segments["edge_r"][0]
            + recovered_state.line_segments["edge_r"][1]
        )
        recovered_state.line_observations_by_landmark["edge_r"] = []
        sync.enforce_mirror_line_segments(
            recovered_state.line_segments,
            recovered_state.landmarks,
            [("edge", "edge_r")],
            np.zeros(3),
            np.array((0.0, 1.0, 0.0)),
            recovered_state.line_observations_by_landmark,
            recovered_state.similarities,
            match_map,
            {},
            {"anchor", "near"},
        )
        mirror_rms = _rms(recovered_state.line_segments["edge"], true_sim)
        self.assertLess(
            mirror_rms,
            12.0,
            f"mirror snap restored locked depth ({mirror_rms:.1f} px)",
        )

    def test_line_residual_is_offset_and_heading(self) -> None:
        """Endpoint RMS equals hypot(midpoint offset, half-length * sin(angle))."""
        matches, _observations, _true, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        calibration = matches[0].calibration
        identity = sync.SimilarityTransform()
        point = np.array((0.5, 0.5, 0.8), dtype=np.float64)
        direction = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        projected = sync._project_world_line_to_image(
            point, direction, calibration, identity
        )
        self.assertIsNotNone(projected)
        u_a, v_a = _project(point + np.array((-0.4, 0.0, 0.0)), calibration)
        u_b, v_b = _project(point + np.array((0.4, 0.0, 0.0)), calibration)
        offset = 12.0
        observation = sync.SyncLineObservation(
            "anchor",
            "edge",
            u_a,
            v_a + offset,
            u_b,
            v_b + offset,
            "edge",
        )
        errors = sync._line_observation_reprojection_errors(
            point, direction, observation, calibration, identity
        )
        rms = float(np.sqrt(np.mean(np.square(errors))))
        self.assertAlmostEqual(rms, offset, delta=1.5)
        line_rmse = sync.observation_reprojection_rmse_px(
            kind="LINE",
            point_a=point + np.array((-0.4, 0.0, 0.0)),
            point_b=point + np.array((0.4, 0.0, 0.0)),
            u=u_a,
            v=v_a + offset,
            u2=u_b,
            v2=v_b + offset,
            calibration=calibration,
            similarity=identity,
        )
        self.assertIsNotNone(line_rmse)
        self.assertAlmostEqual(line_rmse, offset, delta=1.5)
        u_hit, v_hit = _project(point, calibration)
        point_rmse = sync.observation_reprojection_rmse_px(
            kind="POINT",
            point_a=point,
            point_b=None,
            u=u_hit,
            v=v_hit,
            calibration=calibration,
            similarity=identity,
        )
        self.assertIsNotNone(point_rmse)
        self.assertAlmostEqual(point_rmse, 0.0, delta=0.05)

    def test_cloud_pnp_not_vetoed_by_disagreeing_line(self) -> None:
        """A skipped still that fits frozen 3D must register even if a line is off."""
        matches, observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=True
        )
        known_world = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
            "p3": np.array((1.5, 1.0, 0.0), dtype=np.float64),
            "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
            "p5": np.array((-0.5, 1.2, 1.5), dtype=np.float64),
            "p6": np.array((0.8, -0.4, 0.9), dtype=np.float64),
        }
        other_only: dict[str, list] = {}
        for observation in observations:
            if observation.match_id != "other":
                continue
            other_only.setdefault(observation.landmark_id, []).append(observation)
        other_cal = matches[1].calibration
        private_a = true_sim.inverse_point(known_world["p0"])
        private_b = true_sim.inverse_point(known_world["p1"])
        u1, v1 = _project(private_a, other_cal)
        u2, v2 = _project(private_b, other_cal)
        line_observations = {
            "edge": [
                sync.SyncLineObservation(
                    "other",
                    "edge",
                    u1,
                    v1 + 80.0,
                    u2,
                    v2 + 80.0,
                    "edge",
                )
            ]
        }
        solved, detail = sync._relative_pose_from_correspondences(
            "anchor",
            "other",
            other_only,
            {item.match_id: item for item in matches},
            known_world,
            known_lines={
                "edge": (known_world["p0"], known_world["p1"]),
            },
            line_observations_by_landmark=line_observations,
        )
        self.assertIsNotNone(solved, detail)
        self.assertTrue(np.allclose(solved.rotation, true_sim.rotation, atol=0.15))

    def test_recovered_polish_follows_side_line(self) -> None:
        """A back-cluster cannot ignore a long side line when polishing a recovered still."""
        matches, _observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        other = matches[1].calibration
        cluster = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((0.12, 0.02, 0.0), dtype=np.float64),
            "p2": np.array((0.04, 0.14, 0.0), dtype=np.float64),
            "p3": np.array((0.09, 0.08, 0.05), dtype=np.float64),
            "p4": np.array((0.02, 0.06, 0.11), dtype=np.float64),
        }
        line_a = np.array((2.4, -0.5, 0.2), dtype=np.float64)
        line_b = np.array((2.4, 1.5, 0.2), dtype=np.float64)

        def _image_of(point: np.ndarray) -> tuple[float, float]:
            return _project(true_sim.inverse_point(point), other)

        usable = [
            sync.SyncObservation("other", landmark_id, *_image_of(point))
            for landmark_id, point in cluster.items()
        ]
        line_obs = sync.SyncLineObservation(
            "other",
            "edge",
            *_image_of(line_a),
            *_image_of(line_b),
            "edge",
        )
        extra = _rodrigues_z(np.deg2rad(8.0))
        bad = sync.SimilarityTransform(
            scale=true_sim.scale,
            rotation=extra @ true_sim.rotation,
            translation=np.array(true_sim.translation, copy=True),
        )
        direction = line_b - line_a
        midpoint = 0.5 * (line_a + line_b)

        def _line_rmse(similarity) -> float:
            errors = sync._line_observation_reprojection_errors(
                midpoint, direction, line_obs, other, similarity
            )
            return float(np.sqrt(np.mean(np.square(errors))))

        before = _line_rmse(bad)
        self.assertGreater(before, 20.0)
        state = SimpleNamespace(
            recovered=["other"],
            similarities={"other": bad, "anchor": sync.SimilarityTransform()},
            fixed_match_ids=set(),
            usable_observations=usable,
            landmarks=cluster,
            match_map={item.match_id: item for item in matches},
            line_segments={"edge": (line_a, line_b)},
            known_lines={},
            line_observations_by_landmark={"edge": [line_obs]},
            landmark_ids=sorted(cluster),
            anchor_id="anchor",
            lock_rotation=False,
            lock_translation=False,
        )
        solve_module._polish_recovered_poses(state)
        after = _line_rmse(state.similarities["other"])
        self.assertLess(after, 12.0, f"line stayed at {after:.1f} px (was {before:.1f})")
        point_errors = [
            solve_module._pick_reprojection_px(
                state.similarities["other"],
                observation,
                other,
                cluster[observation.landmark_id],
            )
            for observation in usable
        ]
        self.assertLess(float(np.sqrt(np.mean(np.square(point_errors)))), 40.0)
