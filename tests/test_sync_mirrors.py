"""Mirror-pair landmarks across a shared-world plane."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

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
from match_perspective.core.sync.ba import _ba_raw_residuals_and_jacobian
from match_perspective.core.sync import solve as solve_module
from sync_fixtures import _look_at_rotation, _project, _synthetic_scene


class MirrorPairSyncTests(unittest.TestCase):
    """Is Mirror Of pairs in joint BA (pairwise still uses real correspondences)."""

    def test_live_reference_jacobian_moves_plane_with_point(self) -> None:
        normal = np.array((1.0, 0.3, -0.2), dtype=float)
        params = np.array((-1.0, 0.4, 0.2, 1.1, 0.1, 0.3, 0.15, 0.2, 0.1, 0.02))
        options = dict(
            free_match_ids=[], free_landmark_ids=["left", "right", "center"],
            fixed_landmarks={}, anchor_id="anchor", matches={}, observations=[],
            line_constraints=[], lock_scale=True, fixed_scales={},
            mirror_pairs=[("left", "right")],
            mirror_plane=(np.array((8.0, 0.0, 0.0)), normal),
            mirror_landmark_id="center", mirror_slack=0.1, free_plane_offset=True,
        )
        residual, jacobian, _protected = _ba_raw_residuals_and_jacobian(params, **options)
        for column in range(6, 10):
            shifted = params.copy()
            shifted[column] += 1e-6
            changed = _ba_raw_residuals_and_jacobian(shifted, **options)[0]
            np.testing.assert_allclose(jacobian[:, column], (changed - residual) / 1e-6,
                                       rtol=1e-6, atol=1e-4)

    def test_live_reference_jacobian_moves_mirror_lines(self) -> None:
        params = np.array((0.15, 0.2, 0.1, -1.0, 0.3, 0.2, 1.0, 0.2, 0.4))
        options = dict(
            free_match_ids=[], free_landmark_ids=["center"],
            fixed_landmarks={}, anchor_id="anchor", matches={}, observations=[],
            line_constraints=[], lock_scale=True, fixed_scales={},
            free_line_ids=["edge_left", "edge_right"],
            fixed_line_directions={"edge_right": np.array((0.0, 1.0, 0.2))},
            mirror_pairs=[("edge_left", "edge_right")],
            mirror_plane=(np.array((9.0, 0.0, 0.0)), np.array((1.0, 0.3, -0.2))),
            mirror_landmark_id="center",
        )
        residual, jacobian, _protected = _ba_raw_residuals_and_jacobian(params, **options)
        for column in range(3):
            shifted = params.copy()
            shifted[column] += 1e-6
            changed = _ba_raw_residuals_and_jacobian(shifted, **options)[0]
            np.testing.assert_allclose(jacobian[:, column], (changed - residual) / 1e-6,
                                       rtol=1e-6, atol=1e-4)

    def test_unpicked_partner_refreshes_from_current_reference_and_fitted_offset(self) -> None:
        anchor_match = _synthetic_scene(with_ground=False)[0][0]
        line_a = np.array((-1.0, -0.2, 0.3))
        line_b = np.array((-1.0, 0.8, 0.3))
        uv_a = _project(line_a, anchor_match.calibration)
        uv_b = _project(line_b, anchor_match.calibration)
        state = object.__new__(solve_module._SolveState)
        state.__dict__.update(
            mirror_plane=(np.array((8.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0))),
            mirror_landmark_id="center", mirror_offset=0.12,
            mirror_pairs=[("left", "right"), ("line_left", "line_right")],
            landmarks={"center": np.array((-0.12, 0.0, 0.0)),
                       "left": np.array((-1.0, 0.2, 0.3)),
                       "right": np.array((0.76, 0.2, 0.3))},
            observations_by_landmark_all={"left": [sync.SyncObservation("anchor", "left", 10, 20)]},
            line_observations_by_landmark={"line_left": [
                sync.SyncLineObservation("anchor", "line_left", *uv_a, *uv_b)
            ]}, similarities={"anchor": sync.SimilarityTransform()},
            match_map={"anchor": anchor_match}, location_match_ids=None,
            line_segments={"line_left": (line_a, line_b)}, known_lines={},
            known_world={}, fixed_match_ids=set(), valid_observations=[], plane_groups=[],
            plane_slack=0.0, plane_seeded_ids=set(), parallel_pairs=None,
            usable_observations=[], observations_by_landmark={}, landmark_ids=[],
        )
        solve_module._attach_mirror_landmarks(state)
        np.testing.assert_allclose(state.landmarks["right"], (1.0, 0.2, 0.3), atol=1e-12)
        self.assertAlmostEqual(float(np.mean([end[0] for end in state.line_segments["line_right"]])),
                               1.0, delta=0.02)
        state.landmarks["center"][0] = -0.20
        solve_module._attach_mirror_landmarks(state)
        np.testing.assert_allclose(state.landmarks["right"], (0.84, 0.2, 0.3), atol=1e-12)
        self.assertAlmostEqual(float(np.mean([end[0] for end in state.line_segments["line_right"]])),
                               0.84, delta=0.02)

    def test_dropping_a_camera_clears_live_offset_before_reconstruction(self) -> None:
        state = object.__new__(solve_module._SolveState)
        state.__dict__.update(
            mirror_landmark_id="center", mirror_offset=0.14,
            fixed_match_ids=set(), skipped_unregistered=[], free_match_ids=["other"],
            similarities={"other": sync.SimilarityTransform()}, peeled_similarities={},
            usable_observations=[], observations_by_landmark={}, landmark_ids=[],
            line_observations_by_landmark={},
        )
        state.drop_matches(["other"])
        self.assertEqual(state.mirror_offset, 0.0)
        self.assertNotIn("other", state.similarities)

    def test_reflect_point_round_trip(self) -> None:
        """Reflecting twice returns the original point."""
        point = np.array((1.2, -0.4, 0.8), dtype=np.float64)
        origin = np.array((0.1, 0.0, 0.0), dtype=np.float64)
        normal = np.array((1.0, 0.2, 0.0), dtype=np.float64)
        once = sync.reflect_point(point, origin, normal)
        twice = sync.reflect_point(once, origin, normal)
        self.assertTrue(np.allclose(twice, point, atol=1.0e-12))
        self.assertGreater(float(np.linalg.norm(once - point)), 0.5)

    def test_suggested_mirror_partner_name_swaps_left_right(self) -> None:
        """A trailing left/right token flips; other names stay unmatched."""
        self.assertEqual(
            sync.suggested_mirror_partner_name("handle-left"),
            "handle-right",
        )
        self.assertEqual(
            sync.suggested_mirror_partner_name("Handle-Left"),
            "Handle-Right",
        )
        self.assertEqual(
            sync.suggested_mirror_partner_name("DOOR_RIGHT"),
            "DOOR_LEFT",
        )
        self.assertEqual(sync.suggested_mirror_partner_name("left"), "right")
        self.assertIsNone(sync.suggested_mirror_partner_name("handle"))
        self.assertIsNone(sync.suggested_mirror_partner_name(""))

    def test_apply_mirror_seed_averages_two_sided_points(self) -> None:
        """Both reconstructed sides snap onto one reflection; Known 3D stays put."""
        origin = np.array((0.2, 0.0, 0.0), dtype=np.float64)
        normal = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        left = np.array((-1.0, 0.4, 0.8), dtype=np.float64)
        right = np.array((1.4, 0.1, 0.9), dtype=np.float64)
        landmarks = {"left": left.copy(), "right": right.copy(), "pin": left.copy()}
        sync.apply_mirror_seed(
            landmarks,
            [("left", "right")],
            origin,
            normal,
        )
        reflected = sync.reflect_point(landmarks["left"], origin, normal)
        self.assertTrue(np.allclose(landmarks["right"], reflected, atol=1.0e-12))
        sync.apply_mirror_seed(
            landmarks,
            [("pin", "right")],
            origin,
            normal,
            known_ids={"pin"},
        )
        self.assertTrue(np.allclose(landmarks["pin"], left))
        self.assertTrue(
            np.allclose(
                landmarks["right"],
                sync.reflect_point(left, origin, normal),
                atol=1.0e-12,
            )
        )

    def _scene_with_side_features(self):
        """Shared landmarks plus two left-only / right-only pairs across X=0."""
        matches, observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        pairs_true = {
            "left": np.array((-1.0, 1.0, 1.2), dtype=np.float64),
            "right": np.array((1.0, 1.0, 1.2), dtype=np.float64),
            "left_b": np.array((-1.0, -0.5, 0.4), dtype=np.float64),
            "right_b": np.array((1.0, -0.5, 0.4), dtype=np.float64),
        }
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        for name, point in pairs_true.items():
            if name.startswith("left"):
                observations.append(
                    sync.SyncObservation(
                        "anchor", name, *_project(point, anchor_cal)
                    )
                )
            else:
                observations.append(
                    sync.SyncObservation(
                        "other",
                        name,
                        *_project(true_sim.inverse_point(point), other_cal),
                    )
                )
        plane = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
        )
        return matches, observations, pairs_true, plane

    def test_mirror_pair_snaps_one_sided_features(self) -> None:
        """A correctly placed plane reconstructs left-only / right-only features."""
        matches, observations, pairs_true, plane = self._scene_with_side_features()
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            mirror_pairs=[("left", "right"), ("left_b", "right_b")],
            mirror_plane=plane,
            mirror_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        for name, point in pairs_true.items():
            self.assertIn(name, result.landmarks)
            self.assertLess(
                float(np.linalg.norm(result.landmarks[name] - point)),
                0.05,
                msg=name,
            )
        reflected = sync.reflect_point(
            result.landmarks["left"], plane[0], plane[1]
        )
        self.assertTrue(
            np.allclose(result.landmarks["right"], reflected, atol=1.0e-3),
            msg="solved pair is not a reflection",
        )
        self.assertIn("2 mirror", result.message)

    def test_mirror_pairs_ignored_without_plane(self) -> None:
        """Pairs without a Mirror Empty do not block a normal solve."""
        matches, observations, _pairs, _plane = self._scene_with_side_features()
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            mirror_pairs=[("left", "right"), ("left_b", "right_b")],
            mirror_plane=None,
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("ignored — no Mirror Empty", result.message)
        self.assertNotIn("left", result.landmarks)
        self.assertNotIn("right", result.landmarks)

    def _scene_with_two_view_mirrors(self):
        """Known 3D cloud plus two mirror pairs seen in both stills.

        A free two-camera cloud can slide onto a biased Empty with ~0 px
        error; pinning the shared points forces slack to move the plane.
        """
        matches, observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        pairs_true = {
            "left": np.array((-1.0, 1.0, 1.2), dtype=np.float64),
            "right": np.array((1.0, 1.0, 1.2), dtype=np.float64),
            "left_b": np.array((-1.0, -0.5, 0.4), dtype=np.float64),
            "right_b": np.array((1.0, -0.5, 0.4), dtype=np.float64),
        }
        known_world = {
            "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
            "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
            "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
            "p3": np.array((1.5, 1.0, 0.0), dtype=np.float64),
            "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
            "p5": np.array((-0.5, 1.2, 1.5), dtype=np.float64),
            "p6": np.array((0.8, -0.4, 0.9), dtype=np.float64),
        }
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        for name, point in pairs_true.items():
            observations.append(
                sync.SyncObservation(
                    "anchor", name, *_project(point, anchor_cal)
                )
            )
            observations.append(
                sync.SyncObservation(
                    "other",
                    name,
                    *_project(true_sim.inverse_point(point), other_cal),
                )
            )
        plane = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
        )
        return matches, observations, pairs_true, plane, known_world

    def test_mirror_slack_eases_biased_plane(self) -> None:
        """Positive slack slides a slightly biased plane along its normal."""
        matches, observations, pairs_true, plane, known_world = (
            self._scene_with_two_view_mirrors()
        )
        biased = (
            np.array((0.08, 0.0, 0.0), dtype=np.float64),
            plane[1],
        )
        mirror_pairs = [("left", "right"), ("left_b", "right_b")]
        frozen = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            mirror_pairs=mirror_pairs,
            mirror_plane=biased,
            mirror_slack=0.0,
        )
        eased = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            known_world=known_world,
            mirror_pairs=mirror_pairs,
            mirror_plane=biased,
            mirror_slack=0.15,
        )
        self.assertTrue(frozen.success, frozen.message)
        self.assertTrue(eased.success, eased.message)

        def _mid_x(result) -> float:
            return float(
                np.mean(
                    [
                        0.5
                        * (
                            result.landmarks[left_id][0]
                            + result.landmarks[right_id][0]
                        )
                        for left_id, right_id in mirror_pairs
                    ]
                )
            )

        frozen_mid = _mid_x(frozen)
        eased_mid = _mid_x(eased)
        self.assertAlmostEqual(frozen_mid, 0.08, delta=0.03)
        self.assertLess(abs(eased_mid), abs(frozen_mid) - 0.02)
        dist_frozen = sum(
            float(np.linalg.norm(frozen.landmarks[name] - point))
            for name, point in pairs_true.items()
        )
        dist_eased = sum(
            float(np.linalg.norm(eased.landmarks[name] - point))
            for name, point in pairs_true.items()
        )
        self.assertLess(dist_eased, dist_frozen - 0.02)

    def _live_reference_case(self):
        """Independent point picks define a plane omitted by the stored origin."""
        matches, observations, _pairs_true, plane, known_world = (
            self._scene_with_two_view_mirrors()
        )
        true_sim = _synthetic_scene(with_ground=False)[2]
        reference = np.array((0.0, 0.35, 1.0), dtype=np.float64)
        anchor_uv = _project(reference, matches[0].calibration)
        other_uv = _project(true_sim.inverse_point(reference), matches[1].calibration)
        observations.extend((
            sync.SyncObservation("anchor", "center", anchor_uv[0] + 2.0, anchor_uv[1] - 1.0),
            sync.SyncObservation("other", "center", other_uv[0] - 1.5, other_uv[1] + 0.5),
        ))
        options = dict(
            matches=matches, observations=observations, anchor_id="anchor",
            known_world=known_world,
            mirror_pairs=[("left", "right"), ("left_b", "right_b")],
            mirror_plane=(np.array((0.22, 0.0, 0.0)), plane[1]),
        )
        return options

    def test_live_reference_follows_noisy_two_view_point_and_ignores_stored_origin(self):
        """A stale fixed plane misses the same pair midpoint that a live point fits."""
        options = self._live_reference_case()
        live = sync.solve_landmark_sync(**options, mirror_landmark_id="center", mirror_slack=0.0)
        shifted = dict(options, mirror_plane=(np.array((-1.7, 2.4, 0.5)), options["mirror_plane"][1]))
        moved_origin = sync.solve_landmark_sync(**shifted, mirror_landmark_id="center", mirror_slack=0.0)
        stale = sync.solve_landmark_sync(**options, mirror_slack=0.0)
        for result in (live, moved_origin, stale):
            self.assertTrue(result.success, result.message)
        self.assertIn("mirror plane follows point", live.message)
        self.assertLess(abs(float(live.landmarks["center"][0])), 0.04)
        for left, right in options["mirror_pairs"]:
            midpoint = 0.5 * (live.landmarks[left] + live.landmarks[right])
            self.assertLess(abs(float(midpoint[0] - live.landmarks["center"][0])), 0.015)
            stale_midpoint = 0.5 * (stale.landmarks[left] + stale.landmarks[right])
            self.assertGreater(abs(float(stale_midpoint[0] - stale.landmarks["center"][0])), 0.10)
        np.testing.assert_allclose(live.landmarks["center"], moved_origin.landmarks["center"], atol=1e-7)
        np.testing.assert_allclose(live.landmarks["left"], moved_origin.landmarks["left"], atol=1e-7)

    def test_live_reference_soft_slack_keeps_plane_offset_relative_to_current_point(self):
        options = self._live_reference_case()
        reference = np.array((-0.11, 0.35, 1.0))
        true_sim = _synthetic_scene(with_ground=False)[2]
        anchor_uv = _project(reference, options["matches"][0].calibration)
        other_uv = _project(true_sim.inverse_point(reference), options["matches"][1].calibration)
        options["observations"] = [
            item for item in options["observations"] if item.landmark_id != "center"
        ] + [
            sync.SyncObservation("anchor", "center", anchor_uv[0] + 2.0, anchor_uv[1] - 1.0),
            sync.SyncObservation("other", "center", other_uv[0] - 1.5, other_uv[1] + 0.5),
        ]
        result = sync.solve_landmark_sync(**options, mirror_landmark_id="center", mirror_slack=0.2)
        self.assertTrue(result.success, result.message)
        point = result.landmarks["center"]
        midpoint_x = np.mean([
            0.5 * (result.landmarks[left][0] + result.landmarks[right][0])
            for left, right in options["mirror_pairs"]
        ])
        self.assertLess(abs(float(point[0] + 0.11)), 0.04)
        self.assertGreater(float(midpoint_x - point[0]), 0.05)
        self.assertLess(float(midpoint_x - point[0]), 0.2)
        self.assertLess(abs(float(midpoint_x)), 0.04)

    def test_live_reference_rejects_missing_line_pair_member_and_fit_only_source(self):
        options = self._live_reference_case()
        for change, message in (
            ({"mirror_landmark_id": "absent"}, "missing"),
            ({"mirror_landmark_id": "left"}, "member"),
            ({"mirror_landmark_id": "center", "line_observations": [
                sync.SyncLineObservation("anchor", "center", 1, 2, 3, 4)
            ]}, "not a line"),
            ({"mirror_landmark_id": "center", "location_match_ids": {"anchor"}}, "two location-enabled"),
            ({"mirror_landmark_id": "center", "readonly_match_ids": {"other"}}, "two location-enabled"),
        ):
            result = sync.solve_landmark_sync(**(options | change))
            self.assertFalse(result.success)
            self.assertIn(message, result.message)

    def test_mirror_line_pair_seeds_one_sided_edges(self) -> None:
        """A left-only / right-only line pair reconstructs across the plane."""
        matches, observations, true_sim, _center, _shared = _synthetic_scene(
            with_ground=False
        )
        plane = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
        )
        left_a = np.array((-1.2, 0.4, 0.2), dtype=np.float64)
        left_b = np.array((-1.2, 1.1, 0.9), dtype=np.float64)
        right_a = sync.reflect_point(left_a, plane[0], plane[1])
        right_b = sync.reflect_point(left_b, plane[0], plane[1])
        anchor_cal = matches[0].calibration
        other_cal = matches[1].calibration
        ua1, va1 = _project(left_a, anchor_cal)
        ua2, va2 = _project(left_b, anchor_cal)
        ub1, vb1 = _project(true_sim.inverse_point(right_a), other_cal)
        ub2, vb2 = _project(true_sim.inverse_point(right_b), other_cal)
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            line_observations=[
                sync.SyncLineObservation(
                    "anchor", "edge_left", ua1, va1, ua2, va2, "edge_left"
                ),
                sync.SyncLineObservation(
                    "other", "edge_right", ub1, vb1, ub2, vb2, "edge_right"
                ),
            ],
            mirror_pairs=[("edge_left", "edge_right")],
            mirror_plane=plane,
            mirror_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("edge_left", result.line_segments)
        self.assertIn("edge_right", result.line_segments)
        left_seg = result.line_segments["edge_left"]
        right_seg = result.line_segments["edge_right"]
        reflected = sync.reflect_point(
            0.5 * (left_seg[0] + left_seg[1]), plane[0], plane[1]
        )
        mid_right = 0.5 * (right_seg[0] + right_seg[1])
        direction = right_seg[1] - right_seg[0]
        direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        distance = float(np.linalg.norm(np.cross(reflected - mid_right, direction)))
        self.assertLess(distance, 0.08)

    def test_resected_camera_uses_one_view_mirror_line(self) -> None:
        """A skipped still's one-view Is Mirror Of line pulls its resected pose.

        Pairwise growth can PnP from four cloud hits, so this still also gets
        three disagreeing off-plane picks that peel it; floor tags resect it.
        """
        intrinsics = core.CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=400.0,
            cy=300.0,
            image_width=800,
            image_height=600,
        )
        ground_points = (
            np.array((-1.5, -1.2, 0.0)),
            np.array((1.6, -1.0, 0.0)),
            np.array((1.8, 1.4, 0.0)),
            np.array((-1.3, 1.6, 0.0)),
        )
        elevated_points = (
            np.array((-0.6, -0.5, 0.8)),
            np.array((0.5, -0.4, 0.8)),
            np.array((0.4, 0.5, 0.8)),
        )
        plane = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
        )
        left_a = np.array((-1.2, 0.3, 0.2), dtype=np.float64)
        left_b = np.array((-1.2, 1.0, 1.0), dtype=np.float64)
        right_a = sync.reflect_point(left_a, plane[0], plane[1])
        right_b = sync.reflect_point(left_b, plane[0], plane[1])
        target = np.array((0.2, 0.2, 0.3), dtype=np.float64)
        anchor_center = np.array((-3.0, -4.0, 2.2), dtype=np.float64)
        side_center = np.array((3.4, -3.1, 2.3), dtype=np.float64)
        skip_center = np.array((0.2, -5.0, 1.7), dtype=np.float64)
        leftover_center = np.array((0.8, -4.2, 2.4), dtype=np.float64)
        anchor = core.Calibration(
            intrinsics,
            _look_at_rotation(anchor_center, target),
            anchor_center,
        )
        side = core.Calibration(
            intrinsics,
            _look_at_rotation(side_center, target),
            side_center,
        )
        skip_true = core.Calibration(
            intrinsics,
            _look_at_rotation(skip_center, target),
            skip_center,
        )
        leftover = core.Calibration(
            intrinsics,
            _look_at_rotation(leftover_center, target),
            leftover_center,
        )
        matches = [
            sync.SyncMatchInput("anchor", anchor),
            sync.SyncMatchInput("side", side),
            sync.SyncMatchInput("skip", leftover),
        ]
        true_by_id = {"anchor": anchor, "side": side, "skip": skip_true}
        observations: list[sync.SyncObservation] = []
        for index, point in enumerate(ground_points):
            for match_id in ("anchor", "side"):
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"g{index}",
                        *_project(point, true_by_id[match_id]),
                        True,
                    )
                )
            u_coord, v_coord = _project(point, skip_true)
            observations.append(
                sync.SyncObservation(
                    "skip",
                    f"g{index}",
                    u_coord + 18.0,
                    v_coord,
                    True,
                )
            )
        for index, point in enumerate(elevated_points):
            for match_id in ("anchor", "side"):
                observations.append(
                    sync.SyncObservation(
                        match_id,
                        f"e{index}",
                        *_project(point, true_by_id[match_id]),
                        False,
                    )
                )
            u_coord, v_coord = _project(point, skip_true)
            observations.append(
                sync.SyncObservation(
                    "skip",
                    f"e{index}",
                    u_coord + 180.0,
                    v_coord,
                    False,
                )
            )
        ua1, va1 = _project(left_a, anchor)
        ua2, va2 = _project(left_b, anchor)
        us1, vs1 = _project(left_a, side)
        us2, vs2 = _project(left_b, side)
        ur1, vr1 = _project(right_a, skip_true)
        ur2, vr2 = _project(right_b, skip_true)
        line_observations = [
            sync.SyncLineObservation(
                "anchor", "edge_left", ua1, va1, ua2, va2, "edge_left"
            ),
            sync.SyncLineObservation(
                "side", "edge_left", us1, vs1, us2, vs2, "edge_left"
            ),
            sync.SyncLineObservation(
                "skip", "edge_right", ur1, vr1, ur2, vr2, "edge_right"
            ),
        ]
        solve_kwargs = {
            "matches": matches,
            "observations": observations,
            "anchor_id": "anchor",
            "line_observations": line_observations,
            "mirror_plane": plane,
            "mirror_slack": 0.0,
        }
        without_line = sync.solve_landmark_sync(**solve_kwargs)
        with_line = sync.solve_landmark_sync(
            **solve_kwargs,
            mirror_pairs=[("edge_left", "edge_right")],
        )
        self.assertTrue(without_line.success, without_line.message)
        self.assertTrue(with_line.success, with_line.message)
        self.assertIn("skip", without_line.similarities)
        self.assertIn("skip", with_line.similarities)
        self.assertIn("recovered 'skip'", with_line.message)
        self.assertIn("edge_right", with_line.line_segments)
        pose_delta = float(
            np.linalg.norm(
                with_line.similarities["skip"].translation
                - without_line.similarities["skip"].translation
            )
        )
        self.assertGreater(pose_delta, 0.01)
