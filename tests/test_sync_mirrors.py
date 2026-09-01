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
from sync_fixtures import _look_at_rotation, _project, _synthetic_scene


class MirrorPairSyncTests(unittest.TestCase):
    """Is Mirror Of pairs in joint BA (pairwise still uses real correspondences)."""

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
