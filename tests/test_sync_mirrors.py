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

from match_perspective.core import sync
from sync_fixtures import _project, _synthetic_scene


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
