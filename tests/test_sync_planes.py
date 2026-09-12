"""Shared-plane landmark buckets (axis-aligned and free)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package

from match_perspective.core import sync
from sync_fixtures import _project, _synthetic_scene, _three_view_scene
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.solver import calibration


def _append_points(matches, observations, true_sim, points: dict[str, np.ndarray]) -> None:
    """Project extra shared-world points into every still in ``matches``."""
    private = {
        key: true_sim.inverse_point(point) for key, point in points.items()
    }
    for match in matches:
        for landmark_id, point in points.items():
            source = (
                point
                if match.match_id == "anchor" or match.match_id == "third"
                else private[landmark_id]
            )
            observations.append(
                sync.SyncObservation(
                    match.match_id,
                    landmark_id,
                    *_project(source, match.calibration),
                )
            )


class PlaneGroupSyncTests(unittest.TestCase):
    """Is in Plane buckets in joint BA (pairwise still uses 2D correspondences)."""

    def test_normalize_plane_groups_drops_invalid_and_duplicates(self) -> None:
        groups = sync.normalize_plane_groups(
            [
                ("a", "Z", 2),
                ("a", "X", 1),
                ("b", "free", 10),
                ("c", "Q", 1),
                ("d", "Y", 0),
                ("e", "X", 11),
            ]
        )
        self.assertEqual(groups, [("a", "Z", 2), ("b", "FREE", 10)])

    def test_fit_free_plane_recovers_tilted_quad(self) -> None:
        points = np.array(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.4),
                (0.0, 1.5, 0.15),
                (2.0, 1.5, 0.55),
            ),
            dtype=np.float64,
        )
        fitted = sync.fit_free_plane(points)
        self.assertIsNotNone(fitted)
        centroid, normal = fitted
        distances = [abs(float(np.dot(normal, point - centroid))) for point in points]
        self.assertLess(max(distances), 1.0e-12)

    def test_supported_hard_plane_preserves_compatible_fixed_line_direction(self) -> None:
        """A plane refit must retain an exact axis or Known 3D direction."""
        case = generate("mixed_lines", seed=0, noise_px=0.3)
        point_ids = [point["id"] for point in case["request"]["points"][:4]]
        groups = [(key, "Y", 1) for key in point_ids] + [("edge_1", "Y", 1)]
        line_observations = [
            sync.SyncLineObservation(**item)
            for item in case["request"]["line_observations"]
            if item["landmark_id"] == "edge_1"
        ]
        matches = {
            camera["id"]: sync.SyncMatchInput(camera["id"], calibration(camera))
            for camera in case["truth"]["cameras"]
        }
        identity = sync.SimilarityTransform(np.float64(1.0), np.eye(3), np.zeros(3))
        poses = {key: identity for key in matches}
        reference = tuple(
            np.asarray(end, dtype=float)
            for end in case["truth"]["lines"]["edge_0"]
        )
        for name, links in (
            ("world-axis", [("edge_1", "WORLD_AXIS_Z")]),
            ("known-line", [("edge_1", "edge_0")]),
            ("incompatible", [("edge_1", "WORLD_AXIS_Y")]),
            ("no-weight", [("edge_1", "WORLD_AXIS_Z")]),
        ):
            with self.subTest(name=name):
                segment = tuple(
                    np.asarray(end, dtype=float)
                    for end in case["truth"]["lines"]["edge_1"]
                )
                lines = {"edge_1": segment}
                known_lines = {"edge_0": reference} if name == "known-line" else {}
                landmarks = {
                    key: np.asarray(case["truth"]["points"][key], dtype=float)
                    for key in point_ids
                }
                landmarks["edge_1"] = 0.5 * (segment[0] + segment[1])
                items = line_observations if name != "no-weight" else [
                    sync.SyncLineObservation(
                        item.match_id, item.landmark_id,
                        item.u1, item.v1, item.u2, item.v2, weight=0.0,
                    )
                    for item in line_observations
                ]
                sync.enforce_plane_line_segments(
                    lines, landmarks, groups, {"edge_1": items},
                    poses, matches, known_lines,
                    plane_slack=0.0, parallel_pairs=links,
                )
                end_a, end_b = lines["edge_1"]
                direction = end_b - end_a
                direction /= np.linalg.norm(direction)
                self.assertLess(max(abs(end_a[1] + 0.9), abs(end_b[1] + 0.9)), 1e-8)
                sine_z = float(np.linalg.norm(np.cross(direction, [0.0, 0.0, 1.0])))
                if name != "incompatible":
                    self.assertLess(sine_z, 1e-8)

    def test_axis_bucket_shares_z_and_keeps_a_second_bucket_separate(self) -> None:
        matches, observations, true_sim, _c, _s = _synthetic_scene(with_ground=True)
        low = {
            "z_a": np.array((0.4, 0.2, 1.00), dtype=np.float64),
            "z_b": np.array((1.1, -0.3, 1.06), dtype=np.float64),
            "z_c": np.array((-0.2, 1.4, 1.12), dtype=np.float64),
            "z_d": np.array((1.6, 0.8, 1.18), dtype=np.float64),
        }
        high = {
            "h_a": np.array((0.3, -0.6, 2.00), dtype=np.float64),
            "h_b": np.array((1.4, 0.4, 2.08), dtype=np.float64),
        }
        _append_points(matches, observations, true_sim, {**low, **high})
        unconstrained = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(unconstrained.success, unconstrained.message)
        unconstrained_spread = max(
            float(unconstrained.landmarks[name][2]) for name in low
        ) - min(float(unconstrained.landmarks[name][2]) for name in low)
        self.assertGreater(unconstrained_spread, 0.1)

        groups = [(name, "Z", 1) for name in low] + [
            (name, "Z", 2) for name in high
        ]
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=groups,
            plane_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        low_z = [float(result.landmarks[name][2]) for name in low]
        high_z = [float(result.landmarks[name][2]) for name in high]
        self.assertLess(max(low_z) - min(low_z), 0.02)
        self.assertLess(max(high_z) - min(high_z), 0.02)
        self.assertGreater(float(np.mean(high_z)) - float(np.mean(low_z)), 0.5)
        self.assertIn("2 plane", result.message)

    def test_free_bucket_pulls_a_warped_quad_coplanar(self) -> None:
        matches, observations, true_sim, _c, _s, _t = _three_view_scene(
            with_ground=True
        )
        warped = {
            "f_a": np.array((0.2, 0.1, 0.80), dtype=np.float64),
            "f_b": np.array((1.5, 0.0, 0.82), dtype=np.float64),
            "f_c": np.array((0.1, 1.3, 0.84), dtype=np.float64),
            "f_d": np.array((1.4, 1.2, 1.70), dtype=np.float64),
        }
        _append_points(matches, observations, true_sim, warped)
        unconstrained = sync.solve_landmark_sync(
            matches, observations, anchor_id="anchor"
        )
        self.assertTrue(unconstrained.success, unconstrained.message)
        before = np.stack([unconstrained.landmarks[name] for name in warped])
        before_fit = sync.fit_free_plane(before)
        self.assertIsNotNone(before_fit)
        before_centroid, before_normal = before_fit
        before_err = max(
            abs(float(np.dot(before_normal, point - before_centroid)))
            for point in before
        )
        self.assertGreater(before_err, 0.05)

        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=[(name, "FREE", 1) for name in warped],
            plane_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        after = np.stack([result.landmarks[name] for name in warped])
        after_fit = sync.fit_free_plane(after)
        self.assertIsNotNone(after_fit)
        after_centroid, after_normal = after_fit
        after_err = max(
            abs(float(np.dot(after_normal, point - after_centroid)))
            for point in after
        )
        self.assertLess(after_err, 0.03)
        self.assertLess(after_err, 0.5 * before_err)
        self.assertIn("1 plane", result.message)

    def test_three_free_members_do_not_constrain(self) -> None:
        matches, observations, true_sim, _c, _s = _synthetic_scene(with_ground=True)
        points = {
            "t_a": np.array((0.2, 0.1, 0.80), dtype=np.float64),
            "t_b": np.array((1.5, 0.0, 0.95), dtype=np.float64),
            "t_c": np.array((0.1, 1.3, 1.20), dtype=np.float64),
        }
        _append_points(matches, observations, true_sim, points)
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=[(name, "FREE", 1) for name in points],
        )
        self.assertTrue(result.success, result.message)
        self.assertNotIn("plane", result.message)

    def test_plane_slack_allows_axis_spread(self) -> None:
        matches, observations, true_sim, _c, _s = _synthetic_scene(with_ground=True)
        points = {
            "s_a": np.array((0.4, 0.2, 1.00), dtype=np.float64),
            "s_b": np.array((1.1, -0.3, 1.20), dtype=np.float64),
        }
        _append_points(matches, observations, true_sim, points)
        groups = [(name, "Z", 1) for name in points]
        pinned = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=groups,
            plane_slack=0.0,
        )
        eased = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=groups,
            plane_slack=0.15,
        )
        self.assertTrue(pinned.success, pinned.message)
        self.assertTrue(eased.success, eased.message)
        pinned_spread = abs(
            float(pinned.landmarks["s_a"][2]) - float(pinned.landmarks["s_b"][2])
        )
        eased_spread = abs(
            float(eased.landmarks["s_a"][2]) - float(eased.landmarks["s_b"][2])
        )
        self.assertLess(pinned_spread, 0.02)
        self.assertGreater(eased_spread, pinned_spread + 0.02)

    def test_z_plane_keeps_noisy_mirror_pairs(self) -> None:
        """Is in Plane must not drop Is Mirror Of on the same landmarks."""
        matches, observations, true_sim, _c, _s, _t = _three_view_scene(
            with_ground=True
        )
        pairs = {
            "m_left": np.array((-1.0, 0.4, 1.05), dtype=np.float64),
            "m_right": np.array((1.0, 0.4, 1.05), dtype=np.float64),
            "n_left": np.array((-1.3, 1.2, 1.12), dtype=np.float64),
            "n_right": np.array((1.3, 1.2, 1.12), dtype=np.float64),
        }
        _append_points(matches, observations, true_sim, pairs)
        for observation in observations:
            if observation.landmark_id in {"m_right", "n_right"}:
                observation.u += 16.0
        plane = (
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
        )
        result = sync.solve_landmark_sync(
            matches,
            observations,
            anchor_id="anchor",
            plane_groups=[(name, "Z", 1) for name in pairs],
            plane_slack=0.0,
            mirror_pairs=[("m_left", "m_right"), ("n_left", "n_right")],
            mirror_plane=plane,
            mirror_slack=0.0,
        )
        self.assertTrue(result.success, result.message)
        heights = [float(result.landmarks[name][2]) for name in pairs]
        self.assertLess(max(heights) - min(heights), 0.03)
        for left, right in (("m_left", "m_right"), ("n_left", "n_right")):
            reflected = sync.reflect_point(
                result.landmarks[left], plane[0], plane[1]
            )
            gap = float(np.linalg.norm(result.landmarks[right] - reflected))
            self.assertLess(gap, 0.04, msg=f"{left}/{right} gap {gap:.3f}")


if __name__ == "__main__":
    unittest.main()
