"""Unit tests for automatic VP line clustering and axis assignment."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Load modules without executing the add-on's bpy-dependent __init__.
_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package


def _load(name: str):
    module_name = f"match_perspective.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        _ROOT / f"{name}.py",
        submodule_search_locations=[str(_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


core = _load("core")
vp_line_detect = _load("vp_line_detect")


def _segments_toward(
    vanishing_xy: tuple[float, float],
    anchors: list[tuple[float, float]],
    length: float = 80.0,
) -> list[core.LineSegment]:
    """Build finite segments aimed at a vanishing point."""
    vx, vy = vanishing_xy
    segments: list[core.LineSegment] = []
    for ax, ay in anchors:
        direction = np.array([vx - ax, vy - ay], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        direction /= max(norm, 1.0e-9)
        bx = ax + direction[0] * length
        by = ay + direction[1] * length
        segments.append(core.LineSegment(ax, ay, float(bx), float(by)))
    return segments


class VpClusteringTests(unittest.TestCase):
    def test_cluster_three_orthogonal_ish_bundles(self) -> None:
        width, height = 800, 600
        # Three finite VPs: left, right, and below (verticals lean toward bottom).
        left = (-400.0, 300.0)
        right = (1200.0, 280.0)
        vertical = (400.0, 2000.0)
        segments = []
        segments += _segments_toward(
            left,
            [(200, 150), (220, 250), (180, 350), (240, 450)],
            length=120.0,
        )
        segments += _segments_toward(
            right,
            [(550, 140), (580, 240), (560, 340), (600, 440)],
            length=120.0,
        )
        segments += _segments_toward(
            vertical,
            [(300, 100), (400, 120), (500, 110), (350, 200), (450, 220)],
            length=160.0,
        )
        # A few distractors that do not share a VP.
        segments.append(core.LineSegment(50, 50, 120, 80))
        segments.append(core.LineSegment(700, 500, 760, 520))

        clusters = vp_line_detect.cluster_vanishing_points(
            segments,
            seed=1,
            image_width=width,
            image_height=height,
        )
        self.assertGreaterEqual(len(clusters), 3)

        bundles = vp_line_detect.assign_axes_three_point(clusters, width, height)
        self.assertGreaterEqual(len(bundles["x"]), 2)
        self.assertGreaterEqual(len(bundles["z"]), 2)
        self.assertGreaterEqual(len(bundles["y"]), 2)

        # Vertical bundle should be the upright one (internal y / UI Z).
        upright = np.mean(
            [vp_line_detect._segment_uprightness(segment) for segment in bundles["y"]]
        )
        other = np.mean(
            [
                vp_line_detect._segment_uprightness(segment)
                for segment in bundles["x"] + bundles["z"]
            ]
        )
        self.assertGreater(upright, other)

    def test_prefer_far_apart_triad(self) -> None:
        """A near-duplicate VP cluster should lose to a well-separated triad."""
        width, height = 800, 600
        left = (-500.0, 300.0)
        right = (1300.0, 300.0)
        vertical = (400.0, 2500.0)
        near_left = (-480.0, 310.0)
        segments = []
        segments += _segments_toward(
            left, [(180, 120), (200, 220), (190, 320), (210, 420)], length=140.0
        )
        segments += _segments_toward(
            right, [(580, 120), (600, 220), (590, 320), (610, 420)], length=140.0
        )
        segments += _segments_toward(
            vertical,
            [(300, 80), (400, 90), (500, 85), (350, 180)],
            length=180.0,
        )
        # Extra support for a VP almost identical to ``left`` — should not win.
        segments += _segments_toward(
            near_left, [(160, 140), (170, 240), (165, 340)], length=100.0
        )
        clusters = vp_line_detect.cluster_vanishing_points(
            segments,
            seed=2,
            image_width=width,
            image_height=height,
        )
        bundles = vp_line_detect.assign_axes_three_point(clusters, width, height)
        left_vp = core.vanishing_point_from_lines(bundles["x"])
        right_vp = core.vanishing_point_from_lines(bundles["z"])
        self.assertIsNotNone(left_vp)
        self.assertIsNotNone(right_vp)
        separation = vp_line_detect._angular_separation_radians(
            left_vp,
            right_vp,
            0.5 * width,
            0.5 * height,
            float(width),
        )
        # Left vs right should be clearly separated (well over 20°).
        self.assertGreater(separation, np.radians(20.0))

    def test_segment_vp_residual_zero_on_exact_line(self) -> None:
        segment = core.LineSegment(0.0, 0.0, 100.0, 0.0)
        vanishing = np.array([50.0, 0.0, 1.0])
        self.assertLess(vp_line_detect.segment_vp_residual(segment, vanishing), 1.0e-6)

    def test_select_diverse_prefers_spread_over_cluster(self) -> None:
        """Nearby long twins lose to a shorter but angularly far segment."""
        vanishing = np.array([400.0, 2000.0, 1.0])
        # Two nearly overlapping uprights on the left (long).
        close_a = core.LineSegment(200.0, 100.0, 210.0, 400.0)
        close_b = core.LineSegment(208.0, 110.0, 218.0, 410.0)
        # One upright on the far right (slightly shorter).
        far = core.LineSegment(600.0, 120.0, 590.0, 380.0)
        picked = vp_line_detect.select_diverse_segments(
            [close_a, close_b, far],
            vanishing,
            limit=2,
            image_width=800,
            image_height=600,
        )
        self.assertEqual(len(picked), 2)
        # Must include the far line; must not keep both close twins.
        self.assertTrue(
            any(
                abs(segment.x1 - far.x1) < 1.0 and abs(segment.y1 - far.y1) < 1.0
                for segment in picked
            )
        )
        close_count = sum(
            1
            for segment in picked
            if abs(segment.x1 - close_a.x1) < 15.0 or abs(segment.x1 - close_b.x1) < 15.0
        )
        self.assertEqual(close_count, 1)

    def test_debug_rgba_is_black_with_white_stroke(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(
            __import__("cv2"), "createLineSegmentDetector"
        ):
            self.skipTest("OpenCV LSD unavailable")
        rgba = vp_line_detect.render_debug_rgba(
            64,
            48,
            [core.LineSegment(0.0, 24.0, 63.0, 24.0)],
        )
        self.assertEqual(rgba.shape, (48, 64, 4))
        self.assertAlmostEqual(float(rgba[0, 0, 0]), 0.0, delta=0.05)
        self.assertGreater(float(rgba[24, 32, 0]), 0.5)


class VpLsdSmokeTests(unittest.TestCase):
    """Optional OpenCV smoke: LSD recovers drawn strokes."""

    def test_detect_drawn_lines(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(cv2, "createLineSegmentDetector"):
            self.skipTest("OpenCV LSD unavailable")

        canvas = np.full((400, 600), 255, dtype=np.uint8)
        for y in (80, 160, 240, 320):
            cv2.line(canvas, (50, y), (250, 200), 0, 2)
        for y in (80, 160, 240, 320):
            cv2.line(canvas, (550, y), (350, 200), 0, 2)
        for x in (200, 280, 360, 440):
            cv2.line(canvas, (x, 50), (300, 350), 0, 2)

        segments = vp_line_detect.detect_line_segments(canvas)
        self.assertGreaterEqual(len(segments), 6)
        # Length floor should reject tiny fragments.
        min_length = vp_line_detect._min_segment_length(600, 400)
        self.assertTrue(
            all(core.segment_length(segment) >= min_length * 0.95 for segment in segments)
        )

    def test_higher_sensitivity_relaxes_thresholds(self) -> None:
        low = vp_line_detect.edge_detect_settings(0.1)
        high = vp_line_detect.edge_detect_settings(0.9)
        self.assertGreater(low.quant, high.quant)
        self.assertGreater(low.density_th, high.density_th)
        self.assertGreater(low.min_length_frac, high.min_length_frac)
        self.assertGreater(high.clahe_blend, low.clahe_blend)
        self.assertGreater(high.max_candidates, low.max_candidates)

    def test_full_pipeline_outcome(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(cv2, "createLineSegmentDetector"):
            self.skipTest("OpenCV LSD unavailable")

        canvas = np.full((400, 600), 255, dtype=np.uint8)
        for y in (80, 160, 240, 320):
            cv2.line(canvas, (50, y), (250, 200), 0, 2)
        for y in (80, 160, 240, 320):
            cv2.line(canvas, (550, y), (350, 200), 0, 2)
        for x in (200, 280, 360, 440):
            cv2.line(canvas, (x, 50), (300, 350), 0, 2)

        outcome = vp_line_detect.detect_vp_line_bundles(canvas, seed=0)
        self.assertGreaterEqual(outcome.result.candidates, 6)
        self.assertGreaterEqual(outcome.result.clusters, 3)
        self.assertEqual(len(outcome.candidates), outcome.result.candidates)
        self.assertGreaterEqual(len(outcome.bundles["y"]), 2)


if __name__ == "__main__":
    unittest.main()
