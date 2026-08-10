"""Unit tests for VP-line edge / mid-line snap."""

from __future__ import annotations

import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "match_perspective.line_snap",
    _ROOT / "line_snap.py",
    submodule_search_locations=[str(_ROOT)],
)
assert _spec is not None and _spec.loader is not None
line_snap = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.line_snap"] = line_snap
_spec.loader.exec_module(line_snap)


def _vertical_step_edge(width: int = 120, height: int = 80, edge_x: float = 60.0) -> np.ndarray:
    """Left dark / right bright vertical step edge."""
    gray = np.zeros((height, width), dtype=np.float64)
    gray[:, int(edge_x) :] = 1.0
    return gray


def _vertical_dark_line(
    width: int = 120,
    height: int = 80,
    line_x: int = 55,
    thickness: int = 1,
) -> np.ndarray:
    """Bright field with a thin dark vertical stroke (grout-like)."""
    gray = np.ones((height, width), dtype=np.float64) * 0.85
    half = max(thickness // 2, 0)
    gray[:, line_x - half : line_x + half + 1] = 0.1
    return gray


def _vertical_bright_line(
    width: int = 120,
    height: int = 80,
    line_x: int = 70,
) -> np.ndarray:
    gray = np.ones((height, width), dtype=np.float64) * 0.2
    gray[:, line_x] = 0.95
    return gray


class LineSnapTests(unittest.TestCase):
    def test_snaps_to_step_edge(self) -> None:
        edge_x = 60.0
        gray = _vertical_step_edge(edge_x=edge_x)
        # Seed parallel but offset by a few pixels.
        result = line_snap.snap_segment_to_feature(
            gray,
            (edge_x + 3.5, 10.0),
            (edge_x + 3.0, 70.0),
            search_radius=6,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "edge")
        self.assertAlmostEqual(result.point_a[0], edge_x, delta=1.0)
        self.assertAlmostEqual(result.point_b[0], edge_x, delta=1.0)

    def test_snaps_to_dark_line(self) -> None:
        line_x = 55
        gray = _vertical_dark_line(line_x=line_x)
        result = line_snap.snap_segment_to_feature(
            gray,
            (line_x + 2.5, 8.0),
            (line_x + 3.0, 72.0),
            search_radius=6,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "dark_line")
        self.assertAlmostEqual(result.point_a[0], float(line_x), delta=1.0)
        self.assertAlmostEqual(result.point_b[0], float(line_x), delta=1.0)

    def test_snaps_to_bright_line(self) -> None:
        line_x = 70
        gray = _vertical_bright_line(line_x=line_x)
        result = line_snap.snap_segment_to_feature(
            gray,
            (line_x - 2.0, 12.0),
            (line_x - 2.5, 68.0),
            search_radius=6,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "bright_line")
        self.assertAlmostEqual(result.point_a[0], float(line_x), delta=1.0)
        self.assertAlmostEqual(result.point_b[0], float(line_x), delta=1.0)

    def test_rejects_featureless_region(self) -> None:
        gray = np.full((80, 120), 0.4, dtype=np.float64)
        result = line_snap.snap_segment_to_feature(
            gray,
            (40.0, 10.0),
            (42.0, 70.0),
            search_radius=6,
        )
        self.assertIsNone(result)

    def test_rejects_large_direction_mismatch(self) -> None:
        # Horizontal edge, but seed is nearly vertical — should not snap.
        gray = np.zeros((80, 120), dtype=np.float64)
        gray[40:, :] = 1.0
        result = line_snap.snap_segment_to_feature(
            gray,
            (30.0, 10.0),
            (32.0, 70.0),
            search_radius=6,
            max_angle_degrees=8.0,
        )
        self.assertIsNone(result)

    def test_kind_label(self) -> None:
        self.assertEqual(line_snap.kind_label("dark_line"), "dark line")
        self.assertEqual(line_snap.kind_label("edge"), "edge")


if __name__ == "__main__":
    unittest.main()
