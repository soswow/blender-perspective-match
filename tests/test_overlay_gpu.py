"""Unit-geometry overlay transforms stay in pixel space."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package
if "match_perspective.ui" not in sys.modules:
    _ui = types.ModuleType("match_perspective.ui")
    _ui.__path__ = [str(_ROOT / "ui")]
    sys.modules["match_perspective.ui"] = _ui

_spec = importlib.util.spec_from_file_location(
    "match_perspective.ui.overlay_gpu",
    _ROOT / "ui" / "overlay_gpu.py",
)
assert _spec is not None and _spec.loader is not None
overlay_gpu = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.overlay_gpu"] = overlay_gpu
_spec.loader.exec_module(overlay_gpu)


class OverlayGpuTests(unittest.TestCase):
    def test_segment_maps_unit_x_onto_endpoints(self) -> None:
        trs = overlay_gpu.segment_trs(10.0, 20.0, 40.0, 60.0)
        self.assertIsNotNone(trs)
        start = overlay_gpu.apply_segment_trs(trs, 0.0)
        end = overlay_gpu.apply_segment_trs(trs, 1.0)
        self.assertAlmostEqual(start[0], 10.0)
        self.assertAlmostEqual(start[1], 20.0)
        self.assertAlmostEqual(end[0], 40.0)
        self.assertAlmostEqual(end[1], 60.0)

    def test_segment_rejects_degenerate(self) -> None:
        self.assertIsNone(overlay_gpu.segment_trs(3.0, 4.0, 3.0, 4.0))

    def test_rect_maps_unit_square_corners(self) -> None:
        trs = overlay_gpu.rect_trs(8.0, 16.0, 24.0, 48.0)
        self.assertIsNotNone(trs)
        self.assertEqual(overlay_gpu.apply_rect_trs(trs, 0.0, 0.0), (8.0, 16.0))
        self.assertEqual(overlay_gpu.apply_rect_trs(trs, 1.0, 1.0), (24.0, 48.0))

    def test_circle_maps_unit_axes(self) -> None:
        trs = overlay_gpu.circle_trs(100.0, 50.0, 7.0)
        self.assertIsNotNone(trs)
        east = overlay_gpu.apply_circle_trs(trs, 1.0, 0.0)
        north = overlay_gpu.apply_circle_trs(trs, 0.0, 1.0)
        self.assertAlmostEqual(east[0], 107.0)
        self.assertAlmostEqual(east[1], 50.0)
        self.assertAlmostEqual(north[0], 100.0)
        self.assertAlmostEqual(north[1], 57.0)
        self.assertAlmostEqual(math.hypot(east[0] - 100.0, east[1] - 50.0), 7.0)
