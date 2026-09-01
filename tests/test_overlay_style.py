"""Landmark vs VP rubber-band overlay colors."""

from __future__ import annotations

import importlib.util
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
    "match_perspective.ui.overlay_style",
    _ROOT / "ui" / "overlay_style.py",
)
assert _spec is not None and _spec.loader is not None
overlay_style = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.overlay_style"] = overlay_style
_spec.loader.exec_module(overlay_style)


class OverlayStyleTests(unittest.TestCase):
    def test_landmark_line_preview_ignores_vp_axis(self) -> None:
        selected = overlay_style.LANDMARK_COLOR_SELECTED
        self.assertEqual(
            overlay_style.preview_line_color("LANDMARK_LINE", "z"),
            selected,
        )
        self.assertEqual(
            overlay_style.preview_line_color("LANDMARK_LINE", "y"),
            selected,
        )
        self.assertNotEqual(selected, overlay_style.AXIS_COLORS["z"])
        self.assertNotEqual(selected, overlay_style.AXIS_COLORS["y"])

    def test_vp_line_preview_follows_active_axis(self) -> None:
        self.assertEqual(
            overlay_style.preview_line_color("LINE", "z"),
            overlay_style.AXIS_COLORS["z"],
        )
        self.assertEqual(
            overlay_style.preview_line_color("LINE", "y"),
            overlay_style.AXIS_COLORS["y"],
        )
