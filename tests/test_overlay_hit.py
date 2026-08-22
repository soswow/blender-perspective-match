"""Screen-space landmark overlay hit testing."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

# Load overlay_hit without executing the add-on's bpy-dependent __init__.
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
    "match_perspective.ui.overlay_hit",
    _ROOT / "ui" / "overlay_hit.py",
)
assert _spec is not None and _spec.loader is not None
overlay_hit = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.overlay_hit"] = overlay_hit
_spec.loader.exec_module(overlay_hit)


class OverlayHitTests(unittest.TestCase):
    """Protect click-to-select radii and nearest-wins overlap."""

    def test_point_hit_and_miss(self) -> None:
        landmarks = [(0, "POINT", (100.0, 80.0), None)]
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (104.0, 83.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            0,
        )
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (130.0, 80.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            -1,
        )

    def test_nearest_point_wins(self) -> None:
        landmarks = [
            (0, "POINT", (100.0, 80.0), None),
            (1, "POINT", (108.0, 80.0), None),
        ]
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (110.0, 80.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            1,
        )

    def test_line_body_and_endpoint(self) -> None:
        landmarks = [(3, "LINE", (10.0, 10.0), (110.0, 10.0))]
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (60.0, 16.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            3,
        )
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (10.0, 18.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            3,
        )
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (60.0, 40.0),
                landmarks,
                point_radius=12.0,
                line_radius=11.0,
            ),
            -1,
        )

    def test_empty_list_misses(self) -> None:
        self.assertEqual(
            overlay_hit.nearest_landmark_hit(
                (0.0, 0.0),
                [],
                point_radius=12.0,
                line_radius=11.0,
            ),
            -1,
        )
