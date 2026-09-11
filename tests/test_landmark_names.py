"""Duplicate-landmark name suggestions."""

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
    "match_perspective.ui.landmark_names",
    _ROOT / "ui" / "landmark_names.py",
)
assert _spec is not None and _spec.loader is not None
landmark_names = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.landmark_names"] = landmark_names
_spec.loader.exec_module(landmark_names)


class DuplicateLandmarkNameTests(unittest.TestCase):
    """Flip sides or increment a trailing number before falling back to copy."""

    def test_flips_left_right_and_top_bottom(self) -> None:
        suggest = landmark_names.suggested_duplicate_landmark_name
        self.assertEqual(suggest("handle-left", ()), "handle-right")
        self.assertEqual(suggest("Handle-Left", ()), "Handle-Right")
        self.assertEqual(suggest("DOOR_RIGHT", ()), "DOOR_LEFT")
        self.assertEqual(suggest("sill top", ()), "sill bottom")
        self.assertEqual(suggest("Sill Bottom", ()), "Sill Top")
        self.assertEqual(suggest("left", ()), "right")

    def test_increments_trailing_space_number(self) -> None:
        suggest = landmark_names.suggested_duplicate_landmark_name
        self.assertEqual(suggest("window 3", ()), "window 4")
        self.assertEqual(suggest("bay 42", ()), "bay 43")
        self.assertEqual(suggest("post 9", ()), "post 10")

    def test_number_wins_over_side_token(self) -> None:
        self.assertEqual(
            landmark_names.suggested_duplicate_landmark_name("jamb left 2", ()),
            "jamb left 3",
        )

    def test_taken_name_falls_back_to_copy(self) -> None:
        suggest = landmark_names.suggested_duplicate_landmark_name
        self.assertEqual(
            suggest("handle-left", ("handle-right",)),
            "handle-left copy",
        )
        self.assertEqual(suggest("window 3", ("window 4",)), "window 3 copy")
        self.assertEqual(suggest("plain", ()), "plain copy")

    def test_ignores_embedded_side_words(self) -> None:
        suggest = landmark_names.suggested_duplicate_landmark_name
        self.assertEqual(suggest("bright", ()), "bright copy")
        self.assertEqual(suggest("stop", ()), "stop copy")
        self.assertEqual(suggest("window3", ()), "window3 copy")
