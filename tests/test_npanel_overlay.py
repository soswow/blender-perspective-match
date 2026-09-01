"""N-panel overlay visibility is a direct region read, not a heartbeat."""

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
    "match_perspective.ui.npanel",
    _ROOT / "ui" / "npanel.py",
)
assert _spec is not None and _spec.loader is not None
npanel = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.npanel"] = npanel
_spec.loader.exec_module(npanel)


class NPanelOverlayTests(unittest.TestCase):
    def test_open_match_tab_shows_overlay(self) -> None:
        self.assertTrue(
            npanel.shows_perspective_match_tab(
                True, 250, npanel.PERSPECTIVE_MATCH_CATEGORY
            )
        )

    def test_n_key_hides_overlay(self) -> None:
        self.assertFalse(
            npanel.shows_perspective_match_tab(
                False, 250, npanel.PERSPECTIVE_MATCH_CATEGORY
            )
        )

    def test_collapsed_ui_region_hides_overlay(self) -> None:
        self.assertFalse(
            npanel.shows_perspective_match_tab(
                True, 1, npanel.PERSPECTIVE_MATCH_CATEGORY
            )
        )

    def test_other_tab_hides_overlay(self) -> None:
        self.assertFalse(
            npanel.shows_perspective_match_tab(True, 250, "Item")
        )
