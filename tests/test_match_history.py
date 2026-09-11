"""Browser-style Perspective Match visit history."""

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
    "match_perspective.ui.match_history",
    _ROOT / "ui" / "match_history.py",
)
assert _spec is not None and _spec.loader is not None
match_history = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.match_history"] = match_history
_spec.loader.exec_module(match_history)


class MatchHistoryTests(unittest.TestCase):
    """Back/forward visits, cap, and rename/delete bookkeeping."""

    def test_seeds_previous_when_history_is_empty(self) -> None:
        names, index = match_history.record_visit(
            [], -1, "B", previous="A"
        )
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(index, 1)

    def test_skips_duplicate_current_visit(self) -> None:
        names, index = match_history.record_visit(["A", "B"], 1, "B")
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(index, 1)

    def test_new_visit_drops_forward_stack(self) -> None:
        names, index = match_history.record_visit(["A", "B", "C"], 1, "D")
        self.assertEqual(names, ["A", "B", "D"])
        self.assertEqual(index, 2)

    def test_caps_at_ten_oldest_first(self) -> None:
        names: list[str] = []
        index = -1
        for i in range(12):
            names, index = match_history.record_visit(names, index, f"M{i}")
        self.assertEqual(len(names), 10)
        self.assertEqual(names[0], "M2")
        self.assertEqual(names[-1], "M11")
        self.assertEqual(index, 9)

    def test_back_and_forward_wrap(self) -> None:
        names = ["A", "B", "C"]
        self.assertEqual(match_history.step_index(names, 1, -1), 0)
        self.assertEqual(match_history.step_index(names, 0, -1), 2)
        self.assertEqual(match_history.step_index(names, 1, 1), 2)
        self.assertEqual(match_history.step_index(names, 2, 1), 0)
        self.assertEqual(match_history.step_index(["A"], 0, 1), 0)
        self.assertIsNone(match_history.step_index([], -1, -1))

    def test_rename_rewrites_stored_names(self) -> None:
        self.assertEqual(
            match_history.rename_entries(["A", "B", "A"], "A", "Z"),
            ["Z", "B", "Z"],
        )

    def test_remove_adjusts_index(self) -> None:
        names, index = match_history.remove_entries(["A", "B", "C", "B"], 3, "B")
        self.assertEqual(names, ["A", "C"])
        self.assertEqual(index, 1)

    def test_prune_missing_roots(self) -> None:
        names, index = match_history.prune_missing(
            ["A", "B", "C"], 2, {"A", "C"}
        )
        self.assertEqual(names, ["A", "C"])
        self.assertEqual(index, 1)
