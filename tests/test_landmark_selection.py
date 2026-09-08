"""Landmark ↔ viewport selection lookup."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package
if "match_perspective.scene" not in sys.modules:
    _scene = types.ModuleType("match_perspective.scene")
    _scene.__path__ = [str(_ROOT / "scene")]
    sys.modules["match_perspective.scene"] = _scene

_spec = importlib.util.spec_from_file_location(
    "match_perspective.scene.landmark_selection",
    _ROOT / "scene" / "landmark_selection.py",
)
assert _spec is not None and _spec.loader is not None
landmark_selection = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.scene.landmark_selection"] = landmark_selection
_spec.loader.exec_module(landmark_selection)


class _FakeLandmark:
    def __init__(
        self,
        *,
        item_id: str = "",
        name: str = "",
        known_object=None,
        known_object_b=None,
    ) -> None:
        self.item_id = item_id
        self.name = name
        self.known_object = known_object
        self.known_object_b = known_object_b


class _FakeSpace:
    def __init__(self, landmarks) -> None:
        self.landmarks = landmarks


class _FakeObject:
    def __init__(self, name: str, parent: _FakeObject | None = None) -> None:
        self.name = name
        self.parent = parent

    def get(self, _key: str, default=None):
        return default


class _FakeViewLayer:
    def __init__(self, active, selected) -> None:
        self.objects = SimpleNamespace(active=active, selected=selected)


class LandmarkSelectionTests(unittest.TestCase):
    def test_known_child_object_maps_to_landmark(self) -> None:
        child = _FakeObject("pin_child")
        parent = _FakeObject("parent_mesh", parent=None)
        child.parent = parent
        space = _FakeSpace(
            [_FakeLandmark(item_id="lm-1", name="Corner", known_object=child)]
        )
        self.assertEqual(landmark_selection.landmark_index_for_helper(space, child), 0)

    def test_parent_selection_does_not_match_child_known_object(self) -> None:
        child = _FakeObject("pin_child")
        parent = _FakeObject("parent_mesh")
        child.parent = parent
        space = _FakeSpace(
            [_FakeLandmark(item_id="lm-1", name="Corner", known_object=child)]
        )
        self.assertEqual(landmark_selection.landmark_index_for_helper(space, parent), -1)

    def test_viewport_selection_checks_non_active_selected_objects(self) -> None:
        child = _FakeObject("pin_child")
        parent = _FakeObject("parent_mesh")
        child.parent = parent
        space = _FakeSpace(
            [_FakeLandmark(item_id="lm-1", name="Corner", known_object=child)]
        )
        view_layer = _FakeViewLayer(parent, [parent, child])
        self.assertEqual(
            landmark_selection.landmark_index_for_viewport_selection(space, view_layer),
            0,
        )


if __name__ == "__main__":
    unittest.main()
