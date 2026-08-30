"""Landmark sidebar list snapshot: filter, sort, and link icons."""

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
if "match_perspective.ui" not in sys.modules:
    _ui = types.ModuleType("match_perspective.ui")
    _ui.__path__ = [str(_ROOT / "ui")]
    sys.modules["match_perspective.ui"] = _ui

_spec = importlib.util.spec_from_file_location(
    "match_perspective.ui.landmark_list",
    _ROOT / "ui" / "landmark_list.py",
)
assert _spec is not None and _spec.loader is not None
landmark_list = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.ui.landmark_list"] = landmark_list
_spec.loader.exec_module(landmark_list)

BITFLAG = 1 << 30


def _landmark(
    *,
    item_id: str,
    name: str,
    kind: str = "POINT",
    creation_index: int = 0,
    mirror_of: str = "NONE",
    parallel_to: str = "NONE",
    observations: tuple = (),
):
    return SimpleNamespace(
        item_id=item_id,
        name=name,
        kind=kind,
        creation_index=creation_index,
        mirror_of=mirror_of,
        parallel_to=parallel_to,
        observations=observations,
    )


class LandmarkListTests(unittest.TestCase):
    """Protect O(1) row meta against the old per-row full-list scans."""

    def test_mirror_icon_both_directions(self) -> None:
        root_a = object()
        left = _landmark(
            item_id="left",
            name="Left",
            mirror_of="right",
            observations=(SimpleNamespace(match_root=root_a, is_set=True),),
        )
        right = _landmark(item_id="right", name="Right")
        other = _landmark(item_id="other", name="Other")
        rows = landmark_list.collect_landmark_rows((left, right, other), root_a)
        self.assertTrue(rows[0].mirror_linked)
        self.assertTrue(rows[1].mirror_linked)
        self.assertFalse(rows[2].mirror_linked)
        self.assertEqual(rows[0].observation_count, 1)
        self.assertTrue(rows[0].has_pick_in_active)
        self.assertFalse(rows[1].has_pick_in_active)

    def test_parallel_icon_world_axis_and_partner(self) -> None:
        world = _landmark(
            item_id="world",
            name="World",
            kind="LINE",
            parallel_to="WORLD_AXIS_X",
        )
        partner = _landmark(item_id="partner", name="Partner", kind="LINE")
        linked = _landmark(
            item_id="linked",
            name="Linked",
            kind="LINE",
            parallel_to="partner",
        )
        point = _landmark(item_id="pt", name="Point")
        rows = landmark_list.collect_landmark_rows((world, partner, linked, point))
        self.assertTrue(rows[0].parallel_linked)
        self.assertTrue(rows[1].parallel_linked)
        self.assertTrue(rows[2].parallel_linked)
        self.assertFalse(rows[3].parallel_linked)

    def test_filter_empty_means_show_all(self) -> None:
        rows = landmark_list.collect_landmark_rows(
            (_landmark(item_id="a", name="A"),)
        )
        self.assertEqual(
            landmark_list.filter_flags(rows, filter_current=False, bitflag=BITFLAG),
            [],
        )
        self.assertEqual(
            landmark_list.filter_flags(rows, filter_current=True, bitflag=BITFLAG),
            [0],
        )

    def test_sort_creation_then_name(self) -> None:
        rows = landmark_list.collect_landmark_rows(
            (
                _landmark(item_id="b", name="Beta", creation_index=2),
                _landmark(item_id="a", name="Alpha", creation_index=1),
            )
        )
        creation = landmark_list.sort_neworder(rows, sort_alphabetical=False)
        self.assertEqual(creation[1], 0)
        self.assertEqual(creation[0], 1)
        alpha = landmark_list.sort_neworder(rows, sort_alphabetical=True)
        self.assertEqual(alpha[1], 0)
        self.assertEqual(alpha[0], 1)

    def test_legacy_creation_index_keeps_collection_order(self) -> None:
        rows = landmark_list.collect_landmark_rows(
            (
                _landmark(item_id="a", name="A", creation_index=-1),
                _landmark(item_id="b", name="B", creation_index=0),
            )
        )
        self.assertEqual(landmark_list.sort_neworder(rows, sort_alphabetical=False), [])

    def test_enum_entries_skip_self(self) -> None:
        rows = landmark_list.collect_landmark_rows(
            (
                _landmark(item_id="p1", name="A"),
                _landmark(item_id="p2", name="B"),
                _landmark(item_id="l1", name="Edge", kind="LINE"),
            )
        )
        mirror = landmark_list.mirror_of_enum_entries(rows, "p1")
        self.assertEqual([item[0] for item in mirror], ["p2"])
        parallel = landmark_list.parallel_to_enum_entries(rows, "l1")
        self.assertEqual(parallel, ())

    def test_enum_candidates_do_not_resolve_dynamic_links(self) -> None:
        class DynamicLinksMustNotBeRead:
            item_id = "safe"
            name = "Safe"
            kind = "POINT"

            @property
            def mirror_of(self):
                raise AssertionError("dynamic mirror enum was resolved")

            @property
            def parallel_to(self):
                raise AssertionError("dynamic parallel enum was resolved")

        candidates = landmark_list.collect_landmark_enum_candidates(
            (DynamicLinksMustNotBeRead(),)
        )
        self.assertEqual(
            candidates,
            (
                landmark_list.LandmarkEnumCandidate(
                    name="Safe",
                    item_id="safe",
                    kind="POINT",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
