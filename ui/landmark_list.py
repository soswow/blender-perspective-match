"""Sync landmark list: one RNA pass, then O(1) row meta and enum entries.

UIList draw and dynamic EnumProperty items otherwise walk every landmark
(and their observations) on every sidebar redraw, including viewport pan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LandmarkRowMeta:
    """Display facts for one CollectionProperty index."""

    observation_count: int
    has_pick_in_active: bool
    mirror_linked: bool
    parallel_linked: bool
    creation_index: int
    name: str
    item_id: str
    kind: str


def _observation_count(landmark) -> int:
    observations = getattr(landmark, "observations", ())
    return sum(1 for observation in observations if observation.is_set)


def _has_pick_in_match(landmark, root) -> bool:
    if root is None:
        return False
    for observation in getattr(landmark, "observations", ()):
        if observation.match_root == root:
            return bool(observation.is_set)
    return False


def collect_landmark_rows(landmarks, active_root=None) -> tuple[LandmarkRowMeta, ...]:
    """One pass over ``landmarks``: pick counts and mirror/parallel link flags."""
    infos: list[tuple[object, str, str, str, str, int, int, bool]] = []
    mirror_targets: set[str] = set()
    parallel_targets: set[str] = set()
    for landmark in landmarks:
        item_id = str(getattr(landmark, "item_id", "") or "")
        kind = str(getattr(landmark, "kind", "POINT") or "POINT")
        mirror_of = str(getattr(landmark, "mirror_of", "NONE") or "NONE")
        parallel_to = str(getattr(landmark, "parallel_to", "NONE") or "NONE")
        if kind == "POINT" and mirror_of not in {"", "NONE"}:
            mirror_targets.add(mirror_of)
        if kind == "LINE" and parallel_to not in {"", "NONE"}:
            if not parallel_to.startswith("WORLD_AXIS"):
                parallel_targets.add(parallel_to)
        infos.append(
            (
                landmark,
                item_id,
                kind,
                mirror_of,
                parallel_to,
                int(getattr(landmark, "creation_index", -1)),
                _observation_count(landmark),
                _has_pick_in_match(landmark, active_root),
            )
        )

    rows: list[LandmarkRowMeta] = []
    for (
        landmark,
        item_id,
        kind,
        mirror_of,
        parallel_to,
        creation_index,
        observation_count,
        has_pick,
    ) in infos:
        mirror_linked = kind == "POINT" and (
            mirror_of not in {"", "NONE"} or item_id in mirror_targets
        )
        parallel_linked = kind == "LINE" and (
            parallel_to not in {"", "NONE"} or item_id in parallel_targets
        )
        rows.append(
            LandmarkRowMeta(
                observation_count=observation_count,
                has_pick_in_active=has_pick,
                mirror_linked=mirror_linked,
                parallel_linked=parallel_linked,
                creation_index=creation_index,
                name=str(getattr(landmark, "name", "") or ""),
                item_id=item_id,
                kind=kind,
            )
        )
    return tuple(rows)


def filter_flags(
    rows: tuple[LandmarkRowMeta, ...],
    *,
    filter_current: bool,
    bitflag: int,
) -> list[int]:
    """Empty when unfiltered (Blender treats that as show-all)."""
    if not filter_current:
        return []
    return [bitflag if row.has_pick_in_active else 0 for row in rows]


def sort_neworder(
    rows: tuple[LandmarkRowMeta, ...],
    *,
    sort_alphabetical: bool,
) -> list[int]:
    """``neworder[old_index] = new_index``, matching ``UI_UL_list.sort_items_helper``."""
    if not rows:
        return []
    if any(row.creation_index < 0 for row in rows):
        return []
    if sort_alphabetical:
        keyed = list(enumerate(rows))
        keyed.sort(key=lambda item: item[1].name.lower())
    else:
        keyed = list(enumerate(rows))
        keyed.sort(key=lambda item: item[1].creation_index)
    neworder = [0] * len(rows)
    for new_index, (old_index, _row) in enumerate(keyed):
        neworder[old_index] = new_index
    return neworder


def parallel_to_enum_entries(
    rows: tuple[LandmarkRowMeta, ...],
    current_item_id: str,
) -> tuple[tuple[str, str, str], ...]:
    """Line landmarks other than ``current_item_id`` for Is Parallel To."""
    entries: list[tuple[str, str, str]] = []
    for row in rows:
        if row.kind != "LINE" or not row.item_id or row.item_id == current_item_id:
            continue
        entries.append(
            (
                row.item_id,
                row.name or row.item_id[:8],
                "Same 3D direction as this line",
            )
        )
    return tuple(entries)


def mirror_of_enum_entries(
    rows: tuple[LandmarkRowMeta, ...],
    current_item_id: str,
) -> tuple[tuple[str, str, str], ...]:
    """Point landmarks other than ``current_item_id`` for Is Mirror Of."""
    entries: list[tuple[str, str, str]] = []
    for row in rows:
        if row.kind != "POINT" or not row.item_id or row.item_id == current_item_id:
            continue
        entries.append(
            (
                row.item_id,
                row.name or row.item_id[:8],
                "This feature mirrored across the scene Mirror Empty",
            )
        )
    return tuple(entries)
