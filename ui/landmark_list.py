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


@dataclass(frozen=True)
class LandmarkEnumCandidate:
    """Static landmark fields that are safe inside dynamic enum callbacks."""

    name: str
    item_id: str
    kind: str


def collect_landmark_enum_candidates(landmarks) -> tuple[LandmarkEnumCandidate, ...]:
    """Collect enum labels without resolving mirror_of / parallel_to enums."""
    return tuple(
        LandmarkEnumCandidate(
            name=str(getattr(landmark, "name", "") or ""),
            item_id=str(getattr(landmark, "item_id", "") or ""),
            kind=str(getattr(landmark, "kind", "POINT") or "POINT"),
        )
        for landmark in landmarks
    )


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
        mirror_of = stored_mirror_id(landmark)
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


def stored_mirror_id(landmark) -> str:
    """Persisted partner id; ignores the dynamic enum's list index."""
    if hasattr(landmark, "mirror_of_id"):
        value = str(getattr(landmark, "mirror_of_id", "") or "NONE")
        return value or "NONE"
    return str(getattr(landmark, "mirror_of", "NONE") or "NONE")


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


def _normalized_mirror_id(partner_id: str, source_id: str) -> str:
    if not partner_id or partner_id in {"", "NONE"} or partner_id == source_id:
        return "NONE"
    return partner_id


def inbound_mirror_ids(links: dict[str, str], item_id: str) -> tuple[str, ...]:
    """Landmarks whose stored partner is ``item_id``."""
    return tuple(
        other_id
        for other_id, partner_id in links.items()
        if other_id != item_id and partner_id == item_id
    )


def unique_inbound_mirror_id(links: dict[str, str], item_id: str) -> str:
    """Sole inbound partner, else NONE when missing or ambiguous."""
    found = inbound_mirror_ids(links, item_id)
    if len(found) == 1:
        return found[0]
    return "NONE"


def healed_mirror_partner_id(links: dict[str, str], item_id: str) -> str:
    """Stored partner, or the unique landmark that already names this one."""
    current = _normalized_mirror_id(links.get(item_id, "NONE"), item_id)
    if current != "NONE":
        return current
    return unique_inbound_mirror_id(links, item_id)


def legacy_mirror_partner_id(
    point_ids_in_order: tuple[str, ...],
    source_id: str,
    stored_index: int,
) -> str:
    """Decode an old EnumProperty list index (NONE at 0, then other points)."""
    if stored_index <= 0:
        return "NONE"
    others = tuple(item_id for item_id in point_ids_in_order if item_id != source_id)
    if stored_index > len(others):
        return "NONE"
    return others[stored_index - 1]


def mirror_pair_writes(
    links: dict[str, str],
    source_id: str,
    partner_id: str,
) -> dict[str, str]:
    """``item_id → mirror_of`` assignments so a pair is stored on both sides.

    ``links`` is the mapping after ``source_id`` already has ``partner_id``.
    Clearing or stealing a partner writes NONE onto the leftover side.
    """
    partner_id = _normalized_mirror_id(partner_id, source_id)
    writes: dict[str, str] = {}
    if links.get(source_id, "NONE") != partner_id:
        writes[source_id] = partner_id
    for item_id, current in links.items():
        if not item_id or item_id == source_id:
            continue
        current = current or "NONE"
        if item_id == partner_id:
            if current != source_id:
                writes[item_id] = source_id
        elif current == source_id or (
            partner_id != "NONE" and current == partner_id
        ):
            writes[item_id] = "NONE"
    return writes
