"""Visit history for Perspective Match selection (browser-style back/forward)."""

from __future__ import annotations

MATCH_HISTORY_LIMIT = 10


def _clamp_index(names: list[str], index: int) -> int:
    if not names:
        return -1
    return max(0, min(index, len(names) - 1))


def _append(names: list[str], index: int, selected: str) -> tuple[list[str], int]:
    """Drop any forward visits, then append ``selected`` (capped)."""
    if 0 <= index < len(names) and names[index] == selected:
        return names, index
    kept = names[: index + 1] if index >= 0 else []
    kept.append(selected)
    if len(kept) > MATCH_HISTORY_LIMIT:
        kept = kept[-MATCH_HISTORY_LIMIT:]
    return kept, len(kept) - 1


def record_visit(
    names: list[str],
    index: int,
    selected: str,
    *,
    previous: str | None = None,
) -> tuple[list[str], int]:
    """Record a match selection, seeding the prior match when history is empty."""
    if not selected:
        return list(names), index
    working = list(names)
    working_index = index
    if not working and previous and previous != selected:
        working, working_index = _append(working, working_index, previous)
    return _append(working, working_index, selected)


def step_index(names: list[str], index: int, delta: int) -> int | None:
    """Next history index, wrapping at both ends. None when the list is empty."""
    if not names:
        return None
    current = _clamp_index(names, index)
    step = 1 if int(delta) >= 0 else -1
    return (current + step) % len(names)


def rename_entries(names: list[str], old: str, new: str) -> list[str]:
    """Rewrite stored root names after a match rename."""
    if not old or old == new:
        return list(names)
    return [new if name == old else name for name in names]


def remove_entries(
    names: list[str],
    index: int,
    removed: str,
) -> tuple[list[str], int]:
    """Drop every visit to a deleted match and keep the index valid."""
    if not removed:
        return list(names), index
    kept: list[str] = []
    new_index = index
    for visit_index, name in enumerate(names):
        if name == removed:
            if visit_index <= index:
                new_index -= 1
            continue
        kept.append(name)
    if not kept:
        return [], -1
    return kept, _clamp_index(kept, new_index)


def prune_missing(
    names: list[str],
    index: int,
    valid: set[str],
) -> tuple[list[str], int]:
    """Drop visits whose match root no longer exists."""
    kept: list[str] = []
    new_index = index
    for visit_index, name in enumerate(names):
        if name not in valid:
            if visit_index <= index:
                new_index -= 1
            continue
        kept.append(name)
    if not kept:
        return [], -1
    return kept, _clamp_index(kept, new_index)
