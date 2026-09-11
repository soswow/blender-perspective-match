"""Landmark ↔ viewport object lookup (no bpy)."""

from __future__ import annotations

from typing import Any

LANDMARK_HELPER_ID_KEY = "pm_landmark_id"


def safe_identifier(name: str) -> str:
    """Return a compact Blender-friendly id derived from a file stem."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    )
    return (cleaned.strip("_") or "image")[:60]


def _landmark_index_for_single_object(space, obj: Any) -> int:
    """Landmark index for one object if it is a helper or Known 3D pointer target."""
    if obj.name.startswith("PM_LM_"):
        item_id = obj.get(LANDMARK_HELPER_ID_KEY, "")
        if item_id:
            for index, landmark in enumerate(space.landmarks):
                if landmark.item_id == item_id:
                    return index
        for index, landmark in enumerate(space.landmarks):
            base_name = safe_identifier(landmark.name) or landmark.item_id[:8]
            if obj.name == f"PM_LM_{base_name}":
                return index
        return -1
    for index, landmark in enumerate(space.landmarks):
        if landmark.known_object == obj or landmark.known_object_b == obj:
            return index
    return -1


def landmark_index_for_helper(space, obj: Any | None) -> int:
    """Collection index of the landmark this helper or Known 3D object represents, or -1."""
    if obj is None:
        return -1
    current = obj
    while current is not None:
        index = _landmark_index_for_single_object(space, current)
        if index >= 0:
            return index
        current = current.parent
    return -1


def landmark_index_for_viewport_selection(space, view_layer) -> int:
    """Landmark index from the active or any selected viewport object, or -1."""
    active = view_layer.objects.active
    if active is not None:
        index = landmark_index_for_helper(space, active)
        if index >= 0:
            return index
    for obj in view_layer.objects.selected:
        if obj == active:
            continue
        index = landmark_index_for_helper(space, obj)
        if index >= 0:
            return index
    return -1


def landmark_index_for_exclusive_viewport_selection(space, view_layer) -> int:
    """Landmark index only when a single viewport object is selected, else -1."""
    selected = getattr(view_layer.objects, "selected", ())
    if len(selected) != 1:
        return -1
    return landmark_index_for_viewport_selection(space, view_layer)
