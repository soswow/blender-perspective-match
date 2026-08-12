"""Custom UI icons loaded via bpy.utils.previews."""

from __future__ import annotations

from pathlib import Path

import bpy
import bpy.utils.previews

# Preview collections survive as module globals; cleared on unregister.
_preview_collections: dict[str, bpy.utils.previews.ImagePreviewCollection] = {}

_ICONS_DIR = Path(__file__).resolve().parent.parent / "icons"

# name → filename under icons/ (64px for HiDPI; Blender scales as needed)
_ICON_FILES = {
    "vp_lines": "vp-lines-64.png",
    "april_tag": "april-tag-64.png",
}


def icon_id(name: str) -> int:
    """Return a custom icon_id for UILayout icon_value=, or 0 if missing."""
    collection = _preview_collections.get("main")
    if collection is None or name not in collection:
        return 0
    return int(collection[name].icon_id)


def register() -> None:
    """Load PNGs from icons/ into a preview collection."""
    if "main" in _preview_collections:
        return
    collection = bpy.utils.previews.new()
    for name, filename in _ICON_FILES.items():
        path = _ICONS_DIR / filename
        if not path.is_file():
            print(f"Perspective Match: missing icon {path}")
            continue
        collection.load(name, str(path), "IMAGE")
    _preview_collections["main"] = collection


def unregister() -> None:
    """Release preview collections (required to avoid leaks on reload)."""
    for collection in _preview_collections.values():
        bpy.utils.previews.remove(collection)
    _preview_collections.clear()
