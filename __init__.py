"""Perspective Match Blender extension entry point."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import operators, overlay, panel, properties

CLASSES = (
    *properties.CLASSES,
    *operators.CLASSES,
    *panel.CLASSES,
)


@persistent
def _reset_modal_state(_dummy=None) -> None:
    """Clear transient modal flags after file load (safe outside register())."""
    for scene in bpy.data.scenes:
        settings = getattr(scene, "match_perspective", None)
        if settings is not None:
            settings.is_modal = False
            settings.work_mode = "NONE"


def register() -> None:
    """Register RNA classes, scene state, and the viewport overlay."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.match_perspective = bpy.props.PointerProperty(
        type=properties.PMSettings,
    )
    # bpy.data is restricted during register(); reset modal flags on load instead.
    if _reset_modal_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_modal_state)
    overlay.register_viewport_draw_handler()


def unregister() -> None:
    """Remove drawing, scene state, and all registered classes."""
    if _reset_modal_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_modal_state)
    overlay.unregister_viewport_draw_handler()
    if hasattr(bpy.types.Scene, "match_perspective"):
        del bpy.types.Scene.match_perspective
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
