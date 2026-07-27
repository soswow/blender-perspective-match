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
    operators._active_interact = None
    for scene in bpy.data.scenes:
        workspace = getattr(scene, "match_perspective", None)
        if workspace is not None:
            workspace.is_modal = False
            workspace.work_mode = "NONE"


def register() -> None:
    """Register RNA classes, scene/object state, and the viewport overlay."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.match_perspective = bpy.props.PointerProperty(
        type=properties.PMWorkspace,
    )
    bpy.types.Object.pm_session = bpy.props.PointerProperty(
        type=properties.PMSession,
    )
    operators._active_interact = None
    if _reset_modal_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_modal_state)
    overlay.register_viewport_draw_handler()


def unregister() -> None:
    """Remove drawing, scene state, and all registered classes."""
    if _reset_modal_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_modal_state)
    operators._active_interact = None
    overlay.unregister_viewport_draw_handler()
    if hasattr(bpy.types.Object, "pm_session"):
        del bpy.types.Object.pm_session
    if hasattr(bpy.types.Scene, "match_perspective"):
        del bpy.types.Scene.match_perspective
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
