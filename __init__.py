"""Perspective Match Blender extension entry point."""

from __future__ import annotations

import importlib
import sys

import bpy
from bpy.app.handlers import persistent

from . import operators, overlay, panel, properties

CLASSES = (
    *properties.CLASSES,
    *operators.CLASSES,
    *panel.CLASSES,
)

# Dependency order for importlib.reload during development.
_RELOAD_SUBMODULES = (
    "core",
    "properties",
    "scene",
    "distortion",
    "project_io",
    "overlay",
    "operators",
    "panel",
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


def reload_addon() -> None:
    """Unregister, reload package modules from disk, then register again.

    ``bpy.ops.script.reload()`` often leaves Panel / PropertyGroup RNA on the
    old class objects. This path tears registration down first so UI edits show up.
    """
    package_name = __package__
    if not package_name:
        raise RuntimeError("Perspective Match reload requires a package context")

    # Drop modal ownership before classes disappear.
    operators._active_interact = None
    for blender_scene in bpy.data.scenes:
        workspace = getattr(blender_scene, "match_perspective", None)
        if workspace is not None:
            workspace.is_modal = False
            workspace.work_mode = "NONE"

    unregister()

    for submodule_name in _RELOAD_SUBMODULES:
        module_name = f"{package_name}.{submodule_name}"
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)

    importlib.reload(sys.modules[package_name])
    sys.modules[package_name].register()

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


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
