"""Perspective Match Blender extension entry point."""

from __future__ import annotations

import importlib
import sys
import traceback

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

_reload_pending = False


@persistent
def _reset_modal_state(_dummy=None) -> None:
    """Clear transient modal flags after file load (safe outside register())."""
    operators._active_interact = None
    for scene in bpy.data.scenes:
        workspace = getattr(scene, "match_perspective", None)
        if workspace is not None:
            workspace.is_modal = False
            workspace.work_mode = "NONE"


def _clear_modal_flags() -> None:
    """Stop treating any modal interact as active before tearing classes down."""
    operators._active_interact = None
    for blender_scene in bpy.data.scenes:
        workspace = getattr(blender_scene, "match_perspective", None)
        if workspace is not None:
            workspace.is_modal = False
            workspace.work_mode = "NONE"


def reload_addon() -> None:
    """Unregister, reload package modules from disk, then register again.

    Must not run from inside an operator/panel stack belonging to this add-on —
    call ``schedule_reload()`` instead so Blender can finish the current event.
    """
    package_name = __package__
    if not package_name:
        raise RuntimeError("Perspective Match reload requires a package context")

    _clear_modal_flags()
    unregister()

    for submodule_name in _RELOAD_SUBMODULES:
        module_name = f"{package_name}.{submodule_name}"
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)

    importlib.reload(sys.modules[package_name])
    sys.modules[package_name].register()

    window_manager = bpy.context.window_manager
    if window_manager is not None:
        for window in window_manager.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                area.tag_redraw()


def schedule_reload() -> bool:
    """Queue a reload on a short timer after the current operator returns.

    Returns False if a reload is already queued.
    """
    global _reload_pending
    if _reload_pending:
        return False
    _reload_pending = True

    package_name = __package__

    def _run_reload() -> None:
        global _reload_pending
        _reload_pending = False
        try:
            package = sys.modules.get(package_name)
            if package is None:
                print(f"Perspective Match: reload skipped; missing module {package_name}")
                return None
            package.reload_addon()
            print("Perspective Match: reloaded from disk")
        except Exception:
            print("Perspective Match: reload failed")
            traceback.print_exc()
        return None

    bpy.app.timers.register(_run_reload, first_interval=0.1)
    return True


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
    # Be tolerant during development reloads if a class is already gone.
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
