"""Perspective Match Blender extension entry point."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

import bpy
from bpy.app.handlers import persistent

from . import properties
from .ui import icons, operators, overlay, panel

CLASSES = (
    *properties.CLASSES,
    *operators.CLASSES,
    *panel.CLASSES,
)

_reload_pending = False
_addon_keymaps: list[tuple] = []


_IS_DEV_INSTALL: bool | None = None


def is_dev_install() -> bool:
    """True when this package is a git checkout (``./scripts/link-dev.sh``), not a zip.

    Release builds exclude ``/scripts/`` (see ``blender_manifest.toml``), so the
    sidebar reload button stays hidden for Install from Disk users.
    """
    global _IS_DEV_INSTALL
    if _IS_DEV_INSTALL is None:
        root = Path(__file__).resolve().parent
        _IS_DEV_INSTALL = (root / "scripts" / "link-dev.sh").is_file()
    return _IS_DEV_INSTALL


def _register_keymaps() -> None:
    """Ctrl+Alt+NumPad 1–9 / arrows, Ctrl+Cmd+A pick, plate click-to-select."""
    _unregister_keymaps()
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return
    keyconfig = window_manager.keyconfigs.addon
    if keyconfig is None:
        return
    keymap = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
    for index in range(1, 10):
        item = keymap.keymap_items.new(
            "perspective_match.activate_match_slot",
            f"NUMPAD_{index}",
            "PRESS",
            ctrl=True,
            alt=True,
        )
        item.properties.index = index
        _addon_keymaps.append((keymap, item))
    for key, direction in (
        ("RIGHT_ARROW", 1),
        ("DOWN_ARROW", 1),
        ("LEFT_ARROW", -1),
        ("UP_ARROW", -1),
    ):
        item = keymap.keymap_items.new(
            "perspective_match.cycle_match",
            key,
            "PRESS",
            ctrl=True,
            alt=True,
        )
        item.properties.direction = direction
        _addon_keymaps.append((keymap, item))
    # Head of the map so we see LMB before view3d.select; miss returns PASS_THROUGH.
    item = keymap.keymap_items.new(
        "perspective_match.select_overlay_landmark",
        "LEFTMOUSE",
        "PRESS",
        head=True,
    )
    _addon_keymaps.append((keymap, item))
    # Ctrl+Cmd+A (OS key): unused in factory 3D View; poll no-ops unless the
    # Perspective Match tab is open in camera view of the active match.
    item = keymap.keymap_items.new(
        "perspective_match.pick_in_active_match",
        "A",
        "PRESS",
        ctrl=True,
        oskey=True,
    )
    _addon_keymaps.append((keymap, item))


def _unregister_keymaps() -> None:
    for keymap, item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _addon_keymaps.clear()


def _register_classes() -> None:
    """Register RNA classes; skip types that are already live (re-enable / wheels)."""
    for cls in CLASSES:
        if getattr(cls, "is_registered", False):
            continue
        try:
            bpy.utils.register_class(cls)
        except ValueError as error:
            # Same Python class still in RNA after a partial unregister.
            if "already registered" not in str(error):
                raise


def _unregister_classes() -> None:
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


def _attach_pointer_properties() -> None:
    if not hasattr(bpy.types.Scene, "match_perspective"):
        bpy.types.Scene.match_perspective = bpy.props.PointerProperty(
            type=properties.PMWorkspace,
        )
    if not hasattr(bpy.types.Object, "pm_session"):
        bpy.types.Object.pm_session = bpy.props.PointerProperty(
            type=properties.PMSession,
        )


def _detach_pointer_properties() -> None:
    # Drop type properties before unregistering PropertyGroups they reference.
    if hasattr(bpy.types.Object, "pm_session"):
        del bpy.types.Object.pm_session
    if hasattr(bpy.types.Scene, "match_perspective"):
        del bpy.types.Scene.match_perspective


@persistent
def _capture_framing_before_save(_dummy=None) -> None:
    """Persist active match camera-view zoom/pan before writing the .blend."""
    try:
        from . import scene as scene_module

        scene_module.capture_active_match_framing(bpy.context)
    except Exception:
        traceback.print_exc()


@persistent
def _reset_modal_state(_dummy=None) -> None:
    """Clear transient modal flags after file load (safe outside register())."""
    operators._active_interact = None
    operators.request_lens_refine_cancel()
    operators.request_vp_detect_cancel()
    operators._lens_refine_running = False
    operators._lens_refine_cancel = None
    operators._vp_detect_running = False
    operators._vp_detect_cancel = None
    properties.bump_sync_ui_cache()
    for scene_block in bpy.data.scenes:
        workspace = getattr(scene_block, "match_perspective", None)
        if workspace is None:
            continue
        workspace.is_modal = False
        workspace.work_mode = "NONE"
        # Backfill creation_index so Sort A–Z off restores saved add order.
        properties.ensure_landmark_creation_indices(workspace)
        properties.ensure_mirror_pairs(workspace)
    # Re-bind overlay callback + rehydrate active match after .blend load.
    overlay.ensure_viewport_draw_handler()
    # File load clears msgbus subscriptions; re-bind the landmark helper listener.
    operators.register_landmark_selection_listener()
    if not bpy.app.timers.is_registered(_restore_active_match_after_load):
        bpy.app.timers.register(_restore_active_match_after_load, first_interval=0.15)


@persistent
def _refresh_after_history(_scene=None) -> None:
    """Invalidate RNA-derived UI caches after Blender restores Undo / Redo."""
    properties.bump_sync_ui_cache()
    for scene_block in bpy.data.scenes:
        workspace = getattr(scene_block, "match_perspective", None)
        if workspace is None:
            continue
        try:
            properties.reconcile_workspace_refs(workspace)
        except (ReferenceError, RuntimeError):
            # A subsequent redraw rebuilds from the post-history datablocks.
            properties.bump_sync_ui_cache()
    properties.tag_viewport_redraw()


def _restore_active_match_after_load() -> None:
    """Deferred: apply saved camera state and enter camera view after load."""
    context = bpy.context
    if context is None or not hasattr(context, "scene"):
        return None
    try:
        from . import scene as scene_module

        space = properties.workspace(context)
        root = properties.active_root(context)
        if root is None:
            # Pointer may be empty while enum still names a loaded root.
            name = getattr(space, "active_match", "NONE")
            if name and name != "NONE":
                candidate = bpy.data.objects.get(name)
                if properties.is_match_root(candidate):
                    root = candidate
        properties.reconcile_workspace_refs(space)
        if root is not None:
            scene_module.set_active_match(context, root)
        else:
            overlay.ensure_viewport_draw_handler()
    except Exception:
        traceback.print_exc()
    return None


def _clear_modal_flags() -> None:
    """Stop treating any modal interact as active before tearing classes down."""
    operators._active_interact = None
    for blender_scene in bpy.data.scenes:
        workspace = getattr(blender_scene, "match_perspective", None)
        if workspace is not None:
            workspace.is_modal = False
            workspace.work_mode = "NONE"


def reload_addon() -> None:
    """Unregister, drop cached modules, import fresh from disk, register again.

    Must not run from inside an operator/panel stack belonging to this add-on —
    call ``schedule_reload()`` instead so Blender can finish the current event.

    Drops every ``{package}.*`` entry from ``sys.modules`` (instead of only
    ``importlib.reload``) so package-layout moves — e.g. ``scene.py`` →
    ``scene/`` — do not leave a stale module object that shadows the new package.
    """
    package_name = __package__
    if not package_name:
        raise RuntimeError("Perspective Match reload requires a package context")

    package_path = Path(sys.modules[package_name].__file__).resolve().parent

    _clear_modal_flags()
    unregister()

    # Purge the package and all nested modules so the next import sees disk layout.
    prefix = package_name + "."
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(prefix):
            del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Perspective Match reload could not load {package_path}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    package.register()

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


def _notify_optional_opencv() -> None:
    """Import OpenCV off the enable path; Info + console if extras are missing."""
    from .detect import opencv as opencv_support

    message = opencv_support.load_warning()
    # Detect / AprilTag buttons appear once the probe has run.
    properties.tag_viewport_redraw()
    if not message:
        return None
    print(message)
    if bpy.app.background:
        return None

    def _report_to_info() -> None:
        try:
            bpy.ops.perspective_match.report_info("EXEC_DEFAULT", message=message)
        except Exception:
            pass
        return None

    bpy.app.timers.register(_report_to_info, first_interval=0.05)
    return None


def register() -> None:
    """Register RNA classes, scene/object state, and the viewport overlay."""
    _register_classes()
    _attach_pointer_properties()
    operators._active_interact = None
    if _reset_modal_state not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_modal_state)
    if _refresh_after_history not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_refresh_after_history)
    if _refresh_after_history not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_refresh_after_history)
    if _capture_framing_before_save not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_capture_framing_before_save)
    # Existing open file: migrate landmark creation indices once on enable.
    properties.ensure_landmark_creation_indices()
    properties.ensure_mirror_pairs()
    overlay.register_viewport_draw_handler()
    icons.register()
    _register_keymaps()
    operators.register_landmark_selection_listener()
    # ``import cv2`` is slow; do it after Preferences enable returns.
    if not bpy.app.timers.is_registered(_notify_optional_opencv):
        bpy.app.timers.register(_notify_optional_opencv, first_interval=0.01)


def unregister() -> None:
    """Remove drawing, scene state, and all registered classes."""
    # Always reach class teardown: a failure in icons/keymaps used to leave
    # PMLineSegment registered, so the next enable raised "already registered".
    if bpy.app.timers.is_registered(_notify_optional_opencv):
        bpy.app.timers.unregister(_notify_optional_opencv)
    try:
        icons.unregister()
    except Exception:
        traceback.print_exc()
    try:
        operators.unregister_landmark_selection_listener()
    except Exception:
        traceback.print_exc()
    try:
        _unregister_keymaps()
    except Exception:
        traceback.print_exc()
    if _capture_framing_before_save in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_capture_framing_before_save)
    if _reset_modal_state in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_modal_state)
    if _refresh_after_history in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_refresh_after_history)
    if _refresh_after_history in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(_refresh_after_history)
    operators._active_interact = None
    try:
        operators.request_lens_refine_cancel()
        operators.request_vp_detect_cancel()
    except Exception:
        traceback.print_exc()
    try:
        overlay.unregister_viewport_draw_handler()
    except Exception:
        traceback.print_exc()
    try:
        _detach_pointer_properties()
    except Exception:
        traceback.print_exc()
    _unregister_classes()


if __name__ == "__main__":
    register()
