"""Operators for file I/O, solving, and camera-view modal interaction."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from uuid import uuid4

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector

from .. import core, properties, scene
from ..core.sync.mirrors import suggested_mirror_partner_name
from ..detect import apriltags as apriltag_detect
from ..detect import line_snap
from ..detect import opencv as opencv_support
from ..detect import tag_snap
from ..detect import vp_lines as vp_line_detect
from ..scene import distortion
from . import overlay, overlay_hit


def _session(context: bpy.types.Context):
    return properties.active_session(context)


def _workspace(context: bpy.types.Context):
    return properties.workspace(context)


def _pm_controls_camera(context: bpy.types.Context) -> bool:
    settings = _session(context)
    return settings is not None and not scene.uses_adjusted_camera(settings)


def _report_exception(operator: bpy.types.Operator, error: Exception) -> set[str]:
    if isinstance(error, KeyError):
        message = f"Sync internal error (missing {error})"
    else:
        message = str(error)
    settings = properties.active_session(bpy.context)
    if settings is not None:
        settings.error = message
    # Blender status reports truncate; keep the full text on the session/workspace.
    operator.report({"ERROR"}, message[:255] if len(message) > 255 else message)
    return {"CANCELLED"}


def _line_bundles_for_ready_check(settings) -> dict[str, list]:
    """Build solver line bundles, filtered like scene.line_bundles_from_settings."""
    line_bundles: dict[str, list] = {"x": [], "y": [], "z": []}
    for line in settings.lines:
        line_bundles[line.axis].append(
            core.LineSegment(line.x1, line.y1, line.x2, line.y2)
        )
    if settings.vp_mode == "1":
        line_bundles["x"] = []
    elif settings.vp_mode == "2":
        line_bundles["y"] = []
    return line_bundles


def _required_lines_ready(settings, *, for_auto_fov: bool = False) -> bool:
    """True when orientation can be solved from current VP lines.

    ``for_auto_fov=True`` requires the classic multi-line path (focal will be
    unlocked). Otherwise Manual FOV / YAML may use one line per axis.
    """
    line_bundles = _line_bundles_for_ready_check(settings)
    lock_focal = False if for_auto_fov else bool(
        settings.lock_focal or settings.vp_mode == "1"
    )
    return core.can_solve_orientation(
        line_bundles,
        lock_focal=lock_focal,
        vp_mode=settings.vp_mode,
    )


def _lines_needed_message(settings, *, for_auto_fov: bool = False) -> str:
    """Human-readable reason a VP solve cannot run yet."""
    line_bundles = _line_bundles_for_ready_check(settings)
    lock_focal = False if for_auto_fov else bool(
        settings.lock_focal or settings.vp_mode == "1"
    )
    return core.orientation_solve_hint(
        line_bundles,
        lock_focal=lock_focal,
        vp_mode=settings.vp_mode,
    )


def _refine_if_ready(context: bpy.types.Context) -> None:
    settings = _session(context)
    if settings is None:
        return
    if scene.uses_adjusted_camera(settings):
        line_bundles = scene.line_bundles_from_settings(settings)
        if any(line_bundles.values()):
            calibration = scene.calibration_from_settings(settings)
            scene._update_diagnostics(settings, line_bundles, calibration)
        settings.status = "Adjusted Camera kept · VP lines are diagnostic only"
        properties.tag_viewport_redraw(context)
        return
    if _required_lines_ready(settings):
        scene.refine_match(context)
        distortion.sync_undistorted_plate_after_refine(context)
    else:
        settings.status = _lines_needed_message(settings)
        properties.tag_viewport_redraw(context)


def _refine_after_vp_detect(context: bpy.types.Context) -> str:
    """Solve/apply camera from the lines Detect just wrote. Returns status text."""
    settings = _session(context)
    if settings is None:
        return "No active match"
    if scene.uses_adjusted_camera(settings):
        line_bundles = scene.line_bundles_from_settings(settings)
        calibration = scene.calibration_from_settings(settings)
        scene._update_diagnostics(settings, line_bundles, calibration)
        return "Adjusted Camera kept · detected VP lines are diagnostic only"
    if not _required_lines_ready(settings):
        return _lines_needed_message(settings)
    calibration = scene.refine_match(context)
    distortion.sync_undistorted_plate_after_refine(context)
    # Same as Auto from VPs: make sure the matched camera is what the view shows.
    scene.enter_camera_view(context)
    return (
        f"Camera matched · HFOV {calibration.hfov_degrees:.2f}°"
        + (
            f" · λ {calibration.division_lambda:.4f}"
            if abs(calibration.division_lambda) > 1.0e-6
            and not calibration.uses_brown_conrady
            else ""
        )
    )


def _maybe_snap_vp_segment(
    context: bpy.types.Context,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], str | None]:
    """Optionally snap a VP segment; returns endpoints and a status suffix."""
    settings = _session(context)
    if settings is None or not settings.snap_vp_lines_to_edges:
        return point_a, point_b, None
    try:
        snapped = line_snap.snap_segment_in_session(settings, point_a, point_b)
    except Exception as error:
        settings.status = f"Edge snap skipped: {error}"
        properties.tag_viewport_redraw(context)
        return point_a, point_b, None
    if snapped is None:
        return point_a, point_b, None
    label = line_snap.kind_label(snapped.kind)
    return (
        snapped.point_a,
        snapped.point_b,
        f"Snapped to {label} ({snapped.mean_shift_px:.1f}px)",
    )


def _tag_snap_enabled(workspace) -> bool:
    """True when Snap to AprilTag is on and OpenCV has already probed in."""
    if workspace is None or not workspace.snap_landmark_to_apriltag:
        return False
    caps = opencv_support.cached_capabilities()
    return caps is not None and caps.available


def _maybe_snap_landmark_point(
    context: bpy.types.Context,
    image_point: tuple[float, float],
) -> tuple[tuple[float, float], str | None]:
    """Optionally snap a point pick onto a nearby AprilTag-like quad."""
    workspace = _workspace(context)
    settings = _session(context)
    if settings is None or not _tag_snap_enabled(workspace):
        return image_point, None
    try:
        snapped = tag_snap.snap_point_in_session(settings, image_point)
    except Exception as error:
        return image_point, f"AprilTag snap skipped: {error}"
    if snapped is None:
        return image_point, None
    return (
        snapped.center_xy,
        f"Snapped to AprilTag ({snapped.mean_shift_px:.1f}px)",
    )


def _log_selected_vp_line(
    settings,
    index: int,
    operator: bpy.types.Operator | None = None,
) -> None:
    """Report the selected VP line id into Blender's Info editor."""
    if not (0 <= index < len(settings.lines)):
        return
    line = settings.lines[index]
    # Older strokes / some imports may lack an id — mint one on first select.
    if not (line.item_id or "").strip():
        line.item_id = f"blender-line-{uuid4().hex}"
    axis_ui = {"x": "X", "z": "Y", "y": "Z"}.get(line.axis, line.axis)
    message = f"VP line [{index}] id={line.item_id} axis={axis_ui}"
    if operator is not None:
        operator.report({"INFO"}, message)
    else:
        print(f"Perspective Match: {message}")


def _region_contains(region, mouse_x: int, mouse_y: int) -> bool:
    """True when window coords fall inside a visible region rect."""
    # Collapsed sidebars report width/height 1 — treat as not hittable.
    if region.width <= 1 or region.height <= 1:
        return False
    return (
        region.x <= mouse_x < region.x + region.width
        and region.y <= mouse_y < region.y + region.height
    )


def _view3d_under_event(context: bpy.types.Context, event):
    """Return (area, region, space) for the 3D View WINDOW under the mouse.

    With Preferences → Interface → Region Overlap, the N-panel / Toolbar float
    over the WINDOW region (same pixel rect). Prefer those overlay regions so
    sidebar clicks PASS_THROUGH to Blender UI instead of hitting the plate.
    """
    screen = context.screen
    if screen is None:
        return None, None, None
    mouse_x = event.mouse_x
    mouse_y = event.mouse_y
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        # Overlay regions sit on top of WINDOW when region overlap is on.
        for region in area.regions:
            if region.type == "WINDOW":
                continue
            if _region_contains(region, mouse_x, mouse_y):
                return None, None, None
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            if _region_contains(region, mouse_x, mouse_y):
                space = area.spaces.active
                return area, region, space
    return None, None, None


# Live modal instance — is_modal alone can go stale if the handler dies uncleanly.
_active_interact: "PM_OT_interact | None" = None

# Background Refine Lenses job (pure numpy in a worker thread; bpy apply on main).
_lens_refine_lock = threading.Lock()
_lens_refine_cancel: threading.Event | None = None
_lens_refine_running = False
_lens_refine_progress = {"step": 0, "total": 1, "label": ""}
_lens_refine_result_box: dict = {}

_diagnose_sync_lock = threading.Lock()
_diagnose_sync_cancel: threading.Event | None = None
_diagnose_sync_running = False
_diagnose_sync_progress = {"step": 0, "total": 1, "label": ""}
_diagnose_sync_result_box: dict = {}

_vp_detect_lock = threading.Lock()
_vp_detect_cancel: threading.Event | None = None
_vp_detect_running = False
_vp_detect_progress = {"label": ""}
_vp_detect_result_box: dict = {}


def lens_refine_is_running() -> bool:
    """True while a Refine Lenses modal/worker is active."""
    return bool(_lens_refine_running)


def diagnose_sync_is_running() -> bool:
    """True while a Diagnose modal/worker is active."""
    return bool(_diagnose_sync_running)


def vp_detect_is_running() -> bool:
    """True while Detect VP Lines is running in the background."""
    return bool(_vp_detect_running)


def request_vp_detect_cancel() -> bool:
    """Ask the Detect VP Lines worker to stop; True if a job was signalled."""
    event = _vp_detect_cancel
    if event is None:
        return False
    event.set()
    return True


def request_lens_refine_cancel() -> bool:
    """Signal the running Refine Lenses worker to stop. True if one was running."""
    event = _lens_refine_cancel
    if event is None:
        return False
    event.set()
    return True


def request_diagnose_sync_cancel() -> bool:
    """Signal the running Diagnose worker to stop."""
    event = _diagnose_sync_cancel
    if event is None:
        return False
    event.set()
    return True


_NUMPAD_SLOT_KEYS = {
    "NUMPAD_1": 1,
    "NUMPAD_2": 2,
    "NUMPAD_3": 3,
    "NUMPAD_4": 4,
    "NUMPAD_5": 5,
    "NUMPAD_6": 6,
    "NUMPAD_7": 7,
    "NUMPAD_8": 8,
    "NUMPAD_9": 9,
}

_ARROW_CYCLE_KEYS = {
    "LEFT_ARROW": -1,
    "UP_ARROW": -1,
    "RIGHT_ARROW": 1,
    "DOWN_ARROW": 1,
}


def _clear_interact_flags(context: bpy.types.Context) -> None:
    """Reset workspace modal flags without touching a live operator."""
    workspace = _workspace(context)
    workspace.is_modal = False
    workspace.work_mode = "NONE"
    _clear_vp_line_selection(context)


def _clear_vp_line_selection(context: bpy.types.Context) -> None:
    """Drop the selected VP line so endpoint handles disappear outside edit mode."""
    settings = _session(context)
    if settings is None:
        return
    if settings.selected_line_index != -1:
        settings.selected_line_index = -1
        properties.tag_viewport_redraw(context)


def _interact_cursor_for_mode(mode: str, context: bpy.types.Context) -> str:
    """Blender window cursor id for an active Draw / Pick tool mode."""
    if mode == "LINE":
        # Empty-plate draw affordance; hover hit-tests refine this further.
        return "PAINT_CROSS"
    if mode == "PP":
        return "SCROLL_XY"
    if mode == "ORIGIN":
        return "PAINT_CROSS"
    if mode == "LANDMARK":
        landmark = scene.active_landmark(context)
        if landmark is not None and landmark.kind == "LINE":
            return "KNIFE"
        return "DOT"
    return "CROSSHAIR"


def _set_interact_cursor(
    context: bpy.types.Context,
    mode: str,
    *,
    modal: bool,
) -> None:
    """Start/refresh the modal cursor stack; plate hover owns the visible cursor.

    ``modal=True`` pushes DEFAULT so later ``cursor_set("DEFAULT")`` is a real
    arrow (Blender remaps DEFAULT → win->modalcursor while modal is set).
    Do not force the mode cursor here — invoke often runs over a sidebar button,
    and the first plate hover applies PAINT_CROSS / etc.
    """
    del mode  # Mode cursor comes from hover hit-testing, not from this helper.
    window = context.window
    if window is None:
        return
    if modal:
        # See WM_cursor_set in wm_cursors.cc: DEFAULT remaps to modalcursor.
        window.cursor_modal_set("DEFAULT")
    # Forget any cached id so the next plate hover always calls cursor_set.
    if _active_interact is not None:
        _active_interact._hover_cursor = ""


def _restore_interact_cursor(context: bpy.types.Context) -> None:
    """Undo cursor_modal_set from an interact tool session."""
    window = context.window
    if window is None:
        return
    window.cursor_modal_restore()
    if _active_interact is not None:
        _active_interact._hover_cursor = ""


def _perspective_match_sidebar_active(context: bpy.types.Context) -> bool:
    """True when a 3D View has the Perspective Match N-panel tab selected."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return False
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "UI":
                continue
            if region.width > 1 and region.active_panel_category == "Perspective Match":
                return True
    return False


_LANDMARK_SELECT_MSGBUS_OWNER = object()


def _queue_sidebar_landmark_from_selection(*_args) -> None:
    """msgbus notify: apply after the select operator finishes."""
    if bpy.app.timers.is_registered(_apply_sidebar_landmark_from_selection):
        return
    bpy.app.timers.register(_apply_sidebar_landmark_from_selection, first_interval=0.0)


def _apply_sidebar_landmark_from_selection() -> None:
    """If the active object is a PM helper or Known 3D object, select it in the Sync list."""
    context = bpy.context
    if context is None:
        return None
    if not _perspective_match_sidebar_active(context):
        return None
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return None
    space = _workspace(context)
    index = scene.landmark_index_for_helper(space, view_layer.objects.active)
    if index < 0 or space.active_landmark_index == index:
        return None
    _set_active_landmark(context, index)
    return None


def register_landmark_selection_listener() -> None:
    """Watch LayerObjects.active so viewport picks update the Sync list."""
    unregister_landmark_selection_listener()
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_LANDMARK_SELECT_MSGBUS_OWNER,
        args=(),
        notify=_queue_sidebar_landmark_from_selection,
        options={"PERSISTENT"},
    )


def unregister_landmark_selection_listener() -> None:
    """Drop the active-object subscription and any pending apply timer."""
    bpy.msgbus.clear_by_owner(_LANDMARK_SELECT_MSGBUS_OWNER)
    if bpy.app.timers.is_registered(_apply_sidebar_landmark_from_selection):
        bpy.app.timers.unregister(_apply_sidebar_landmark_from_selection)


def _overlay_landmark_hit_index(context: bpy.types.Context, mouse: Vector) -> int:
    """Collection index of the landmark pick under ``mouse``, or -1."""
    space = _workspace(context)
    if not space.show_landmark_overlay:
        return -1
    root = properties.active_root(context)
    if root is None:
        return -1
    items: list[
        tuple[int, str, tuple[float, float], tuple[float, float] | None]
    ] = []
    for index, landmark in enumerate(space.landmarks):
        observation = scene.observation_for_match(landmark, root)
        if observation is None or not observation.is_set:
            continue
        if landmark.kind == "LINE":
            point_a = scene.image_to_region(context, observation.x, observation.y)
            point_b = scene.image_to_region(
                context, observation.x2, observation.y2
            )
            if point_a is None or point_b is None:
                continue
            items.append(
                (index, "LINE", (point_a.x, point_a.y), (point_b.x, point_b.y))
            )
            continue
        point = scene.image_to_region(context, observation.x, observation.y)
        if point is None:
            continue
        items.append((index, "POINT", (point.x, point.y), None))
    scale = overlay.ui_scale()
    return overlay_hit.nearest_landmark_hit(
        (float(mouse.x), float(mouse.y)),
        items,
        point_radius=12.0 * scale,
        line_radius=11.0 * scale,
    )


def _set_active_landmark(context: bpy.types.Context, index: int) -> None:
    """Select a landmark in the sidebar list and refresh the overlay."""
    space = _workspace(context)
    if index < 0 or index >= len(space.landmarks):
        return
    space.active_landmark_index = index
    properties.tag_viewport_redraw(context)


def _match_slot_from_event(event) -> int | None:
    """Ctrl+Alt+NumPad 1–9 → 1-based match slot, else None."""
    if event.value != "PRESS" or not event.ctrl or not event.alt:
        return None
    return _NUMPAD_SLOT_KEYS.get(event.type)


def _match_cycle_from_event(event) -> int | None:
    """Ctrl+Alt+Arrow → +1 (next) or -1 (previous), else None."""
    if event.value != "PRESS" or not event.ctrl or not event.alt:
        return None
    return _ARROW_CYCLE_KEYS.get(event.type)


def cancel_active_interact(context: bpy.types.Context) -> bool:
    """Cancel Draw / Pick Origin / PP / Landmark modal. True if one was live."""
    global _active_interact
    active = _active_interact
    if active is None:
        workspace = _workspace(context)
        if workspace.is_modal:
            _clear_interact_flags(context)
            overlay.clear_preview(context)
            # Stale flag only — no live modal_set stack to restore.
            if context.window is not None:
                context.window.cursor_set("DEFAULT")
        return False
    # Drop in-progress gestures (half-drawn lines, PP drag, etc.) then exit.
    active._cancel_drag(context)
    active._finish(context, cancelled=True)
    return True


def activate_match_by_slot(
    context: bpy.types.Context,
    index: int,
    *,
    report=None,
) -> set[str]:
    """Activate the Nth match root (1-based, name-sorted). Cancels live tools."""
    roots = properties.iter_match_roots()
    if index < 1 or index > len(roots):
        message = (
            f"No match in slot {index} "
            f"({len(roots)} match{'es' if len(roots) != 1 else ''})"
        )
        if report is not None:
            report({"WARNING"}, message)
        return {"CANCELLED"}
    root = roots[index - 1]
    try:
        # set_active_match cancels any Draw / Pick modal before switching.
        scene.set_active_match(context, root)
    except Exception as error:
        if report is not None:
            report({"ERROR"}, str(error))
        return {"CANCELLED"}
    return {"FINISHED"}


def activate_match_by_delta(
    context: bpy.types.Context,
    delta: int,
    *,
    report=None,
) -> set[str]:
    """Activate the previous/next match in the name-sorted list (wraps)."""
    roots = properties.iter_match_roots()
    if not roots:
        if report is not None:
            report({"WARNING"}, "No Perspective Match cameras")
        return {"CANCELLED"}
    step = 1 if int(delta) >= 0 else -1
    current = properties.active_root(context)
    try:
        index = roots.index(current)
        index = (index + step) % len(roots)
    except ValueError:
        index = 0 if step > 0 else len(roots) - 1
    return activate_match_by_slot(context, index + 1, report=report)


def _delete_selected_item(context: bpy.types.Context) -> bool:
    """Delete the selected VP line. Return whether something was removed."""
    settings = _session(context)
    if settings is None:
        return False
    if 0 <= settings.selected_line_index < len(settings.lines):
        settings.lines.remove(settings.selected_line_index)
        settings.selected_line_index = -1
        _refine_if_ready(context)
        return True
    return False


class PM_OT_load_image(bpy.types.Operator, ImportHelper):
    """Load a still into the active match camera."""

    bl_idname = "perspective_match.load_image"
    bl_label = "Open Reference Image"
    bl_description = "Load a still into this match"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.exr;*.webp",
        options={"HIDDEN"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        self._context_scene = context.scene
        try:
            scene.bind_reference_image(context, self.filepath)
        except Exception as error:
            return _report_exception(self, error)
        self.report({"INFO"}, f"Loaded {Path(self.filepath).name}")
        return {"FINISHED"}


class PM_OT_replace_image(bpy.types.Operator, ImportHelper):
    """Replace the active match still without clearing lines or landmarks."""

    bl_idname = "perspective_match.replace_image"
    bl_label = "Replace Reference Image"
    bl_description = (
        "Swap the reference still on the active match. Keeps VP lines, origin, "
        "calibration, and landmarks. New image must match the current pixel size"
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.exr;*.webp",
        options={"HIDDEN"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = properties.active_session(context)
        return settings is not None and settings.image is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        self._context_scene = context.scene
        try:
            scene.replace_reference_image(context, self.filepath)
        except Exception as error:
            return _report_exception(self, error)
        self.report({"INFO"}, f"Replaced with {Path(self.filepath).name}")
        return {"FINISHED"}


class PM_OT_import_ros_yaml(bpy.types.Operator, ImportHelper):
    """Import ROS camera_info YAML intrinsics into the active match."""

    bl_idname = "perspective_match.import_ros_yaml"
    bl_label = "Import YAML"
    bl_description = (
        "Import ROS camera_info YAML: lock Manual FOV from fx, set principal "
        "point from cx/cy, and apply OpenCV D (plumb_bob / rational) or "
        "fitzgibbon_lambda when D is zero"
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".yaml"
    filter_glob: bpy.props.StringProperty(
        default="*.yaml;*.yml",
        options={"HIDDEN"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and not scene.uses_adjusted_camera(settings)
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            message = scene.apply_ros_camera_info_yaml(context, self.filepath)
        except Exception as error:
            return _report_exception(self, error)
        self.report({"INFO"}, message[:255] if len(message) > 255 else message)
        return {"FINISHED"}


class PM_OT_refine(bpy.types.Operator):
    """Refine and apply the matched camera from current VP lines."""

    bl_idname = "perspective_match.refine"
    bl_label = "Refine Camera"
    bl_description = (
        "Unlock Manual FOV and solve focal / orientation from VP lines. "
        "Needs two axes with at least two lines each in 3-point mode. "
        "When Use Known 3D is on, also polish FOV, principal point, and camera "
        "position from landmark picks without changing VP orientation"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Stay clickable with a still loaded so a disabled button is not a silent dead-end.
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and not scene.uses_adjusted_camera(settings)
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None or settings.image is None:
            self.report({"ERROR"}, "Load a reference image first")
            return {"CANCELLED"}
        if not _required_lines_ready(settings, for_auto_fov=True):
            message = _lines_needed_message(settings, for_auto_fov=True)
            settings.status = message
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        # "Auto from VPs" means FOV comes from geometry, not the manual lock.
        settings.lock_focal = False
        try:
            calibration = scene.refine_match(context)
            from ..scene import distortion

            distortion.sync_undistorted_plate_after_refine(context)
        except Exception as error:
            return _report_exception(self, error)
        scene.enter_camera_view(context)
        self.report({"INFO"}, f"Matched camera at {calibration.hfov_degrees:.2f}° HFOV")
        return {"FINISHED"}


class PM_OT_camera_view(bpy.types.Operator):
    """Enter the managed match camera view."""

    bl_idname = "perspective_match.camera_view"
    bl_label = "View Match Camera"
    bl_description = "Switch this 3D View to the Perspective Match camera"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and settings is not None
            and settings.camera_object is not None
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene.ensure_match_ready(context)
        scene.enter_camera_view(context, restore_framing=True)
        return {"FINISHED"}


class PM_OT_apply_manual_fov(bpy.types.Operator):
    """Apply the manual horizontal FOV and optionally refine orientation."""

    bl_idname = "perspective_match.apply_manual_fov"
    bl_label = "Apply Manual FOV"
    bl_description = "Apply horizontal FOV and keep it locked during camera refinement"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _pm_controls_camera(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            self.report({"ERROR"}, "Create or activate a match camera first")
            return {"CANCELLED"}
        settings.lock_focal = True
        try:
            if _required_lines_ready(settings):
                scene.refine_match(context)
            else:
                scene.apply_manual_fov(context)
            distortion.sync_undistorted_plate_after_refine(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class PM_OT_reset_camera(bpy.types.Operator):
    """Reset PP/distortion and use the current manual FOV."""

    bl_idname = "perspective_match.reset_camera"
    bl_label = "Reset Camera"
    bl_description = "Reset principal point and distortion, then solve orientation at manual FOV"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _pm_controls_camera(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            self.report({"ERROR"}, "Create or activate a match camera first")
            return {"CANCELLED"}
        settings.cx = settings.image_width * 0.5
        settings.cy = settings.image_height * 0.5
        settings.division_lambda = 0.0
        settings.lambda_saturated = False
        settings.brown_conrady = (0.0,) * 8
        settings.lock_focal = True
        scene.invalidate_undistorted_cache(settings)
        try:
            if _required_lines_ready(settings):
                scene.refine_match(context)
            else:
                scene.apply_manual_fov(context)
            distortion.sync_undistorted_plate_after_refine(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        settings.status = "Camera reset to manual FOV and centered principal point"
        return {"FINISHED"}


class PM_OT_edit_pp_offset(bpy.types.Operator):
    """Type principal-point offsets in pixels from image center."""

    bl_idname = "perspective_match.edit_pp_offset"
    bl_label = "Edit PP Offset"
    bl_description = (
        "Enter principal-point offsets in pixels from the image center "
        "(same values shown as PP offset X, Y)"
    )
    bl_options = {"REGISTER", "UNDO"}

    offset_x: bpy.props.FloatProperty(
        name="Offset X",
        description="Horizontal principal-point offset from image center (pixels)",
        default=0.0,
        precision=1,
        step=10,
    )
    offset_y: bpy.props.FloatProperty(
        name="Offset Y",
        description="Vertical principal-point offset from image center (pixels)",
        default=0.0,
        precision=1,
        step=10,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and not scene.uses_adjusted_camera(settings)
        )

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        settings = _session(context)
        if settings is None or settings.image is None:
            return {"CANCELLED"}
        # Prefill from the same center-relative values shown in the panel.
        self.offset_x = float(settings.cx - settings.image_width * 0.5)
        self.offset_y = float(settings.cy - settings.image_height * 0.5)
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, _context: bpy.types.Context) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "offset_x")
        layout.prop(self, "offset_y")
        layout.label(text="Pixels from image center", icon="INFO")

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None or settings.image is None:
            self.report({"ERROR"}, "Load a reference image first")
            return {"CANCELLED"}
        width = max(int(settings.image_width), 1)
        height = max(int(settings.image_height), 1)
        image_point = (
            width * 0.5 + float(self.offset_x),
            height * 0.5 + float(self.offset_y),
        )
        try:
            scene.set_principal_point(context, image_point, finalize=True)
            distortion.sync_undistorted_plate_after_refine(context)
        except Exception as error:
            return _report_exception(self, error)
        self.report(
            {"INFO"},
            f"PP offset {self.offset_x:+.1f}, {self.offset_y:+.1f} px",
        )
        return {"FINISHED"}


class PM_OT_clear_axis(bpy.types.Operator):
    """Delete all VP lines on the selected colored axis."""

    bl_idname = "perspective_match.clear_axis"
    bl_label = "Clear Axis"
    bl_description = "Delete every line on the active axis"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            return {"CANCELLED"}
        for index in reversed(range(len(settings.lines))):
            if settings.lines[index].axis == settings.active_axis:
                settings.lines.remove(index)
        settings.selected_line_index = -1
        _refine_if_ready(context)
        return {"FINISHED"}


class PM_OT_detect_vp_lines(bpy.types.Operator):
    """Detect vanishing-point line bundles automatically (3-point mode only)."""

    bl_idname = "perspective_match.detect_vp_lines"
    bl_label = "Detect VP Lines"
    bl_description = (
        "Find straight edges, cluster them into three vanishing points, and "
        "replace the current VP lines (3-point perspective only). "
        "Runs in the background — Esc to cancel. "
        "Applies the matched camera when enough lines exist. "
        "Edit or delete strokes afterward as usual"
    )
    # No UNDO on the long modal: Blender's modal undo often keeps CollectionProperty
    # line edits but rolls back Object.matrix_local from apply_camera. We push an
    # explicit undo step in _finish_job covering lines + camera together.
    bl_options = {"REGISTER"}

    _timer = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if vp_detect_is_running():
            return False
        caps = opencv_support.cached_capabilities()
        if caps is None or not caps.line_segment_detector:
            return False
        settings = properties.active_session(context)
        return (
            settings is not None
            and settings.image is not None
            and str(settings.vp_mode) == "3"
            and properties.active_root(context) is not None
        )

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        global _vp_detect_cancel, _vp_detect_running, _vp_detect_result_box
        global _vp_detect_progress

        settings = properties.active_session(context)
        if settings is None:
            return {"CANCELLED"}

        try:
            # Copy pixels on the main thread — bpy / Image access is not thread-safe.
            gray = vp_line_detect.load_detection_gray(settings)
        except vp_line_detect.VpLineDependencyError as error:
            return _report_exception(self, error)
        except Exception as error:
            return _report_exception(self, error)

        # Detach so the worker owns a contiguous buffer with no Blender aliases.
        gray = np.ascontiguousarray(gray.copy())
        sensitivity = float(settings.vp_detect_sensitivity)
        cancel_event = threading.Event()
        result_box: dict = {"done": False}
        progress_state = {"label": "Detecting edges…"}

        with _vp_detect_lock:
            if _vp_detect_running:
                self.report({"WARNING"}, "Detect VP Lines already running")
                return {"CANCELLED"}
            _vp_detect_cancel = cancel_event
            _vp_detect_running = True
            _vp_detect_result_box = result_box
            _vp_detect_progress = progress_state

        settings.status = "Detect VP Lines running… Esc to cancel"
        settings.error = ""
        properties.tag_viewport_redraw(context)

        def _worker() -> None:
            try:
                if cancel_event.is_set():
                    result_box["cancelled"] = True
                    return
                progress_state["label"] = "Detecting edges…"
                outcome = vp_line_detect.detect_vp_line_bundles(
                    gray,
                    sensitivity=sensitivity,
                )
                if cancel_event.is_set():
                    result_box["cancelled"] = True
                    return
                progress_state["label"] = "Building debug plate…"
                debug_rgba = vp_line_detect.render_debug_rgba(
                    int(gray.shape[1]),
                    int(gray.shape[0]),
                    outcome.candidates,
                )
                result_box["outcome"] = outcome
                result_box["debug_rgba"] = debug_rgba
                result_box["sensitivity"] = sensitivity
            except Exception as error:
                result_box["error"] = error
            finally:
                result_box["done"] = True

        thread = threading.Thread(
            target=_worker,
            name="PM-DetectVpLines",
            daemon=True,
        )
        thread.start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)
        self._timer = window_manager.event_timer_add(0.1, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Scripted calls may use EXEC_DEFAULT; still run as a background modal.
        return self.invoke(context, None)

    def _finish_job(self, context: bpy.types.Context, *, cancelled: bool) -> set[str]:
        global _vp_detect_cancel, _vp_detect_running

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            window_manager.progress_end()
        except Exception:
            pass

        result_box = _vp_detect_result_box
        with _vp_detect_lock:
            _vp_detect_running = False
            _vp_detect_cancel = None

        settings = properties.active_session(context)
        if result_box.get("error") is not None:
            error = result_box["error"]
            if settings is not None:
                settings.status = str(error)
            properties.tag_viewport_redraw(context)
            return _report_exception(
                self,
                error if isinstance(error, Exception) else Exception(str(error)),
            )

        if cancelled or result_box.get("cancelled"):
            message = "Detect VP Lines cancelled"
            if settings is not None:
                settings.status = message
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        outcome = result_box.get("outcome")
        debug_rgba = result_box.get("debug_rgba")
        if outcome is None or settings is None:
            message = "Detect VP Lines failed"
            if settings is not None:
                settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        try:
            # One undo step for lines + camera (modal UNDO is unreliable for pose).
            try:
                bpy.ops.ed.undo_push(message="Detect VP Lines")
            except Exception:
                pass
            vp_line_detect.apply_vp_line_bundles(settings, outcome.bundles)
            if debug_rgba is not None:
                vp_line_detect.install_debug_rgba_plate(settings, debug_rgba)
                settings.vp_detect_sensitivity_baked = float(
                    result_box.get("sensitivity", settings.vp_detect_sensitivity)
                )
                if settings.view_vp_detect_debug:
                    scene.refresh_background_projection(context)
            refine_status = _refine_after_vp_detect(context)
        except Exception as error:
            return _report_exception(self, error)

        counts = outcome.result.counts
        # UI axis order: X (red) · Y (green/internal z) · Z (blue/internal y).
        detect_message = (
            f"Detected VP lines · X {counts.get('x', 0)} · "
            f"Y {counts.get('z', 0)} · Z {counts.get('y', 0)} "
            f"({outcome.result.candidates} edges → {outcome.result.clusters} clusters)"
        )
        settings.error = ""
        if refine_status.startswith("Camera matched"):
            settings.status = f"{detect_message} · {refine_status}"
            self.report({"INFO"}, settings.status)
        else:
            # Lines written but orientation still blocked — surface both facts.
            settings.status = f"{detect_message} · {refine_status}"
            self.report({"WARNING"}, settings.status)
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        settings = properties.active_session(context)
        if event.type in {"ESC"} and event.value == "PRESS":
            request_vp_detect_cancel()
            if settings is not None:
                settings.status = "Cancelling Detect VP Lines…"
            properties.tag_viewport_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        label = str(_vp_detect_progress.get("label", ""))
        if settings is not None and label:
            settings.status = f"Detect VP Lines · {label}"
        try:
            context.window_manager.progress_update(50)
        except Exception:
            pass
        properties.tag_viewport_redraw(context)

        if not _vp_detect_result_box.get("done"):
            return {"PASS_THROUGH"}

        return self._finish_job(
            context,
            cancelled=bool(_vp_detect_cancel and _vp_detect_cancel.is_set()),
        )


class PM_OT_toggle_vp_detect_debug(bpy.types.Operator):
    """Toggle the black/white auto-detected edge debug plate."""

    bl_idname = "perspective_match.toggle_vp_detect_debug"
    bl_label = "Debug auto detected edges"
    bl_description = (
        "Toggle a black plate with white strokes for every auto-detected edge. "
        "Runs edge detection on first use, then reuses that plate. Esc cancels"
    )
    bl_options = {"REGISTER", "UNDO"}

    _timer = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if vp_detect_is_running():
            return False
        caps = opencv_support.cached_capabilities()
        if caps is None or not caps.line_segment_detector:
            return False
        settings = properties.active_session(context)
        return (
            settings is not None
            and settings.image is not None
            and properties.active_root(context) is not None
        )

    def _show_or_hide(self, context: bpy.types.Context, enabling: bool) -> set[str]:
        settings = properties.active_session(context)
        if settings is None:
            return {"CANCELLED"}
        try:
            vp_line_detect.set_vp_detect_debug_view(context, enabling)
        except Exception as error:
            return _report_exception(self, error)
        message = (
            "Debug auto detected edges on"
            if enabling
            else "Debug auto detected edges off"
        )
        settings.status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        global _vp_detect_cancel, _vp_detect_running, _vp_detect_result_box
        global _vp_detect_progress

        settings = properties.active_session(context)
        if settings is None:
            return {"CANCELLED"}

        # Already showing → hide and reuse the cached plate next time.
        if settings.view_vp_detect_debug:
            return self._show_or_hide(context, False)

        # Plate from a prior Detect / debug scan at the same sensitivity → show it.
        plate_matches = (
            settings.vp_detect_debug_image is not None
            and abs(
                float(settings.vp_detect_sensitivity_baked)
                - float(settings.vp_detect_sensitivity)
            )
            < 1.0e-4
        )
        if plate_matches:
            return self._show_or_hide(context, True)

        # First use (or sensitivity changed): detect edges, then show the plate.
        try:
            gray = vp_line_detect.load_detection_gray(settings)
        except vp_line_detect.VpLineDependencyError as error:
            return _report_exception(self, error)
        except Exception as error:
            return _report_exception(self, error)

        gray = np.ascontiguousarray(gray.copy())
        sensitivity = float(settings.vp_detect_sensitivity)
        cancel_event = threading.Event()
        result_box: dict = {"done": False, "mode": "edges_only"}
        progress_state = {"label": "Detecting edges…"}

        with _vp_detect_lock:
            if _vp_detect_running:
                self.report({"WARNING"}, "Edge detection already running")
                return {"CANCELLED"}
            _vp_detect_cancel = cancel_event
            _vp_detect_running = True
            _vp_detect_result_box = result_box
            _vp_detect_progress = progress_state

        settings.status = "Detecting edges for debug… Esc to cancel"
        settings.error = ""
        properties.tag_viewport_redraw(context)

        def _worker() -> None:
            try:
                if cancel_event.is_set():
                    result_box["cancelled"] = True
                    return
                _candidates, debug_rgba = vp_line_detect.detect_edges_for_debug(
                    gray,
                    sensitivity=sensitivity,
                )
                if cancel_event.is_set():
                    result_box["cancelled"] = True
                    return
                result_box["debug_rgba"] = debug_rgba
                result_box["edge_count"] = len(_candidates)
                result_box["sensitivity"] = sensitivity
            except Exception as error:
                result_box["error"] = error
            finally:
                result_box["done"] = True

        thread = threading.Thread(
            target=_worker,
            name="PM-DetectVpEdgesDebug",
            daemon=True,
        )
        thread.start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)
        self._timer = window_manager.event_timer_add(0.1, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        return self.invoke(context, None)

    def _finish_edge_scan(
        self, context: bpy.types.Context, *, cancelled: bool
    ) -> set[str]:
        global _vp_detect_cancel, _vp_detect_running

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            window_manager.progress_end()
        except Exception:
            pass

        result_box = _vp_detect_result_box
        with _vp_detect_lock:
            _vp_detect_running = False
            _vp_detect_cancel = None

        settings = properties.active_session(context)
        if result_box.get("error") is not None:
            error = result_box["error"]
            if settings is not None:
                settings.status = str(error)
            properties.tag_viewport_redraw(context)
            return _report_exception(
                self,
                error if isinstance(error, Exception) else Exception(str(error)),
            )

        if cancelled or result_box.get("cancelled"):
            message = "Edge debug scan cancelled"
            if settings is not None:
                settings.status = message
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        debug_rgba = result_box.get("debug_rgba")
        if debug_rgba is None or settings is None:
            message = "Edge debug scan failed"
            if settings is not None:
                settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        try:
            vp_line_detect.install_debug_rgba_plate(settings, debug_rgba)
            settings.vp_detect_sensitivity_baked = float(
                result_box.get("sensitivity", settings.vp_detect_sensitivity)
            )
            vp_line_detect.set_vp_detect_debug_view(context, True)
        except Exception as error:
            return _report_exception(self, error)

        edge_count = int(result_box.get("edge_count", 0))
        message = f"Debug auto detected edges on · {edge_count} edges"
        settings.status = message
        settings.error = ""
        self.report({"INFO"}, message)
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        settings = properties.active_session(context)
        if event.type in {"ESC"} and event.value == "PRESS":
            request_vp_detect_cancel()
            if settings is not None:
                settings.status = "Cancelling edge debug scan…"
            properties.tag_viewport_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        label = str(_vp_detect_progress.get("label", ""))
        if settings is not None and label:
            settings.status = f"Debug edges · {label}"
        try:
            context.window_manager.progress_update(50)
        except Exception:
            pass
        properties.tag_viewport_redraw(context)

        if not _vp_detect_result_box.get("done"):
            return {"PASS_THROUGH"}

        return self._finish_edge_scan(
            context,
            cancelled=bool(_vp_detect_cancel and _vp_detect_cancel.is_set()),
        )


class PM_OT_delete_selected(bpy.types.Operator):
    """Delete the selected VP line."""

    bl_idname = "perspective_match.delete_selected"
    bl_label = "Delete Selected"
    bl_description = "Delete the selected VP line"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        if settings is None:
            return False
        return 0 <= settings.selected_line_index < len(settings.lines)

    def execute(self, context: bpy.types.Context) -> set[str]:
        if not _delete_selected_item(context):
            self.report({"WARNING"}, "Nothing selected to delete")
            return {"CANCELLED"}
        return {"FINISHED"}


class PM_OT_clear_placement(bpy.types.Operator):
    """Clear the origin pick."""

    bl_idname = "perspective_match.clear_placement"
    bl_label = "Clear Origin"
    bl_description = "Clear the picked ground origin"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _pm_controls_camera(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            return {"CANCELLED"}
        settings.origin_is_set = False
        calibration = scene.calibration_from_settings(settings)
        calibration.camera_center = core.default_camera_center(calibration.rotation_w2c)
        scene.apply_camera(context.scene, settings, calibration)
        properties.tag_viewport_redraw(context)
        return {"FINISHED"}


class PM_OT_generate_undistorted(bpy.types.Operator):
    """Generate a transparent expanded undistorted PNG in post-processed/."""

    bl_idname = "perspective_match.generate_undistorted"
    bl_label = "Generate Undistorted Plate"
    bl_description = "Remap the still with current lens distortion and activate it as camera background"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and core.has_lens_distortion(
                settings.division_lambda,
                tuple(settings.brown_conrady),
            )
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            image = distortion.generate_undistorted_plate(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Generated {Path(image.filepath_raw).name}")
        return {"FINISHED"}


class PM_OT_estimate_distortion(bpy.types.Operator):
    """Estimate Fitzgibbon λ once from VP lines and show the undistorted plate."""

    bl_idname = "perspective_match.estimate_distortion"
    bl_label = "Estimate Distortion"
    bl_description = (
        "Estimate division λ from VP lines (≥3 concurrent segments on one axis), "
        "then generate and show an undistorted plate. Does not re-run when lines "
        "change — press again to re-fit. Works with Manual FOV (λ at the locked "
        "focal). When Use Known 3D is on, also polish FOV, principal point, and "
        "camera position from landmark picks without changing VP orientation"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and not scene.uses_adjusted_camera(settings)
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        try:
            distortion.estimate_distortion(context)
        except Exception as error:
            return _report_exception(self, error)
        scene.enter_camera_view(context)
        if settings is not None and abs(settings.division_lambda) > 1.0e-8:
            self.report(
                {"INFO"},
                f"Estimated λ {settings.division_lambda:.5f}",
            )
        elif settings is not None and settings.lambda_saturated:
            self.report({"WARNING"}, "Estimate saturated; pinhole retained")
        else:
            self.report(
                {"WARNING"},
                "Need ≥3 concurrent segments on one axis to estimate λ",
            )
        return {"FINISHED"}


class PM_OT_use_undistorted_plate(bpy.types.Operator):
    """Show the undistorted pinhole plate for the current distortion model."""

    bl_idname = "perspective_match.use_undistorted_plate"
    bl_label = "Undistorted Plate"
    bl_description = (
        "Remap the still with the current lens model (imported D or estimated λ) "
        "and show that pinhole plate as the camera background"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and core.has_lens_distortion(
                settings.division_lambda,
                tuple(settings.brown_conrady),
            )
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            distortion.use_undistorted_plate(context)
        except Exception as error:
            return _report_exception(self, error)
        scene.enter_camera_view(context)
        self.report({"INFO"}, "Viewing undistorted plate")
        return {"FINISHED"}


class PM_OT_use_original_plate(bpy.types.Operator):
    """Show the original still. Imported D stays; estimated λ is cleared."""

    bl_idname = "perspective_match.use_original_plate"
    bl_label = "Original Plate"
    bl_description = (
        "Show the original reference image. Imported Brown–Conrady D is kept "
        "(use Undistorted Plate to remap again). Estimated λ is cleared and the camera is re-solved"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return settings is not None and settings.image is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            distortion.use_original_plate(context)
        except Exception as error:
            return _report_exception(self, error)
        scene.enter_camera_view(context)
        self.report({"INFO"}, "Restored original plate")
        return {"FINISHED"}


class PM_OT_toggle_undistorted(bpy.types.Operator):
    """Toggle original and undistorted camera backgrounds."""

    bl_idname = "perspective_match.toggle_undistorted"
    bl_label = "Toggle Undistorted"
    bl_description = "Switch between the original still and undistorted pinhole plate"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            return {"CANCELLED"}
        try:
            distortion.set_undistorted_view(context, not settings.view_undistorted)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class PM_OT_apply_view_lighting(bpy.types.Operator):
    """Bake display-only exposure/contrast into a post-processed view plate."""

    bl_idname = "perspective_match.apply_view_lighting"
    bl_label = "Apply Lighting"
    bl_description = (
        "Bake exposure and contrast into post-processed/<stem>-pm-view.png "
        "and use it as the camera background"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return settings is not None and settings.image is not None and bool(settings.image_path)

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            image = distortion.apply_view_lighting(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"View plate {Path(image.filepath_raw).name}")
        return {"FINISHED"}


class PM_OT_reset_view_lighting(bpy.types.Operator):
    """Restore the original still and clear baked view lighting."""

    bl_idname = "perspective_match.reset_view_lighting"
    bl_label = "Reset Lighting"
    bl_description = "Discard the view plate and restore the original camera background"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return settings is not None and (
            settings.view_lighting_applied
            or abs(settings.view_exposure) > 1.0e-6
            or abs(settings.view_contrast - 1.0) > 1.0e-6
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            distortion.reset_view_lighting(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "View lighting reset")
        return {"FINISHED"}


def _distance_to_segment(
    point: Vector,
    start: Vector,
    end: Vector,
) -> float:
    delta = end - start
    squared = delta.length_squared
    if squared < 1.0e-8:
        return (point - start).length
    ratio = max(0.0, min(1.0, float((point - start).dot(delta) / squared)))
    return (point - (start + delta * ratio)).length


class PM_OT_interact(bpy.types.Operator):
    """Persistent modal tool for VP lines and origin picking."""

    bl_idname = "perspective_match.interact"
    bl_label = "Perspective Match Tool"
    bl_description = "Interact with Perspective Match guides in camera view"
    # Persistent modal: one undo step for the whole session when Esc finishes.
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(
        items=(
            ("LINE", "VP Lines", "Draw and edit VP lines"),
            ("ORIGIN", "Origin", "Pick the ground origin"),
            ("PP", "Principal Point", "Drag the principal point on the plate"),
            ("LANDMARK", "Landmark", "Pick the active landmark in this match"),
        ),
        default="LINE",
    )

    _drag_kind: str = ""
    _start: tuple[float, float] | None = None
    _original: tuple[float, ...] | None = None
    _edit_index: int = -1
    _edit_endpoint: int = 0
    # Last cursor id applied during hover (avoids redundant cursor_set calls).
    _hover_cursor: str = ""

    @classmethod
    def description(cls, context: bpy.types.Context, properties) -> str:
        mode = getattr(properties, "mode", "LINE")
        if mode == "LINE":
            settings = _session(context)
            if settings is not None and not _required_lines_ready(settings):
                return _lines_needed_message(settings)
            return "Draw and edit vanishing-point lines in camera view"
        if mode == "ORIGIN":
            return "Pick the ground origin on the reference plate"
        if mode == "PP":
            return "Drag the principal point on the plate"
        if mode == "LANDMARK":
            if _tag_snap_enabled(_workspace(context)):
                return (
                    "Pick the active landmark in this match; "
                    "point clicks snap to a nearby AprilTag-like quad"
                )
            return "Pick the active landmark in this match"
        return cls.bl_description

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
        )

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        global _active_interact
        settings = _session(context)
        workspace = _workspace(context)
        if settings is None:
            return {"CANCELLED"}
        if scene.uses_adjusted_camera(settings) and self.mode in {"ORIGIN", "PP"}:
            self.report(
                {"WARNING"},
                "Adjusted Camera controls placement and projection for this match",
            )
            return {"CANCELLED"}

        # Re-click while a live tool runs: switch mode / refresh instead of stacking.
        if _active_interact is not None:
            active = _active_interact
            previous_mode = active.mode
            active.mode = self.mode
            workspace.work_mode = self.mode
            workspace.is_modal = True
            scene.enter_camera_view(context)
            # Leaving VP edit drops selection so handles do not linger in other tools.
            if previous_mode == "LINE" and self.mode != "LINE":
                _clear_vp_line_selection(context)
            # Already inside a modal_set session — refresh stack without nesting.
            _set_interact_cursor(context, self.mode, modal=False)
            active._refresh_cursor_for_event(context, event)
            settings.status = active._status_prompt()
            properties.tag_viewport_redraw(context)
            self.report({"INFO"}, "Perspective Match tool already active")
            return {"CANCELLED"}

        # Flag left behind after a dead modal — clear and start fresh.
        if workspace.is_modal:
            _clear_interact_flags(context)

        scene.enter_camera_view(context)
        workspace.work_mode = self.mode
        workspace.is_modal = True
        self._drag_kind = ""
        self._edit_index = -1
        self._hover_cursor = ""
        _active_interact = self
        context.window_manager.modal_handler_add(self)
        _set_interact_cursor(context, self.mode, modal=True)
        # Button/keymap invoke: apply hover only when already over the plate.
        self._refresh_cursor_for_event(context, event)
        settings.status = self._status_prompt()
        properties.tag_viewport_redraw(context)
        return {"RUNNING_MODAL"}

    def _apply_hover_cursor(self, context: bpy.types.Context, cursor: str) -> None:
        """Set the window cursor (always; Blender no-ops if it is already set)."""
        # Do not skip when self._hover_cursor matches — Blender UI can change the
        # visible cursor (sidebar DEFAULT) without updating our cache.
        window = context.window
        if window is None:
            return
        window.cursor_set(cursor)
        self._hover_cursor = cursor

    def _refresh_cursor_for_event(
        self,
        context: bpy.types.Context,
        event,
    ) -> None:
        """Apply plate hover cursor when the event is over the 3D View window."""
        area, region, space = _view3d_under_event(context, event)
        if area is None or region is None or space is None:
            self._hover_cursor = ""
            return
        with context.temp_override(area=area, region=region, space_data=space):
            self._update_hover_cursor(context, event, region)

    def _update_hover_cursor(
        self,
        context: bpy.types.Context,
        event,
        region,
    ) -> None:
        """VP-line cursors: paint-cross draw, hand-point select, hand ends, closed drag."""
        if self.mode == "LINE":
            # Grab cursor only while actually drawing or dragging an endpoint.
            if self._drag_kind in {"LINE_ENDPOINT", "NEW_LINE"}:
                self._apply_hover_cursor(context, "HAND_CLOSED")
                return
            # Click-select (body hit) is not a grab — show selected-body cursor.
            if self._drag_kind == "SELECT":
                self._apply_hover_cursor(context, "DEFAULT")
                return
            cursor = "PAINT_CROSS"
            if scene.is_camera_view(context):
                settings = _session(context)
                mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
                hit_index, endpoint = self._hit_line(context, mouse)
                if hit_index >= 0:
                    if endpoint:
                        # Control-point handle on the selected line.
                        cursor = "HAND"
                    elif (
                        settings is not None
                        and hit_index == settings.selected_line_index
                    ):
                        # Already selected segment body — no special action.
                        cursor = "DEFAULT"
                    else:
                        # Click will select this unselected line.
                        cursor = "HAND_POINT"
            self._apply_hover_cursor(context, cursor)
            return

        if self._drag_kind:
            return
        cursor = _interact_cursor_for_mode(self.mode, context)
        if scene.is_camera_view(context):
            mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
            if self.mode == "LANDMARK":
                hit_index = _overlay_landmark_hit_index(context, mouse)
                space = _workspace(context)
                if hit_index >= 0 and hit_index != space.active_landmark_index:
                    cursor = "HAND_POINT"
                else:
                    landmark = scene.active_landmark(context)
                    if landmark is not None and landmark.kind == "LINE":
                        if self._hit_landmark_line(context, mouse) != 0:
                            cursor = "HAND"
        self._apply_hover_cursor(context, cursor)

    def _status_prompt(self) -> str:
        if self.mode == "LINE":
            return "Drag VP segments · click one to select · Esc exits"
        if self.mode == "LANDMARK":
            landmark = scene.active_landmark(bpy.context)
            name = landmark.name if landmark is not None else "(none)"
            if landmark is not None and landmark.kind == "LINE":
                return (
                    f"Drag line '{name}' · click another pick to select · Esc exits"
                )
            return (
                f"Click landmark '{name}' · click another pick to select · Esc exits"
            )
        if self.mode == "PP":
            return "Drag principal point (light blue) · Esc exits"
        return "Click the world origin on the ground plane"

    def _finish(self, context: bpy.types.Context, *, cancelled: bool = False) -> set[str]:
        global _active_interact
        if _active_interact is self:
            _active_interact = None
        workspace = _workspace(context)
        settings = _session(context)
        workspace.is_modal = False
        workspace.work_mode = "NONE"
        _restore_interact_cursor(context)
        # Drop leftover header text from older builds that used header_text_set.
        if context.screen is not None:
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.header_text_set(None)
        overlay.clear_preview(context)
        # Always clear VP selection on exit — handles must not linger after Esc.
        _clear_vp_line_selection(context)
        if not cancelled and settings is not None:
            settings.status = "Perspective Match tool finished"
        properties.tag_viewport_redraw(context)
        return {"CANCELLED" if cancelled else "FINISHED"}

    def _event_image_point(self, context, event, region) -> tuple[float, float] | None:
        """Map mouse to image pixels using the WINDOW region under the cursor."""
        # Prefer window→region math; mouse_region_* can be wrong when the modal
        # was started from the sidebar and context.region is stale.
        region_x = float(event.mouse_x - region.x)
        region_y = float(event.mouse_y - region.y)
        return scene.region_to_image(context, region_x, region_y, clamp=False)

    def _hit_line(self, context, mouse: Vector) -> tuple[int, int]:
        settings = _session(context)
        selected = settings.selected_line_index
        # Screen-space grab radii track Retina / UI scale with the overlay handles.
        handle_radius = 12.0 * overlay.ui_scale()
        body_radius = 11.0 * overlay.ui_scale()
        if 0 <= selected < len(settings.lines):
            line = settings.lines[selected]
            for endpoint_index, image_point in enumerate(((line.x1, line.y1), (line.x2, line.y2))):
                screen = scene.image_to_region(context, image_point[0], image_point[1])
                if screen is not None and (screen - mouse).length <= handle_radius:
                    return selected, endpoint_index + 1
        best_index = -1
        best_distance = body_radius
        for index, line in enumerate(settings.lines):
            point_a = scene.image_to_region(context, line.x1, line.y1)
            point_b = scene.image_to_region(context, line.x2, line.y2)
            if point_a is None or point_b is None:
                continue
            distance = _distance_to_segment(mouse, point_a, point_b)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index, 0

    def _begin_drag(self, context, event, image_point: tuple[float, float], region) -> None:
        settings = _session(context)
        mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
        self._start = image_point
        hit_index, endpoint = self._hit_line(context, mouse)
        if hit_index >= 0:
            settings.selected_line_index = hit_index
            settings.active_axis = settings.lines[hit_index].axis
            _log_selected_vp_line(settings, hit_index, self)
            self._edit_index = hit_index
            if endpoint:
                line = settings.lines[hit_index]
                self._drag_kind = "LINE_ENDPOINT"
                self._edit_endpoint = endpoint
                self._original = (line.x1, line.y1, line.x2, line.y2)
            else:
                self._drag_kind = "SELECT"
            properties.tag_viewport_redraw(context)
            return
        settings.selected_line_index = -1
        self._drag_kind = "NEW_LINE"
        overlay.set_preview(context, "LINE", image_point, image_point)

    def _hit_landmark_line(
        self,
        context,
        mouse: Vector,
    ) -> int:
        """Return 1/2 for an endpoint hit on the active LINE observation, else 0."""
        landmark = scene.active_landmark(context)
        root = properties.active_root(context)
        if landmark is None or landmark.kind != "LINE" or root is None:
            return 0
        observation = scene.observation_for_match(landmark, root)
        if observation is None or not observation.is_set:
            return 0
        for endpoint_index, image_point in enumerate(
            ((observation.x, observation.y), (observation.x2, observation.y2))
        ):
            screen = scene.image_to_region(context, image_point[0], image_point[1])
            if screen is not None and (screen - mouse).length <= 12.0 * overlay.ui_scale():
                return endpoint_index + 1
        point_a = scene.image_to_region(context, observation.x, observation.y)
        point_b = scene.image_to_region(context, observation.x2, observation.y2)
        if point_a is None or point_b is None:
            return 0
        if _distance_to_segment(mouse, point_a, point_b) < 11.0 * overlay.ui_scale():
            # Segment body hit — treat as select / no new draw; user can grab ends.
            return -1
        return 0

    def _begin_landmark_line_drag(
        self,
        context,
        event,
        image_point: tuple[float, float],
        region,
    ) -> None:
        mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
        endpoint = self._hit_landmark_line(context, mouse)
        landmark = scene.active_landmark(context)
        root = properties.active_root(context)
        observation = (
            scene.observation_for_match(landmark, root)
            if landmark is not None and root is not None
            else None
        )
        if endpoint > 0 and observation is not None:
            self._drag_kind = "LANDMARK_LINE_ENDPOINT"
            self._edit_endpoint = endpoint
            self._original = (
                observation.x,
                observation.y,
                observation.x2,
                observation.y2,
            )
            properties.tag_viewport_redraw(context)
            return
        if endpoint < 0:
            # Clicked the segment body — keep existing line; don't replace it.
            settings = _session(context)
            if settings is not None:
                settings.status = self._status_prompt()
            self._drag_kind = ""
            properties.tag_viewport_redraw(context)
            return
        self._start = image_point
        self._drag_kind = "LANDMARK_LINE"
        overlay.set_preview(context, "LINE", image_point, image_point)

    def _update_drag(self, context, image_point: tuple[float, float]) -> None:
        settings = _session(context)
        if self._drag_kind in {"NEW_LINE", "LANDMARK_LINE"}:
            overlay.set_preview(context, "LINE", self._start, image_point)
        elif self._drag_kind == "LINE_ENDPOINT":
            line = settings.lines[self._edit_index]
            if self._edit_endpoint == 1:
                line.x1, line.y1 = image_point
            else:
                line.x2, line.y2 = image_point
            properties.tag_viewport_redraw(context)
        elif self._drag_kind == "LANDMARK_LINE_ENDPOINT":
            landmark = scene.active_landmark(context)
            root = properties.active_root(context)
            observation = (
                scene.observation_for_match(landmark, root)
                if landmark is not None and root is not None
                else None
            )
            if observation is None:
                return
            if self._edit_endpoint == 1:
                observation.x, observation.y = image_point
            else:
                observation.x2, observation.y2 = image_point
            properties.tag_viewport_redraw(context)
        elif self._drag_kind == "PP":
            # Keep the current plate and remap stable during the gesture. PP is
            # applied once on release, when an active undistorted view can be
            # rebuilt once instead of falling back to the source on every move.
            overlay.set_preview(context, "PP", image_point, image_point)

    def _complete_landmark_line_drag(
        self,
        context,
        image_point: tuple[float, float],
    ) -> None:
        settings = _session(context)
        if self._drag_kind == "LANDMARK_LINE_ENDPOINT":
            # Endpoint already updated live; just finish the gesture.
            self._drag_kind = ""
            self._original = None
            if settings is not None:
                settings.status = self._status_prompt()
            properties.tag_viewport_redraw(context)
            return
        if self._start is not None:
            length = math.hypot(
                image_point[0] - self._start[0],
                image_point[1] - self._start[1],
            )
            if length >= 8.0:
                try:
                    scene.set_landmark_line_observation(
                        context,
                        self._start,
                        image_point,
                    )
                except Exception as error:
                    self.report({"ERROR"}, str(error))
        self._drag_kind = ""
        self._start = None
        overlay.clear_preview(context)
        if settings is not None:
            settings.status = self._status_prompt()
        properties.tag_viewport_redraw(context)

    def _complete_drag(self, context, image_point: tuple[float, float]) -> None:
        settings = _session(context)
        snap_status: str | None = None
        drag_kind = self._drag_kind
        start = self._start
        edit_index = self._edit_index

        # End the gesture before refine/plate work so the closed-hand cursor
        # cannot stick for the duration of that (sometimes multi-second) work.
        self._drag_kind = ""
        self._start = None
        self._original = None
        overlay.clear_preview(context)
        if drag_kind in {"NEW_LINE", "LINE_ENDPOINT"}:
            self._apply_hover_cursor(context, "WAIT")

        if drag_kind == "NEW_LINE" and start is not None:
            if math.hypot(image_point[0] - start[0], image_point[1] - start[1]) >= 8.0:
                start_point, end_point, snap_status = _maybe_snap_vp_segment(
                    context,
                    start,
                    image_point,
                )
                line = settings.lines.add()
                line.item_id = f"blender-line-{uuid4().hex}"
                line.axis = settings.active_axis
                line.x1, line.y1 = start_point
                line.x2, line.y2 = end_point
                settings.selected_line_index = len(settings.lines) - 1
                _log_selected_vp_line(settings, settings.selected_line_index, self)
                _refine_if_ready(context)
        elif drag_kind == "LINE_ENDPOINT":
            line = settings.lines[edit_index]
            start_point, end_point, snap_status = _maybe_snap_vp_segment(
                context,
                (line.x1, line.y1),
                (line.x2, line.y2),
            )
            line.x1, line.y1 = start_point
            line.x2, line.y2 = end_point
            _refine_if_ready(context)
        elif drag_kind == "PP":
            try:
                scene.set_principal_point(context, image_point, finalize=True)
                distortion.sync_undistorted_plate_after_refine(context)
            except Exception as error:
                self.report({"ERROR"}, str(error))
            if settings is not None:
                settings.status = self._status_prompt()
        if snap_status is not None and settings is not None:
            settings.status = snap_status
            properties.tag_viewport_redraw(context)

    def _cancel_drag(self, context) -> bool:
        if not self._drag_kind:
            return False
        settings = _session(context)
        if self._original is not None and self._drag_kind == "LINE_ENDPOINT":
            if self._edit_index >= 0:
                line = settings.lines[self._edit_index]
                line.x1, line.y1, line.x2, line.y2 = self._original
        if (
            self._original is not None
            and self._drag_kind == "LANDMARK_LINE_ENDPOINT"
        ):
            landmark = scene.active_landmark(context)
            root = properties.active_root(context)
            observation = (
                scene.observation_for_match(landmark, root)
                if landmark is not None and root is not None
                else None
            )
            if observation is not None:
                (
                    observation.x,
                    observation.y,
                    observation.x2,
                    observation.y2,
                ) = self._original
        self._drag_kind = ""
        self._original = None
        self._start = None
        overlay.clear_preview(context)
        return True

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        workspace = _workspace(context)
        settings = _session(context)
        # External clears (match switch / unload) or lost session end the tool.
        if settings is None or not workspace.is_modal or _active_interact is not self:
            return self._finish(context, cancelled=True)
        # Ctrl+Alt+NumPad / arrows must be handled here — modal handlers see keys first.
        slot = _match_slot_from_event(event)
        if slot is not None:
            result = activate_match_by_slot(context, slot, report=self.report)
            # Successful switch already finished this modal via set_active_match.
            if result == {"FINISHED"}:
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        cycle = _match_cycle_from_event(event)
        if cycle is not None:
            result = activate_match_by_delta(context, cycle, report=self.report)
            if result == {"FINISHED"}:
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._cancel_drag(context):
                return {"RUNNING_MODAL"}
            # Completed edits belong to this modal undo step; normal Esc exits cleanly.
            return self._finish(context)
        if event.type in {"DEL", "BACK_SPACE"} and event.value == "PRESS":
            _delete_selected_item(context)
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE", "N"}:
            return {"PASS_THROUGH"}

        area, region, space = _view3d_under_event(context, event)
        # Sidebar / toolbar / other editors: pass through so UI stays clickable
        # (critical with Region Overlap, where N-panel floats over WINDOW).
        if area is None or region is None or space is None:
            if self._drag_kind:
                # Drag left the viewport — cancel the in-progress gesture.
                if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                    self._cancel_drag(context)
                    return {"RUNNING_MODAL"}
                return {"RUNNING_MODAL"}
            # Blender owns the cursor over UI (DEFAULT). Forget our last id so
            # returning to the plate always re-applies PAINT_CROSS / hover state
            # instead of skipping cursor_set as a no-op.
            self._hover_cursor = ""
            return {"PASS_THROUGH"}

        with context.temp_override(area=area, region=region, space_data=space):
            # Allow MMB pan/orbit and idle mouse move without fighting navigation.
            if event.type == "MIDDLEMOUSE":
                return {"PASS_THROUGH"}
            if event.type == "MOUSEMOVE" and not self._drag_kind:
                self._update_hover_cursor(context, event, region)
                return {"PASS_THROUGH"}

            # LMB / active drag: pin back to the match camera if navigation left it.
            if not scene.is_camera_view(context):
                scene.enter_camera_view(context)
            if not scene.is_camera_view(context):
                return {"PASS_THROUGH"}

            image_point = self._event_image_point(context, event, region)
            if event.type == "MOUSEMOVE" and self._drag_kind:
                if image_point is not None:
                    self._update_drag(context, image_point)
                self._update_hover_cursor(context, event, region)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                # Consume LMB so Blender object selection cannot race the tool.
                if image_point is None:
                    settings.status = "Click inside the camera frame"
                    properties.tag_viewport_redraw(context)
                    return {"RUNNING_MODAL"}
                if self.mode == "ORIGIN":
                    try:
                        scene.set_origin(context, image_point)
                    except Exception as error:
                        self.report({"ERROR"}, str(error))
                        return {"RUNNING_MODAL"}
                    return self._finish(context)
                if self.mode == "PP":
                    settings_local = _session(context)
                    self._drag_kind = "PP"
                    self._original = (
                        float(settings_local.cx),
                        float(settings_local.cy),
                    )
                    overlay.set_preview(context, "PP", image_point, image_point)
                    return {"RUNNING_MODAL"}
                if self.mode == "LANDMARK":
                    mouse = Vector(
                        (event.mouse_x - region.x, event.mouse_y - region.y)
                    )
                    hit_index = _overlay_landmark_hit_index(context, mouse)
                    space = _workspace(context)
                    if (
                        hit_index >= 0
                        and hit_index != space.active_landmark_index
                    ):
                        # Clicking another pick selects it instead of moving the active one.
                        _set_active_landmark(context, hit_index)
                        settings.status = self._status_prompt()
                        return {"RUNNING_MODAL"}
                    landmark = scene.active_landmark(context)
                    if landmark is not None and landmark.kind == "LINE":
                        self._begin_landmark_line_drag(
                            context, event, image_point, region
                        )
                        return {"RUNNING_MODAL"}
                    try:
                        snapped_point, snap_note = _maybe_snap_landmark_point(
                            context, image_point
                        )
                        scene.set_landmark_observation(context, snapped_point)
                    except Exception as error:
                        self.report({"ERROR"}, str(error))
                        return {"RUNNING_MODAL"}
                    # Stay in the tool so the same landmark can be picked in other matches
                    # after switching Active Match from the sidebar.
                    settings.status = self._status_prompt()
                    if snap_note:
                        settings.status = f"{settings.status} · {snap_note}"
                    properties.tag_viewport_redraw(context)
                    return {"RUNNING_MODAL"}
                self._begin_drag(context, event, image_point, region)
                self._update_hover_cursor(context, event, region)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self._drag_kind:
                if image_point is not None:
                    if self._drag_kind in {
                        "LANDMARK_LINE",
                        "LANDMARK_LINE_ENDPOINT",
                    }:
                        self._complete_landmark_line_drag(context, image_point)
                    else:
                        self._complete_drag(context, image_point)
                else:
                    self._cancel_drag(context)
                # Refresh after the gesture so release without move is correct.
                self._update_hover_cursor(context, event, region)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context) -> None:
        self._cancel_drag(context)
        self._finish(context, cancelled=True)


class PM_OT_pick_in_active_match(bpy.types.Operator):
    """Keymap wrapper for Pick in Active Match (Ctrl+Cmd+A)."""

    bl_idname = "perspective_match.pick_in_active_match"
    bl_label = "Pick in Active Match"
    bl_description = (
        "Start picking the active landmark on the plate. "
        "Available in camera view while the Perspective Match sidebar tab is open"
    )
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Leave Ctrl+Cmd+A to Blender unless the N-panel tab is selected,
        # a match still is loaded, a landmark is active, and we are in that
        # match's camera view (not a free orbit or another camera).
        if not _perspective_match_sidebar_active(context):
            return False
        if scene.active_landmark(context) is None:
            return False
        return PM_OT_interact.poll(context) and scene.is_camera_view(context)

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        result = bpy.ops.perspective_match.interact("INVOKE_DEFAULT", mode="LANDMARK")
        # Inner operator owns the modal handler; do not return RUNNING_MODAL.
        if result == {"RUNNING_MODAL"}:
            return {"FINISHED"}
        return result


class PM_OT_activate_match_slot(bpy.types.Operator):
    """Activate a match by 1-based index in the name-sorted match list."""

    bl_idname = "perspective_match.activate_match_slot"
    bl_label = "Activate Match Slot"
    bl_description = (
        "Switch to a Perspective Match by slot (Ctrl+Alt+NumPad 1–9). "
        "Re-selecting the current match keeps live zoom/pan. "
        "Cancels any active Draw / Pick tool"
    )
    bl_options = {"UNDO"}

    index: bpy.props.IntProperty(
        name="Slot",
        description="1-based index into the name-sorted match list",
        default=1,
        min=1,
        max=9,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Only claim the shortcut while the Perspective Match sidebar tab is active.
        return _perspective_match_sidebar_active(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        return activate_match_by_slot(context, int(self.index), report=self.report)


class PM_OT_cycle_match(bpy.types.Operator):
    """Activate the previous or next match in the name-sorted list."""

    bl_idname = "perspective_match.cycle_match"
    bl_label = "Cycle Match"
    bl_description = (
        "Switch to the previous or next Perspective Match "
        "(Ctrl+Alt+Arrow). Wraps around the name-sorted list"
    )
    bl_options = {"UNDO"}

    direction: bpy.props.IntProperty(
        name="Direction",
        description="+1 next, -1 previous",
        default=1,
        min=-1,
        max=1,
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _perspective_match_sidebar_active(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        return activate_match_by_delta(
            context, int(self.direction), report=self.report
        )


class PM_OT_new_match_camera(bpy.types.Operator):
    """Create a new Empty + Camera match session and activate it."""

    bl_idname = "perspective_match.new_match_camera"
    bl_label = "New Match Camera"
    bl_description = (
        "Create a new Perspective Match camera session and open the "
        "reference image file dialog. If the previous match has Manual FOV "
        "(or YAML / 1-point K), its intrinsics and distortion are copied"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        previous = properties.active_root(context)
        previous_session = previous.pm_session if previous is not None else None
        copied_manual_k = bool(
            previous_session is not None
            and (previous_session.lock_focal or previous_session.vp_mode == "1")
            and float(previous_session.fx) > 1.0
        )
        try:
            root = scene.create_match_camera(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if copied_manual_k:
            self.report({"INFO"}, f"Created {root.name} (manual K copied)")
        else:
            self.report({"INFO"}, f"Created {root.name}")
        # Chain into Open Image so a new match always starts with a still.
        bpy.ops.perspective_match.load_image("INVOKE_DEFAULT")
        return {"FINISHED"}


class PM_OT_bulk_create_matches(bpy.types.Operator):
    """Create a match camera for each still in a folder."""

    bl_idname = "perspective_match.bulk_create_matches"
    bl_label = "Bulk Create"
    bl_description = (
        "Create a Perspective Match for each still in a folder. Skips images "
        "that already have a match. Copies the complete Manual FOV / YAML / "
        "1-point camera model from the active match onto every new one"
    )
    bl_options = {"REGISTER", "UNDO"}

    directory: bpy.props.StringProperty(
        name="Folder",
        description="Folder of stills to turn into match cameras",
        subtype="DIR_PATH",
    )
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", options={"HIDDEN"})
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        settings = properties.active_session(context)
        if settings is not None and settings.image_path:
            parent = Path(bpy.path.abspath(settings.image_path)).expanduser().parent
            if parent.is_dir():
                self.directory = str(parent)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        folder = (self.directory or "").strip()
        if not folder:
            self.report({"ERROR"}, "Choose a folder of stills")
            return {"CANCELLED"}
        try:
            created, skipped = scene.bulk_create_match_cameras(context, folder)
        except Exception as error:
            return _report_exception(self, error)
        if created == 0 and skipped == 0:
            self.report({"WARNING"}, "No stills in that folder")
            return {"CANCELLED"}
        created_label = "match" if created == 1 else "matches"
        skipped_bit = f", {skipped} skipped" if skipped else ""
        self.report({"INFO"}, f"Created {created} {created_label}{skipped_bit}")
        return {"FINISHED"}


class PM_OT_reference_image_label(bpy.types.Operator):
    """Filename label whose tooltip is the full reference image path."""

    bl_idname = "perspective_match.reference_image_label"
    bl_label = "Reference Image"
    bl_description = "Full path of the reference image"
    bl_options = {"INTERNAL"}

    path: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def description(cls, _context, properties) -> str:
        path = getattr(properties, "path", "") or ""
        return path if path else "Full path of the reference image"

    def execute(self, _context: bpy.types.Context) -> set[str]:
        # Label-only control — click is a no-op; tooltip carries the path.
        return {"FINISHED"}


class PM_OT_report_info(bpy.types.Operator):
    """Write a line into the Info editor (used when optional OpenCV is missing)."""

    bl_idname = "perspective_match.report_info"
    bl_label = "Perspective Match Info"
    bl_options = {"INTERNAL", "REGISTER"}

    message: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, _context: bpy.types.Context) -> set[str]:
        text = (self.message or "").strip()
        if text:
            self.report({"INFO"}, text[:255])
        return {"FINISHED"}


class PM_OT_unload_match(bpy.types.Operator):
    """Unload the active match session from the UI without deleting objects."""

    bl_idname = "perspective_match.unload_match"
    bl_label = "Unload"
    bl_description = "Unload the active Perspective Match session from the sidebar"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return properties.active_root(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene.unload_match(context)
        self.report({"INFO"}, "Perspective Match session unloaded")
        return {"FINISHED"}


class PM_OT_delete_match(bpy.types.Operator):
    """Delete the active match hierarchy after confirmation."""

    bl_idname = "perspective_match.delete_match"
    bl_label = "Delete Match"
    bl_description = (
        "Delete the active Perspective Match (collection, Origin, Camera) "
        "and its landmark picks"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return properties.active_root(context) is not None

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        root = properties.active_root(context)
        if root is None:
            return {"CANCELLED"}
        label = scene.match_prefix(root)
        return context.window_manager.invoke_confirm(
            self,
            event,
            title="Delete Match?",
            message=f"Delete {label} and its camera? This cannot be undone easily.",
            confirm_text="Delete",
            icon="WARNING",
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            prefix = scene.delete_match(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Deleted {prefix}")
        return {"FINISHED"}


class PM_OT_rename_match(bpy.types.Operator):
    """Rename the active match hierarchy (collection, Origin, Camera)."""

    bl_idname = "perspective_match.rename_match"
    bl_label = "Rename Match"
    bl_description = (
        "Rename the active Perspective Match after it was created. "
        "Defaults from the image stem when you open a still; change it anytime"
    )
    bl_options = {"REGISTER", "UNDO"}

    new_name: bpy.props.StringProperty(
        name="Name",
        description="Stored as PM_<name> (collection / Origin / Camera)",
        default="",
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return properties.active_root(context) is not None

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        root = properties.active_root(context)
        prefix = scene.match_prefix(root)
        # Prefill without the PM_ prefix so the dialog shows a short label.
        self.new_name = prefix[3:] if prefix.startswith("PM_") else prefix
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context: bpy.types.Context) -> None:
        layout = self.layout
        # Focus the name field and select its text so the next keystroke replaces it.
        layout.activate_init = True
        layout.prop(self, "new_name")
        layout.label(text="Saved as PM_<name>_Origin", icon="INFO")

    def execute(self, context: bpy.types.Context) -> set[str]:
        root = properties.active_root(context)
        if root is None:
            return {"CANCELLED"}
        try:
            renamed = scene.rename_match(context, root, self.new_name)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Renamed to {scene.match_prefix(renamed)}")
        return {"FINISHED"}


class PM_OT_reload(bpy.types.Operator):
    """Dev helper: unregister, reload modules from disk, and register again."""

    bl_idname = "perspective_match.reload"
    bl_label = "Reload Perspective Match"
    bl_description = (
        "Reload this extension from disk after the current UI event finishes "
        "(safer than System → Reload Scripts for panel/operator edits). "
        "Only available when the extension is a linked git checkout."
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, _context: bpy.types.Context) -> bool:
        from .. import is_dev_install

        return is_dev_install()

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Never unregister from inside this operator — that can SIGSEGV Blender.
        # Schedule the tear-down on a timer so this execute can return first.
        # Operators live under ui/; schedule_reload is on the extension root package.
        from .. import schedule_reload

        if not schedule_reload():
            self.report({"WARNING"}, "Perspective Match reload already queued")
            return {"CANCELLED"}
        self.report({"INFO"}, "Perspective Match reload queued")
        return {"FINISHED"}


class PM_OT_select_overlay_landmark(bpy.types.Operator):
    """Select a landmark by clicking its pick on the plate (sidebar tab open)."""

    bl_idname = "perspective_match.select_overlay_landmark"
    bl_label = "Select Overlay Landmark"
    bl_description = (
        "Select a landmark by clicking its pick on the reference plate. "
        "Active while the Perspective Match sidebar tab is open"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Live Draw / Pick tools own LMB; the modal handles select there.
        if _active_interact is not None:
            return False
        if not _perspective_match_sidebar_active(context):
            return False
        settings = _session(context)
        space = _workspace(context)
        return (
            settings is not None
            and settings.image is not None
            and space.show_landmark_overlay
            and len(space.landmarks) > 0
        )

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        # Leave modified clicks to Blender (Shift/Ctrl select, etc.).
        if event.shift or event.ctrl or event.alt or event.oskey:
            return {"PASS_THROUGH"}
        area, region, space_data = _view3d_under_event(context, event)
        if area is None or region is None or space_data is None:
            return {"PASS_THROUGH"}
        with context.temp_override(
            area=area, region=region, space_data=space_data
        ):
            if not scene.is_camera_view(context):
                return {"PASS_THROUGH"}
            mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
            index = _overlay_landmark_hit_index(context, mouse)
            if index < 0:
                return {"PASS_THROUGH"}
            _set_active_landmark(context, index)
        return {"FINISHED"}


class PM_OT_add_landmark(bpy.types.Operator):
    """Create a new named sync landmark."""

    bl_idname = "perspective_match.add_landmark"
    bl_label = "Add Landmark"
    bl_description = "Add a point or line landmark that can be picked in multiple stills"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(
        name="Kind",
        items=properties.LANDMARK_KIND_ITEMS,
        default="POINT",
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        landmark = space.landmarks.add()
        landmark.item_id = f"landmark-{uuid4().hex}"
        landmark.kind = self.kind
        if self.kind == "LINE":
            landmark.name = f"Line {len(space.landmarks)}"
        else:
            landmark.name = f"Landmark {len(space.landmarks)}"
        properties.ensure_landmark_creation_indices(space)
        space.active_landmark_index = len(space.landmarks) - 1
        properties.tag_sync_ui_redraw(context)
        return {"FINISHED"}


class PM_OT_add_landmarks_from_selected(bpy.types.Operator):
    """Create Known 3D landmarks from the current object selection."""

    bl_idname = "perspective_match.add_landmarks_from_selected"
    bl_label = "Landmarks from Selected"
    bl_description = (
        "Create one sync landmark per selected object as Known 3D and "
        "auto-project into the anchor still"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return bool(context.selected_objects)

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        created_landmarks = []
        for obj in context.selected_objects:
            if properties.is_match_root(obj):
                continue
            if obj.type == "CAMERA":
                continue
            landmark = space.landmarks.add()
            landmark.item_id = f"landmark-{uuid4().hex}"
            landmark.name = obj.name
            landmark.known_object = obj
            created_landmarks.append(landmark)
        if not created_landmarks:
            self.report({"ERROR"}, "Select Empties or meshes (not match cameras)")
            return {"CANCELLED"}
        properties.ensure_landmark_creation_indices(space)
        space.active_landmark_index = len(space.landmarks) - 1
        projected, target = scene.auto_project_known_landmarks(
            context,
            created_landmarks,
        )
        properties.tag_sync_ui_redraw(context)
        if projected and target is not None:
            self.report(
                {"INFO"},
                f"Added {len(created_landmarks)} Known 3D · "
                f"projected {projected} into {target.name}",
            )
        elif projected == 0:
            self.report(
                {"WARNING"},
                f"Added {len(created_landmarks)} Known 3D · "
                "could not project into the anchor (set Anchor, solve camera, "
                "check Empties are in front of the camera)",
            )
        else:
            self.report({"INFO"}, f"Added {len(created_landmarks)} Known 3D landmark(s)")
        return {"FINISHED"}


class PM_OT_landmark_use_selected(bpy.types.Operator):
    """Assign the active object as Known 3D for the active landmark."""

    bl_idname = "perspective_match.landmark_use_selected"
    bl_label = "Use Selected as Known 3D"
    bl_description = (
        "Link the active object’s world location to this landmark as fixed "
        "shared 3D (then pick the feature in other stills in 2D)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        landmark = scene.active_landmark(context)
        obj = context.active_object
        return (
            landmark is not None
            and obj is not None
            and not properties.is_match_root(obj)
            and obj.type != "CAMERA"
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        landmark = scene.active_landmark(context)
        obj = context.active_object
        if landmark is None or obj is None:
            return {"CANCELLED"}
        landmark.known_object = obj
        if not landmark.name or landmark.name.startswith("Landmark "):
            landmark.name = obj.name
        projected, target = scene.auto_project_known_landmarks(context, [landmark])
        properties.tag_sync_ui_redraw(context)
        if projected and target is not None:
            self.report(
                {"INFO"},
                f"Known 3D ← {obj.name} · projected into {target.name}",
            )
        else:
            self.report({"INFO"}, f"Known 3D ← {obj.name}")
        return {"FINISHED"}


class PM_OT_landmark_clear_known(bpy.types.Operator):
    """Clear the Known 3D object link on the active landmark."""

    bl_idname = "perspective_match.landmark_clear_known"
    bl_label = "Clear Known 3D"
    bl_description = "Remove the Blender object link from the active landmark"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        landmark = scene.active_landmark(context)
        return landmark is not None and (
            landmark.known_object is not None or landmark.known_object_b is not None
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        landmark = scene.active_landmark(context)
        if landmark is None:
            return {"CANCELLED"}
        landmark.known_object = None
        landmark.known_object_b = None
        properties.tag_viewport_redraw(context)
        return {"FINISHED"}


class PM_OT_use_selected_mirror(bpy.types.Operator):
    """Assign the active object as the scene Mirror Empty."""

    bl_idname = "perspective_match.use_selected_mirror"
    bl_label = "Use Selected as Mirror Empty"
    bl_description = (
        "Use the active object as the shared mirror plane for every "
        "Is Mirror Of pair"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type != "CAMERA"

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        obj = context.active_object
        if obj is None:
            return {"CANCELLED"}
        space.mirror_object = obj
        properties.tag_viewport_redraw(context)
        return {"FINISHED"}


class PM_OT_clear_mirror(bpy.types.Operator):
    """Clear the scene Mirror Empty."""

    bl_idname = "perspective_match.clear_mirror"
    bl_label = "Clear Mirror Empty"
    bl_description = "Remove the shared mirror plane Empty"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        space = properties.workspace(context)
        return getattr(space, "mirror_object", None) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        space.mirror_object = None
        properties.tag_viewport_redraw(context)
        return {"FINISHED"}


def _named_mirror_partner(context: bpy.types.Context, landmark):
    """Other point landmark whose name is this one's left/right swap."""
    wanted = suggested_mirror_partner_name(landmark.name or "")
    if not wanted or not landmark.item_id:
        return None
    wanted_key = wanted.casefold()
    space = properties.workspace(context)
    for other in space.landmarks:
        if other.kind != "POINT":
            continue
        if other.item_id == landmark.item_id or not other.item_id:
            continue
        if (other.name or "").strip().casefold() == wanted_key:
            return other
    return None


class PM_OT_guess_mirror_partner(bpy.types.Operator):
    """Set Is Mirror Of from a left/right name swap."""

    bl_idname = "perspective_match.guess_mirror_partner"
    bl_label = "Guess Mirror Partner"
    bl_description = (
        "Set Is Mirror Of from a matching landmark whose name ends with "
        "the opposite of this one's left/right suffix"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        landmark = scene.active_landmark(context)
        if landmark is None or landmark.kind != "POINT":
            return False
        partner = _named_mirror_partner(context, landmark)
        return partner is not None and landmark.mirror_of != partner.item_id

    def execute(self, context: bpy.types.Context) -> set[str]:
        landmark = scene.active_landmark(context)
        if landmark is None or landmark.kind != "POINT":
            return {"CANCELLED"}
        partner = _named_mirror_partner(context, landmark)
        if partner is None:
            self.report(
                {"WARNING"},
                "No point landmark whose name swaps left/right with this one",
            )
            return {"CANCELLED"}
        landmark.mirror_of = partner.item_id
        properties.tag_viewport_redraw(context)
        self.report({"INFO"}, f"Is Mirror Of set to '{partner.name}'")
        return {"FINISHED"}


class PM_OT_remove_landmark(bpy.types.Operator):
    """Delete the active sync landmark."""

    bl_idname = "perspective_match.remove_landmark"
    bl_label = "Remove Landmark"
    bl_description = "Delete the active landmark and its picks"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        space = properties.workspace(context)
        return 0 <= space.active_landmark_index < len(space.landmarks)

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        index = space.active_landmark_index
        removed_id = space.landmarks[index].item_id
        for landmark in space.landmarks:
            if getattr(landmark, "mirror_of", "NONE") == removed_id:
                landmark.mirror_of = "NONE"
        space.landmarks.remove(index)
        space.active_landmark_index = min(index, len(space.landmarks) - 1)
        scene.sync_landmark_empties(context)
        properties.tag_sync_ui_redraw(context)
        return {"FINISHED"}


class PM_OT_find_apriltag_landmarks(bpy.types.Operator):
    """Detect supported AprilTag markers and assign them to sync landmarks."""

    bl_idname = "perspective_match.find_apriltag_landmarks"
    bl_label = "Find AprilTags"
    bl_description = (
        "Scan the active match still for AprilTag 25h9 and 36h10 markers. "
        "Assign each tag centre to a family-qualified landmark "
        "(create the landmark when missing)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        caps = opencv_support.cached_capabilities()
        if caps is None or not caps.apriltags:
            return False
        settings = properties.active_session(context)
        return (
            settings is not None
            and settings.image is not None
            and properties.active_root(context) is not None
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            result = apriltag_detect.find_and_assign_apriltags(context)
        except apriltag_detect.AprilTagDependencyError as error:
            return _report_exception(self, error)
        except Exception as error:
            return _report_exception(self, error)

        settings = properties.active_session(context)
        if result.detected == 0:
            message = "No AprilTag 25h9 or 36h10 markers found in this still"
            if settings is not None:
                settings.status = message
            self.report({"WARNING"}, message)
            return {"FINISHED"}

        parts = [f"Found {result.detected}"]
        if result.updated:
            parts.append(f"updated {result.updated}")
        if result.created:
            parts.append(f"created {result.created}")
        if result.skipped:
            parts.append(f"skipped {result.skipped} line")
        message = " · ".join(parts)
        if settings is not None:
            settings.status = message
            settings.error = ""
        self.report({"INFO"}, message)
        return {"FINISHED"}


class PM_OT_duplicate_landmark(bpy.types.Operator):
    """Duplicate the active landmark without picks, Known 3D links, or solved positions."""

    bl_idname = "perspective_match.duplicate_landmark"
    bl_label = "Duplicate Landmark"
    bl_description = (
        "Duplicate the active landmark (keeps type, On Ground, Use in Sync, "
        "Sync Weight). "
        "Clears picks, Known 3D links, parallel links, mirror links, and solved "
        "positions. "
        "Sets Pick Confidence from the source landmark's picks when available"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        space = _workspace(context)
        return 0 <= space.active_landmark_index < len(space.landmarks)

    def execute(self, context: bpy.types.Context) -> set[str]:
        space = _workspace(context)
        source = scene.active_landmark(context)
        if source is None:
            return {"CANCELLED"}

        # Prefer confidence from the active match pick, else any set pick.
        pick_confidence = None
        active_root = properties.active_root(context)
        for observation in source.observations:
            if not observation.is_set:
                continue
            if active_root is not None and observation.match_root == active_root:
                pick_confidence = observation.confidence
                break
            if pick_confidence is None:
                pick_confidence = observation.confidence

        duplicate = space.landmarks.add()
        duplicate.item_id = f"landmark-{uuid4().hex}"
        duplicate.name = f"{source.name} copy"
        duplicate.kind = source.kind
        duplicate.on_ground = bool(source.on_ground)
        duplicate.use_in_sync = bool(source.use_in_sync)
        duplicate.sync_weight = float(getattr(source, "sync_weight", 1.0))
        # Intentionally leave Known 3D / parallel / observations / positions empty.
        properties.ensure_landmark_creation_indices(space)
        space.active_landmark_index = len(space.landmarks) - 1
        if pick_confidence is not None:
            space.landmark_pick_confidence = pick_confidence
        properties.tag_sync_ui_redraw(context)
        self.report({"INFO"}, f"Duplicated '{source.name}' → '{duplicate.name}'")
        return {"FINISHED"}


class PM_OT_clear_landmark_observation(bpy.types.Operator):
    """Clear the active match's pick on the active landmark."""

    bl_idname = "perspective_match.clear_landmark_observation"
    bl_label = "Clear Pick"
    bl_description = "Remove this match's observation from the active landmark"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        landmark = scene.active_landmark(context)
        root = properties.active_root(context)
        if landmark is None or root is None:
            return False
        return scene.observation_for_match(landmark, root) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        if not scene.clear_landmark_observation_for_active(context):
            self.report({"WARNING"}, "No pick to clear")
            return {"CANCELLED"}
        return {"FINISHED"}


class PM_OT_solve_sync(bpy.types.Operator):
    """Register non-anchor matches to the anchor via the landmark graph."""

    bl_idname = "perspective_match.solve_sync"
    bl_label = "Solve Sync"
    bl_description = (
        "Register non-anchor match Empties from 2D landmarks and/or Known 3D "
        "Blender objects. On Ground / Known 3D pin absolute scale vs the anchor. "
        "With locked K, 4+ shared On Ground picks across 3+ images can "
        "initialize orientation without VPs. Matches without Pick Origin get "
        "one from their first On Ground pick"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if diagnose_sync_is_running():
            return False
        anchor = properties.anchor_root(context)
        return (
            anchor is not None
            and properties.match_sync_enabled(anchor)
            and len(properties.iter_sync_enabled_roots()) >= 2
            and len(properties.workspace(context).landmarks) >= 1
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        workspace = _workspace(context)
        try:
            result = scene.solve_and_apply_sync(context)
        except Exception as error:
            message = (
                f"Sync internal error (missing {error})"
                if isinstance(error, KeyError)
                else str(error)
            )
            workspace.sync_status = message
            return _report_exception(self, error)
        # Clear a previous failure banner on success.
        settings = properties.active_session(context)
        if settings is not None:
            settings.error = ""
        self.report({"INFO"}, result.message)
        return {"FINISHED"}


class PM_OT_diagnose_sync(bpy.types.Operator):
    """Report sync quality without moving match Empties."""

    bl_idname = "perspective_match.diagnose_sync"
    bl_label = "Diagnose"
    bl_description = (
        "Run the sync solver and list per-landmark RMSE plus Known 3D "
        "consistency checks without applying a pose. Auto-sets missing "
        "origins and, with locked K plus 4+ shared On Ground picks across 3+ "
        "images, can initialize orientation like Solve Sync. Runs in the "
        "background — Esc or Cancel to stop"
    )
    bl_options = {"REGISTER"}

    _timer = None
    _prep = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if diagnose_sync_is_running() or lens_refine_is_running():
            return False
        return PM_OT_solve_sync.poll(context)

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        global _diagnose_sync_cancel, _diagnose_sync_running
        global _diagnose_sync_progress, _diagnose_sync_result_box

        workspace = _workspace(context)
        try:
            prep = scene.prepare_diagnose_sync(context)
        except Exception as error:
            workspace.sync_status = str(error)
            return _report_exception(self, error)

        cancel_event = threading.Event()
        result_box: dict = {"done": False}
        progress_state = {"step": 0, "total": 6, "label": "Starting…"}
        with _diagnose_sync_lock:
            if _diagnose_sync_running:
                self.report({"WARNING"}, "Diagnose already running")
                return {"CANCELLED"}
            _diagnose_sync_cancel = cancel_event
            _diagnose_sync_running = True
            _diagnose_sync_progress = progress_state
            _diagnose_sync_result_box = result_box

        self._prep = prep
        workspace.sync_status = "Diagnose running… Esc to cancel"
        properties.tag_viewport_redraw(context)

        def _on_progress(step: int, total: int, label: str) -> None:
            progress_state["step"] = int(step)
            progress_state["total"] = max(int(total), 1)
            progress_state["label"] = str(label)

        def _worker() -> None:
            from ..core import sync as sync_module

            try:
                result_box["result"] = scene.run_diagnose_sync(
                    prep,
                    cancel_check=cancel_event.is_set,
                    progress_callback=_on_progress,
                )
            except sync_module.SyncCancelled:
                result_box["cancelled"] = True
            except Exception as error:
                result_box["error"] = error
            finally:
                result_box["done"] = True

        threading.Thread(
            target=_worker,
            name="PM-DiagnoseSync",
            daemon=True,
        ).start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, 6)
        self._timer = window_manager.event_timer_add(0.1, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_job(self, context: bpy.types.Context, *, cancelled: bool) -> set[str]:
        global _diagnose_sync_cancel, _diagnose_sync_running

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            window_manager.progress_end()
        except Exception:
            pass

        workspace = _workspace(context)
        result_box = _diagnose_sync_result_box
        with _diagnose_sync_lock:
            _diagnose_sync_running = False
            _diagnose_sync_cancel = None

        if result_box.get("error") is not None:
            error = result_box["error"]
            workspace.sync_status = str(error)
            properties.tag_viewport_redraw(context)
            return _report_exception(
                self,
                error if isinstance(error, Exception) else Exception(str(error)),
            )

        if cancelled or result_box.get("cancelled"):
            workspace.sync_status = "Diagnose cancelled"
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, "Diagnose cancelled")
            return {"CANCELLED"}

        result = result_box.get("result")
        if result is None:
            workspace.sync_status = "Diagnose cancelled"
            properties.tag_viewport_redraw(context)
            return {"CANCELLED"}

        try:
            scene.apply_diagnose_sync_result(context, self._prep, result)
        except Exception as error:
            workspace.sync_status = str(error)
            return _report_exception(self, error)
        settings = properties.active_session(context)
        if settings is not None:
            settings.error = ""
        self.report({"INFO"} if result.success else {"WARNING"}, workspace.sync_status)
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        workspace = _workspace(context)
        if event.type == "ESC" and event.value == "PRESS":
            request_diagnose_sync_cancel()
            workspace.sync_status = "Cancelling Diagnose…"
            properties.tag_viewport_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = _diagnose_sync_progress
        total = max(int(progress.get("total", 1)), 1)
        step = min(int(progress.get("step", 0)), total)
        label = str(progress.get("label", ""))
        workspace.sync_status = f"Diagnose {step}/{total} · {label}"
        try:
            context.window_manager.progress_update(step)
        except Exception:
            pass
        properties.tag_viewport_redraw(context)
        if not _diagnose_sync_result_box.get("done"):
            return {"PASS_THROUGH"}
        return self._finish_job(
            context,
            cancelled=bool(
                _diagnose_sync_cancel and _diagnose_sync_cancel.is_set()
            ),
        )

    def cancel(self, context: bpy.types.Context) -> None:
        request_diagnose_sync_cancel()
        for _ in range(50):
            if _diagnose_sync_result_box.get("done"):
                break
            time.sleep(0.02)
        self._finish_job(context, cancelled=True)

    def execute(self, context: bpy.types.Context) -> set[str]:
        workspace = _workspace(context)
        try:
            result = scene.diagnose_sync(context)
        except Exception as error:
            workspace.sync_status = str(error)
            return _report_exception(self, error)
        level = {"INFO"} if result.success else {"WARNING"}
        self.report(level, workspace.sync_status)
        return {"FINISHED"}


class PM_OT_cancel_diagnose_sync(bpy.types.Operator):
    """Stop the running Diagnose background job."""

    bl_idname = "perspective_match.cancel_diagnose_sync"
    bl_label = "Cancel Diagnose"
    bl_description = "Cancel the running Diagnose solve (Esc also works)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return diagnose_sync_is_running()

    def execute(self, context: bpy.types.Context) -> set[str]:
        if not request_diagnose_sync_cancel():
            return {"CANCELLED"}
        workspace = _workspace(context)
        workspace.sync_status = "Cancelling Diagnose…"
        properties.tag_viewport_redraw(context)
        self.report({"INFO"}, "Cancelling Diagnose…")
        return {"FINISHED"}


class PM_OT_refine_lenses(bpy.types.Operator):
    """Adjust match focals so landmark sync and VP lines agree better."""

    bl_idname = "perspective_match.refine_lenses"
    bl_label = "Refine Lenses"
    bl_description = (
        "Search focal length so landmark sync and VP lines agree better, "
        "then Solve Sync. With Same Lens, one scale is applied "
        "to every still (works without VP lines). Otherwise each unlocked "
        "match is searched on its own. Runs in the background — Esc or "
        "Cancel Refine to stop"
    )
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _prep = None
    _progress_max = 1

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if lens_refine_is_running() or diagnose_sync_is_running():
            return False
        return PM_OT_solve_sync.poll(context)

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        global _lens_refine_cancel, _lens_refine_running, _lens_refine_result_box
        global _lens_refine_progress

        workspace = _workspace(context)
        try:
            prep = scene.prepare_lens_refine(context)
        except Exception as error:
            message = str(error)
            workspace.sync_status = message
            return _report_exception(self, error)

        from ..core import lens_refine

        if prep.share_lens:
            search_count = len(prep.lens_inputs)
        else:
            search_count = sum(1 for item in prep.lens_inputs if not item.freeze_focal)
        total = lens_refine.estimate_refine_evaluation_count(
            search_count,
            share_lens=bool(prep.share_lens),
        )
        cancel_event = threading.Event()
        result_box: dict = {"done": False}
        progress_state = {"step": 0, "total": max(total, 1), "label": "Starting…"}

        with _lens_refine_lock:
            if _lens_refine_running:
                self.report({"WARNING"}, "Refine Lenses already running")
                return {"CANCELLED"}
            _lens_refine_cancel = cancel_event
            _lens_refine_running = True
            _lens_refine_result_box = result_box
            _lens_refine_progress = progress_state

        self._prep = prep
        self._progress_max = max(total, 1)
        workspace.lens_refine_progress = 0.0
        workspace.sync_status = "Refine Lenses running… Esc to cancel"
        properties.tag_viewport_redraw(context)

        def _on_progress(step: int, total_steps: int, label: str) -> None:
            progress_state["step"] = int(step)
            progress_state["total"] = max(int(total_steps), 1)
            progress_state["label"] = str(label)

        def _worker() -> None:
            try:
                refine_result = lens_refine.refine_lenses_from_landmarks(
                    prep.lens_inputs,
                    prep.observations,
                    anchor_id=prep.anchor_id,
                    known_world=prep.known_world,
                    line_observations=prep.line_observations,
                    known_lines=prep.known_lines,
                    parallel_pairs=prep.parallel_pairs,
                    fx_span=prep.fx_span,
                    lock_rotation=prep.lock_rotation,
                    lock_translation=prep.lock_translation,
                    share_lens=prep.share_lens,
                    ground_slack=prep.ground_slack,
                    known_3d_slack=prep.known_3d_slack,
                    mirror_pairs=prep.mirror_pairs,
                    mirror_plane=prep.mirror_plane,
                    mirror_slack=prep.mirror_slack,
                    cancel_check=cancel_event.is_set,
                    progress_callback=_on_progress,
                )
                result_box["result"] = refine_result
            except Exception as error:
                result_box["error"] = error
            finally:
                result_box["done"] = True

        thread = threading.Thread(
            target=_worker,
            name="PM-RefineLenses",
            daemon=True,
        )
        thread.start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, self._progress_max)
        self._timer = window_manager.event_timer_add(0.1, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_job(self, context: bpy.types.Context, *, cancelled: bool) -> set[str]:
        global _lens_refine_cancel, _lens_refine_running

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            window_manager.progress_end()
        except Exception:
            pass

        workspace = _workspace(context)
        result_box = _lens_refine_result_box
        prep = self._prep

        with _lens_refine_lock:
            _lens_refine_running = False
            _lens_refine_cancel = None

        if result_box.get("error") is not None:
            error = result_box["error"]
            workspace.sync_status = str(error)
            workspace.lens_refine_progress = 0.0
            properties.tag_viewport_redraw(context)
            return _report_exception(self, error if isinstance(error, Exception) else Exception(str(error)))

        refine_result = result_box.get("result")
        if refine_result is None:
            workspace.sync_status = "Lens refine cancelled"
            workspace.lens_refine_progress = 0.0
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, "Lens refine cancelled")
            return {"CANCELLED"}

        if refine_result.cancelled or cancelled:
            # Discard partial lens changes — leave the scene as it was.
            workspace.sync_status = refine_result.message
            workspace.lens_refine_progress = 0.0
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, refine_result.message)
            return {"CANCELLED"}

        try:
            _refine, sync_result = scene.apply_lens_refine_result(
                context,
                refine_result,
                prep.root_by_name,
            )
        except Exception as error:
            return _report_exception(self, error)

        settings = properties.active_session(context)
        if settings is not None:
            settings.error = ""
        success = refine_result.improved or (
            sync_result is not None and sync_result.success
        )
        self.report(
            {"INFO"} if success else {"WARNING"},
            workspace.sync_status or refine_result.message,
        )
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        workspace = _workspace(context)
        if event.type in {"ESC"} and event.value == "PRESS":
            request_lens_refine_cancel()
            workspace.sync_status = "Cancelling lens refine…"
            properties.tag_viewport_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = _lens_refine_progress
        total = max(int(progress.get("total", 1)), 1)
        step = int(progress.get("step", 0))
        label = str(progress.get("label", ""))
        workspace.lens_refine_progress = min(max(step / total, 0.0), 1.0)
        if label:
            workspace.sync_status = f"Refine Lenses {step}/{total} · {label}"
        try:
            context.window_manager.progress_update(min(step, total))
        except Exception:
            pass
        properties.tag_viewport_redraw(context)

        if not _lens_refine_result_box.get("done"):
            return {"PASS_THROUGH"}

        return self._finish_job(
            context,
            cancelled=bool(_lens_refine_cancel and _lens_refine_cancel.is_set()),
        )

    def cancel(self, context: bpy.types.Context) -> None:
        request_lens_refine_cancel()
        # Wait briefly so the worker can exit cleanly before apply is skipped.
        for _ in range(50):
            if _lens_refine_result_box.get("done"):
                break
            time.sleep(0.02)
        self._finish_job(context, cancelled=True)

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Scripting / redo: run blocking on the main thread.
        workspace = _workspace(context)
        try:
            refine_result, sync_result = scene.refine_lenses_and_sync(context)
        except Exception as error:
            message = str(error)
            workspace.sync_status = message
            return _report_exception(self, error)
        settings = properties.active_session(context)
        if settings is not None:
            settings.error = ""
        success = refine_result.improved or (
            sync_result is not None and sync_result.success
        )
        self.report(
            {"INFO"} if success else {"WARNING"},
            workspace.sync_status or refine_result.message,
        )
        return {"FINISHED"}


class PM_OT_cancel_refine_lenses(bpy.types.Operator):
    """Stop the running Refine Lenses background job."""

    bl_idname = "perspective_match.cancel_refine_lenses"
    bl_label = "Cancel Refine"
    bl_description = "Cancel the running Refine Lenses search (Esc also works)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return lens_refine_is_running()

    def execute(self, context: bpy.types.Context) -> set[str]:
        if not request_lens_refine_cancel():
            return {"CANCELLED"}
        workspace = _workspace(context)
        workspace.sync_status = "Cancelling lens refine…"
        properties.tag_viewport_redraw(context)
        self.report({"INFO"}, "Cancelling Refine Lenses…")
        return {"FINISHED"}


class PM_OT_clear_sync(bpy.types.Operator):
    """Reset match Empty sync transforms and landmark Empties."""

    bl_idname = "perspective_match.clear_sync"
    bl_label = "Clear Sync"
    bl_description = "Reset all match root Empties to identity and clear landmark Empties"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return not diagnose_sync_is_running()

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene.clear_sync_transforms(context)
        self.report({"INFO"}, "Sync cleared")
        return {"FINISHED"}


CLASSES = (
    PM_OT_new_match_camera,
    PM_OT_bulk_create_matches,
    PM_OT_reference_image_label,
    PM_OT_report_info,
    PM_OT_unload_match,
    PM_OT_delete_match,
    PM_OT_rename_match,
    PM_OT_reload,
    PM_OT_load_image,
    PM_OT_replace_image,
    PM_OT_import_ros_yaml,
    PM_OT_refine,
    PM_OT_camera_view,
    PM_OT_apply_manual_fov,
    PM_OT_reset_camera,
    PM_OT_edit_pp_offset,
    PM_OT_clear_axis,
    PM_OT_detect_vp_lines,
    PM_OT_toggle_vp_detect_debug,
    PM_OT_delete_selected,
    PM_OT_clear_placement,
    PM_OT_generate_undistorted,
    PM_OT_estimate_distortion,
    PM_OT_use_undistorted_plate,
    PM_OT_use_original_plate,
    PM_OT_toggle_undistorted,
    PM_OT_apply_view_lighting,
    PM_OT_reset_view_lighting,
    PM_OT_add_landmark,
    PM_OT_add_landmarks_from_selected,
    PM_OT_landmark_use_selected,
    PM_OT_landmark_clear_known,
    PM_OT_use_selected_mirror,
    PM_OT_clear_mirror,
    PM_OT_guess_mirror_partner,
    PM_OT_remove_landmark,
    PM_OT_find_apriltag_landmarks,
    PM_OT_duplicate_landmark,
    PM_OT_clear_landmark_observation,
    PM_OT_select_overlay_landmark,
    PM_OT_solve_sync,
    PM_OT_diagnose_sync,
    PM_OT_cancel_diagnose_sync,
    PM_OT_refine_lenses,
    PM_OT_cancel_refine_lenses,
    PM_OT_clear_sync,
    PM_OT_activate_match_slot,
    PM_OT_cycle_match,
    PM_OT_interact,
    PM_OT_pick_in_active_match,
)
