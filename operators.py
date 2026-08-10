"""Operators for file I/O, solving, and camera-view modal interaction."""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector

from . import apriltag_detect, core, distortion, feature_detect, overlay, properties, scene


def _session(context: bpy.types.Context):
    return properties.active_session(context)


def _workspace(context: bpy.types.Context):
    return properties.workspace(context)


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


def _axis_line_counts(settings) -> dict[str, int]:
    counts = {"x": 0, "y": 0, "z": 0}
    for line in settings.lines:
        counts[line.axis] += 1
    return counts


def _required_lines_ready(settings) -> bool:
    counts = _axis_line_counts(settings)
    if settings.vp_mode == "1":
        # Depth (Blender Z / axis y) + horizontals on Y (axis z).
        return counts["y"] >= 2 and counts["z"] >= 2
    if settings.vp_mode == "2":
        # Two horizontals (X × Y); uprights are parallel / not a finite VP.
        return counts["x"] >= 2 and counts["z"] >= 2
    # 3-point: any two axes with ≥2 lines; the third world axis is derived.
    return sum(1 for axis in ("x", "y", "z") if counts[axis] >= 2) >= 2


def _lines_needed_message(settings) -> str:
    """Human-readable reason Auto-from-VPs cannot run yet."""
    counts = _axis_line_counts(settings)
    summary = f"X {counts['x']} · Y {counts['z']} · Z {counts['y']}"
    if settings.vp_mode == "1":
        return f"1-point needs 2+ Y and 2+ Z lines ({summary})"
    if settings.vp_mode == "2":
        return f"2-point needs 2+ X and 2+ Y lines ({summary})"
    return f"3-point needs 2+ lines on any two axes ({summary})"


def _refine_if_ready(context: bpy.types.Context) -> None:
    settings = _session(context)
    if settings is None:
        return
    if _required_lines_ready(settings):
        scene.refine_match(context)
        distortion.sync_undistorted_plate_after_refine(context)
    else:
        settings.status = _lines_needed_message(settings)
        properties.tag_viewport_redraw(context)


def _view3d_under_event(context: bpy.types.Context, event):
    """Return (area, region, space) for the 3D View WINDOW under the mouse."""
    screen = context.screen
    if screen is None:
        return None, None, None
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            if (
                region.x <= event.mouse_x < region.x + region.width
                and region.y <= event.mouse_y < region.y + region.height
            ):
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

# Background Find Auto Features job (OpenCV in a worker thread; bpy apply on main).
_auto_feature_lock = threading.Lock()
_auto_feature_cancel: threading.Event | None = None
_auto_feature_running = False
_auto_feature_progress = {"step": 0, "total": 1, "label": ""}
_auto_feature_result_box: dict = {}


def lens_refine_is_running() -> bool:
    """True while a Refine Lenses modal/worker is active."""
    return bool(_lens_refine_running)


def auto_feature_is_running() -> bool:
    """True while a Find Auto Features modal/worker is active."""
    return bool(_auto_feature_running)


def request_lens_refine_cancel() -> bool:
    """Signal the running Refine Lenses worker to stop. True if one was running."""
    event = _lens_refine_cancel
    if event is None:
        return False
    event.set()
    return True


def request_auto_feature_cancel() -> bool:
    """Signal the running Find Auto Features worker to stop. True if one was running."""
    event = _auto_feature_cancel
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


def _clear_interact_flags(context: bpy.types.Context) -> None:
    """Reset workspace modal flags without touching a live operator."""
    workspace = _workspace(context)
    workspace.is_modal = False
    workspace.work_mode = "NONE"


def _interact_cursor_for_mode(mode: str, context: bpy.types.Context) -> str:
    """Blender window cursor id for an active Draw / Pick tool mode."""
    if mode == "LINE":
        return "KNIFE"
    if mode == "PP":
        return "SCROLL_XY"
    if mode == "ORIGIN":
        return "PICK_AREA"
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
    """Apply the mode cursor; ``modal=True`` stacks via cursor_modal_set."""
    window = context.window
    if window is None:
        return
    cursor = _interact_cursor_for_mode(mode, context)
    if modal:
        window.cursor_modal_set(cursor)
    else:
        window.cursor_set(cursor)


def _restore_interact_cursor(context: bpy.types.Context) -> None:
    """Undo cursor_modal_set from an interact tool session."""
    window = context.window
    if window is None:
        return
    window.cursor_modal_restore()


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


def _match_slot_from_event(event) -> int | None:
    """Ctrl+Alt+NumPad 1–9 → 1-based match slot, else None."""
    if event.value != "PRESS" or not event.ctrl or not event.alt:
        return None
    return _NUMPAD_SLOT_KEYS.get(event.type)


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
    if report is not None:
        report({"INFO"}, f"Active match {index}: {root.name}")
    return {"FINISHED"}


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
        "point from cx/cy, and apply fitzgibbon_lambda when present. "
        "Brown–Conrady distortion coefficients are skipped"
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
        return settings is not None and settings.image is not None

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
        "Needs two axes with at least two lines each in 3-point mode"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Stay clickable with a still loaded so a disabled button is not a silent dead-end.
        settings = _session(context)
        return settings is not None and settings.image is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None or settings.image is None:
            self.report({"ERROR"}, "Load a reference image first")
            return {"CANCELLED"}
        if not _required_lines_ready(settings):
            message = _lines_needed_message(settings)
            settings.status = message
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        # "Auto from VPs" means FOV comes from geometry, not the manual lock.
        settings.lock_focal = False
        try:
            calibration = scene.refine_match(context)
            from . import distortion

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

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = _session(context)
        if settings is None:
            self.report({"ERROR"}, "Create or activate a match camera first")
            return {"CANCELLED"}
        settings.cx = settings.image_width * 0.5
        settings.cy = settings.image_height * 0.5
        settings.division_lambda = 0.0
        settings.lambda_saturated = False
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
        return settings is not None and settings.image is not None

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
    """Generate a transparent expanded undistorted PNG beside the source."""

    bl_idname = "perspective_match.generate_undistorted"
    bl_label = "Generate Undistorted Plate"
    bl_description = "Remap the still with division λ and activate it as camera background"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and abs(settings.division_lambda) > 1.0e-8
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
        "change — press again to re-fit. Works with Manual FOV (λ at the locked focal)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return settings is not None and settings.image is not None

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


class PM_OT_use_original_plate(bpy.types.Operator):
    """Clear distortion and restore the original still."""

    bl_idname = "perspective_match.use_original_plate"
    bl_label = "Original Plate"
    bl_description = (
        "Clear λ, re-solve the camera, and show the original reference image"
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
    """Bake display-only exposure/contrast into a sibling view plate."""

    bl_idname = "perspective_match.apply_view_lighting"
    bl_label = "Apply Lighting"
    bl_description = (
        "Bake exposure and contrast into <stem>-pm-view.png and use it as the camera background"
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
    # Throttle full VP re-orient while dragging the principal point.
    _pp_last_reorient: float = 0.0
    _PP_REORIENT_INTERVAL = 0.08

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

        # Re-click while a live tool runs: switch mode / refresh instead of stacking.
        if _active_interact is not None:
            active = _active_interact
            active.mode = self.mode
            workspace.work_mode = self.mode
            workspace.is_modal = True
            scene.enter_camera_view(context)
            # Already inside a modal_set session — swap cursor without nesting.
            _set_interact_cursor(context, self.mode, modal=False)
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
        _active_interact = self
        context.window_manager.modal_handler_add(self)
        _set_interact_cursor(context, self.mode, modal=True)
        settings.status = self._status_prompt()
        properties.tag_viewport_redraw(context)
        return {"RUNNING_MODAL"}

    def _status_prompt(self) -> str:
        if self.mode == "LINE":
            return "Drag VP segments · click one to select · Esc exits"
        if self.mode == "LANDMARK":
            landmark = scene.active_landmark(bpy.context)
            name = landmark.name if landmark is not None else "(none)"
            if landmark is not None and landmark.kind == "LINE":
                return (
                    f"Drag line '{name}' · pull endpoints to edit · Esc exits"
                )
            return f"Click landmark '{name}' in this match · Esc exits"
        if self.mode == "PP":
            return "Drag principal point (violet) · Esc exits"
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
        # Leaving the line tool clears selection so handles don't linger.
        if settings is not None and self.mode == "LINE":
            settings.selected_line_index = -1
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
        if 0 <= selected < len(settings.lines):
            line = settings.lines[selected]
            for endpoint_index, image_point in enumerate(((line.x1, line.y1), (line.x2, line.y2))):
                screen = scene.image_to_region(context, image_point[0], image_point[1])
                if screen is not None and (screen - mouse).length <= 12.0:
                    return selected, endpoint_index + 1
        best_index = -1
        best_distance = 11.0
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
            if screen is not None and (screen - mouse).length <= 12.0:
                return endpoint_index + 1
        point_a = scene.image_to_region(context, observation.x, observation.y)
        point_b = scene.image_to_region(context, observation.x2, observation.y2)
        if point_a is None or point_b is None:
            return 0
        if _distance_to_segment(mouse, point_a, point_b) < 11.0:
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
            # Shift every move; rebuild orientation on a throttle so geometry
            # tracks the release "snap" without refining on every mouse event.
            now = time.monotonic()
            reorient = (now - self._pp_last_reorient) >= self._PP_REORIENT_INTERVAL
            try:
                scene.set_principal_point(
                    context, image_point, finalize=reorient
                )
                if reorient:
                    self._pp_last_reorient = now
            except Exception as error:
                self.report({"ERROR"}, str(error))

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
        if self._drag_kind == "NEW_LINE" and self._start is not None:
            if math.hypot(image_point[0] - self._start[0], image_point[1] - self._start[1]) >= 8.0:
                line = settings.lines.add()
                line.item_id = f"blender-line-{uuid4().hex}"
                line.axis = settings.active_axis
                line.x1, line.y1 = self._start
                line.x2, line.y2 = image_point
                settings.selected_line_index = len(settings.lines) - 1
                _refine_if_ready(context)
        elif self._drag_kind == "LINE_ENDPOINT":
            _refine_if_ready(context)
        elif self._drag_kind == "PP":
            try:
                scene.set_principal_point(context, image_point, finalize=True)
            except Exception as error:
                self.report({"ERROR"}, str(error))
            if settings is not None:
                settings.status = self._status_prompt()
        self._drag_kind = ""
        self._start = None
        self._original = None
        overlay.clear_preview(context)

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
        if self._original is not None and self._drag_kind == "PP":
            # Restore pre-drag PP without a second orientation solve yet —
            # finalize=False keeps R; Esc then exits the tool.
            try:
                scene.set_principal_point(
                    context,
                    (float(self._original[0]), float(self._original[1])),
                    finalize=False,
                )
            except Exception:
                pass
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
        # Ctrl+Alt+NumPad must be handled here — modal handlers see keys first.
        slot = _match_slot_from_event(event)
        if slot is not None:
            result = activate_match_by_slot(context, slot, report=self.report)
            # Successful switch already finished this modal via set_active_match.
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
        # Sidebar / other editors keep normal UI while the tool is idle.
        if area is None or region is None or space is None:
            if self._drag_kind:
                # Drag left the viewport — cancel the in-progress gesture.
                if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                    self._cancel_drag(context)
                    return {"RUNNING_MODAL"}
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        with context.temp_override(area=area, region=region, space_data=space):
            # Allow MMB pan/orbit and idle mouse move without fighting navigation.
            if event.type == "MIDDLEMOUSE":
                return {"PASS_THROUGH"}
            if event.type == "MOUSEMOVE" and not self._drag_kind:
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
                    self._pp_last_reorient = 0.0
                    try:
                        # First sample reorients immediately (interval gate open).
                        scene.set_principal_point(
                            context, image_point, finalize=True
                        )
                        self._pp_last_reorient = time.monotonic()
                    except Exception as error:
                        self._drag_kind = ""
                        self._original = None
                        self.report({"ERROR"}, str(error))
                    return {"RUNNING_MODAL"}
                if self.mode == "LANDMARK":
                    landmark = scene.active_landmark(context)
                    if landmark is not None and landmark.kind == "LINE":
                        self._begin_landmark_line_drag(
                            context, event, image_point, region
                        )
                        return {"RUNNING_MODAL"}
                    try:
                        scene.set_landmark_observation(context, image_point)
                    except Exception as error:
                        self.report({"ERROR"}, str(error))
                        return {"RUNNING_MODAL"}
                    # Stay in the tool so the same landmark can be picked in other matches
                    # after switching Active Match from the sidebar.
                    settings.status = self._status_prompt()
                    properties.tag_viewport_redraw(context)
                    return {"RUNNING_MODAL"}
                self._begin_drag(context, event, image_point, region)
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
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context) -> None:
        self._cancel_drag(context)
        self._finish(context, cancelled=True)


class PM_OT_activate_match_slot(bpy.types.Operator):
    """Activate a match by 1-based index in the name-sorted match list."""

    bl_idname = "perspective_match.activate_match_slot"
    bl_label = "Activate Match Slot"
    bl_description = (
        "Switch to a Perspective Match by slot (Ctrl+Alt+NumPad 1–9). "
        "Cancels any active Draw / Pick tool"
    )
    bl_options = {"REGISTER", "UNDO"}

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


class PM_OT_new_match_camera(bpy.types.Operator):
    """Create a new Empty + Camera match session and activate it."""

    bl_idname = "perspective_match.new_match_camera"
    bl_label = "New Match Camera"
    bl_description = (
        "Create a new Perspective Match camera session and open the "
        "reference image file dialog"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            root = scene.create_match_camera(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {root.name}")
        # Chain into Open Image so a new match always starts with a still.
        bpy.ops.perspective_match.load_image("INVOKE_DEFAULT")
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
        "(safer than System → Reload Scripts for panel/operator edits)"
    )
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Never unregister from inside this operator — that can SIGSEGV Blender.
        # Schedule the tear-down on a timer so this execute can return first.
        package = sys.modules[__package__]
        if not package.schedule_reload():
            self.report({"WARNING"}, "Perspective Match reload already queued")
            return {"CANCELLED"}
        self.report({"INFO"}, "Perspective Match reload queued")
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
        properties.tag_viewport_redraw(context)
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
        properties.tag_viewport_redraw(context)
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
        properties.tag_viewport_redraw(context)
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
        space.landmarks.remove(index)
        space.active_landmark_index = min(index, len(space.landmarks) - 1)
        scene.sync_landmark_empties(context)
        properties.tag_viewport_redraw(context)
        return {"FINISHED"}


class PM_OT_find_apriltag_landmarks(bpy.types.Operator):
    """Detect AprilTag 25h9 markers and assign them to sync landmarks."""

    bl_idname = "perspective_match.find_apriltag_landmarks"
    bl_label = "Find AprilTags"
    bl_description = (
        "Scan the active match still for AprilTag 25h9 markers. "
        "Assign each tag centre to a landmark named idNN-25h9 "
        "(create the landmark when missing)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
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
            message = "No AprilTag 25h9 markers found in this still"
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


class PM_OT_find_auto_features(bpy.types.Operator):
    """Detect and match automatic features across sync-enabled stills (background)."""

    bl_idname = "perspective_match.find_auto_features"
    bl_label = "Find Auto Features"
    bl_description = (
        "Detect ORB/SIFT features in sync-enabled stills, match them with RANSAC, "
        "and store anonymous auto tracks (runs in the background — Esc to cancel)"
    )
    bl_options = {"REGISTER", "UNDO"}

    _timer = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if auto_feature_is_running() or lens_refine_is_running():
            return False
        roots = [
            root
            for root in properties.iter_sync_enabled_roots()
            if root.pm_session.image is not None
        ]
        return len(roots) >= 2

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        global _auto_feature_cancel, _auto_feature_running, _auto_feature_result_box
        global _auto_feature_progress

        workspace = _workspace(context)
        # Prefer on-disk paths so cv2.imread runs in the worker (keeps UI free).
        # Only pull Blender pixels on the main thread when no readable path exists.
        sources: list[feature_detect.ImageSource] = []
        try:
            feature_detect._import_cv2()
            for root in properties.iter_sync_enabled_roots():
                session = root.pm_session
                if session.image is None:
                    continue
                image_path = (getattr(session, "image_path", "") or "").strip()
                path_ok = False
                if image_path:
                    path_ok = Path(image_path).expanduser().is_file()
                if path_ok:
                    sources.append(
                        feature_detect.ImageSource(
                            match_id=root.name,
                            image_path=image_path,
                        )
                    )
                    continue
                # Blender buffer — must be read on the main thread.
                gray = feature_detect.load_match_gray(session)
                sources.append(
                    feature_detect.ImageSource(
                        match_id=root.name,
                        gray=gray,
                    )
                )
        except feature_detect.FeatureDetectDependencyError as error:
            return _report_exception(self, error)
        except Exception as error:
            return _report_exception(self, error)

        if len(sources) < 2:
            self.report({"WARNING"}, "Need at least two sync-enabled stills")
            return {"CANCELLED"}

        detector = workspace.auto_feature_detector
        max_features = int(workspace.auto_feature_max_features)
        match_ratio = float(workspace.auto_feature_match_ratio)
        ransac_px = float(workspace.auto_feature_ransac_px)
        keep_percentile = float(workspace.auto_feature_keep_percent)
        max_orphans = int(workspace.auto_feature_max_orphans)
        cancel_event = threading.Event()
        result_box: dict = {"done": False}
        progress_state = {"step": 0, "total": 1, "label": "Starting…"}

        with _auto_feature_lock:
            if _auto_feature_running:
                self.report({"WARNING"}, "Find Auto Features already running")
                return {"CANCELLED"}
            _auto_feature_cancel = cancel_event
            _auto_feature_running = True
            _auto_feature_result_box = result_box
            _auto_feature_progress = progress_state

        workspace.auto_feature_progress = 0.0
        workspace.auto_feature_status = "Find Auto Features running… Esc to cancel"
        workspace.sync_status = workspace.auto_feature_status
        properties.tag_viewport_redraw(context)

        def _on_progress(step: int, total_steps: int, label: str) -> None:
            progress_state["step"] = int(step)
            progress_state["total"] = max(int(total_steps), 1)
            progress_state["label"] = str(label)

        def _worker() -> None:
            try:
                job_result = feature_detect.build_tracks_from_sources(
                    sources,
                    detector=detector,
                    max_features=max_features,
                    ratio=match_ratio,
                    ransac_px=ransac_px,
                    keep_percentile=keep_percentile,
                    max_orphans_per_match=max_orphans,
                    cancel_check=cancel_event.is_set,
                    progress_callback=_on_progress,
                )
                result_box["result"] = job_result
            except Exception as error:
                result_box["error"] = error
            finally:
                result_box["done"] = True

        thread = threading.Thread(
            target=_worker,
            name="PM-AutoFeatures",
            daemon=True,
        )
        thread.start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)
        self._timer = window_manager.event_timer_add(0.1, window=context.window)
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_job(self, context: bpy.types.Context, *, cancelled: bool) -> set[str]:
        global _auto_feature_cancel, _auto_feature_running

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            window_manager.progress_end()
        except Exception:
            pass

        workspace = _workspace(context)
        result_box = _auto_feature_result_box

        with _auto_feature_lock:
            _auto_feature_running = False
            _auto_feature_cancel = None

        if result_box.get("error") is not None:
            error = result_box["error"]
            workspace.auto_feature_progress = 0.0
            workspace.auto_feature_status = str(error)
            workspace.sync_status = str(error)
            properties.tag_viewport_redraw(context)
            return _report_exception(
                self,
                error if isinstance(error, Exception) else Exception(str(error)),
            )

        job_result = result_box.get("result")
        if job_result is None or cancelled:
            workspace.auto_feature_progress = 0.0
            message = "Auto feature detection cancelled"
            workspace.auto_feature_status = message
            workspace.sync_status = message
            properties.tag_viewport_redraw(context)
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        try:
            feature_detect.apply_feature_tracks(context, job_result)
        except Exception as error:
            return _report_exception(self, error)

        workspace.auto_feature_progress = 0.0
        workspace.sync_status = job_result.message
        self.report({"INFO"}, job_result.message)
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        workspace = _workspace(context)
        if event.type in {"ESC"} and event.value == "PRESS":
            request_auto_feature_cancel()
            workspace.auto_feature_status = "Cancelling auto features…"
            workspace.sync_status = workspace.auto_feature_status
            properties.tag_viewport_redraw(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = _auto_feature_progress
        total = max(int(progress.get("total", 1)), 1)
        step = int(progress.get("step", 0))
        label = str(progress.get("label", ""))
        workspace.auto_feature_progress = min(max(step / total, 0.0), 1.0)
        if label:
            status = f"Auto features {step}/{total} · {label}"
            workspace.auto_feature_status = status
            workspace.sync_status = status
        try:
            context.window_manager.progress_update(min(int(100 * step / total), 100))
        except Exception:
            pass
        properties.tag_viewport_redraw(context)

        if not _auto_feature_result_box.get("done"):
            return {"PASS_THROUGH"}

        return self._finish_job(
            context,
            cancelled=bool(_auto_feature_cancel and _auto_feature_cancel.is_set()),
        )

    def cancel(self, context: bpy.types.Context) -> None:
        request_auto_feature_cancel()
        for _ in range(50):
            if _auto_feature_result_box.get("done"):
                break
            time.sleep(0.02)
        self._finish_job(context, cancelled=True)

    def execute(self, context: bpy.types.Context) -> set[str]:
        # Sidebar / redo should still use the background modal path.
        if context.window is not None:
            return self.invoke(context, None)
        workspace = _workspace(context)
        try:
            sources = []
            for root in properties.iter_sync_enabled_roots():
                session = root.pm_session
                if session.image is None:
                    continue
                sources.append(
                    feature_detect.ImageSource(
                        match_id=root.name,
                        image_path=getattr(session, "image_path", "") or "",
                        gray=feature_detect.load_match_gray(session),
                    )
                )
            if len(sources) < 2:
                self.report({"WARNING"}, "Need at least two sync-enabled stills")
                return {"CANCELLED"}
            job_result = feature_detect.build_tracks_from_sources(
                sources,
                detector=workspace.auto_feature_detector,
                max_features=int(workspace.auto_feature_max_features),
                ratio=float(workspace.auto_feature_match_ratio),
                ransac_px=float(workspace.auto_feature_ransac_px),
                keep_percentile=float(workspace.auto_feature_keep_percent),
                max_orphans_per_match=int(workspace.auto_feature_max_orphans),
            )
            feature_detect.apply_feature_tracks(context, job_result)
        except Exception as error:
            return _report_exception(self, error)
        workspace.sync_status = job_result.message
        self.report({"INFO"}, job_result.message)
        return {"FINISHED"}


class PM_OT_cancel_auto_features(bpy.types.Operator):
    """Stop the running Find Auto Features background job."""

    bl_idname = "perspective_match.cancel_auto_features"
    bl_label = "Cancel Auto Features"
    bl_description = "Cancel the running Find Auto Features job (Esc also works)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return auto_feature_is_running()

    def execute(self, context: bpy.types.Context) -> set[str]:
        if not request_auto_feature_cancel():
            return {"CANCELLED"}
        workspace = _workspace(context)
        workspace.auto_feature_status = "Cancelling auto features…"
        workspace.sync_status = workspace.auto_feature_status
        properties.tag_viewport_redraw(context)
        self.report({"INFO"}, "Cancelling Find Auto Features…")
        return {"FINISHED"}


class PM_OT_clear_auto_features(bpy.types.Operator):
    """Remove all automatic feature tracks."""

    bl_idname = "perspective_match.clear_auto_features"
    bl_label = "Clear Auto Features"
    bl_description = "Remove all automatically detected feature tracks"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if auto_feature_is_running():
            return False
        return len(properties.workspace(context).auto_tracks) > 0

    def execute(self, context: bpy.types.Context) -> set[str]:
        count = feature_detect.clear_auto_tracks(context)
        message = f"Cleared {count} auto tracks" if count else "No auto tracks"
        _workspace(context).sync_status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class PM_OT_duplicate_landmark(bpy.types.Operator):
    """Duplicate the active landmark without picks, Known 3D links, or solved positions."""

    bl_idname = "perspective_match.duplicate_landmark"
    bl_label = "Duplicate Landmark"
    bl_description = (
        "Duplicate the active landmark (keeps type, On Ground, Use in Sync). "
        "Clears picks, Known 3D links, parallel links, and solved positions. "
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
        # Intentionally leave Known 3D / parallel / observations / positions empty.
        properties.ensure_landmark_creation_indices(space)
        space.active_landmark_index = len(space.landmarks) - 1
        if pick_confidence is not None:
            space.landmark_pick_confidence = pick_confidence
        properties.tag_viewport_redraw(context)
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
        "Blender objects. On Ground / Known 3D pin absolute scale vs the anchor"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        anchor = properties.anchor_root(context)
        return (
            anchor is not None
            and properties.match_sync_enabled(anchor)
            and len(properties.iter_sync_enabled_roots()) >= 2
            and scene.sync_has_usable_features(context)
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
        "consistency checks without applying a pose"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return PM_OT_solve_sync.poll(context)

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


class PM_OT_refine_lenses(bpy.types.Operator):
    """Adjust match focals so landmark sync and VP lines agree better."""

    bl_idname = "perspective_match.refine_lenses"
    bl_label = "Refine Lenses"
    bl_description = (
        "Search each unlocked match's focal length (re-orient from VP lines) "
        "to lower sync reprojection error, then Solve Sync. "
        "Runs in the background — Esc or Cancel Refine to stop. "
        "Skips 1-point matches and matches without enough VP lines"
    )
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _prep = None
    _progress_max = 1

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if lens_refine_is_running():
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

        from . import lens_refine

        free_count = sum(1 for item in prep.lens_inputs if not item.freeze_focal)
        total = lens_refine.estimate_refine_evaluation_count(free_count)
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

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene.clear_sync_transforms(context)
        self.report({"INFO"}, "Sync cleared")
        return {"FINISHED"}


CLASSES = (
    PM_OT_new_match_camera,
    PM_OT_reference_image_label,
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
    PM_OT_delete_selected,
    PM_OT_clear_placement,
    PM_OT_generate_undistorted,
    PM_OT_estimate_distortion,
    PM_OT_use_original_plate,
    PM_OT_toggle_undistorted,
    PM_OT_apply_view_lighting,
    PM_OT_reset_view_lighting,
    PM_OT_add_landmark,
    PM_OT_add_landmarks_from_selected,
    PM_OT_landmark_use_selected,
    PM_OT_landmark_clear_known,
    PM_OT_remove_landmark,
    PM_OT_find_apriltag_landmarks,
    PM_OT_find_auto_features,
    PM_OT_cancel_auto_features,
    PM_OT_clear_auto_features,
    PM_OT_duplicate_landmark,
    PM_OT_clear_landmark_observation,
    PM_OT_solve_sync,
    PM_OT_diagnose_sync,
    PM_OT_refine_lenses,
    PM_OT_cancel_refine_lenses,
    PM_OT_clear_sync,
    PM_OT_activate_match_slot,
    PM_OT_interact,
)
