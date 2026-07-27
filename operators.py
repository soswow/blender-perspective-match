"""Operators for file I/O, solving, and camera-view modal interaction."""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector

from . import core, distortion, overlay, project_io, properties, scene


def _session(context: bpy.types.Context):
    return properties.active_session(context)


def _workspace(context: bpy.types.Context):
    return properties.workspace(context)


def _report_exception(operator: bpy.types.Operator, error: Exception) -> set[str]:
    settings = properties.active_session(bpy.context)
    if settings is not None:
        settings.error = str(error)
    operator.report({"ERROR"}, str(error))
    return {"CANCELLED"}


def _required_lines_ready(settings) -> bool:
    counts = {"x": 0, "y": 0, "z": 0}
    for line in settings.lines:
        counts[line.axis] += 1
    if settings.vp_mode == "1":
        return counts["y"] >= 2 and counts["z"] >= 2
    if settings.vp_mode == "2":
        return counts["x"] >= 2 and counts["z"] >= 2
    return counts["y"] >= 2 and counts["z"] >= 2


def _refine_if_ready(context: bpy.types.Context) -> None:
    settings = _session(context)
    if settings is None:
        return
    if _required_lines_ready(settings):
        scene.refine_match(context)
    else:
        settings.status = "Draw at least two lines for each required axis"
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


def _clear_interact_flags(context: bpy.types.Context) -> None:
    """Reset workspace modal flags without touching a live operator."""
    workspace = _workspace(context)
    workspace.is_modal = False
    workspace.work_mode = "NONE"


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
    bl_description = "Load a still into the active Perspective Match camera"
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


class PM_OT_import_project(bpy.types.Operator, ImportHelper):
    """Import a desktop-compatible Perspective Match project."""

    bl_idname = "perspective_match.import_project"
    bl_label = "Import Project"
    bl_description = "Import a .pmproj project into the active match camera"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".pmproj"
    filter_glob: bpy.props.StringProperty(default="*.pmproj", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        self._context_scene = context.scene
        try:
            project_io.load_project(context, self.filepath)
        except Exception as error:
            return _report_exception(self, error)
        self.report({"INFO"}, f"Imported {Path(self.filepath).name}")
        return {"FINISHED"}


class PM_OT_refine(bpy.types.Operator):
    """Refine and apply the matched camera from current VP lines."""

    bl_idname = "perspective_match.refine"
    bl_label = "Refine Camera"
    bl_description = "Solve focal, principal point, and camera orientation from VP lines"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return (
            settings is not None
            and settings.image is not None
            and _required_lines_ready(settings)
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        self._context_scene = context.scene
        try:
            calibration = scene.refine_match(context)
        except Exception as error:
            return _report_exception(self, error)
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
        scene.enter_camera_view(context)
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
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        settings.status = "Camera reset to manual FOV and centered principal point"
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


class PM_OT_apply_placement(bpy.types.Operator):
    """Reapply the picked origin."""

    bl_idname = "perspective_match.apply_placement"
    bl_label = "Apply Origin"
    bl_description = "Recompute camera position from the picked origin"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = _session(context)
        return settings is not None and settings.origin_is_set

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            scene.reapply_placement(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
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
        ),
        default="LINE",
    )

    _drag_kind: str = ""
    _start: tuple[float, float] | None = None
    _original: tuple[float, ...] | None = None
    _edit_index: int = -1
    _edit_endpoint: int = 0

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
            context.window.cursor_set("CROSSHAIR")
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
        context.window.cursor_set("CROSSHAIR")
        settings.status = self._status_prompt()
        properties.tag_viewport_redraw(context)
        return {"RUNNING_MODAL"}

    def _status_prompt(self) -> str:
        if self.mode == "LINE":
            return "Drag VP segments · click one to select · Esc exits"
        return "Click the world origin on the ground plane"

    def _finish(self, context: bpy.types.Context, *, cancelled: bool = False) -> set[str]:
        global _active_interact
        if _active_interact is self:
            _active_interact = None
        workspace = _workspace(context)
        settings = _session(context)
        workspace.is_modal = False
        workspace.work_mode = "NONE"
        if context.window is not None:
            context.window.cursor_set("DEFAULT")
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

    def _update_drag(self, context, image_point: tuple[float, float]) -> None:
        settings = _session(context)
        if self._drag_kind == "NEW_LINE":
            overlay.set_preview(context, "LINE", self._start, image_point)
        elif self._drag_kind == "LINE_ENDPOINT":
            line = settings.lines[self._edit_index]
            if self._edit_endpoint == 1:
                line.x1, line.y1 = image_point
            else:
                line.x2, line.y2 = image_point
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
        self._drag_kind = ""
        self._start = None
        self._original = None
        overlay.clear_preview(context)

    def _cancel_drag(self, context) -> bool:
        if not self._drag_kind:
            return False
        settings = _session(context)
        if self._original is not None and self._edit_index >= 0:
            if self._drag_kind == "LINE_ENDPOINT":
                line = settings.lines[self._edit_index]
                line.x1, line.y1, line.x2, line.y2 = self._original
        self._drag_kind = ""
        self._original = None
        overlay.clear_preview(context)
        return True

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        workspace = _workspace(context)
        settings = _session(context)
        # External clears (match switch / unload) or lost session end the tool.
        if settings is None or not workspace.is_modal or _active_interact is not self:
            return self._finish(context, cancelled=True)
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
                self._begin_drag(context, event, image_point, region)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self._drag_kind:
                if image_point is not None:
                    self._complete_drag(context, image_point)
                else:
                    self._cancel_drag(context)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context) -> None:
        self._cancel_drag(context)
        self._finish(context, cancelled=True)


class PM_OT_new_match_camera(bpy.types.Operator):
    """Create a new Empty + Camera match session and activate it."""

    bl_idname = "perspective_match.new_match_camera"
    bl_label = "New Match Camera"
    bl_description = "Create a new Perspective Match camera session"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            root = scene.create_match_camera(context)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {root.name}")
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


CLASSES = (
    PM_OT_new_match_camera,
    PM_OT_unload_match,
    PM_OT_load_image,
    PM_OT_import_project,
    PM_OT_refine,
    PM_OT_camera_view,
    PM_OT_apply_manual_fov,
    PM_OT_reset_camera,
    PM_OT_clear_axis,
    PM_OT_delete_selected,
    PM_OT_apply_placement,
    PM_OT_clear_placement,
    PM_OT_generate_undistorted,
    PM_OT_toggle_undistorted,
    PM_OT_apply_view_lighting,
    PM_OT_reset_view_lighting,
    PM_OT_interact,
)
