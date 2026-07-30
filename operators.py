"""Operators for file I/O, solving, and camera-view modal interaction."""

from __future__ import annotations

import math
import sys
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
            ("LANDMARK", "Landmark", "Pick the active landmark in this match"),
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
        if self.mode == "LANDMARK":
            landmark = scene.active_landmark(bpy.context)
            name = landmark.name if landmark is not None else "(none)"
            if landmark is not None and landmark.kind == "LINE":
                return (
                    f"Drag line '{name}' · pull endpoints to edit · Esc exits"
                )
            return f"Click landmark '{name}' in this match · Esc exits"
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
        return (
            properties.anchor_root(context) is not None
            and len(properties.iter_match_roots()) >= 2
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
    PM_OT_unload_match,
    PM_OT_reload,
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
    PM_OT_add_landmark,
    PM_OT_add_landmarks_from_selected,
    PM_OT_landmark_use_selected,
    PM_OT_landmark_clear_known,
    PM_OT_remove_landmark,
    PM_OT_clear_landmark_observation,
    PM_OT_solve_sync,
    PM_OT_diagnose_sync,
    PM_OT_clear_sync,
    PM_OT_interact,
)
