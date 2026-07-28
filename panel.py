"""Blender-native staged sidebar UI for Perspective Match."""

from __future__ import annotations

from pathlib import Path

import bpy

from . import properties


def _axis_counts(settings) -> dict[str, int]:
    counts = {"x": 0, "y": 0, "z": 0}
    for line in settings.lines:
        counts[line.axis] += 1
    return counts


def _panel_lines_ready(settings) -> bool:
    counts = _axis_counts(settings)
    if settings.vp_mode == "1":
        return counts["y"] >= 2 and counts["z"] >= 2
    if settings.vp_mode == "2":
        return counts["x"] >= 2 and counts["z"] >= 2
    return sum(1 for axis in ("x", "y", "z") if counts[axis] >= 2) >= 2


def _panel_lines_hint(settings) -> str:
    counts = _axis_counts(settings)
    summary = f"X {counts['x']} · Y {counts['z']} · Z {counts['y']}"
    if settings.vp_mode == "1":
        return f"Need 2+ Y and 2+ Z ({summary})"
    if settings.vp_mode == "2":
        return f"Need 2+ X and 2+ Y ({summary})"
    return f"Need 2+ lines on two axes ({summary})"


class VIEW3D_PT_perspective_match(bpy.types.Panel):
    """Perspective Match sidebar panel."""

    bl_label = "Perspective Match"
    bl_idname = "VIEW3D_PT_perspective_match"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Perspective Match"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        workspace = properties.workspace(context)
        settings = properties.active_session(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        header = layout.box()
        header.label(text="Match Cameras", icon="CAMERA_DATA")
        row = header.row(align=True)
        row.operator("perspective_match.new_match_camera", icon="ADD")
        row.operator("perspective_match.unload_match", text="", icon="X")
        header.prop(workspace, "active_match", text="")

        if settings is None:
            header.label(text="Create or select a match camera", icon="INFO")
            return

        image_box = layout.box()
        image_box.label(text="1. Reference Image", icon="IMAGE_DATA")
        row = image_box.row(align=True)
        row.operator("perspective_match.load_image", text="Open Image", icon="FILE_IMAGE")
        row.operator("perspective_match.import_project", text="Import Project", icon="FILE_FOLDER")
        if settings.image is None:
            image_box.label(text="Load a still or .pmproj into this match", icon="INFO")
            return
        image_box.label(text=Path(settings.image_path).name, icon="CHECKMARK")
        image_box.label(text=f"{settings.image_width} × {settings.image_height} px")
        image_box.operator("perspective_match.camera_view", icon="CAMERA_DATA")

        perspective_box = layout.box()
        perspective_box.label(text="2. Perspective", icon="ORIENTATION_GLOBAL")
        perspective_box.prop(settings, "vp_mode", expand=True)
        if settings.vp_mode == "1":
            tip = "1-point: Y + Z lines; FOV stays manual"
        elif settings.vp_mode == "2":
            tip = "2-point: X + Y horizontals; Z uprights not used"
        else:
            tip = "3-point: any two axes (2+ lines each); the third is derived"
        perspective_box.label(text=tip, icon="INFO")

        line_box = layout.box()
        line_header = line_box.row()
        line_header.label(text="3. VP Lines & Camera", icon="TRACKING")
        line_header.prop(
            settings,
            "show_vp_overlay",
            text="",
            icon="HIDE_OFF" if settings.show_vp_overlay else "HIDE_ON",
            emboss=False,
        )
        line_box.prop(settings, "active_axis", expand=True)
        counts = _axis_counts(settings)
        required = (
            f"X {counts['x']} · Y {counts['z']} · Z {counts['y']}"
        )
        line_box.label(text=required, icon="DRIVER_DISTANCE")
        row = line_box.row(align=True)
        draw_row = row.row(align=True)
        draw_row.operator_context = "INVOKE_REGION_WIN"
        operator = draw_row.operator(
            "perspective_match.interact",
            text="Draw / Edit Lines",
            icon="GREASEPENCIL",
        )
        operator.mode = "LINE"
        row.operator("perspective_match.delete_selected", text="", icon="TRASH")
        row.operator("perspective_match.clear_axis", text="", icon="X")
        if workspace.is_modal and workspace.work_mode == "LINE":
            line_box.label(text="Line tool active — Esc exits", icon="MOUSE_LMB")

        focal_row = line_box.row(align=True)
        focal_row.prop(settings, "lock_focal", text="Manual FOV", toggle=True)
        focal_row.operator("perspective_match.refine", text="Auto from VPs", icon="FILE_REFRESH")
        if not _panel_lines_ready(settings):
            line_box.label(text=_panel_lines_hint(settings), icon="INFO")
        manual_column = line_box.column(align=True)
        manual_column.enabled = settings.lock_focal or settings.vp_mode == "1"
        manual_column.prop(settings, "hfov_degrees")
        manual_column.operator("perspective_match.apply_manual_fov", icon="CHECKMARK")
        line_box.operator("perspective_match.reset_camera", icon="LOOP_BACK")

        if settings.fov_xy > 0.0 or settings.fov_zy > 0.0 or settings.fov_zx > 0.0:
            diagnostics = line_box.column(align=True)
            diagnostics.label(text="HFOV by VP pair:")
            if settings.fov_xy > 0.0:
                diagnostics.label(text=f"XY: {settings.fov_xy:.2f}°")
            if settings.fov_zy > 0.0:
                diagnostics.label(text=f"ZY: {settings.fov_zy:.2f}°")
            if settings.fov_zx > 0.0:
                diagnostics.label(text=f"ZX: {settings.fov_zx:.2f}°")
            diagnostics.label(text=f"Axis residual: {settings.residual_degrees:.2f}°")
        if settings.camera_object is not None and settings.camera_object.data is not None:
            line_box.label(
                text=(
                    f"Lens: {settings.camera_object.data.lens:.2f} mm · "
                    f"HFOV {settings.hfov_degrees:.2f}°"
                ),
                icon="CAMERA_DATA",
            )
        if abs(settings.cx - settings.image_width * 0.5) > 0.5 or abs(
            settings.cy - settings.image_height * 0.5
        ) > 0.5:
            line_box.label(
                text=(
                    f"PP offset: {settings.cx - settings.image_width * 0.5:+.1f}, "
                    f"{settings.cy - settings.image_height * 0.5:+.1f} px"
                ),
                icon="PIVOT_CURSOR",
            )

        origin_box = layout.box()
        origin_box.label(text="4. Origin", icon="PIVOT_CURSOR")
        row = origin_box.row(align=True)
        pick_row = row.row(align=True)
        pick_row.operator_context = "INVOKE_REGION_WIN"
        operator = pick_row.operator("perspective_match.interact", text="Pick Origin", icon="PIVOT_CURSOR")
        operator.mode = "ORIGIN"
        apply_row = origin_box.row(align=True)
        apply_row.enabled = settings.origin_is_set
        apply_row.operator("perspective_match.apply_placement", icon="CHECKMARK")
        apply_row.operator("perspective_match.clear_placement", text="", icon="X")
        if settings.origin_is_set:
            origin_box.label(text="Origin set", icon="CHECKMARK")
        if workspace.is_modal and workspace.work_mode == "ORIGIN":
            origin_box.label(text="Origin tool active — click ground point", icon="MOUSE_LMB")

        distortion_box = layout.box()
        distortion_box.label(text="Lens Distortion", icon="MOD_SIMPLEDEFORM")
        distortion_box.prop(settings, "estimate_distortion")
        row = distortion_box.row(align=True)
        row.operator("perspective_match.refine", text="Estimate λ", icon="FILE_REFRESH")
        row.enabled = settings.estimate_distortion
        distortion_box.label(text=f"Division λ: {settings.division_lambda:.5f}")
        if settings.lambda_saturated:
            distortion_box.label(text="Estimate saturated; pinhole retained", icon="ERROR")
        row = distortion_box.row(align=True)
        row.enabled = abs(settings.division_lambda) > 1.0e-8
        row.operator("perspective_match.generate_undistorted", icon="IMAGE_DATA")
        if settings.undistorted_image is not None:
            row.operator(
                "perspective_match.toggle_undistorted",
                text="Original" if settings.view_undistorted else "Undistorted",
                icon="UV_SYNC_SELECT",
            )

        view_box = layout.box()
        view_box.label(text="View", icon="IMAGE_RGB")
        view_box.label(text="Display only — does not affect the solve", icon="INFO")
        view_box.prop(settings, "view_exposure")
        view_box.prop(settings, "view_contrast")
        row = view_box.row(align=True)
        row.operator("perspective_match.apply_view_lighting", icon="CHECKMARK")
        row.operator("perspective_match.reset_view_lighting", text="", icon="LOOP_BACK")
        if settings.view_lighting_applied:
            view_box.label(text=Path(settings.view_path).name or "View plate active", icon="CHECKMARK")
        view_box.prop(settings, "overlay_opacity")
        view_box.prop(settings, "controls_opacity")

        if settings.error:
            error_box = layout.box()
            error_box.alert = True
            error_box.label(text=settings.error, icon="ERROR")
        layout.label(text=settings.status, icon="INFO")
        layout.separator()
        layout.operator("perspective_match.reload", icon="FILE_REFRESH")


CLASSES = (VIEW3D_PT_perspective_match,)
