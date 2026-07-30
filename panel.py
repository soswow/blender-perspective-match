"""Blender-native staged sidebar UI for Perspective Match."""

from __future__ import annotations

from pathlib import Path

import bpy

from . import properties, scene


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


def _observation_count(landmark) -> int:
    return sum(1 for observation in landmark.observations if observation.is_set)


def _landmark_is_parallel_linked(landmark, workspace) -> bool:
    """True when this line is parallel-linked to another (either direction)."""
    if landmark.kind != "LINE":
        return False
    if landmark.parallel_to and landmark.parallel_to != "NONE":
        return True
    for other in workspace.landmarks:
        if other.item_id == landmark.item_id:
            continue
        if other.kind == "LINE" and other.parallel_to == landmark.item_id:
            return True
    return False


class PM_UL_landmarks(bpy.types.UIList):
    """Compact landmark list for the sync section."""

    bl_idname = "PM_UL_landmarks"

    def draw_item(
        self,
        _context,
        layout,
        _data,
        item,
        _icon,
        _active_data,
        _active_property,
        _index=0,
        _flt_flag=0,
    ) -> None:
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="EMPTY_AXIS")
        if item.kind == "LINE":
            row.label(text="", icon="MESH_DATA")
            if _landmark_is_parallel_linked(item, _data):
                row.label(text="", icon="LINKED")
        if item.known_object is not None:
            row.label(text="", icon="PIVOT_CURSOR")
        elif item.on_ground:
            row.label(text="", icon="ORIENTATION_VIEW")
        count = _observation_count(item)
        if item.rmse_px > 0.5:
            row.label(text=f"{count} · {item.rmse_px:.0f}px")
        else:
            row.label(text=f"{count}")

    def filter_items(self, context, data, propname):
        """Alphabetical when Sort A–Z is on; otherwise creation / add order.

        Read-only: Blender forbids writing Scene ID data from UIList draw.
        creation_index is assigned on add / file load, not here.
        """
        landmarks = getattr(data, propname)
        helper_funcs = bpy.types.UI_UL_list
        flt_flags = []
        flt_neworder = []
        if not landmarks:
            return flt_flags, flt_neworder
        if getattr(data, "landmarks_sort_alphabetical", False):
            flt_neworder = helper_funcs.sort_items_by_name(landmarks, "name")
        elif any(landmark.creation_index < 0 for landmark in landmarks):
            # Legacy / not yet migrated — keep collection order (no ID writes).
            flt_neworder = []
        else:
            keyed = [
                (index, int(landmark.creation_index))
                for index, landmark in enumerate(landmarks)
            ]
            flt_neworder = helper_funcs.sort_items_helper(
                keyed, lambda item: item[1], False
            )
        return flt_flags, flt_neworder


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
        else:
            self._draw_active_match(layout, context, workspace, settings)

        self._draw_sync(layout, context, workspace, settings)

        if settings is not None and settings.error:
            error_box = layout.box()
            error_box.alert = True
            error_box.label(text=settings.error, icon="ERROR")
        if settings is not None:
            layout.label(text=settings.status, icon="INFO")
        elif workspace.sync_status:
            layout.label(text=workspace.sync_status, icon="INFO")
        layout.separator()
        layout.operator("perspective_match.reload", icon="FILE_REFRESH")

    def _draw_active_match(self, layout, context, workspace, settings) -> None:
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
        operator = pick_row.operator(
            "perspective_match.interact",
            text="Pick Origin",
            icon="PIVOT_CURSOR",
        )
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
            view_box.label(
                text=Path(settings.view_path).name or "View plate active",
                icon="CHECKMARK",
            )
        view_box.prop(settings, "overlay_opacity")

    def _draw_sync(self, layout, context, workspace, settings) -> None:
        sync_box = layout.box()
        sync_header = sync_box.row()
        sync_header.label(text="5. Sync Matches", icon="LINKED")
        sync_header.prop(
            workspace,
            "show_landmark_overlay",
            text="",
            icon="HIDE_OFF" if workspace.show_landmark_overlay else "HIDE_ON",
            emboss=False,
        )
        match_count = len(properties.iter_match_roots())
        if match_count < 2:
            sync_box.label(text="Create at least two matched cameras", icon="INFO")
            return

        sync_box.prop(workspace, "anchor_match", text="Anchor")
        sync_box.label(
            text="Points or lines across stills; Known 3D Empties optional",
            icon="INFO",
        )

        list_row = sync_box.row()
        list_row.template_list(
            "PM_UL_landmarks",
            "",
            workspace,
            "landmarks",
            workspace,
            "active_landmark_index",
            rows=3,
        )
        list_column = list_row.column(align=True)
        add_point = list_column.operator(
            "perspective_match.add_landmark",
            text="",
            icon="ADD",
        )
        add_point.kind = "POINT"
        add_line = list_column.operator(
            "perspective_match.add_landmark",
            text="",
            icon="MESH_DATA",
        )
        add_line.kind = "LINE"
        list_column.operator(
            "perspective_match.add_landmarks_from_selected",
            text="",
            icon="IMPORT",
        )
        list_column.operator("perspective_match.remove_landmark", text="", icon="REMOVE")
        # Separate control group: display order only (does not reorder storage).
        list_column.separator()
        list_column.prop(
            workspace,
            "landmarks_sort_alphabetical",
            text="",
            icon="SORTALPHA",
            toggle=True,
        )

        landmark = scene.active_landmark(context)
        if landmark is not None:
            sync_box.prop(landmark, "kind")
            if landmark.kind == "POINT":
                sync_box.prop(landmark, "on_ground")
            known_row = sync_box.row(align=True)
            known_row.prop(landmark, "known_object", text="Known 3D")
            known_row.operator(
                "perspective_match.landmark_use_selected",
                text="",
                icon="EYEDROPPER",
            )
            known_row.operator(
                "perspective_match.landmark_clear_known",
                text="",
                icon="X",
            )
            if landmark.kind == "LINE":
                sync_box.prop(landmark, "known_object_b", text="Known 3D B")
                sync_box.prop(landmark, "parallel_to", text="Is Parallel To")
                sync_box.label(
                    text="Optional: two Empties = metric edge; else draw in ≥3 stills",
                    icon="INFO",
                )
            if landmark.known_object is not None:
                location = landmark.known_object.matrix_world.to_translation()
                sync_box.label(
                    text=(
                        f"World ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
                    ),
                    icon="EMPTY_AXIS",
                )
            pick_row = sync_box.row(align=True)
            pick_row.operator_context = "INVOKE_REGION_WIN"
            pick_enabled = pick_row.row(align=True)
            pick_enabled.enabled = (
                settings is not None and settings.image is not None
            )
            pick_label = (
                "Draw Line in Active Match"
                if landmark.kind == "LINE"
                else "Pick in Active Match"
            )
            operator = pick_enabled.operator(
                "perspective_match.interact",
                text=pick_label,
                icon="EYEDROPPER",
            )
            operator.mode = "LANDMARK"
            pick_row.operator(
                "perspective_match.clear_landmark_observation",
                text="",
                icon="X",
            )
            sync_box.prop(workspace, "landmark_pick_confidence")
            if workspace.is_modal and workspace.work_mode == "LANDMARK":
                sync_box.label(
                    text="Landmark tool active — switch matches to pick more",
                    icon="MOUSE_LMB",
                )

            # Compact per-match pick status for the active landmark.
            for root in properties.iter_match_roots():
                observation = scene.observation_for_match(landmark, root)
                row = sync_box.row(align=True)
                if observation is not None and observation.is_set:
                    if landmark.kind == "LINE":
                        row.label(
                            text=(
                                f"{root.name}: "
                                f"({observation.x:.0f},{observation.y:.0f})–"
                                f"({observation.x2:.0f},{observation.y2:.0f})"
                            ),
                            icon="CHECKMARK",
                        )
                    else:
                        row.label(
                            text=f"{root.name}: ({observation.x:.0f}, {observation.y:.0f})",
                            icon="CHECKMARK",
                        )
                    row.prop(observation, "confidence", text="")
                else:
                    row.label(text=f"{root.name}: —", icon="DOT")
            if landmark.has_position or landmark.rmse_px > 0.5:
                detail = f"Last sync RMSE {landmark.rmse_px:.2f} px"
                if landmark.has_position:
                    detail += (
                        f" · ({landmark.position[0]:.2f}, "
                        f"{landmark.position[1]:.2f}, "
                        f"{landmark.position[2]:.2f})"
                    )
                else:
                    detail += " · diagnose/reject"
                sync_box.label(text=detail, icon="EMPTY_AXIS")

        row = sync_box.row(align=True)
        row.operator("perspective_match.solve_sync", icon="FILE_REFRESH")
        row.operator("perspective_match.diagnose_sync", text="Diagnose", icon="INFO")
        row.operator("perspective_match.clear_sync", text="Clear", icon="X")
        empties_row = sync_box.row(align=True)
        empties_row.prop(workspace, "show_landmark_empties", text="Landmark Empties")
        size_row = empties_row.row(align=True)
        size_row.enabled = workspace.show_landmark_empties
        size_row.prop(workspace, "landmark_empty_size", text="Size")
        if workspace.sync_status:
            status_column = sync_box.column(align=True)
            # Blender labels do not wrap — split so the full message is readable.
            wrap_width = 48
            text = workspace.sync_status
            while text:
                status_column.label(text=text[:wrap_width])
                text = text[wrap_width:]
        if settings is not None and settings.sync_is_applied:
            sync_box.label(
                text=(
                    f"This match sync RMSE {settings.sync_rmse_px:.2f} px · "
                    f"s={settings.sync_scale:.3f}"
                ),
                icon="CHECKMARK",
            )


CLASSES = (
    PM_UL_landmarks,
    VIEW3D_PT_perspective_match,
)
