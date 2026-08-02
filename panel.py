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
        if self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="EMPTY_AXIS")
            return

        row = layout.row(align=True)
        # Icon toggle — plain Bool+emboss=False reserves a half-row and centers the name.
        sync_icon = "CHECKBOX_HLT" if item.use_in_sync else "CHECKBOX_DEHLT"
        row.prop(item, "use_in_sync", text="", emboss=False, icon=sync_icon)
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
        if not item.use_in_sync:
            row.label(text=f"{count} · off")
        elif item.rmse_px > 0.5:
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


def _section(
    layout,
    idname: str,
    title: str,
    icon: str = "NONE",
    *,
    default_closed: bool = False,
):
    """Create a Blender-native collapsible section; body is None when collapsed."""
    header, body = layout.panel(idname, default_closed=default_closed)
    header.label(text=title, icon=icon)
    return header, body


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

        _header, cameras = _section(
            layout, "PM_match_cameras", "Match Cameras", "CAMERA_DATA"
        )
        if cameras is not None:
            row = cameras.row(align=True)
            row.operator("perspective_match.new_match_camera", icon="ADD")
            row.operator("perspective_match.unload_match", text="", icon="X")
            cameras.prop(workspace, "active_match", text="")
            if settings is None:
                cameras.label(text="Create or select a match camera", icon="INFO")

        if settings is not None:
            self._draw_active_match(layout, context, workspace, settings)

        self._draw_sync(layout, context, workspace, settings)

        if settings is not None and settings.error:
            layout.alert = True
            layout.label(text=settings.error, icon="ERROR")
            layout.alert = False
        if settings is not None:
            layout.label(text=settings.status, icon="INFO")
        elif workspace.sync_status:
            layout.label(text=workspace.sync_status, icon="INFO")
        layout.separator()
        layout.operator("perspective_match.reload", icon="FILE_REFRESH")

    def _draw_active_match(self, layout, context, workspace, settings) -> None:
        _header, image = _section(
            layout, "PM_reference_image", "1. Reference Image", "IMAGE_DATA"
        )
        if image is not None:
            row = image.row(align=True)
            row.operator(
                "perspective_match.load_image", text="Open Image", icon="FILE_IMAGE"
            )
            row.operator(
                "perspective_match.import_project",
                text="Import Project",
                icon="FILE_FOLDER",
            )
            if settings.image is None:
                image.label(
                    text="Load a still or .pmproj into this match", icon="INFO"
                )
            else:
                image.label(text=Path(settings.image_path).name, icon="CHECKMARK")
                image.label(
                    text=f"{settings.image_width} × {settings.image_height} px"
                )
                image.operator("perspective_match.camera_view", icon="CAMERA_DATA")

        if settings.image is None:
            return

        _header, perspective = _section(
            layout, "PM_perspective", "2. Perspective", "ORIENTATION_GLOBAL"
        )
        if perspective is not None:
            perspective.prop(settings, "vp_mode", expand=True)
            if settings.vp_mode == "1":
                tip = "1-point: Y + Z lines; FOV stays manual"
            elif settings.vp_mode == "2":
                tip = "2-point: X + Y horizontals; Z uprights not used"
            else:
                tip = "3-point: any two axes (2+ lines each); the third is derived"
            perspective.label(text=tip, icon="INFO")

        line_header, line_body = _section(
            layout, "PM_vp_lines", "3. VP Lines", "TRACKING"
        )
        line_header.prop(
            settings,
            "show_vp_overlay",
            text="",
            icon="HIDE_OFF" if settings.show_vp_overlay else "HIDE_ON",
            emboss=False,
        )
        if line_body is not None:
            line_body.prop(settings, "active_axis", expand=True)
            counts = _axis_counts(settings)
            required = f"X {counts['x']} · Y {counts['z']} · Z {counts['y']}"
            line_body.label(text=required, icon="DRIVER_DISTANCE")
            row = line_body.row(align=True)
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
                line_body.label(text="Line tool active — Esc exits", icon="MOUSE_LMB")
            if not _panel_lines_ready(settings):
                line_body.label(text=_panel_lines_hint(settings), icon="INFO")

        _header, origin = _section(layout, "PM_origin", "4. Origin", "PIVOT_CURSOR")
        if origin is not None:
            row = origin.row(align=True)
            pick_row = row.row(align=True)
            pick_row.operator_context = "INVOKE_REGION_WIN"
            operator = pick_row.operator(
                "perspective_match.interact",
                text="Pick Origin",
                icon="PIVOT_CURSOR",
            )
            operator.mode = "ORIGIN"
            clear_row = row.row(align=True)
            clear_row.enabled = settings.origin_is_set
            clear_row.operator(
                "perspective_match.clear_placement", text="", icon="X"
            )
            if settings.origin_is_set:
                origin.label(text="Origin set", icon="CHECKMARK")
            if workspace.is_modal and workspace.work_mode == "ORIGIN":
                origin.label(
                    text="Origin tool active — click ground point", icon="MOUSE_LMB"
                )

        _header, camera = _section(layout, "PM_camera", "5. Camera", "CAMERA_DATA")
        if camera is not None:
            focal_row = camera.row(align=True)
            focal_row.prop(settings, "lock_focal", text="Manual FOV", toggle=True)
            focal_row.operator(
                "perspective_match.refine", text="Auto from VPs", icon="FILE_REFRESH"
            )
            manual_column = camera.column(align=True)
            manual_column.enabled = settings.lock_focal or settings.vp_mode == "1"
            manual_column.prop(settings, "hfov_degrees")
            manual_column.operator(
                "perspective_match.apply_manual_fov", icon="CHECKMARK"
            )
            camera.operator("perspective_match.reset_camera", icon="LOOP_BACK")

            if settings.fov_xy > 0.0 or settings.fov_zy > 0.0 or settings.fov_zx > 0.0:
                diagnostics = camera.column(align=True)
                diagnostics.label(text="HFOV by VP pair:")
                if settings.fov_xy > 0.0:
                    diagnostics.label(text=f"XY: {settings.fov_xy:.2f}°")
                if settings.fov_zy > 0.0:
                    diagnostics.label(text=f"ZY: {settings.fov_zy:.2f}°")
                if settings.fov_zx > 0.0:
                    diagnostics.label(text=f"ZX: {settings.fov_zx:.2f}°")
                diagnostics.label(
                    text=f"Axis residual: {settings.residual_degrees:.2f}°"
                )
            if (
                settings.camera_object is not None
                and settings.camera_object.data is not None
            ):
                camera.label(
                    text=(
                        f"Lens: {settings.camera_object.data.lens:.2f} mm · "
                        f"HFOV {settings.hfov_degrees:.2f}°"
                    ),
                    icon="CAMERA_DATA",
                )
            if abs(settings.cx - settings.image_width * 0.5) > 0.5 or abs(
                settings.cy - settings.image_height * 0.5
            ) > 0.5:
                camera.label(
                    text=(
                        f"PP offset: "
                        f"{settings.cx - settings.image_width * 0.5:+.1f}, "
                        f"{settings.cy - settings.image_height * 0.5:+.1f} px"
                    ),
                    icon="PIVOT_CURSOR",
                )

            pp_row = camera.row(align=True)
            pp_row.operator_context = "INVOKE_REGION_WIN"
            pp_operator = pp_row.operator(
                "perspective_match.interact",
                text="Manual PP Offset",
                icon="PIVOT_CURSOR",
            )
            pp_operator.mode = "PP"
            if workspace.is_modal and workspace.work_mode == "PP":
                camera.label(
                    text="PP tool active — drag violet crosshair",
                    icon="MOUSE_LMB",
                )

            camera.prop(settings, "estimate_distortion")
            camera.label(text=f"Division λ: {settings.division_lambda:.5f}")
            if settings.lambda_saturated:
                camera.label(
                    text="Estimate saturated; pinhole retained", icon="ERROR"
                )
            if settings.view_undistorted or settings.estimate_distortion:
                camera.operator(
                    "perspective_match.use_original_plate",
                    text="Original Plate",
                    icon="IMAGE_DATA",
                )

        _header, view = _section(
            layout, "PM_view", "View", "IMAGE_RGB", default_closed=True
        )
        if view is not None:
            view.label(text="Display only — does not affect the solve", icon="INFO")
            view.prop(settings, "view_exposure")
            view.prop(settings, "view_contrast")
            row = view.row(align=True)
            row.operator("perspective_match.apply_view_lighting", icon="CHECKMARK")
            row.operator(
                "perspective_match.reset_view_lighting", text="", icon="LOOP_BACK"
            )
            if settings.view_lighting_applied:
                view.label(
                    text=Path(settings.view_path).name or "View plate active",
                    icon="CHECKMARK",
                )
            view.prop(settings, "overlay_opacity")

    def _draw_sync(self, layout, context, workspace, settings) -> None:
        sync_header, sync_body = _section(
            layout, "PM_sync", "6. Sync Matches", "LINKED"
        )
        sync_header.prop(
            workspace,
            "show_landmark_overlay",
            text="",
            icon="HIDE_OFF" if workspace.show_landmark_overlay else "HIDE_ON",
            emboss=False,
        )
        if sync_body is None:
            return

        match_count = len(properties.iter_match_roots())
        if match_count < 2:
            sync_body.label(text="Create at least two matched cameras", icon="INFO")
            return

        sync_body.prop(workspace, "anchor_match", text="Anchor")
        sync_body.label(
            text="Points or lines across stills; Known 3D Empties optional",
            icon="INFO",
        )

        list_row = sync_body.row()
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
            sync_body.prop(landmark, "use_in_sync")
            sync_body.prop(landmark, "kind")
            if landmark.kind == "POINT":
                sync_body.prop(landmark, "on_ground")
            known_row = sync_body.row(align=True)
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
                sync_body.prop(landmark, "known_object_b", text="Known 3D B")
                sync_body.prop(landmark, "parallel_to", text="Is Parallel To")
                sync_body.label(
                    text="Optional: two Empties = metric edge; else draw in ≥3 stills",
                    icon="INFO",
                )
            if landmark.known_object is not None:
                location = landmark.known_object.matrix_world.to_translation()
                sync_body.label(
                    text=(
                        f"World ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
                    ),
                    icon="EMPTY_AXIS",
                )
            pick_row = sync_body.row(align=True)
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
            sync_body.prop(workspace, "landmark_pick_confidence")
            if workspace.is_modal and workspace.work_mode == "LANDMARK":
                sync_body.label(
                    text="Landmark tool active — switch matches to pick more",
                    icon="MOUSE_LMB",
                )

            # Compact per-match pick status for the active landmark.
            for root in properties.iter_match_roots():
                observation = scene.observation_for_match(landmark, root)
                row = sync_body.row(align=True)
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
                            text=(
                                f"{root.name}: "
                                f"({observation.x:.0f}, {observation.y:.0f})"
                            ),
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
                sync_body.label(text=detail, icon="EMPTY_AXIS")

        row = sync_body.row(align=True)
        row.operator("perspective_match.solve_sync", icon="FILE_REFRESH")
        row.operator("perspective_match.diagnose_sync", text="Diagnose", icon="INFO")
        row.operator("perspective_match.clear_sync", text="Clear", icon="X")
        refine_row = sync_body.row(align=True)
        refine_row.operator(
            "perspective_match.refine_lenses",
            icon="CAMERA_DATA",
        )
        span = refine_row.row(align=True)
        span.ui_units_x = 4
        span.prop(workspace, "lens_refine_span_percent", text="%")
        empties_row = sync_body.row(align=True)
        empties_row.prop(workspace, "show_landmark_empties", text="Landmark Empties")
        size_row = empties_row.row(align=True)
        size_row.enabled = workspace.show_landmark_empties
        size_row.prop(workspace, "landmark_empty_size", text="Size")
        if workspace.sync_status:
            status_column = sync_body.column(align=True)
            # Blender labels do not wrap — split so the full message is readable.
            wrap_width = 48
            text = workspace.sync_status
            while text:
                status_column.label(text=text[:wrap_width])
                text = text[wrap_width:]
        if settings is not None and settings.sync_is_applied:
            sync_body.label(
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
