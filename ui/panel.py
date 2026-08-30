"""Blender-native staged sidebar UI for Perspective Match."""

from __future__ import annotations

from pathlib import Path
import textwrap

import bpy

from .. import core, properties, scene
from ..detect import opencv as opencv_support
from . import icons, landmark_list, operators, overlay


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

        rows = properties.landmark_sidebar_rows(_data, _context)
        row_meta = rows[_index] if 0 <= _index < len(rows) else None

        row = layout.row(align=True)
        # Icon toggle — plain Bool+emboss=False reserves a half-row and centers the name.
        sync_icon = "CHECKBOX_HLT" if item.use_in_sync else "CHECKBOX_DEHLT"
        row.prop(item, "use_in_sync", text="", emboss=False, icon=sync_icon)
        # Name gets ~2/3; status icons / count share the remaining third.
        split = row.split(factor=0.67, align=True)
        split.prop(item, "name", text="", emboss=False, icon="EMPTY_AXIS")
        meta = split.row(align=True)
        meta.alignment = "RIGHT"
        if item.kind == "LINE":
            meta.label(text="", icon="MESH_DATA")
            if row_meta is not None and row_meta.parallel_linked:
                meta.label(text="", icon="LINKED")
        elif row_meta is not None and row_meta.mirror_linked:
            meta.label(text="", icon="MOD_MIRROR")
        if item.known_object is not None:
            meta.label(text="", icon="PIVOT_CURSOR")
        elif item.on_ground:
            meta.label(text="", icon="ORIENTATION_VIEW")
        count = 0 if row_meta is None else row_meta.observation_count
        weight = float(getattr(item, "sync_weight", 1.0))
        weight_mark = f" · ×{weight:g}" if abs(weight - 1.0) > 0.05 else ""
        if not item.use_in_sync:
            meta.label(text=f"{count} · off{weight_mark}")
        elif item.rmse_px > 0.5:
            meta.label(text=f"{count} · {item.rmse_px:.0f}px{weight_mark}")
        else:
            meta.label(text=f"{count}{weight_mark}")

    def filter_items(self, context, data, propname):
        """Filter to the active match, then apply the selected list order.

        Read-only: Blender forbids writing Scene ID data from UIList draw.
        creation_index is assigned on add / file load, not here.
        """
        landmarks = getattr(data, propname)
        if not landmarks:
            return [], []
        rows = properties.landmark_sidebar_rows(data, context)
        flt_flags = landmark_list.filter_flags(
            rows,
            filter_current=bool(getattr(data, "landmarks_filter_current_match", False)),
            bitflag=self.bitflag_filter_item,
        )
        flt_neworder = landmark_list.sort_neworder(
            rows,
            sort_alphabetical=bool(getattr(data, "landmarks_sort_alphabetical", False)),
        )
        return flt_flags, flt_neworder


def _section(
    layout,
    idname: str,
    title: str,
    icon: str = "NONE",
    *,
    icon_value: int = 0,
    default_closed: bool = False,
):
    """Create a Blender-native collapsible section; body is None when collapsed."""
    header, body = layout.panel(idname, default_closed=default_closed)
    if icon_value:
        header.label(text=title, icon_value=icon_value)
    else:
        header.label(text=title, icon=icon)
    return header, body


def _section_eye(header, data, prop: str) -> None:
    """Eye toggle on a section header; empty label, hide on/off icons."""
    header.prop(
        data,
        prop,
        text="",
        icon="HIDE_OFF" if getattr(data, prop) else "HIDE_ON",
        emboss=False,
    )


def _mode_tool_active(workspace, mode: str) -> bool:
    return bool(workspace.is_modal and workspace.work_mode == mode)


class VIEW3D_PT_perspective_match(bpy.types.Panel):
    """Perspective Match sidebar panel."""

    bl_label = "Perspective Match"
    bl_idname = "VIEW3D_PT_perspective_match"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Perspective Match"

    def draw(self, context: bpy.types.Context) -> None:
        overlay.note_sidebar_draw(context)
        layout = self.layout
        workspace = properties.workspace(context)
        settings = properties.active_session(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Match Cameras stays open; other sections default closed so a new
        # match with an image does not flood the sidebar.
        _cameras_header, cameras = _section(
            layout, "PM_match_cameras", "Match Cameras", "CAMERA_DATA"
        )
        if cameras is not None:
            create_row = cameras.row(align=True)
            create_row.operator("perspective_match.new_match_camera", icon="ADD")
            create_row.operator(
                "perspective_match.bulk_create_matches",
                text="Bulk Create",
                icon="FILE_FOLDER",
            )
            cameras.prop(workspace, "active_match", text="")
            actions = cameras.row(align=True)
            actions.enabled = settings is not None
            actions.operator(
                "perspective_match.rename_match", text="Rename", icon="FONT_DATA"
            )
            actions.operator(
                "perspective_match.unload_match", text="Unload", icon="X"
            )
            actions.operator(
                "perspective_match.delete_match", text="Delete", icon="TRASH"
            )
            actions.operator(
                "perspective_match.camera_view",
                text="View Match Camera",
                icon="CAMERA_DATA",
            )

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
        from .. import is_dev_install

        if is_dev_install():
            layout.separator()
            layout.operator("perspective_match.reload", icon="FILE_REFRESH")

    def _draw_active_match(self, layout, context, workspace, settings) -> None:
        _image_header, image = _section(
            layout,
            "PM_reference_image",
            "Reference Image",
            "IMAGE_DATA",
            default_closed=True,
        )
        if image is not None:
            if settings.image is not None:
                # Emboss-free operator so the filename can carry a path tooltip.
                name_row = image.row(align=True)
                name_row.alignment = "LEFT"
                label_op = name_row.operator(
                    "perspective_match.reference_image_label",
                    text=(
                        f"{Path(settings.image_path).name} "
                        f"[{settings.image_width} × {settings.image_height} px]"
                    ),
                    emboss=False,
                )
                label_op.path = settings.image_path
            row = image.row(align=True)
            row.operator(
                "perspective_match.load_image", text="Open Image", icon="FILE_IMAGE"
            )
            if settings.image is not None:
                row.operator(
                    "perspective_match.replace_image",
                    text="Replace Image",
                    icon="FILE_REFRESH",
                )

        if settings.image is None:
            return

        line_header, line_body = _section(
            layout,
            "PM_vp_lines",
            "Vanishing Point Lines",
            icon_value=icons.icon_id("vp_lines"),
            default_closed=True,
        )
        _section_eye(line_header, settings, "show_vp_overlay")
        if line_body is not None:
            line_body.prop(settings, "vp_mode", expand=True)
            line_body.prop(settings, "active_axis", expand=True)
            polarity_row = line_body.row()
            polarity_row.enabled = not scene.uses_adjusted_camera(settings)
            polarity_row.prop(settings, "flip_xy_vp_polarity")
            row = line_body.row(align=True)
            draw_row = row.row(align=True)
            draw_row.operator_context = "INVOKE_REGION_WIN"
            operator = draw_row.operator(
                "perspective_match.interact",
                text="Draw / Edit Lines",
                icon="GREASEPENCIL",
                depress=_mode_tool_active(workspace, "LINE"),
            )
            operator.mode = "LINE"
            row.operator("perspective_match.delete_selected", text="", icon="TRASH")
            row.operator("perspective_match.clear_axis", text="", icon="X")
            caps = opencv_support.cached_capabilities()
            if caps is not None and caps.line_segment_detector:
                detect_row = line_body.row(align=True)
                detect_row.operator_context = "INVOKE_DEFAULT"
                detect_row.operator(
                    "perspective_match.detect_vp_lines",
                    text="Detect VP Lines",
                    icon="VIEWZOOM",
                )
                line_body.prop(
                    settings,
                    "vp_detect_sensitivity",
                    text="Edge Sensitivity",
                    slider=True,
                )
                debug_row = line_body.row(align=True)
                debug_row.operator_context = "INVOKE_DEFAULT"
                debug_row.operator(
                    "perspective_match.toggle_vp_detect_debug",
                    text="Debug auto detected edges",
                    icon="SEQ_HISTOGRAM",
                    depress=bool(settings.view_vp_detect_debug),
                )
            line_body.prop(settings, "snap_vp_lines_to_edges")
            line_body.prop(settings, "show_vp_error_labels")

        adjusted_camera = scene.uses_adjusted_camera(settings)

        origin_header, origin = _section(
            layout, "PM_origin", "Origin", "PIVOT_CURSOR", default_closed=True
        )
        _section_eye(origin_header, settings, "show_origin_overlay")
        if origin is not None:
            origin.enabled = not adjusted_camera
            row = origin.row(align=True)
            pick_row = row.row(align=True)
            pick_row.operator_context = "INVOKE_REGION_WIN"
            operator = pick_row.operator(
                "perspective_match.interact",
                text="Re-pick Origin" if settings.origin_is_set else "Pick Origin",
                icon="PIVOT_CURSOR",
                depress=_mode_tool_active(workspace, "ORIGIN"),
            )
            operator.mode = "ORIGIN"
            clear_row = row.row(align=True)
            clear_row.enabled = settings.origin_is_set
            clear_row.operator(
                "perspective_match.clear_placement", text="", icon="X"
            )


        camera_header, camera = _section(
            layout, "PM_camera", "Camera", "CAMERA_DATA", default_closed=True
        )
        _section_eye(camera_header, settings, "show_camera_overlay")
        if camera is not None:
            camera.prop(settings, "camera_control", expand=True)
            if adjusted_camera:
                camera.label(
                    text="Live camera transform and FOV are preserved",
                    icon="LOCKED",
                )
            # Full-width toolbar: property_split would shrink buttons to the value column.
            camera_actions = camera.column(align=True)
            camera_actions.use_property_split = False
            camera_actions.enabled = not adjusted_camera
            focal_row = camera_actions.row(align=True)
            focal_row.prop(settings, "lock_focal", text="Manual FOV", toggle=True)
            focal_row.operator(
                "perspective_match.refine", text="Auto from VPs", icon="FILE_REFRESH"
            )
            focal_row.operator(
                "perspective_match.import_ros_yaml",
                text="Import YAML",
                icon="IMPORT",
            )
            camera_actions.prop(settings, "use_known_3d_in_camera")
            manual_column = camera.column(align=True)
            manual_column.enabled = (
                not adjusted_camera
                and (settings.lock_focal or settings.vp_mode == "1")
            )
            manual_column.prop(settings, "hfov_degrees")
            manual_column.operator(
                "perspective_match.apply_manual_fov", icon="CHECKMARK"
            )
            reset_row = camera.row()
            reset_row.enabled = not adjusted_camera
            reset_row.operator("perspective_match.reset_camera", icon="LOOP_BACK")

            if settings.fov_xy > 0.0 or settings.fov_zy > 0.0 or settings.fov_zx > 0.0:
                diagnostics = camera.column(align=True)
                pair_parts = []
                if settings.fov_xy > 0.0:
                    pair_parts.append(f"XY: {settings.fov_xy:.2f}°")
                if settings.fov_zy > 0.0:
                    pair_parts.append(f"ZY: {settings.fov_zy:.2f}°")
                if settings.fov_zx > 0.0:
                    pair_parts.append(f"ZX: {settings.fov_zx:.2f}°")
                diagnostics.label(
                    text="HFOV by VP pair: " + ", ".join(pair_parts)
                )
                diagnostics.label(
                    text=f"Axis residual: {settings.residual_degrees:.2f}°"
                )
                if settings.vp_line_rms_px >= 0.0:
                    diagnostics.label(
                        text=f"VP line RMSE: {settings.vp_line_rms_px:.2f} px"
                    )
            elif settings.vp_line_rms_px >= 0.0:
                camera.label(
                    text=f"VP line RMSE: {settings.vp_line_rms_px:.2f} px"
                )
            effective = scene.calibration_from_settings(settings)
            if (
                settings.camera_object is not None
                and settings.camera_object.data is not None
            ):
                camera.label(
                    text=(
                        f"Lens: {settings.camera_object.data.lens:.2f} mm · "
                        f"HFOV {effective.hfov_degrees:.2f}°"
                    ),
                    icon="CAMERA_DATA",
                )
            intrinsics = effective.intrinsics
            if abs(intrinsics.cx - settings.image_width * 0.5) > 0.5 or abs(
                intrinsics.cy - settings.image_height * 0.5
            ) > 0.5:
                camera.label(
                    text=(
                        f"PP offset: "
                        f"{intrinsics.cx - settings.image_width * 0.5:+.1f}, "
                        f"{intrinsics.cy - settings.image_height * 0.5:+.1f} px"
                    ),
                    icon="PIVOT_CURSOR",
                )

            pp_row = camera.row(align=True)
            pp_row.enabled = not adjusted_camera
            drag_row = pp_row.row(align=True)
            drag_row.operator_context = "INVOKE_REGION_WIN"
            pp_operator = drag_row.operator(
                "perspective_match.interact",
                text="Manual PP Offset",
                icon="PIVOT_CURSOR",
                depress=_mode_tool_active(workspace, "PP"),
            )
            pp_operator.mode = "PP"
            # Icon-only: type offsets instead of dragging the crosshair.
            edit_row = pp_row.row(align=True)
            edit_row.operator_context = "INVOKE_DEFAULT"
            edit_row.operator(
                "perspective_match.edit_pp_offset",
                text="",
                icon="GREASEPENCIL",
            )

            distortion_row = camera.row()
            distortion_row.enabled = not adjusted_camera
            distortion_row.operator(
                "perspective_match.estimate_distortion",
                text="Estimate Distortion",
                icon="MOD_SIMPLEDEFORM",
            )
            if core.has_brown_conrady(tuple(settings.brown_conrady)):
                k1, k2, p1, p2, k3 = tuple(settings.brown_conrady[:5])
                camera.label(text=f"Imported D: k1 {k1:.4g}  k2 {k2:.4g}  k3 {k3:.4g}")
                if abs(p1) > 1.0e-8 or abs(p2) > 1.0e-8:
                    camera.label(text=f"p1 {p1:.4g}  p2 {p2:.4g}")
            else:
                camera.label(text=f"Division λ: {settings.division_lambda:.5f}")
            if settings.lambda_saturated:
                camera.label(
                    text="Estimate saturated; pinhole retained", icon="ERROR"
                )
            if settings.view_undistorted or core.has_lens_distortion(
                settings.division_lambda,
                tuple(settings.brown_conrady),
            ):
                plates = camera.row(align=True)
                plates.enabled = not adjusted_camera
                plates.operator(
                    "perspective_match.use_undistorted_plate",
                    text="Undistorted Plate",
                    icon="MOD_UVPROJECT",
                )
                plates.operator(
                    "perspective_match.use_original_plate",
                    text="Original Plate",
                    icon="IMAGE_DATA",
                )

        _header, view = _section(
            layout, "PM_view", "View", "IMAGE_RGB", default_closed=True
        )
        if view is not None:
            view.prop(settings, "overlay_opacity")
            view.prop(settings, "view_exposure")
            view.prop(settings, "view_contrast")
            row = view.row(align=True)
            row.operator("perspective_match.apply_view_lighting", icon="CHECKMARK")
            row.operator(
                "perspective_match.reset_view_lighting", text="", icon="LOOP_BACK"
            )

    def _draw_sync(self, layout, context, workspace, settings) -> None:
        sync_header, sync_body = _section(
            layout, "PM_sync", "Sync Matches", "LINKED", default_closed=True
        )
        _section_eye(sync_header, workspace, "show_landmark_overlay")
        if sync_body is None:
            return

        if settings is None:
            return

        # Opt-in for this match — when off, hide the rest of the sync UI.
        # Last-run badge sits on the same row (check vs X after Solve Sync).
        enable_row = sync_body.row(align=True)
        enable_row.use_property_split = False
        enable_row.prop(settings, "sync_enabled", text="Enable sync for current match")
        status = enable_row.row(align=True)
        status.alignment = "RIGHT"
        if settings.sync_last_ok:
            status.label(text="Synced", icon="CHECKMARK")
        else:
            status.label(text="Not synced", icon="X")
        if not settings.sync_enabled:
            return

        lock_pose_row = sync_body.row()
        active_root = properties.active_root(context)
        lock_pose_row.enabled = active_root != properties.anchor_root(context)
        lock_pose_row.prop(settings, "sync_lock_pose", text="Lock Pose in Sync")

        match_count = len(properties.iter_match_roots())
        enabled_count = len(properties.iter_sync_enabled_roots())
        if match_count < 2 or enabled_count < 2:
            return

        sync_body.prop(workspace, "anchor_match", text="Anchor")

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
        
        list_column.operator(
            "perspective_match.duplicate_landmark",
            text="",
            icon="DUPLICATE",
        )

        list_column.operator("perspective_match.remove_landmark", text="", icon="REMOVE")
        
        list_column.separator()

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
        caps = opencv_support.cached_capabilities()
        if caps is not None and caps.apriltags:
            list_column.operator(
                "perspective_match.find_apriltag_landmarks",
                text="",
                icon_value=icons.icon_id("april_tag"),
            )
        
        
        # Separate control group: list display / overlay helpers (not storage).
        list_column.separator()

        list_column.prop(
            workspace,
            "landmarks_sort_alphabetical",
            text="",
            icon="SORTALPHA",
            toggle=True,
        )
        list_column.prop(
            workspace,
            "landmarks_filter_current_match",
            text="",
            icon="FILTER",
            toggle=True,
        )
        list_column.prop(
            workspace,
            "show_landmark_labels",
            text="",
            icon="FONTPREVIEW",
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
            if landmark.kind == "POINT":
                mirror_of_row = sync_body.row(align=True)
                mirror_of_row.prop(landmark, "mirror_of", text="Is Mirror Of")
                mirror_of_row.operator(
                    "perspective_match.guess_mirror_partner",
                    text="",
                    icon="SHADERFX",
                )
            if landmark.known_object is not None:
                location = landmark.known_object.matrix_world.to_translation()
                sync_body.label(
                    text=(
                        f"World ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
                    ),
                    icon="EMPTY_AXIS",
                )
            sync_body.prop(landmark, "sync_weight", slider=True)
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
                depress=_mode_tool_active(workspace, "LANDMARK"),
            )
            operator.mode = "LANDMARK"
            pick_row.operator(
                "perspective_match.clear_landmark_observation",
                text="",
                icon="X",
            )
            if landmark.kind == "POINT":
                caps = opencv_support.cached_capabilities()
                if caps is not None and caps.available:
                    sync_body.prop(workspace, "snap_landmark_to_apriltag")
            _header, confidence_body = _section(
                sync_body,
                "PM_landmark_confidence",
                "Pick Confidence",
                default_closed=True,
            )
            if confidence_body is not None:
                confidence_body.prop(workspace, "landmark_pick_confidence")

                # Compact per-match pick status for the active landmark.
                for root in properties.iter_match_roots():
                    observation = scene.observation_for_match(landmark, root)
                    label = scene.match_prefix(root)
                    row = confidence_body.row(align=True)
                    if observation is not None and observation.is_set:
                        if landmark.kind == "LINE":
                            row.label(
                                text=(
                                    f"{label}: "
                                    f"({observation.x:.0f},{observation.y:.0f})–"
                                    f"({observation.x2:.0f},{observation.y2:.0f})"
                                ),
                                icon="CHECKMARK",
                            )
                        else:
                            row.label(
                                text=(
                                    f"{label}: "
                                    f"({observation.x:.0f}, {observation.y:.0f})"
                                ),
                                icon="CHECKMARK",
                            )
                        row.prop(observation, "confidence", text="")
                    else:
                        row.label(text=f"{label}: —", icon="DOT")
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
                    confidence_body.label(text=detail, icon="EMPTY_AXIS")

        row = sync_body.row(align=True)
        if operators.diagnose_sync_is_running():
            status = row.row(align=True)
            status.enabled = False
            status.label(text="Diagnosing…", icon="INFO")
            row.operator(
                "perspective_match.cancel_diagnose_sync",
                text="Cancel",
                icon="X",
            )
        else:
            row.operator("perspective_match.solve_sync", icon="FILE_REFRESH")
            row.operator_context = "INVOKE_DEFAULT"
            row.operator(
                "perspective_match.diagnose_sync",
                text="Diagnose",
                icon="INFO",
            )
            row.operator("perspective_match.clear_sync", text="Clear", icon="X")
        report_row = sync_body.row(align=True)
        report_row.operator(
            "perspective_match.open_sync_report",
            text="Open Report",
            icon="URL",
        )
        report_row.operator(
            "perspective_match.export_sync_report",
            text="Export",
            icon="EXPORT",
        )
        # Full-width rows — avoid property-split pushing controls to mid-panel.
        lock_row = sync_body.row(align=True)
        lock_row.use_property_split = False
        lock_row.prop(workspace, "lock_rotation", text="Lock Rotation")
        lock_row.prop(workspace, "lock_translation", text="Lock Translation")
        slack_row = sync_body.row(align=True)
        slack_row.use_property_split = False
        slack_row.prop(workspace, "ground_slack", text="Ground Slack")
        slack_row.prop(workspace, "known_3d_slack", text="Known 3D Slack")
        mirror_row = sync_body.row(align=True)
        mirror_row.use_property_split = False
        mirror_row.prop(workspace, "mirror_object", text="Mirror Empty")
        mirror_row.operator(
            "perspective_match.use_selected_mirror",
            text="",
            icon="EYEDROPPER",
        )
        mirror_row.operator(
            "perspective_match.clear_mirror",
            text="",
            icon="X",
        )
        mirror_opts = sync_body.row(align=True)
        mirror_opts.use_property_split = False
        mirror_opts.prop(workspace, "mirror_plane", text="Plane")
        mirror_opts.prop(workspace, "mirror_slack", text="Mirror Slack")
        opts_row = sync_body.row(align=True)
        opts_row.use_property_split = False
        opts_row.prop(workspace, "share_lens", text="Same Lens")
        span = opts_row.row(align=True)
        span.ui_units_x = 4
        span.prop(workspace, "lens_refine_span_percent", text="%")
        refine_row = sync_body.row(align=True)
        if operators.lens_refine_is_running():
            opts_row.enabled = False
            progress = refine_row.row(align=True)
            progress.enabled = False
            progress.prop(
                workspace,
                "lens_refine_progress",
                text="Refining",
                slider=True,
            )
            refine_row.operator(
                "perspective_match.cancel_refine_lenses",
                text="Cancel",
                icon="X",
            )
        else:
            refine_row.operator_context = "INVOKE_DEFAULT"
            refine_row.enabled = not any(
                scene.uses_adjusted_camera(root.pm_session)
                for root in properties.iter_sync_enabled_roots()
            )
            refine_row.operator(
                "perspective_match.refine_lenses",
                icon="CAMERA_DATA",
            )
        empties_row = sync_body.row(align=True)
        empties_row.use_property_split = False
        empties_row.prop(workspace, "show_landmark_empties", text="Landmark Empties")
        size_row = empties_row.row(align=True)
        size_row.enabled = workspace.show_landmark_empties
        size_row.prop(workspace, "landmark_empty_size", text="Size")
        origin_hide_row = sync_body.row(align=True)
        origin_hide_row.use_property_split = False
        origin_hide_row.prop(settings, "hide_origin_empty", text="Hide Origin Empty")
        if workspace.sync_status:
            status_column = sync_body.column(align=True)
            # Blender labels do not wrap; keep word boundaries in the compact status.
            status_lines = textwrap.wrap(workspace.sync_status, width=48)
            for index, line in enumerate(status_lines):
                status_column.label(
                    text=line,
                    icon="INFO" if index == 0 else "NONE",
                )
        if settings is not None and settings.sync_last_ok and settings.sync_is_applied:
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
