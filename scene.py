"""Blender scene integration and image/camera coordinate mapping."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from . import core, properties

CV_CAMERA_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])


def safe_identifier(name: str) -> str:
    """Return a compact Blender-friendly id derived from a file stem."""
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return (cleaned.strip("_") or "image")[:60]


def line_bundles_from_settings(settings: properties.PMSettings) -> dict[core.AxisId, list[core.LineSegment]]:
    """Convert RNA line collection to solver dataclasses, filtered by VP mode."""
    output: dict[core.AxisId, list[core.LineSegment]] = {"x": [], "y": [], "z": []}
    for item in settings.lines:
        output[item.axis].append(core.LineSegment(item.x1, item.y1, item.x2, item.y2))
    if settings.vp_mode == "1":
        output["x"] = []
    elif settings.vp_mode == "2":
        output["y"] = []
    return output


def calibration_from_settings(settings: properties.PMSettings) -> core.Calibration:
    """Build an immutable solver calibration from scene settings."""
    intrinsics = core.CameraIntrinsics(
        fx=max(settings.fx, 1.0),
        fy=max(settings.fy, 1.0),
        cx=settings.cx,
        cy=settings.cy,
        image_width=max(settings.image_width, 1),
        image_height=max(settings.image_height, 1),
    )
    rotation = np.asarray(settings.rotation_w2c, dtype=np.float64).reshape(3, 3)
    center = np.asarray(settings.camera_center, dtype=np.float64)
    return core.Calibration(
        intrinsics=intrinsics,
        rotation_w2c=rotation,
        camera_center=center,
        division_lambda=settings.division_lambda,
        lambda_saturated=settings.lambda_saturated,
    )


def store_calibration(settings: properties.PMSettings, calibration: core.Calibration) -> None:
    """Copy a solver result into persistent RNA properties."""
    intrinsics = calibration.intrinsics
    settings.fx = intrinsics.fx
    settings.fy = intrinsics.fy
    settings.cx = intrinsics.cx
    settings.cy = intrinsics.cy
    settings.hfov_degrees = calibration.hfov_degrees
    settings.rotation_w2c = tuple(float(value) for value in calibration.rotation_w2c.reshape(-1))
    settings.camera_center = tuple(float(value) for value in calibration.camera_center)
    settings.division_lambda = calibration.division_lambda
    settings.lambda_saturated = calibration.lambda_saturated


def _remove_previous_match(settings: properties.PMSettings) -> None:
    """Remove the previous managed collection and objects when loading a new still."""
    cached_image = settings.undistorted_image
    collection = settings.match_collection
    if collection is None:
        return
    for obj in list(collection.objects):
        _remove_managed_object(obj)
    bpy.data.collections.remove(collection)
    settings.match_collection = None
    settings.root_object = None
    settings.camera_object = None
    settings.undistorted_image = None
    if cached_image is not None and cached_image.users == 0:
        bpy.data.images.remove(cached_image)


def _remove_managed_object(obj: bpy.types.Object) -> None:
    """Remove an owned object and its now-unused data/material datablocks."""
    data = obj.data
    materials = list(data.materials) if isinstance(data, bpy.types.Mesh) else []
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)
    elif isinstance(data, bpy.types.Camera) and data.users == 0:
        bpy.data.cameras.remove(data)
    for material in materials:
        if material.users == 0:
            bpy.data.materials.remove(material)


def _link_only(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def _parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    matrix_world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = matrix_world


def setup_reference_image(
    context: bpy.types.Context,
    image_path: str,
) -> bpy.types.Object:
    """Create a managed camera/background hierarchy for a reference image."""
    settings = context.scene.match_perspective
    absolute_path = str(Path(bpy.path.abspath(image_path)).expanduser().resolve())
    image = bpy.data.images.load(absolute_path, check_existing=True)
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("The selected image has no readable pixel dimensions")

    _remove_previous_match(settings)
    stem = safe_identifier(Path(absolute_path).stem)
    prefix = f"PM_{stem}"
    collection = bpy.data.collections.new(prefix)
    context.scene.collection.children.link(collection)

    root = bpy.data.objects.new(f"{prefix}_Origin", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 0.5
    collection.objects.link(root)

    camera_data = bpy.data.cameras.new(f"{prefix}_Camera")
    camera_object = bpy.data.objects.new(f"{prefix}_Camera", camera_data)
    collection.objects.link(camera_object)
    _parent_keep_world(camera_object, root)

    camera_data.type = "PERSP"
    camera_data.lens_unit = "MILLIMETERS"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.show_background_images = True
    camera_data.background_images.clear()
    background = camera_data.background_images.new()
    background.image = image
    background.show_background_image = True
    background.alpha = 1.0
    # Keep the plate visible without covering modeled geometry.
    background.display_depth = "BACK"
    background.frame_method = "STRETCH"
    if hasattr(image, "use_view_as_render"):
        image.use_view_as_render = True

    settings.image = image
    settings.image_path = absolute_path
    settings.image_width = width
    settings.image_height = height
    settings.source_image_width = width
    settings.match_collection = collection
    settings.root_object = root
    settings.camera_object = camera_object
    settings.lines.clear()
    settings.surfaces.clear()
    settings.selected_line_index = -1
    settings.selected_surface_index = -1
    settings.is_modal = False
    settings.work_mode = "NONE"
    settings.origin_is_set = False
    settings.scale_point_count = 0
    settings.solved_scale = 1.0
    settings.project_path = ""
    settings.source_session_json = ""
    settings.undistorted_path = ""
    settings.undistorted_image = None
    settings.undistorted_width = 0
    settings.undistorted_height = 0
    settings.undistorted_offset_x = 0.0
    settings.undistorted_offset_y = 0.0
    settings.view_undistorted = False
    settings.error = ""

    focal = core.focal_from_hfov(settings.hfov_degrees, width)
    initial_center = np.array((0.0, -5.0, 1.7), dtype=np.float64)
    forward_world = -initial_center
    forward_world /= np.linalg.norm(forward_world)
    right_world = np.cross(forward_world, np.array((0.0, 0.0, 1.0)))
    right_world /= np.linalg.norm(right_world)
    up_world = np.cross(right_world, forward_world)
    initial_rotation = np.stack([right_world, -up_world, forward_world], axis=0)
    calibration = core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=width * 0.5,
            cy=height * 0.5,
            image_width=width,
            image_height=height,
        ),
        rotation_w2c=initial_rotation,
        camera_center=initial_center,
    )
    store_calibration(settings, calibration)
    apply_camera(context.scene, settings, calibration)
    context.scene.camera = camera_object
    context.scene.render.resolution_x = width
    context.scene.render.resolution_y = height
    context.scene.render.resolution_percentage = 100
    context.scene.render.pixel_aspect_x = 1.0
    context.scene.render.pixel_aspect_y = 1.0
    settings.status = "Choose a perspective mode, then draw VP lines"
    enter_camera_view(context)
    properties.tag_viewport_redraw(context)
    return camera_object


def apply_camera(
    blender_scene: bpy.types.Scene,
    settings: properties.PMSettings,
    calibration: core.Calibration,
) -> None:
    """Apply solved intrinsics and OpenCV extrinsics to the managed Blender camera."""
    camera_object = settings.camera_object
    if camera_object is None or camera_object.type != "CAMERA":
        raise ValueError("Load a reference image before refining the camera")
    intrinsics = calibration.intrinsics
    source_width = max(float(settings.source_image_width), 1.0)
    use_undistorted = (
        settings.view_undistorted
        and settings.undistorted_image is not None
        and settings.undistorted_width > 0
        and settings.undistorted_height > 0
    )
    plate_width = (
        float(settings.undistorted_width)
        if use_undistorted
        else float(intrinsics.image_width)
    )
    plate_height = (
        int(settings.undistorted_height)
        if use_undistorted
        else int(intrinsics.image_height)
    )
    plate_cx = (
        intrinsics.cx - settings.undistorted_offset_x
        if use_undistorted
        else intrinsics.cx
    )
    plate_cy = (
        intrinsics.cy - settings.undistorted_offset_y
        if use_undistorted
        else intrinsics.cy
    )
    camera_data = camera_object.data
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0 * plate_width / source_width
    camera_data.lens = intrinsics.fx * 36.0 / source_width
    camera_data.shift_x = (plate_width * 0.5 - plate_cx) / plate_width
    camera_data.shift_y = (plate_cy - plate_height * 0.5) / plate_width
    if len(camera_data.background_images) > 0:
        camera_data.background_images[0].image = (
            settings.undistorted_image if use_undistorted else settings.image
        )

    camera_to_world_blender = calibration.rotation_w2c.T @ CV_CAMERA_TO_BLENDER_CAMERA
    matrix_world = Matrix(camera_to_world_blender.tolist()).to_4x4()
    matrix_world.translation = Vector(calibration.camera_center.tolist())
    camera_object.matrix_world = matrix_world
    blender_scene.camera = camera_object
    blender_scene.render.resolution_x = int(plate_width)
    blender_scene.render.resolution_y = plate_height
    blender_scene.render.resolution_percentage = 100
    store_calibration(settings, calibration)


def refine_match(context: bpy.types.Context) -> core.Calibration:
    """Refine and apply camera state from all current VP lines."""
    settings = context.scene.match_perspective
    if settings.image is None:
        raise ValueError("Load a reference image first")
    line_bundles = line_bundles_from_settings(settings)
    if not any(len(segments) >= 2 for segments in line_bundles.values()):
        raise ValueError("Draw at least two lines for two required axes")

    previous_calibration = calibration_from_settings(settings)
    intrinsics = previous_calibration.intrinsics
    if settings.lock_focal or settings.vp_mode == "1":
        focal = core.focal_from_hfov(settings.hfov_degrees, intrinsics.image_width)
        intrinsics.fx = focal
        intrinsics.fy = focal

    calibration = core.refine_camera(
        line_bundles,
        intrinsics,
        lock_focal=settings.lock_focal or settings.vp_mode == "1",
        estimate_principal_point=settings.vp_mode == "3" and not settings.lock_focal,
        estimate_distortion=settings.estimate_distortion and not settings.lock_focal,
        initial_division_lambda=previous_calibration.division_lambda,
    )

    if settings.origin_is_set:
        origin = tuple(float(value) for value in settings.origin_image)
        point_a = (
            tuple(float(value) for value in settings.scale_point_a)
            if settings.scale_point_count >= 1
            else None
        )
        point_b = (
            tuple(float(value) for value in settings.scale_point_b)
            if settings.scale_point_count >= 2
            else None
        )
        calibration.camera_center, settings.solved_scale = core.apply_origin_and_scale(
            calibration,
            origin,
            point_a,
            point_b,
            settings.measured_length,
        )
    else:
        calibration.camera_center = core.default_camera_center(calibration.rotation_w2c)

    if _intrinsics_or_distortion_changed(previous_calibration, calibration):
        invalidate_undistorted_cache(settings)
    apply_camera(context.scene, settings, calibration)
    _update_diagnostics(settings, line_bundles, calibration)
    rebuild_surface_meshes(context, calibration)
    settings.status = (
        f"Camera matched · HFOV {calibration.hfov_degrees:.2f}°"
        + (f" · λ {calibration.division_lambda:.4f}" if abs(calibration.division_lambda) > 1.0e-6 else "")
    )
    settings.error = ""
    properties.tag_viewport_redraw(context)
    return calibration


def _update_diagnostics(
    settings: properties.PMSettings,
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
    calibration: core.Calibration,
) -> None:
    """Update compact plane-FOV and angular-residual panel diagnostics."""
    working_lines = core.undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    working = core.collect_vanishing_points(working_lines)
    estimates = core.focal_estimates_by_pair(
        working,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
    )
    settings.fov_xy = (
        core.hfov_from_focal(estimates["XY"], calibration.intrinsics.image_width)
        if "XY" in estimates
        else 0.0
    )
    settings.fov_zy = (
        core.hfov_from_focal(estimates["ZY"], calibration.intrinsics.image_width)
        if "ZY" in estimates
        else 0.0
    )
    settings.fov_zx = (
        core.hfov_from_focal(estimates["ZX"], calibration.intrinsics.image_width)
        if "ZX" in estimates
        else 0.0
    )
    directions = {
        axis: core._normalized_direction(vanishing, calibration.intrinsics)
        for axis, vanishing in working.items()
    }
    axis_columns = {"x": 0, "z": 1, "y": 2}
    errors = []
    for axis, direction in directions.items():
        cosine = abs(float(np.dot(direction, calibration.rotation_w2c[:, axis_columns[axis]])))
        errors.append(float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))))
    settings.residual_degrees = max(errors, default=0.0)


def apply_manual_fov(context: bpy.types.Context) -> None:
    """Apply the current manual HFOV before or without enough VP lines."""
    settings = context.scene.match_perspective
    calibration = calibration_from_settings(settings)
    previous_focal = calibration.intrinsics.fx
    focal = core.focal_from_hfov(settings.hfov_degrees, settings.image_width)
    calibration.intrinsics.fx = focal
    calibration.intrinsics.fy = focal
    if abs(previous_focal - focal) > 1.0e-6:
        invalidate_undistorted_cache(settings)
    apply_camera(context.scene, settings, calibration)
    if settings.origin_is_set:
        reapply_placement(context)
    else:
        rebuild_surface_meshes(context, calibration)
    settings.status = f"Manual HFOV {settings.hfov_degrees:.2f}°"
    properties.tag_viewport_redraw(context)


def _intrinsics_or_distortion_changed(
    previous: core.Calibration,
    current: core.Calibration,
) -> bool:
    """Return whether a cached undistortion remap is no longer valid."""
    first = previous.intrinsics
    second = current.intrinsics
    return (
        abs(first.fx - second.fx) > 1.0e-6
        or abs(first.fy - second.fy) > 1.0e-6
        or abs(first.cx - second.cx) > 1.0e-6
        or abs(first.cy - second.cy) > 1.0e-6
        or abs(previous.division_lambda - current.division_lambda) > 1.0e-9
    )


def invalidate_undistorted_cache(settings: properties.PMSettings) -> None:
    """Switch to source plate and discard a remap made with stale intrinsics."""
    cached_image = settings.undistorted_image
    settings.view_undistorted = False
    if (
        settings.camera_object is not None
        and len(settings.camera_object.data.background_images) > 0
    ):
        settings.camera_object.data.background_images[0].image = settings.image
    settings.undistorted_image = None
    settings.undistorted_path = ""
    settings.undistorted_width = 0
    settings.undistorted_height = 0
    settings.undistorted_offset_x = 0.0
    settings.undistorted_offset_y = 0.0
    if cached_image is not None and cached_image.users == 0:
        bpy.data.images.remove(cached_image)


def set_origin_or_scale(
    context: bpy.types.Context,
    image_point: tuple[float, float],
    mode: str,
) -> None:
    """Store an origin or scale point and reapply camera placement."""
    settings = context.scene.match_perspective
    if mode == "ORIGIN":
        settings.origin_image = image_point
        settings.origin_is_set = True
        settings.status = "Origin set on the ground plane"
    elif settings.scale_point_count == 0:
        settings.scale_point_a = image_point
        settings.scale_point_count = 1
        settings.status = "Scale point 1 set · pick point 2"
        properties.tag_viewport_redraw(context)
        return
    else:
        settings.scale_point_b = image_point
        settings.scale_point_count = 2
        settings.status = "Scale reference set"
    reapply_placement(context)


def reapply_placement(context: bpy.types.Context) -> None:
    """Apply placement to the current solved camera without requiring a new VP solve."""
    settings = context.scene.match_perspective
    calibration = calibration_from_settings(settings)
    if not settings.origin_is_set:
        return
    calibration.camera_center = core.default_camera_center(calibration.rotation_w2c)
    point_a = (
        tuple(float(value) for value in settings.scale_point_a)
        if settings.scale_point_count >= 1
        else None
    )
    point_b = (
        tuple(float(value) for value in settings.scale_point_b)
        if settings.scale_point_count >= 2
        else None
    )
    calibration.camera_center, settings.solved_scale = core.apply_origin_and_scale(
        calibration,
        tuple(float(value) for value in settings.origin_image),
        point_a,
        point_b,
        settings.measured_length,
    )
    apply_camera(context.scene, settings, calibration)
    rebuild_surface_meshes(context, calibration)
    properties.tag_viewport_redraw(context)


def surface_ideal_geometry(
    settings: properties.PMSettings,
    surface: properties.PMSurface,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[tuple[float, float], tuple[float, float]]],
] | None:
    """Build a surface in ideal pinhole coordinates from distorted stored data."""
    calibration = calibration_from_settings(settings)
    ideal_lines = core.undistort_line_bundles(
        line_bundles_from_settings(settings),
        calibration.intrinsics,
        calibration.division_lambda,
    )
    vanishing_points = core.collect_vanishing_points(ideal_lines)
    resolved = core.surface_vanishing_points(surface.plane, vanishing_points)
    if resolved is None:
        return None
    diagonal = core.undistort_points(
        np.array([[surface.x1, surface.y1], [surface.x2, surface.y2]]),
        calibration.intrinsics.fx,
        calibration.intrinsics.fy,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
        calibration.division_lambda,
    )
    corners = core.perspective_rectangle_corners(
        (float(diagonal[0, 0]), float(diagonal[0, 1])),
        (float(diagonal[1, 0]), float(diagonal[1, 1])),
        resolved[0],
        resolved[1],
    )
    if corners is None:
        return None
    return corners, core.surface_grid(corners, surface.divisions)


def surface_image_geometry(
    settings: properties.PMSettings,
    surface: properties.PMSurface,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[tuple[float, float], tuple[float, float]]],
] | None:
    """Return perspective surface corners and its equal-world-spacing grid."""
    ideal_geometry = surface_ideal_geometry(settings, surface)
    if ideal_geometry is None:
        return None
    ideal_corners, ideal_grid = ideal_geometry
    calibration = calibration_from_settings(settings)

    def to_storage(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        mapped = core.distort_points(
            np.asarray(points, dtype=np.float64),
            calibration.intrinsics.fx,
            calibration.intrinsics.fy,
            calibration.intrinsics.cx,
            calibration.intrinsics.cy,
            calibration.division_lambda,
        )
        return [(float(point[0]), float(point[1])) for point in mapped]

    storage_corners = to_storage(ideal_corners)
    storage_grid = []
    for point_a, point_b in ideal_grid:
        mapped = to_storage([point_a, point_b])
        storage_grid.append((mapped[0], mapped[1]))
    return storage_corners, storage_grid


def rebuild_surface_meshes(
    context: bpy.types.Context,
    calibration: core.Calibration | None = None,
) -> None:
    """Create or update all managed surface mesh objects."""
    settings = context.scene.match_perspective
    if settings.match_collection is None:
        return
    solved = calibration or calibration_from_settings(settings)
    prefix = settings.match_collection.name
    plane_colors = {
        "xz": (0.77, 0.94, 0.30, 0.35),
        "yz": (0.94, 0.76, 0.30, 0.35),
        "yx": (1.0, 0.42, 0.29, 0.35),
    }
    for index, surface in enumerate(settings.surfaces):
        ideal_geometry = surface_ideal_geometry(settings, surface)
        if ideal_geometry is None:
            remove_surface_mesh(surface)
            continue
        world_corners = core.reconstruct_surface_world(
            surface.plane,
            ideal_geometry[0],
            solved,
        )
        if world_corners is None:
            remove_surface_mesh(surface)
            continue
        obj = surface.mesh_object
        if obj is None or obj.name not in bpy.data.objects:
            mesh = bpy.data.meshes.new(f"{prefix}_Surface_{index:02d}")
            obj = bpy.data.objects.new(f"{prefix}_Surface_{index:02d}", mesh)
            settings.match_collection.objects.link(obj)
            if settings.root_object is not None:
                _parent_keep_world(obj, settings.root_object)
            material = bpy.data.materials.new(f"{prefix}_SurfaceMat_{index:02d}")
            material.diffuse_color = plane_colors[surface.plane]
            mesh.materials.append(material)
            surface.mesh_object = obj
        mesh = obj.data
        mesh.clear_geometry()
        mesh.from_pydata(world_corners, [], [(0, 1, 2, 3)])
        mesh.update()


def remove_surface_mesh(surface: properties.PMSurface) -> None:
    """Delete the mesh object managed by one surface item."""
    if surface.mesh_object is not None:
        obj = surface.mesh_object
        surface.mesh_object = None
        _remove_managed_object(obj)


def enter_camera_view(context: bpy.types.Context) -> None:
    """Switch the current 3D View to the managed scene camera."""
    if context.area is None or context.area.type != "VIEW_3D":
        return
    context.space_data.camera = context.scene.match_perspective.camera_object
    region_3d = getattr(context.space_data, "region_3d", None)
    if region_3d is not None:
        region_3d.view_perspective = "CAMERA"


def is_camera_view(context: bpy.types.Context) -> bool:
    """Return whether input occurs in the managed camera view."""
    settings = context.scene.match_perspective
    region_3d = getattr(context.space_data, "region_3d", None)
    return (
        context.area is not None
        and context.area.type == "VIEW_3D"
        and context.region is not None
        and context.region.type == "WINDOW"
        and region_3d is not None
        and region_3d.view_perspective == "CAMERA"
        and settings.camera_object is not None
    )


def camera_frame_bounds(
    context: bpy.types.Context,
) -> tuple[float, float, float, float] | None:
    """Project camera-frame corners to region pixels as left, right, bottom, top."""
    if not is_camera_view(context):
        return None
    camera_object = context.scene.match_perspective.camera_object
    region_3d = context.space_data.region_3d
    frame_local = camera_object.data.view_frame(scene=context.scene)
    projected = []
    for corner in frame_local:
        screen = view3d_utils.location_3d_to_region_2d(
            context.region,
            region_3d,
            camera_object.matrix_world @ corner,
        )
        if screen is not None:
            projected.append(screen)
    if len(projected) != 4:
        return None
    x_values = [float(point.x) for point in projected]
    y_values = [float(point.y) for point in projected]
    return min(x_values), max(x_values), min(y_values), max(y_values)


def image_to_region(
    context: bpy.types.Context,
    image_x: float,
    image_y: float,
) -> Vector | None:
    """Map top-left-origin image pixels into current camera-view region pixels."""
    bounds = camera_frame_bounds(context)
    settings = context.scene.match_perspective
    if bounds is None or settings.image_width <= 0 or settings.image_height <= 0:
        return None
    left, right, bottom, top = bounds
    display_x, display_y, display_width, display_height = _storage_to_display(
        settings,
        image_x,
        image_y,
    )
    return Vector(
        (
            left + display_x / display_width * (right - left),
            top - display_y / display_height * (top - bottom),
        )
    )


def ideal_to_region(
    context: bpy.types.Context,
    ideal_x: float,
    ideal_y: float,
) -> Vector | None:
    """Map ideal pinhole pixels to the active original or undistorted plate."""
    settings = context.scene.match_perspective
    if settings.view_undistorted:
        bounds = camera_frame_bounds(context)
        if bounds is None:
            return None
        left, right, bottom, top = bounds
        display_width, display_height = _display_size(settings)
        display_x = ideal_x - settings.undistorted_offset_x
        display_y = ideal_y - settings.undistorted_offset_y
        return Vector(
            (
                left + display_x / display_width * (right - left),
                top - display_y / display_height * (top - bottom),
            )
        )
    storage = core.distort_points(
        np.array([[ideal_x, ideal_y]], dtype=np.float64),
        settings.fx,
        settings.fy,
        settings.cx,
        settings.cy,
        settings.division_lambda,
    )[0]
    return image_to_region(context, float(storage[0]), float(storage[1]))


def region_to_image(
    context: bpy.types.Context,
    region_x: float,
    region_y: float,
    *,
    clamp: bool = True,
) -> tuple[float, float] | None:
    """Map camera-view region pixels into top-left-origin source image pixels."""
    bounds = camera_frame_bounds(context)
    settings = context.scene.match_perspective
    if bounds is None:
        return None
    left, right, bottom, top = bounds
    if right - left < 1.0e-6 or top - bottom < 1.0e-6:
        return None
    normalized_x = (region_x - left) / (right - left)
    normalized_y = (top - region_y) / (top - bottom)
    if not clamp and not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
        return None
    normalized_x = min(1.0, max(0.0, normalized_x))
    normalized_y = min(1.0, max(0.0, normalized_y))
    display_width, display_height = _display_size(settings)
    display_x = normalized_x * display_width
    display_y = normalized_y * display_height
    return _display_to_storage(settings, display_x, display_y)


def _display_size(settings: properties.PMSettings) -> tuple[float, float]:
    """Return the currently displayed plate size."""
    if (
        settings.view_undistorted
        and settings.undistorted_image is not None
        and settings.undistorted_width > 0
        and settings.undistorted_height > 0
    ):
        return float(settings.undistorted_width), float(settings.undistorted_height)
    return float(max(settings.image_width, 1)), float(max(settings.image_height, 1))


def _storage_to_display(
    settings: properties.PMSettings,
    image_x: float,
    image_y: float,
) -> tuple[float, float, float, float]:
    """Map stored distorted source pixels to the visible plate."""
    display_width, display_height = _display_size(settings)
    if not settings.view_undistorted or abs(settings.division_lambda) < 1.0e-12:
        return image_x, image_y, display_width, display_height
    mapped = core.undistort_points(
        np.array([[image_x, image_y]], dtype=np.float64),
        settings.fx,
        settings.fy,
        settings.cx,
        settings.cy,
        settings.division_lambda,
    )[0]
    return (
        float(mapped[0] - settings.undistorted_offset_x),
        float(mapped[1] - settings.undistorted_offset_y),
        display_width,
        display_height,
    )


def _display_to_storage(
    settings: properties.PMSettings,
    display_x: float,
    display_y: float,
) -> tuple[float, float]:
    """Map the visible plate back to stored distorted source pixels."""
    if not settings.view_undistorted or abs(settings.division_lambda) < 1.0e-12:
        return display_x, display_y
    ideal = np.array(
        [[
            display_x + settings.undistorted_offset_x,
            display_y + settings.undistorted_offset_y,
        ]],
        dtype=np.float64,
    )
    mapped = core.distort_points(
        ideal,
        settings.fx,
        settings.fy,
        settings.cx,
        settings.cy,
        settings.division_lambda,
    )[0]
    return float(mapped[0]), float(mapped[1])


def refresh_background_projection(context: bpy.types.Context) -> None:
    """Reapply camera projection after switching original/undistorted plate."""
    settings = context.scene.match_perspective
    apply_camera(context.scene, settings, calibration_from_settings(settings))
    properties.tag_viewport_redraw(context)
