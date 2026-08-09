"""Blender scene integration and multi-match camera lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from . import core, properties

CV_CAMERA_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])


def safe_identifier(name: str) -> str:
    """Return a compact Blender-friendly id derived from a file stem."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    )
    return (cleaned.strip("_") or "image")[:60]


def line_bundles_from_settings(
    settings: properties.PMSession,
) -> dict[core.AxisId, list[core.LineSegment]]:
    """Convert RNA line collection to solver dataclasses, filtered by VP mode."""
    output: dict[core.AxisId, list[core.LineSegment]] = {"x": [], "y": [], "z": []}
    for item in settings.lines:
        output[item.axis].append(core.LineSegment(item.x1, item.y1, item.x2, item.y2))
    if settings.vp_mode == "1":
        output["x"] = []
    elif settings.vp_mode == "2":
        output["y"] = []
    return output


def calibration_from_settings(settings: properties.PMSession) -> core.Calibration:
    """Build an immutable solver calibration from session settings."""
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


def store_calibration(settings: properties.PMSession, calibration: core.Calibration) -> None:
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


def _parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    matrix_world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = matrix_world


def _next_match_index() -> int:
    used = set()
    for root in properties.iter_match_roots():
        name = root.name
        if name.startswith("PM_Match_"):
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                used.add(int(suffix))
    index = 1
    while index in used:
        index += 1
    return index


def _default_calibration(hfov_degrees: float, width: int = 1920, height: int = 1080) -> core.Calibration:
    focal = core.focal_from_hfov(hfov_degrees, width)
    initial_center = np.array((0.0, -5.0, 1.7), dtype=np.float64)
    forward_world = -initial_center
    forward_world /= np.linalg.norm(forward_world)
    right_world = np.cross(forward_world, np.array((0.0, 0.0, 1.0)))
    right_world /= np.linalg.norm(right_world)
    up_world = np.cross(right_world, forward_world)
    initial_rotation = np.stack([right_world, -up_world, forward_world], axis=0)
    return core.Calibration(
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


def create_match_camera(context: bpy.types.Context) -> bpy.types.Object:
    """Create a new Empty + Camera match hierarchy and make it active."""
    index = _next_match_index()
    prefix = f"PM_Match_{index:03d}"
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

    session = root.pm_session
    session.is_match_root = True
    session.camera_object = camera_object
    session.match_collection = collection
    session.image = None
    session.image_path = ""
    session.image_width = 1920
    session.image_height = 1080
    session.source_image_width = 1920
    session.lines.clear()
    session.selected_line_index = -1
    session.origin_is_set = False
    session.project_path = ""
    session.source_session_json = ""
    session.undistorted_path = ""
    session.undistorted_image = None
    session.undistorted_width = 0
    session.undistorted_height = 0
    session.undistorted_offset_x = 0.0
    session.undistorted_offset_y = 0.0
    session.view_undistorted = False
    session.view_lighting_applied = False
    session.view_image = None
    session.view_path = ""
    session.view_exposure = 0.0
    session.view_contrast = 1.0
    session.view_baked_exposure = 0.0
    session.view_baked_contrast = 1.0
    session.error = ""
    session.status = "Load a reference image or project"

    calibration = _default_calibration(session.hfov_degrees)
    store_calibration(session, calibration)
    apply_camera(context.scene, session, calibration)
    set_active_match(context, root)
    session.status = "Load a reference image or project"
    properties.tag_viewport_redraw(context)
    return root


def set_active_match(context: bpy.types.Context, root: bpy.types.Object) -> None:
    """Activate a match root and switch the viewport to its camera."""
    if not properties.is_match_root(root):
        raise ValueError("Not a Perspective Match root")
    # Cancel Draw / Pick Origin / PP / Landmark tools before changing session.
    from . import operators as operators_module

    operators_module.cancel_active_interact(context)
    space = properties.workspace(context)
    previous = space.active_root
    # Persist the outgoing match's camera-view zoom/pan before replacing it.
    if (
        previous is not None
        and previous != root
        and properties.is_match_root(previous)
    ):
        capture_camera_view_framing(context, previous.pm_session)
    space.active_root = root
    # Keep the dropdown in sync without re-entering its update callback.
    properties.sync_active_match_enum(space, root.name)
    space.is_modal = False
    space.work_mode = "NONE"
    session = root.pm_session
    camera = session.camera_object
    if camera is not None:
        context.scene.camera = camera
    # Rehydrate image pointer + lens/pose so overlays map after .blend load.
    ensure_match_ready(context)
    # Keep scene render size aligned with the plate being edited.
    if session.image_width > 0 and session.image_height > 0:
        use_undistorted = (
            session.view_undistorted
            and session.undistorted_width > 0
            and session.undistorted_height > 0
        )
        context.scene.render.resolution_x = (
            session.undistorted_width if use_undistorted else session.image_width
        )
        context.scene.render.resolution_y = (
            session.undistorted_height if use_undistorted else session.image_height
        )
        context.scene.render.resolution_percentage = 100
    enter_camera_view(context, restore_framing=True)
    properties.tag_viewport_redraw(context)


def unload_match(context: bpy.types.Context) -> None:
    """Clear the active editing session without deleting match objects."""
    from . import operators as operators_module

    operators_module.cancel_active_interact(context)
    space = properties.workspace(context)
    previous = space.active_root
    if properties.is_match_root(previous):
        capture_camera_view_framing(context, previous.pm_session)
    space.active_root = None
    properties.sync_active_match_enum(space, "NONE")
    space.is_modal = False
    space.work_mode = "NONE"
    properties.tag_viewport_redraw(context)


def match_prefix(root: bpy.types.Object) -> str:
    """Collection / hierarchy prefix for a match root (``PM_*`` without ``_Origin``)."""
    name = root.name
    if name.endswith("_Origin"):
        return name[: -len("_Origin")]
    return name


def _prefix_from_label(label: str) -> str:
    """Build a ``PM_*`` hierarchy prefix from a user-facing rename label."""
    cleaned = safe_identifier(label.strip())
    # Accept either "Kitchen" or "PM_Kitchen" without doubling the prefix.
    if cleaned.upper().startswith("PM_"):
        cleaned = cleaned[3:] or "match"
    return f"PM_{cleaned}"


def _prefix_is_taken(prefix: str, root: bpy.types.Object) -> bool:
    """True when another match already owns this collection / Origin / Camera name."""
    session = root.pm_session
    collection = bpy.data.collections.get(prefix)
    if collection is not None and collection != session.match_collection:
        return True
    origin = bpy.data.objects.get(f"{prefix}_Origin")
    if origin is not None and origin != root:
        return True
    camera = bpy.data.objects.get(f"{prefix}_Camera")
    if camera is not None and camera != session.camera_object:
        return True
    return False


def _unique_prefix(stem: str) -> str:
    base = f"PM_{safe_identifier(stem)}"
    if base not in bpy.data.collections and f"{base}_Origin" not in bpy.data.objects:
        return base
    index = 2
    while True:
        candidate = f"{base}_{index:02d}"
        if (
            candidate not in bpy.data.collections
            and f"{candidate}_Origin" not in bpy.data.objects
        ):
            return candidate
        index += 1


def _unique_prefix_for_rename(label: str, root: bpy.types.Object) -> str:
    """Like ``_unique_prefix``, but ignores the match being renamed."""
    base = _prefix_from_label(label)
    if not _prefix_is_taken(base, root):
        return base
    index = 2
    while True:
        candidate = f"{base}_{index:02d}"
        if not _prefix_is_taken(candidate, root):
            return candidate
        index += 1


def _rename_match_hierarchy(root: bpy.types.Object, prefix: str) -> None:
    session = root.pm_session
    collection = session.match_collection
    camera = session.camera_object
    if collection is not None:
        collection.name = prefix
    root.name = f"{prefix}_Origin"
    if camera is not None:
        camera.name = f"{prefix}_Camera"
        if camera.data is not None:
            camera.data.name = f"{prefix}_Camera"


def rename_match(
    context: bpy.types.Context,
    root: bpy.types.Object,
    label: str,
) -> bpy.types.Object:
    """Rename an existing match hierarchy from a user label.

    Updates the collection, Origin Empty, and Camera datablock names together,
    then refreshes active/anchor dropdown identifiers. Landmark observations keep
    working because they store Object pointers, not name strings.
    """
    if not properties.is_match_root(root):
        raise ValueError("Not a Perspective Match root")
    if not label or not label.strip():
        raise ValueError("Match name cannot be empty")

    prefix = _unique_prefix_for_rename(label, root)
    if match_prefix(root) == prefix:
        return root

    _rename_match_hierarchy(root, prefix)
    space = properties.workspace(context)
    if space.active_root == root:
        properties.sync_active_match_enum(space, root.name)
    if space.anchor_root == root:
        properties.sync_anchor_match_enum(space, root.name)
    properties.tag_viewport_redraw(context)
    return root


def _clear_derived_plates(session: properties.PMSession) -> None:
    """Drop view-lighting / undistorted caches without touching lines or origin."""
    invalidate_undistorted_cache(session)
    cached_view = session.view_image
    session.view_lighting_applied = False
    session.view_image = None
    session.view_path = ""
    session.view_exposure = 0.0
    session.view_contrast = 1.0
    session.view_baked_exposure = 0.0
    session.view_baked_contrast = 1.0
    if cached_view is not None and cached_view.users == 0:
        bpy.data.images.remove(cached_view)


def _reset_session_edit_state(session: properties.PMSession) -> None:
    session.lines.clear()
    session.selected_line_index = -1
    session.origin_is_set = False
    session.project_path = ""
    session.source_session_json = ""
    clear_similarity_on_session(session)
    _clear_derived_plates(session)
    session.error = ""


def bind_reference_image(context: bpy.types.Context, image_path: str) -> bpy.types.Object:
    """Attach a still to the active match camera without affecting other matches."""
    root = properties.active_root(context)
    if root is None:
        root = create_match_camera(context)
    session = root.pm_session
    camera_object = session.camera_object
    if camera_object is None or camera_object.type != "CAMERA":
        raise ValueError("Active match has no camera")

    absolute_path = str(Path(bpy.path.abspath(image_path)).expanduser().resolve())
    image = bpy.data.images.load(absolute_path, check_existing=True)
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("The selected image has no readable pixel dimensions")

    _reset_session_edit_state(session)
    root.matrix_world = Matrix.Identity(4)

    camera_data = camera_object.data
    camera_data.show_background_images = True
    camera_data.background_images.clear()
    background = camera_data.background_images.new()
    background.image = image
    background.show_background_image = True
    background.alpha = 1.0
    background.display_depth = "BACK"
    background.frame_method = "STRETCH"
    if hasattr(image, "use_view_as_render"):
        image.use_view_as_render = True

    session.image = image
    session.image_path = absolute_path
    session.image_width = width
    session.image_height = height
    session.source_image_width = width

    prefix = _unique_prefix(Path(absolute_path).stem)
    _rename_match_hierarchy(root, prefix)
    space = properties.workspace(context)
    if space.active_root == root:
        properties.sync_active_match_enum(space, root.name)
    if space.anchor_root == root:
        properties.sync_anchor_match_enum(space, root.name)

    calibration = _default_calibration(session.hfov_degrees, width, height)
    store_calibration(session, calibration)
    apply_camera(context.scene, session, calibration)
    context.scene.render.resolution_x = width
    context.scene.render.resolution_y = height
    context.scene.render.resolution_percentage = 100
    context.scene.render.pixel_aspect_x = 1.0
    context.scene.render.pixel_aspect_y = 1.0
    session.status = "Choose a perspective mode, then draw VP lines"
    set_active_match(context, root)
    return camera_object


# Compatibility alias for older call sites / smoke tests.
setup_reference_image = bind_reference_image


def replace_reference_image(
    context: bpy.types.Context, image_path: str
) -> bpy.types.Object:
    """Swap the still on the active match; keep VP lines, origin, calibration, landmarks.

    Requires the same pixel dimensions as the current plate so existing image-space
    picks stay valid. Derived plates (view lighting / undistorted) are discarded.
    """
    root = properties.active_root(context)
    if root is None:
        raise ValueError("Create or activate a match camera first")
    session = root.pm_session
    if session.image is None:
        raise ValueError("No reference image to replace — use Open Image first")
    camera_object = session.camera_object
    if camera_object is None or camera_object.type != "CAMERA":
        raise ValueError("Active match has no camera")

    absolute_path = str(Path(bpy.path.abspath(image_path)).expanduser().resolve())
    image = bpy.data.images.load(absolute_path, check_existing=True)
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("The selected image has no readable pixel dimensions")
    if width != int(session.image_width) or height != int(session.image_height):
        raise ValueError(
            f"New image is {width}×{height}; current plate is "
            f"{int(session.image_width)}×{int(session.image_height)}. "
            "Replace Image needs the same size — use Open Image to start over."
        )

    _clear_derived_plates(session)
    session.error = ""

    camera_data = camera_object.data
    camera_data.show_background_images = True
    if len(camera_data.background_images) == 0:
        background = camera_data.background_images.new()
        background.alpha = 1.0
        background.display_depth = "BACK"
        background.frame_method = "STRETCH"
    else:
        background = camera_data.background_images[0]
    background.image = image
    background.show_background_image = True
    if hasattr(image, "use_view_as_render"):
        image.use_view_as_render = True

    session.image = image
    session.image_path = absolute_path
    # Keep width/height, calibration, lines, origin, and sync transform.
    session.status = f"Replaced plate with {Path(absolute_path).name} (lines kept)"
    refresh_background_projection(context)
    properties.tag_viewport_redraw(context)
    return camera_object


def apply_camera(
    blender_scene: bpy.types.Scene,
    settings: properties.PMSession,
    calibration: core.Calibration,
    *,
    update_scene_camera: bool = True,
) -> None:
    """Apply solved intrinsics and OpenCV extrinsics to the managed Blender camera.

    The private-world pose is written as the camera's *local* matrix so a sync
    transform on the match root Empty can move the whole rig into shared space.

    When ``update_scene_camera`` is False, only this match's camera datablock and
    RNA are updated (used when writing several matches during lens refine).
    """
    camera_object = settings.camera_object
    if camera_object is None or camera_object.type != "CAMERA":
        raise ValueError("Active match has no camera")
    intrinsics = calibration.intrinsics
    source_width = max(float(settings.source_image_width or intrinsics.image_width), 1.0)
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
    if len(camera_data.background_images) > 0 and settings.image is not None:
        if use_undistorted:
            camera_data.background_images[0].image = settings.undistorted_image
        elif (
            settings.view_lighting_applied
            and settings.view_image is not None
        ):
            camera_data.background_images[0].image = settings.view_image
        else:
            camera_data.background_images[0].image = settings.image

    # Private pose as local; root Empty carries optional sync similarity.
    camera_object.matrix_parent_inverse = Matrix.Identity(4)
    camera_object.matrix_local = private_camera_matrix(calibration)
    store_calibration(settings, calibration)
    if update_scene_camera:
        blender_scene.camera = camera_object
        blender_scene.render.resolution_x = int(plate_width)
        blender_scene.render.resolution_y = plate_height
        blender_scene.render.resolution_percentage = 100


def private_camera_matrix(calibration: core.Calibration) -> Matrix:
    """Blender local matrix for a private-world OpenCV calibration."""
    camera_to_world_blender = calibration.rotation_w2c.T @ CV_CAMERA_TO_BLENDER_CAMERA
    matrix_world = Matrix(camera_to_world_blender.tolist()).to_4x4()
    matrix_world.translation = Vector(calibration.camera_center.tolist())
    return matrix_world


def refine_match(
    context: bpy.types.Context,
    *,
    estimate_distortion: bool = False,
) -> core.Calibration:
    """Refine and apply camera state from all current VP lines.

    Pass ``estimate_distortion=True`` only from the Estimate Distortion button.
    Ordinary VP-line / FOV refines keep the stored λ without re-fitting it.
    """
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    if settings.image is None:
        raise ValueError("Load a reference image first")
    line_bundles = line_bundles_from_settings(settings)
    ready_axes = sum(1 for segments in line_bundles.values() if len(segments) >= 2)
    if ready_axes < 2:
        raise ValueError("Draw at least two lines on each of two axes")

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
        # λ is button-only; locked Manual FOV still allows a one-shot estimate.
        estimate_distortion=bool(estimate_distortion),
        initial_division_lambda=previous_calibration.division_lambda,
    )

    if settings.origin_is_set:
        origin = tuple(float(value) for value in settings.origin_image)
        calibration.camera_center, _scale = core.apply_origin_and_scale(
            calibration,
            origin,
        )
    else:
        calibration.camera_center = core.default_camera_center(calibration.rotation_w2c)

    if _intrinsics_or_distortion_changed(previous_calibration, calibration):
        invalidate_undistorted_cache(settings)
    apply_camera(context.scene, settings, calibration)
    _update_diagnostics(settings, line_bundles, calibration)
    settings.status = (
        f"Camera matched · HFOV {calibration.hfov_degrees:.2f}°"
        + (
            f" · λ {calibration.division_lambda:.4f}"
            if abs(calibration.division_lambda) > 1.0e-6
            else ""
        )
    )
    settings.error = ""
    properties.tag_viewport_redraw(context)
    return calibration


def _update_diagnostics(
    settings: properties.PMSession,
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
    calibration: core.Calibration,
) -> None:
    """Update compact plane-FOV and VP residual panel diagnostics."""
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
    settings.residual_degrees = core.vp_angular_residual_degrees(
        calibration,
        line_bundles,
    )
    has_segments = any(segments for segments in line_bundles.values())
    settings.vp_line_rms_px = (
        core.vp_line_residual_rms(calibration, line_bundles)
        if has_segments
        else -1.0
    )


def apply_manual_fov(context: bpy.types.Context) -> None:
    """Apply the current manual HFOV before or without enough VP lines."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    settings.lock_focal = True
    line_bundles = line_bundles_from_settings(settings)
    ready_axes = sum(1 for segments in line_bundles.values() if len(segments) >= 2)
    # With enough lines, re-orient at the locked FOV (keeps stored λ; no re-fit).
    if ready_axes >= 2:
        refine_match(context)
        from . import distortion

        distortion.sync_undistorted_plate_after_refine(context)
        return
    calibration = calibration_from_settings(settings)
    previous_focal = calibration.intrinsics.fx
    width = max(settings.image_width, 1)
    focal = core.focal_from_hfov(settings.hfov_degrees, width)
    calibration.intrinsics.fx = focal
    calibration.intrinsics.fy = focal
    if abs(previous_focal - focal) > 1.0e-6:
        invalidate_undistorted_cache(settings)
    apply_camera(context.scene, settings, calibration)
    if any(segments for segments in line_bundles.values()):
        _update_diagnostics(settings, line_bundles, calibration)
    if settings.origin_is_set:
        reapply_placement(context)
    settings.status = f"Manual HFOV {settings.hfov_degrees:.2f}°"
    properties.tag_viewport_redraw(context)


def apply_ros_camera_info_yaml(context: bpy.types.Context, filepath: str) -> str:
    """Import ROS camera_info YAML: HFOV, principal point, optional Fitzgibbon λ.

    Brown–Conrady / plumb_bob coefficients are ignored. When ``fitzgibbon_lambda``
    is present it becomes ``division_lambda`` (resolution-invariant in the
    normalized plane — not scaled with image size). A non-zero imported λ also
    builds/shows the undistorted plate (without re-fitting λ from VP lines).
    """
    from . import distortion, ros_camera_info

    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    if settings.image is None:
        raise ValueError("Load a reference image first")

    text = Path(filepath).read_text(encoding="utf-8")
    info = ros_camera_info.parse_ros_camera_info_yaml(text)
    fx, fy, cx, cy, scaled = ros_camera_info.scale_intrinsics_to_image(
        info,
        int(settings.image_width),
        int(settings.image_height),
    )

    previous = calibration_from_settings(settings)
    settings.lock_focal = True
    settings.fx = float(fx)
    settings.fy = float(fy)
    settings.cx = float(cx)
    settings.cy = float(cy)
    settings.hfov_degrees = core.hfov_from_focal(fx, max(int(settings.image_width), 1))

    imported_lambda = info.fitzgibbon_lambda is not None
    if imported_lambda:
        settings.division_lambda = float(info.fitzgibbon_lambda)
        settings.lambda_saturated = False

    line_bundles = line_bundles_from_settings(settings)
    ready_axes = sum(1 for segments in line_bundles.values() if len(segments) >= 2)
    if ready_axes >= 2:
        # Locked FOV + imported PP; keep YAML λ (do not re-fit from VP lines).
        refine_match(context, estimate_distortion=False)
    else:
        calibration = calibration_from_settings(settings)
        if _intrinsics_or_distortion_changed(previous, calibration):
            invalidate_undistorted_cache(settings)
        apply_camera(context.scene, settings, calibration)
        if any(segments for segments in line_bundles.values()):
            _update_diagnostics(settings, line_bundles, calibration)
        if settings.origin_is_set:
            reapply_placement(context)

    # Undistorted background: only sync/generate after intrinsics + λ are final.
    if imported_lambda and abs(settings.division_lambda) > 1.0e-8:
        try:
            distortion.generate_undistorted_plate(context)
        except Exception as error:
            settings.status = f"Imported λ; plate failed: {error}"
            print(f"Perspective Match: undistorted plate failed: {error}")
    else:
        distortion.sync_undistorted_plate_after_refine(context)

    offset_x = settings.cx - settings.image_width * 0.5
    offset_y = settings.cy - settings.image_height * 0.5
    label = info.camera_name or Path(filepath).name
    parts = [
        f"Imported {label}",
        f"HFOV {settings.hfov_degrees:.2f}°",
        f"PP {offset_x:+.1f},{offset_y:+.1f} px",
    ]
    if imported_lambda:
        parts.append(f"λ {settings.division_lambda:.5f}")
    if scaled:
        parts.append(
            f"scaled from {info.image_width}×{info.image_height}"
        )
    # plumb_bob k1… is not Fitzgibbon λ.
    if info.distortion_coefficients and any(
        abs(value) > 1.0e-12 for value in info.distortion_coefficients
    ):
        parts.append("plumb_bob D skipped")
    message = " · ".join(parts)
    settings.status = message
    settings.error = ""
    properties.tag_viewport_redraw(context)
    return message


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


def invalidate_undistorted_cache(settings: properties.PMSession) -> None:
    """Discard a remap made with stale intrinsics; restore lit or original plate."""
    cached_image = settings.undistorted_image
    settings.view_undistorted = False
    if (
        settings.camera_object is not None
        and len(settings.camera_object.data.background_images) > 0
        and settings.image is not None
    ):
        if settings.view_lighting_applied and settings.view_image is not None:
            settings.camera_object.data.background_images[0].image = settings.view_image
        else:
            settings.camera_object.data.background_images[0].image = settings.image
    settings.undistorted_image = None
    settings.undistorted_path = ""
    settings.undistorted_width = 0
    settings.undistorted_height = 0
    settings.undistorted_offset_x = 0.0
    settings.undistorted_offset_y = 0.0
    if cached_image is not None and cached_image.users == 0:
        bpy.data.images.remove(cached_image)


def set_origin(
    context: bpy.types.Context,
    image_point: tuple[float, float],
) -> None:
    """Store the ground origin and reapply camera placement."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    settings.origin_image = image_point
    settings.origin_is_set = True
    settings.status = "Origin set on the ground plane"
    reapply_placement(context)


def principal_point_is_off_center(settings: properties.PMSession) -> bool:
    """True when cx/cy differ from the plate center by more than half a pixel."""
    if settings.image_width <= 0 or settings.image_height <= 0:
        return False
    return (
        abs(settings.cx - settings.image_width * 0.5) > 0.5
        or abs(settings.cy - settings.image_height * 0.5) > 0.5
    )


def set_principal_point(
    context: bpy.types.Context,
    image_point: tuple[float, float],
    *,
    finalize: bool = True,
) -> None:
    """Move the principal point in source-image pixels.

    ``finalize=False`` updates shift/intrinsics only (cheap drag samples).
    ``finalize=True`` also rebuilds orientation from VP lines at the locked
    focal — used on release and on throttled samples during drag.
    """
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    if settings.image is None:
        raise ValueError("Load a reference image first")

    width = max(int(settings.image_width), 1)
    height = max(int(settings.image_height), 1)
    cx = float(np.clip(image_point[0], 0.0, float(width)))
    cy = float(np.clip(image_point[1], 0.0, float(height)))

    previous = calibration_from_settings(settings)
    settings.cx = cx
    settings.cy = cy

    line_bundles = line_bundles_from_settings(settings)
    ready_axes = sum(1 for segments in line_bundles.values() if len(segments) >= 2)

    if finalize and ready_axes >= 2:
        intrinsics = previous.intrinsics
        intrinsics.cx = cx
        intrinsics.cy = cy
        calibration = core.refine_camera(
            line_bundles,
            intrinsics,
            lock_focal=True,
            estimate_principal_point=False,
            estimate_distortion=False,
            initial_division_lambda=previous.division_lambda,
        )
        calibration.division_lambda = previous.division_lambda
        calibration.lambda_saturated = previous.lambda_saturated
        if settings.origin_is_set:
            origin = tuple(float(value) for value in settings.origin_image)
            calibration.camera_center, _scale = core.apply_origin_and_scale(
                calibration,
                origin,
            )
        else:
            calibration.camera_center = core.default_camera_center(
                calibration.rotation_w2c
            )
        if _intrinsics_or_distortion_changed(previous, calibration):
            invalidate_undistorted_cache(settings)
        apply_camera(context.scene, settings, calibration)
        _update_diagnostics(settings, line_bundles, calibration)
    else:
        calibration = calibration_from_settings(settings)
        if _intrinsics_or_distortion_changed(previous, calibration):
            invalidate_undistorted_cache(settings)
        apply_camera(context.scene, settings, calibration)

    offset_x = settings.cx - width * 0.5
    offset_y = settings.cy - height * 0.5
    settings.status = f"PP offset {offset_x:+.1f}, {offset_y:+.1f} px"
    properties.tag_viewport_redraw(context)


def reapply_placement(context: bpy.types.Context) -> None:
    """Apply origin placement to the current solved camera without a new VP solve."""
    settings = properties.active_session(context)
    if settings is None:
        return
    calibration = calibration_from_settings(settings)
    if not settings.origin_is_set:
        return
    calibration.camera_center = core.default_camera_center(calibration.rotation_w2c)
    calibration.camera_center, _scale = core.apply_origin_and_scale(
        calibration,
        tuple(float(value) for value in settings.origin_image),
    )
    apply_camera(context.scene, settings, calibration)
    properties.tag_viewport_redraw(context)


def enter_camera_view(
    context: bpy.types.Context,
    *,
    restore_framing: bool = False,
) -> None:
    """Switch the current 3D View to the active match camera.

    When ``restore_framing`` is True, reapply this match's last saved
    camera-view zoom/pan (used when switching matches or clicking Camera View).
    Modal re-pins leave framing alone so live pan/zoom is not reset.
    """
    if context.area is None or context.area.type != "VIEW_3D":
        return
    settings = properties.active_session(context)
    if settings is None or settings.camera_object is None:
        return
    context.space_data.camera = settings.camera_object
    region_3d = getattr(context.space_data, "region_3d", None)
    if region_3d is not None:
        region_3d.view_perspective = "CAMERA"
        if restore_framing:
            restore_camera_view_framing(context, settings)


def _iter_view3d_spaces(context: bpy.types.Context):
    """Yield (space, region_3d) for every 3D View in the current window set."""
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return
    for window in window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            region_3d = getattr(space, "region_3d", None)
            if region_3d is not None:
                yield space, region_3d


def _region_3d_for_session(context: bpy.types.Context, session) -> object | None:
    """Find a camera-view RegionView3D showing this match camera, if any."""
    camera = session.camera_object if session is not None else None
    if camera is None:
        return None
    # Prefer the calling context when it is already that camera view.
    space = getattr(context, "space_data", None)
    if space is not None and getattr(space, "type", None) == "VIEW_3D":
        region_3d = getattr(space, "region_3d", None)
        if (
            region_3d is not None
            and region_3d.view_perspective == "CAMERA"
            and space.camera == camera
        ):
            return region_3d
    for space, region_3d in _iter_view3d_spaces(context):
        if region_3d.view_perspective == "CAMERA" and space.camera == camera:
            return region_3d
    return None


def capture_camera_view_framing(
    context: bpy.types.Context,
    session=None,
) -> bool:
    """Store camera-view zoom/pan from the viewport onto the match session."""
    settings = session if session is not None else properties.active_session(context)
    if settings is None:
        return False
    region_3d = _region_3d_for_session(context, settings)
    if region_3d is None:
        return False
    settings.view_camera_zoom = float(
        max(-30.0, min(600.0, region_3d.view_camera_zoom))
    )
    settings.view_camera_offset = (
        float(region_3d.view_camera_offset[0]),
        float(region_3d.view_camera_offset[1]),
    )
    return True


def restore_camera_view_framing(
    context: bpy.types.Context,
    session=None,
) -> bool:
    """Apply a match session's saved camera-view zoom/pan to the 3D View."""
    settings = session if session is not None else properties.active_session(context)
    if settings is None:
        return False
    space = getattr(context, "space_data", None)
    if space is None or getattr(space, "type", None) != "VIEW_3D":
        return False
    region_3d = getattr(space, "region_3d", None)
    if region_3d is None:
        return False
    region_3d.view_camera_zoom = float(
        max(-30.0, min(600.0, settings.view_camera_zoom))
    )
    region_3d.view_camera_offset = (
        float(settings.view_camera_offset[0]),
        float(settings.view_camera_offset[1]),
    )
    return True


def capture_active_match_framing(context: bpy.types.Context | None = None) -> bool:
    """Capture framing for the active match (used before .blend save)."""
    blender_context = context or bpy.context
    if blender_context is None:
        return False
    return capture_camera_view_framing(blender_context)


def _is_derived_display_image(
    settings: properties.PMSession,
    image: bpy.types.Image | None,
) -> bool:
    """True when ``image`` is a lit/undistorted plate, not the solver source still."""
    if image is None:
        return False
    if settings.view_image is not None and image == settings.view_image:
        return True
    if settings.undistorted_image is not None and image == settings.undistorted_image:
        return True
    name = (image.name or "").lower()
    if (
        name.endswith(".pm-view")
        or ".pm-view." in name
        or name.endswith(".undistorted")
        or ".undistorted." in name
    ):
        return True
    filepath = getattr(image, "filepath", "") or getattr(image, "filepath_raw", "") or ""
    if not filepath:
        return False
    try:
        resolved = str(
            Path(bpy.path.abspath(filepath)).expanduser().resolve()
        ).lower()
    except Exception:
        return False
    return resolved.endswith("-pm-view.png") or resolved.endswith(".undistorted.png")


def _resolved_source_image_path(settings: properties.PMSession) -> str | None:
    """Return the on-disk original still path, or None if missing/unusable."""
    if not settings.image_path:
        return None
    try:
        absolute_path = str(
            Path(bpy.path.abspath(settings.image_path)).expanduser().resolve()
        )
    except Exception:
        return None
    lower = absolute_path.lower()
    # Never treat a baked view/undistorted sibling as the solver source.
    if lower.endswith("-pm-view.png") or lower.endswith(".undistorted.png"):
        return None
    if not Path(absolute_path).is_file():
        return None
    return absolute_path


def ensure_session_image(settings: properties.PMSession) -> bool:
    """Restore the reference Image pointer after .blend load if it was lost.

    Prefer ``image_path`` on disk. Camera backgrounds often still hold the lit
    ``*-pm-view`` plate after a pointer loss — never adopt those as the source
    still, or Apply Lighting / match switching will double-expose.
    """
    image = settings.image
    if (
        image is not None
        and image.name in bpy.data.images
        and not _is_derived_display_image(settings, image)
    ):
        if settings.image_width <= 0 or settings.image_height <= 0:
            settings.image_width = int(image.size[0])
            settings.image_height = int(image.size[1])
            if settings.source_image_width <= 0:
                settings.source_image_width = settings.image_width
        return True

    # Pointer missing or pointing at a derived plate — clear and recover.
    if _is_derived_display_image(settings, image):
        settings.image = None

    absolute_path = _resolved_source_image_path(settings)
    if absolute_path is not None:
        loaded = bpy.data.images.load(absolute_path, check_existing=True)
        settings.image = loaded
        settings.image_width = int(loaded.size[0])
        settings.image_height = int(loaded.size[1])
        if settings.source_image_width <= 0:
            settings.source_image_width = settings.image_width
        settings.image_path = absolute_path
        # Leave the camera background alone; ensure_match_ready → apply_camera
        # selects original / lit / undistorted from session flags.
        return True

    camera = settings.camera_object
    if camera is not None and camera.type == "CAMERA":
        for background in camera.data.background_images:
            candidate = background.image
            if candidate is None or candidate.name not in bpy.data.images:
                continue
            if _is_derived_display_image(settings, candidate):
                continue
            settings.image = candidate
            settings.image_width = int(candidate.size[0])
            settings.image_height = int(candidate.size[1])
            if settings.source_image_width <= 0:
                settings.source_image_width = settings.image_width
            if not settings.image_path and candidate.filepath:
                candidate_path = str(
                    Path(bpy.path.abspath(candidate.filepath)).expanduser().resolve()
                )
                lower = candidate_path.lower()
                if not (
                    lower.endswith("-pm-view.png")
                    or lower.endswith(".undistorted.png")
                ):
                    settings.image_path = candidate_path
            return True
    return False


def ensure_match_ready(context: bpy.types.Context) -> bool:
    """Rehydrate image + Blender camera from persisted session (safe after .blend load).

    Overlay mapping depends on lens/shift/local matrix matching stored intrinsics.
    After file load those can drift from the RNA session — re-apply before drawing.
    """
    settings = properties.active_session(context)
    if settings is None or settings.camera_object is None:
        return False
    ensure_session_image(settings)
    if settings.image is None:
        return False
    # Restore / rebuild the lit plate before falling back to the original still.
    # Clearing view_lighting_applied too early left the EV slider intact while the
    # camera background snapped back to the bright source on match switch.
    if settings.view_lighting_applied:
        from . import distortion as distortion_module

        try:
            distortion_module.ensure_match_view_plate(context)
        except Exception:
            pass
        if settings.view_image is None or settings.view_image.name not in bpy.data.images:
            settings.view_lighting_applied = False
            settings.view_image = None
    if settings.image_width <= 0 or settings.image_height <= 0:
        settings.image_width = max(int(settings.image.size[0]), 1)
        settings.image_height = max(int(settings.image.size[1]), 1)
    if settings.fx <= 0.0 or settings.fy <= 0.0:
        calibration = _default_calibration(
            settings.hfov_degrees,
            settings.image_width,
            settings.image_height,
        )
        store_calibration(settings, calibration)
    try:
        apply_camera(context.scene, settings, calibration_from_settings(settings))
    except Exception:
        return False
    return True


def is_camera_view(context: bpy.types.Context) -> bool:
    """Return whether input occurs in the active match camera view."""
    settings = properties.active_session(context)
    region_3d = getattr(context.space_data, "region_3d", None)
    space_camera = getattr(context.space_data, "camera", None)
    return (
        settings is not None
        and context.area is not None
        and context.area.type == "VIEW_3D"
        and context.region is not None
        and context.region.type == "WINDOW"
        and region_3d is not None
        and region_3d.view_perspective == "CAMERA"
        and settings.camera_object is not None
        and space_camera == settings.camera_object
    )


def camera_frame_bounds(
    context: bpy.types.Context,
) -> tuple[float, float, float, float] | None:
    """Project camera-frame corners to region pixels as left, right, bottom, top."""
    if not is_camera_view(context):
        return None
    settings = properties.active_session(context)
    camera_object = settings.camera_object
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
    settings = properties.active_session(context)
    if bounds is None or settings is None or settings.image_width <= 0 or settings.image_height <= 0:
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
    settings = properties.active_session(context)
    if settings is None:
        return None
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
        max(float(settings.fx), 1.0e-6),
        max(float(settings.fy), 1.0e-6),
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
    settings = properties.active_session(context)
    if bounds is None or settings is None:
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


def _display_size(settings: properties.PMSession) -> tuple[float, float]:
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
    settings: properties.PMSession,
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
    settings: properties.PMSession,
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
    settings = properties.active_session(context)
    if settings is None:
        return
    apply_camera(context.scene, settings, calibration_from_settings(settings))
    properties.tag_viewport_redraw(context)


def _identity_root_matrix() -> Matrix:
    return Matrix.Identity(4)


def _similarity_to_matrix(similarity) -> Matrix:
    """Convert a sync SimilarityTransform into a Blender matrix."""
    return Matrix(similarity.matrix().tolist())


def store_similarity_on_session(session: properties.PMSession, similarity) -> None:
    """Persist a solved similarity onto the match session RNA."""
    session.sync_is_applied = True
    session.sync_scale = float(similarity.scale)
    rotation = Matrix(similarity.rotation.tolist()).to_quaternion()
    session.sync_rotation = (rotation.w, rotation.x, rotation.y, rotation.z)
    session.sync_translation = tuple(float(value) for value in similarity.translation)


def clear_similarity_on_session(session: properties.PMSession) -> None:
    """Reset stored sync transform RNA to identity."""
    session.sync_is_applied = False
    session.sync_scale = 1.0
    session.sync_rotation = (1.0, 0.0, 0.0, 0.0)
    session.sync_translation = (0.0, 0.0, 0.0)
    session.sync_rmse_px = 0.0


def apply_similarity_to_root(root: bpy.types.Object, similarity) -> None:
    """Write a similarity onto a match root Empty."""
    root.matrix_world = _similarity_to_matrix(similarity)
    store_similarity_on_session(root.pm_session, similarity)


def reset_root_sync_transform(root: bpy.types.Object) -> None:
    """Put a match root Empty back at the identity shared-frame pose."""
    root.matrix_world = _identity_root_matrix()
    clear_similarity_on_session(root.pm_session)


def ensure_landmark_collection(context: bpy.types.Context) -> bpy.types.Collection:
    """Return (creating if needed) the collection that holds landmark Empties."""
    name = "PM_Sync_Landmarks"
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        context.scene.collection.children.link(collection)
    return collection


def sync_landmark_empties(context: bpy.types.Context) -> None:
    """Create/update/remove landmark Empties (points) and edge meshes (lines)."""
    space = properties.workspace(context)
    collection = ensure_landmark_collection(context)
    keep_names: set[str] = set()

    def _remove_object(obj: bpy.types.Object) -> None:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and getattr(data, "users", 1) == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)

    for landmark in space.landmarks:
        # Disabled landmarks stay out of the sync solve and out of the helper collection.
        if (
            not landmark.has_position
            or not landmark.use_in_sync
            or not space.show_landmark_empties
        ):
            continue
        base_name = safe_identifier(landmark.name) or landmark.item_id[:8]
        object_name = f"PM_LM_{base_name}"
        keep_names.add(object_name)

        if landmark.kind == "LINE" and landmark.has_line_segment:
            point_a = Vector(landmark.position)
            point_b = Vector(landmark.position_b)
            mesh = bpy.data.meshes.get(object_name)
            if mesh is None:
                mesh = bpy.data.meshes.new(object_name)
            # Rebuild a single-edge segment so the solved line is obvious in 3D.
            mesh.clear_geometry()
            mesh.from_pydata(
                [tuple(point_a), tuple(point_b)],
                [(0, 1)],
                [],
            )
            mesh.update()

            obj = bpy.data.objects.get(object_name)
            # Object type is fixed at creation — replace Empty with a Mesh object.
            if obj is not None and obj.type != "MESH":
                _remove_object(obj)
                obj = None
            if obj is None:
                obj = bpy.data.objects.new(object_name, mesh)
                collection.objects.link(obj)
            else:
                if obj.data != mesh:
                    old_data = obj.data
                    obj.data = mesh
                    if (
                        old_data is not None
                        and old_data != mesh
                        and getattr(old_data, "users", 1) == 0
                    ):
                        bpy.data.meshes.remove(old_data)
                if obj.name not in collection.objects:
                    collection.objects.link(obj)
            obj.location = (0.0, 0.0, 0.0)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (1.0, 1.0, 1.0)
            obj.hide_viewport = False
            obj.hide_render = True
            obj.display_type = "WIRE"
            continue

        empty = bpy.data.objects.get(object_name)
        # Replace a leftover line mesh with an Empty for point landmarks.
        if empty is not None and empty.type != "EMPTY":
            _remove_object(empty)
            empty = None
        if empty is None:
            empty = bpy.data.objects.new(object_name, None)
            empty.empty_display_type = "PLAIN_AXES"
            collection.objects.link(empty)
        elif empty.name not in collection.objects:
            collection.objects.link(empty)
        empty.empty_display_size = float(space.landmark_empty_size)
        empty.location = Vector(landmark.position)
        empty.hide_viewport = False
        empty.hide_render = True

    # Remove stale landmark helpers that belong to this collection.
    for obj in list(collection.objects):
        if obj.name.startswith("PM_LM_") and obj.name not in keep_names:
            _remove_object(obj)


def clear_landmark_empties(context: bpy.types.Context) -> None:
    """Delete all PM landmark Empties and line meshes."""
    collection = bpy.data.collections.get("PM_Sync_Landmarks")
    if collection is None:
        return
    for obj in list(collection.objects):
        if obj.name.startswith("PM_LM_"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and getattr(data, "users", 1) == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)


def active_landmark(context: bpy.types.Context | None = None):
    """Return the active landmark PropertyGroup, if any."""
    space = properties.workspace(context)
    index = space.active_landmark_index
    if 0 <= index < len(space.landmarks):
        return space.landmarks[index]
    return None


def observation_for_match(landmark, root: bpy.types.Object | None):
    """Return the observation entry for a match root inside a landmark."""
    if root is None:
        return None
    for observation in landmark.observations:
        if observation.match_root == root:
            return observation
    return None


def _private_point_from_world(
    root: bpy.types.Object,
    world_location,
) -> np.ndarray:
    """Map a Blender world point into the match's private frame via the root Empty."""
    local = root.matrix_world.inverted() @ Vector(
        (float(world_location[0]), float(world_location[1]), float(world_location[2]))
    )
    return np.array((local.x, local.y, local.z), dtype=np.float64)


def project_known_object_into_match(landmark, root: bpy.types.Object) -> bool:
    """Project Known 3D point or line endpoints into this match's still.

    Known Empties live in Blender/shared world. The match root Empty carries the
    sync transform, so world→private is ``root.matrix_world.inverted()``.
    """
    from . import sync as sync_module

    known_object = landmark.known_object
    if known_object is None or known_object.name not in bpy.data.objects:
        return False
    session = root.pm_session
    if session.image is None or session.fx <= 0.0:
        return False
    calibration = calibration_from_settings(session)

    def _project_world(world_location):
        private_point = _private_point_from_world(root, world_location)
        return sync_module.project_private_point(private_point, calibration)

    world_a = known_object.matrix_world.to_translation()
    projected_a = _project_world(world_a)
    if projected_a is None:
        return False

    observation = observation_for_match(landmark, root)
    if observation is None:
        observation = landmark.observations.add()
        observation.match_root = root
    observation.x = float(projected_a[0])
    observation.y = float(projected_a[1])
    observation.confidence = "HIGH"

    if landmark.kind == "LINE":
        known_b = landmark.known_object_b
        if known_b is None or known_b.name not in bpy.data.objects:
            return False
        projected_b = _project_world(known_b.matrix_world.to_translation())
        if projected_b is None:
            return False
        observation.x2 = float(projected_b[0])
        observation.y2 = float(projected_b[1])
        observation.is_set = True
        return True

    observation.is_set = True
    return True


def auto_project_known_landmarks(
    context: bpy.types.Context,
    landmarks,
) -> tuple[int, bpy.types.Object | None]:
    """Project Known 3D landmarks into the anchor (else active) match image.

    Returns ``(projected_count, target_root)``.
    """
    target = properties.anchor_root(context) or properties.active_root(context)
    if target is None:
        return 0, None
    projected = 0
    for landmark in landmarks:
        if project_known_object_into_match(landmark, target):
            projected += 1
    if projected:
        properties.tag_viewport_redraw(context)
    return projected, target


def set_landmark_observation(
    context: bpy.types.Context,
    image_point: tuple[float, float],
) -> None:
    """Store a point pick for the active landmark on the active match."""
    landmark = active_landmark(context)
    root = properties.active_root(context)
    if landmark is None:
        raise ValueError("Select a landmark first")
    if landmark.kind == "LINE":
        raise ValueError("Active landmark is a Line — drag a segment instead")
    if root is None:
        raise ValueError("Activate a match camera first")
    observation = observation_for_match(landmark, root)
    if observation is None:
        observation = landmark.observations.add()
        observation.match_root = root
    observation.x, observation.y = float(image_point[0]), float(image_point[1])
    observation.is_set = True
    observation.confidence = properties.workspace(context).landmark_pick_confidence
    settings = properties.active_session(context)
    if settings is not None:
        settings.status = f"Landmark '{landmark.name}' picked in {root.name}"
    properties.tag_viewport_redraw(context)


def set_landmark_line_observation(
    context: bpy.types.Context,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> None:
    """Store a line-segment pick for the active LINE landmark."""
    landmark = active_landmark(context)
    root = properties.active_root(context)
    if landmark is None:
        raise ValueError("Select a landmark first")
    if landmark.kind != "LINE":
        raise ValueError("Active landmark is not a Line")
    if root is None:
        raise ValueError("Activate a match camera first")
    observation = observation_for_match(landmark, root)
    if observation is None:
        observation = landmark.observations.add()
        observation.match_root = root
    observation.x, observation.y = float(point_a[0]), float(point_a[1])
    observation.x2, observation.y2 = float(point_b[0]), float(point_b[1])
    observation.is_set = True
    observation.confidence = properties.workspace(context).landmark_pick_confidence
    settings = properties.active_session(context)
    if settings is not None:
        settings.status = f"Line '{landmark.name}' drawn in {root.name}"
    properties.tag_viewport_redraw(context)


def clear_landmark_observation_for_active(context: bpy.types.Context) -> bool:
    """Remove the active match's observation from the active landmark."""
    landmark = active_landmark(context)
    root = properties.active_root(context)
    if landmark is None or root is None:
        return False
    for index, observation in enumerate(list(landmark.observations)):
        if observation.match_root == root:
            landmark.observations.remove(index)
            properties.tag_viewport_redraw(context)
            return True
    return False


def build_sync_problem(context: bpy.types.Context):
    """Collect frozen match calibrations, landmark observations, and known 3D."""
    from . import sync as sync_module

    roots = properties.iter_match_roots()
    matches = []
    for root in roots:
        session = root.pm_session
        if not getattr(session, "sync_enabled", True):
            continue
        if session.image is None or session.fx <= 0.0:
            continue
        matches.append(
            sync_module.SyncMatchInput(
                match_id=root.name,
                calibration=calibration_from_settings(session),
            )
        )
    match_ids = {item.match_id for item in matches}

    observations = []
    line_observations = []
    known_world: dict[str, object] = {}
    known_lines: dict[str, tuple] = {}
    parallel_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    space = properties.workspace(context)
    for landmark in space.landmarks:
        # Disabled landmarks keep picks for debugging but stay out of the graph.
        if not getattr(landmark, "use_in_sync", True):
            continue
        known_object = landmark.known_object
        known_object_b = getattr(landmark, "known_object_b", None)
        if landmark.kind == "LINE":
            other_id = getattr(landmark, "parallel_to", "NONE")
            if other_id and other_id != "NONE" and other_id != landmark.item_id:
                pair = tuple(sorted((landmark.item_id, other_id)))
                if pair not in seen_pairs:
                    # Confirm the target still exists as an enabled Line landmark.
                    target = next(
                        (
                            item
                            for item in space.landmarks
                            if item.item_id == other_id
                            and item.kind == "LINE"
                            and getattr(item, "use_in_sync", True)
                        ),
                        None,
                    )
                    if target is not None:
                        seen_pairs.add(pair)
                        parallel_pairs.append(pair)
            if (
                known_object is not None
                and known_object.name in bpy.data.objects
                and known_object_b is not None
                and known_object_b.name in bpy.data.objects
            ):
                location_a = known_object.matrix_world.to_translation()
                location_b = known_object_b.matrix_world.to_translation()
                known_lines[landmark.item_id] = (
                    (float(location_a.x), float(location_a.y), float(location_a.z)),
                    (float(location_b.x), float(location_b.y), float(location_b.z)),
                )
        elif known_object is not None and known_object.name in bpy.data.objects:
            # World-space location of the Empty / object origin (anchor frame).
            location = known_object.matrix_world.to_translation()
            known_world[landmark.item_id] = (
                float(location.x),
                float(location.y),
                float(location.z),
            )
        for observation in landmark.observations:
            root = observation.match_root
            if (
                not observation.is_set
                or root is None
                or root.name not in match_ids
            ):
                continue
            if landmark.kind == "LINE":
                line_observations.append(
                    sync_module.SyncLineObservation(
                        match_id=root.name,
                        landmark_id=landmark.item_id,
                        u1=float(observation.x),
                        v1=float(observation.y),
                        u2=float(observation.x2),
                        v2=float(observation.y2),
                        landmark_name=landmark.name or landmark.item_id,
                        weight=sync_module.confidence_weight(observation.confidence),
                    )
                )
            else:
                observations.append(
                    sync_module.SyncObservation(
                        match_id=root.name,
                        landmark_id=landmark.item_id,
                        u=float(observation.x),
                        v=float(observation.y),
                        on_ground=bool(landmark.on_ground),
                        landmark_name=landmark.name or landmark.item_id,
                        weight=sync_module.confidence_weight(observation.confidence),
                    )
                )
    return (
        matches,
        observations,
        known_world,
        line_observations,
        known_lines,
        parallel_pairs,
    )


def known_anchor_pick_warnings(context: bpy.types.Context) -> list[str]:
    """Flag Known 3D Empties whose stored anchor pick disagrees with re-projection.

    Large deltas usually mean the Empty moved after auto-project, or the
    anchor camera/intrinsics changed without re-projecting.
    """
    from . import sync as sync_module

    anchor = properties.anchor_root(context)
    if anchor is None:
        return []
    session = anchor.pm_session
    if not getattr(session, "sync_enabled", True):
        return []
    if session.image is None or session.fx <= 0.0:
        return []
    calibration = calibration_from_settings(session)
    warnings: list[str] = []
    for landmark in properties.workspace(context).landmarks:
        if not getattr(landmark, "use_in_sync", True):
            continue
        if landmark.known_object is None:
            continue
        observation = observation_for_match(landmark, anchor)
        if observation is None or not observation.is_set:
            continue
        world = landmark.known_object.matrix_world.to_translation()
        private_point = _private_point_from_world(anchor, world)
        projected = sync_module.project_private_point(private_point, calibration)
        if projected is None:
            warnings.append(
                f"{landmark.name or landmark.item_id}: Known 3D off-screen in anchor"
            )
            continue
        delta = float(
            (projected[0] - observation.x) ** 2 + (projected[1] - observation.y) ** 2
        ) ** 0.5
        if delta > 5.0:
            warnings.append(
                f"{landmark.name or landmark.item_id}: Empty vs anchor pick "
                f"{delta:.0f}px — re-run Landmarks from Selected / Use Selected"
            )
    return warnings


def _apply_sync_landmark_diagnostics(context: bpy.types.Context, result) -> None:
    """Write per-landmark RMSE onto the workspace even when sync is rejected."""
    space = properties.workspace(context)
    for landmark in space.landmarks:
        rmse = result.per_landmark_rmse_px.get(landmark.item_id)
        if rmse is None:
            continue
        landmark.rmse_px = float(rmse)


def diagnose_sync(context: bpy.types.Context):
    """Run sync solve for diagnostics without applying Empty transforms."""
    from . import sync as sync_module

    space = properties.workspace(context)
    anchor = properties.anchor_root(context)
    if anchor is None:
        raise ValueError("Choose an anchor match first")

    warnings = known_anchor_pick_warnings(context)
    matches, observations, known_world, line_observations, known_lines, parallel_pairs = (
        build_sync_problem(context)
    )
    if not getattr(anchor.pm_session, "sync_enabled", True):
        raise ValueError(
            "Anchor match has sync disabled — enable it or choose another anchor"
        )
    if not matches:
        raise ValueError("No sync-enabled solved matches available")
    if len(matches) < 2:
        raise ValueError("Need at least two sync-enabled matches")
    if anchor.name not in {item.match_id for item in matches}:
        raise ValueError("Anchor match needs a solved camera")

    result = sync_module.solve_landmark_sync(
        matches,
        observations,
        anchor_id=anchor.name,
        known_world=known_world,
        line_observations=line_observations,
        known_lines=known_lines,
        parallel_pairs=parallel_pairs,
        lock_rotation=bool(space.lock_rotation),
        lock_translation=bool(space.lock_translation),
    )
    if result.mean_reprojection_px > 8.0 or not result.success:
        result.leave_one_out = sync_module.leave_one_out_landmark_report(
            matches,
            observations,
            anchor_id=anchor.name,
            known_world=known_world,
            line_observations=line_observations,
            known_lines=known_lines,
            parallel_pairs=parallel_pairs,
            top_k=5,
            baseline=result if result.per_landmark_rmse_px else None,
            lock_rotation=bool(space.lock_rotation),
            lock_translation=bool(space.lock_translation),
        )
    _apply_sync_landmark_diagnostics(context, result)

    parts = []
    skipped_matches = sum(
        1
        for root in properties.iter_match_roots()
        if not getattr(root.pm_session, "sync_enabled", True)
    )
    if skipped_matches:
        parts.append(f"{skipped_matches} match(es) sync-disabled")
    excluded = sum(
        1 for landmark in space.landmarks if not getattr(landmark, "use_in_sync", True)
    )
    if excluded:
        parts.append(f"{excluded} landmark(s) excluded from sync")
    if warnings:
        parts.append("Known 3D: " + "; ".join(warnings[:3]))
    parts.append(result.message)
    if result.per_landmark_rmse_px:
        ranked = sorted(
            result.per_landmark_rmse_px.items(), key=lambda item: -item[1]
        )[:5]
        name_by_id = {
            landmark.item_id: (landmark.name or landmark.item_id)
            for landmark in space.landmarks
        }
        bits = [
            f"{name_by_id.get(landmark_id, landmark_id[:8])} {rmse:.1f}px"
            for landmark_id, rmse in ranked
        ]
        parts.append("Per-landmark: " + ", ".join(bits))
    if result.leave_one_out:
        helpful = [
            f"{name} {with_rmse:.0f}→{without_rmse:.0f}px"
            for name, with_rmse, without_rmse in result.leave_one_out[:3]
            if without_rmse < with_rmse
        ]
        if helpful:
            parts.append("Leave-one-out: " + ", ".join(helpful))
    if result.downweighted_landmark_ids:
        name_by_id = {
            landmark.item_id: (landmark.name or landmark.item_id)
            for landmark in space.landmarks
        }
        bits = [
            name_by_id.get(landmark_id, landmark_id[:8])
            for landmark_id in result.downweighted_landmark_ids[:4]
        ]
        parts.append("Auto-downweighted: " + ", ".join(bits))
    if result.success:
        parts.append("Pose OK to apply via Solve Sync (not applied by Diagnose)")
    space.sync_status = " | ".join(parts)
    properties.tag_viewport_redraw(context)
    return result


def solve_and_apply_sync(context: bpy.types.Context):
    """Run landmark sync and write similarities onto match root Empties."""
    from . import sync as sync_module

    space = properties.workspace(context)
    anchor = properties.anchor_root(context)
    if anchor is None:
        raise ValueError("Choose an anchor match first")

    warnings = known_anchor_pick_warnings(context)
    matches, observations, known_world, line_observations, known_lines, parallel_pairs = (
        build_sync_problem(context)
    )
    if not getattr(anchor.pm_session, "sync_enabled", True):
        raise ValueError(
            "Anchor match has sync disabled — enable it or choose another anchor"
        )
    if not matches:
        raise ValueError("No sync-enabled solved matches available")
    if len(matches) < 2:
        raise ValueError("Need at least two sync-enabled matches")
    if anchor.name not in {item.match_id for item in matches}:
        raise ValueError("Anchor match needs a solved camera")

    result = sync_module.solve_landmark_sync(
        matches,
        observations,
        anchor_id=anchor.name,
        known_world=known_world,
        line_observations=line_observations,
        known_lines=known_lines,
        parallel_pairs=parallel_pairs,
        lock_rotation=bool(space.lock_rotation),
        lock_translation=bool(space.lock_translation),
    )
    _apply_sync_landmark_diagnostics(context, result)
    message = result.message
    skipped_matches = sum(
        1
        for root in properties.iter_match_roots()
        if not getattr(root.pm_session, "sync_enabled", True)
    )
    if skipped_matches:
        message = f"{skipped_matches} match(es) sync-disabled · " + message
    excluded = sum(
        1 for landmark in space.landmarks if not getattr(landmark, "use_in_sync", True)
    )
    if excluded:
        message = f"{excluded} landmark(s) excluded · " + message
    if warnings:
        message = "Known 3D warn: " + "; ".join(warnings[:2]) + " | " + message
    space.sync_status = message
    if not result.success:
        raise ValueError(message)

    root_by_name = {root.name: root for root in properties.iter_match_roots()}
    for match_id, similarity in result.similarities.items():
        root = root_by_name.get(match_id)
        if root is None:
            continue
        apply_similarity_to_root(root, similarity)
        root.pm_session.sync_rmse_px = float(
            result.per_match_rmse_px.get(match_id, 0.0)
        )

    landmark_by_id = {landmark.item_id: landmark for landmark in space.landmarks}
    for landmark_id, position in result.landmarks.items():
        landmark = landmark_by_id.get(landmark_id)
        if landmark is None:
            continue
        landmark.position = tuple(float(value) for value in position)
        landmark.has_position = True
        landmark.rmse_px = float(result.per_landmark_rmse_px.get(landmark_id, 0.0))
        segment = result.line_segments.get(landmark_id)
        if segment is not None:
            landmark.position = tuple(float(value) for value in segment[0])
            landmark.position_b = tuple(float(value) for value in segment[1])
            landmark.has_line_segment = True
        else:
            landmark.has_line_segment = False

    sync_landmark_empties(context)
    properties.tag_viewport_redraw(context)
    return result


def clear_sync_transforms(context: bpy.types.Context) -> None:
    """Reset all match root Empties and clear solved landmark positions."""
    space = properties.workspace(context)
    for root in properties.iter_match_roots():
        reset_root_sync_transform(root)
    for landmark in space.landmarks:
        landmark.has_position = False
        landmark.has_line_segment = False
        landmark.rmse_px = 0.0
    clear_landmark_empties(context)
    space.sync_status = "Sync cleared"
    properties.tag_viewport_redraw(context)


def refine_lenses_and_sync(context: bpy.types.Context):
    """Search per-match fx (VP prior) to lower sync RMSE, then apply sync.

    Matches with 1-point mode or fewer than two VP axes stay frozen.
    Manual FOV matches are included so multi-view RMSE can adjust their fx.
    Blocking convenience wrapper — the UI operator runs this work in a thread.
    """
    prep = prepare_lens_refine(context)
    from . import lens_refine

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
    )
    return apply_lens_refine_result(context, refine_result, prep.root_by_name)


@dataclass
class LensRefinePrep:
    """bpy-free inputs gathered on the main thread for a lens refine job."""

    lens_inputs: list
    observations: list
    known_world: dict
    line_observations: list
    known_lines: dict
    parallel_pairs: list
    anchor_id: str
    fx_span: float
    root_by_name: dict
    lock_rotation: bool = False
    lock_translation: bool = False


def prepare_lens_refine(context: bpy.types.Context) -> LensRefinePrep:
    """Validate sync state and build pure-data inputs for lens refine."""
    from . import lens_refine

    space = properties.workspace(context)
    anchor = properties.anchor_root(context)
    if anchor is None:
        raise ValueError("Choose an anchor match first")

    matches_pack, observations, known_world, line_observations, known_lines, parallel_pairs = (
        build_sync_problem(context)
    )
    if not getattr(anchor.pm_session, "sync_enabled", True):
        raise ValueError(
            "Anchor match has sync disabled — enable it or choose another anchor"
        )
    if not matches_pack:
        raise ValueError("No sync-enabled solved matches available")
    if len(matches_pack) < 2:
        raise ValueError("Need at least two sync-enabled matches")
    if anchor.name not in {item.match_id for item in matches_pack}:
        raise ValueError("Anchor match needs a solved camera")
    if not observations and not line_observations:
        raise ValueError("Add landmark picks before refining lenses")

    root_by_name = {root.name: root for root in properties.iter_match_roots()}
    lens_inputs: list[lens_refine.MatchLensInput] = []
    for item in matches_pack:
        root = root_by_name.get(item.match_id)
        if root is None:
            continue
        settings = root.pm_session
        line_bundles = line_bundles_from_settings(settings)
        ready_axes = sum(1 for segments in line_bundles.values() if len(segments) >= 2)
        freeze = settings.vp_mode == "1" or ready_axes < 2
        # Manual FOV matches stay searchable — Refine Lenses exists to adjust fx.
        origin_image = None
        if settings.origin_is_set:
            origin_image = (
                float(settings.origin_image[0]),
                float(settings.origin_image[1]),
            )
        lens_inputs.append(
            lens_refine.MatchLensInput(
                match_id=item.match_id,
                line_bundles=line_bundles,
                intrinsics=item.calibration.intrinsics,
                division_lambda=float(item.calibration.division_lambda),
                origin_image=origin_image,
                freeze_focal=freeze,
                base_calibration=item.calibration,
            )
        )

    if not lens_inputs:
        raise ValueError("No matches available for lens refine")

    return LensRefinePrep(
        lens_inputs=lens_inputs,
        observations=observations,
        known_world=known_world,
        line_observations=line_observations,
        known_lines=known_lines,
        parallel_pairs=parallel_pairs,
        anchor_id=anchor.name,
        fx_span=max(float(space.lens_refine_span_percent), 1.0) / 100.0,
        root_by_name=root_by_name,
        lock_rotation=bool(space.lock_rotation),
        lock_translation=bool(space.lock_translation),
    )


def apply_lens_refine_result(context: bpy.types.Context, refine_result, root_by_name: dict):
    """Write refined calibrations and re-run Solve Sync (main thread only)."""
    space = properties.workspace(context)
    if refine_result.cancelled:
        space.sync_status = refine_result.message
        space.lens_refine_progress = 0.0
        properties.tag_viewport_redraw(context)
        return refine_result, None

    # Write private calibrations without flipping the scene camera each time.
    active_root = properties.active_root(context)
    for match_id, calibration in refine_result.calibrations.items():
        root = root_by_name.get(match_id)
        if root is None:
            continue
        settings = root.pm_session
        previous = calibration_from_settings(settings)
        if _intrinsics_or_distortion_changed(previous, calibration):
            invalidate_undistorted_cache(settings)
        apply_camera(
            context.scene,
            settings,
            calibration,
            update_scene_camera=False,
        )
        _update_diagnostics(settings, line_bundles_from_settings(settings), calibration)

    if active_root is not None:
        apply_camera(
            context.scene,
            active_root.pm_session,
            calibration_from_settings(active_root.pm_session),
            update_scene_camera=True,
        )

    # Re-run the normal apply path so Empty transforms match the new lenses.
    try:
        sync_result = solve_and_apply_sync(context)
        message = refine_result.message + " · " + sync_result.message
        space.sync_status = message
        space.lens_refine_progress = 0.0
        properties.tag_viewport_redraw(context)
        return refine_result, sync_result
    except Exception as error:
        # Lenses may still have improved even if the hard reject remains.
        space.sync_status = refine_result.message + " · " + str(error)
        space.lens_refine_progress = 0.0
        properties.tag_viewport_redraw(context)
        # Surface a synthetic failed sync result so the operator can WARN, not ERROR.
        from . import sync as sync_module

        failed = sync_module.SyncSolveResult(
            similarities={},
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message=str(error),
            success=False,
        )
        return refine_result, failed
