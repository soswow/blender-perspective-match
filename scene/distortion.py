"""Self-contained NumPy image remapping for the division lens model."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

from .. import core, properties, scene


def compute_canvas(
    width: int,
    height: int,
    intrinsics: core.CameraIntrinsics,
    division_lambda: float,
    *,
    samples_per_edge: int = 64,
    max_scale: float = 2.5,
) -> tuple[int, int, float, float]:
    """Return expanded undistorted width, height, and ideal-space offsets."""
    if abs(division_lambda) < 1.0e-15:
        return width, height, 0.0, 0.0
    sample_count = max(8, samples_per_edge)
    horizontal = np.linspace(0.0, float(width - 1), sample_count)
    vertical = np.linspace(0.0, float(height - 1), sample_count)
    border = np.vstack(
        [
            np.column_stack([horizontal, np.zeros(sample_count)]),
            np.column_stack([horizontal, np.full(sample_count, float(height - 1))]),
            np.column_stack([np.zeros(sample_count), vertical]),
            np.column_stack([np.full(sample_count, float(width - 1)), vertical]),
        ]
    )
    ideal = core.undistort_points(
        border,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        division_lambda,
    )
    ideal = ideal[np.isfinite(ideal).all(axis=1)]
    if len(ideal) == 0:
        return width, height, 0.0, 0.0
    minimum_x = float(np.floor(min(float(np.min(ideal[:, 0])), 0.0)))
    minimum_y = float(np.floor(min(float(np.min(ideal[:, 1])), 0.0)))
    maximum_x = float(np.ceil(max(float(np.max(ideal[:, 0])), float(width - 1))))
    maximum_y = float(np.ceil(max(float(np.max(ideal[:, 1])), float(height - 1))))
    output_width = int(maximum_x - minimum_x) + 1
    output_height = int(maximum_y - minimum_y) + 1
    if output_width > int(width * max_scale) or output_height > int(height * max_scale):
        return width, height, 0.0, 0.0
    return output_width, output_height, minimum_x, minimum_y


def _image_pixels_top_left(image: bpy.types.Image) -> np.ndarray:
    """Read Blender's bottom-up float pixel buffer as top-left-origin RGBA.

    Blender may leave file-backed images unloaded until pixels are touched. Some
    RGB stills also report alpha 0; force opaque alpha so remapped PNGs are not
    fully transparent (which many viewers composite as black).
    """
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Reference image has no pixel dimensions")

    # Ensure file pixels are resident before reading.
    if not image.has_data:
        try:
            image.reload()
        except Exception:
            pass

    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    if not np.isfinite(buffer).all():
        raise ValueError("Reference image contains non-finite pixel values")
    if float(np.max(np.abs(buffer))) < 1.0e-8:
        raise ValueError(
            "Reference image pixel buffer is empty — reload the still and try again"
        )

    rgba = np.flipud(buffer.reshape(height, width, 4))
    # Match the desktop OpenCV path: treat the still as opaque for remapping.
    rgba = np.array(rgba, dtype=np.float32, copy=True, order="C")
    rgba[:, :, 3] = 1.0
    return rgba


def remap_rgba(
    source_rgba: np.ndarray,
    intrinsics: core.CameraIntrinsics,
    division_lambda: float,
    canvas: tuple[int, int, float, float],
) -> np.ndarray:
    """Bilinearly remap a top-left-origin RGBA image without OpenCV."""
    source_height, source_width = source_rgba.shape[:2]
    output_width, output_height, offset_x, offset_y = canvas
    output = np.zeros((output_height, output_width, 4), dtype=np.float32)
    chunk_rows = 128
    covered = 0
    for row_start in range(0, output_height, chunk_rows):
        row_end = min(output_height, row_start + chunk_rows)
        y_values, x_values = np.meshgrid(
            np.arange(row_start, row_end, dtype=np.float64),
            np.arange(output_width, dtype=np.float64),
            indexing="ij",
        )
        ideal = np.column_stack(
            [x_values.reshape(-1) + offset_x, y_values.reshape(-1) + offset_y]
        )
        observed = core.distort_points(
            ideal,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            division_lambda,
        )
        source_x = observed[:, 0]
        source_y = observed[:, 1]
        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0.0)
            & (source_y >= 0.0)
            & (source_x <= source_width - 1)
            & (source_y <= source_height - 1)
        )
        if not np.any(valid):
            continue
        covered += int(np.count_nonzero(valid))
        x0 = np.floor(source_x[valid]).astype(np.int64)
        y0 = np.floor(source_y[valid]).astype(np.int64)
        x1 = np.minimum(x0 + 1, source_width - 1)
        y1 = np.minimum(y0 + 1, source_height - 1)
        x_weight = (source_x[valid] - x0).astype(np.float32)[:, None]
        y_weight = (source_y[valid] - y0).astype(np.float32)[:, None]
        top = source_rgba[y0, x0] * (1.0 - x_weight) + source_rgba[y0, x1] * x_weight
        bottom = source_rgba[y1, x0] * (1.0 - x_weight) + source_rgba[y1, x1] * x_weight
        sampled = top * (1.0 - y_weight) + bottom * y_weight
        flat_chunk = output[row_start:row_end].reshape(-1, 4)
        flat_chunk[np.flatnonzero(valid)] = sampled

    if covered == 0:
        raise ValueError(
            "Undistort remap produced an empty plate — check FOV, principal point, and λ"
        )
    return output


def default_view_path(source_path: str, match_key: str = "") -> str:
    """Return the sibling display plate path: ``<stem>[-match]-pm-view.png``."""
    path = Path(source_path)
    token = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in match_key
    ).strip("_")[:48]
    if token:
        return str(path.with_name(f"{path.stem}-{token}-pm-view.png"))
    return str(path.with_name(f"{path.stem}-pm-view.png"))


def default_output_path(source_path: str, match_key: str = "") -> str:
    """Return the conventional cached undistorted PNG path beside the source image."""
    path = Path(source_path)
    token = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in match_key
    ).strip("_")[:48]
    if token:
        return str(path.with_name(f"{path.stem}-{token}.undistorted.png"))
    return str(path.with_name(f"{path.stem}.undistorted.png"))


def _plate_key(settings: properties.PMSession) -> str:
    """Stable per-match token so view/undistorted plates are never shared."""
    camera = settings.camera_object
    if camera is not None and camera.name:
        return camera.name
    return "PM_orphan"


def apply_exposure_contrast(
    rgba: np.ndarray,
    exposure_ev: float,
    contrast: float,
) -> np.ndarray:
    """Apply desktop-compatible display-only brightness then contrast."""
    # CSS filter order used by Perspective Match Studio: brightness(2^EV) then contrast.
    gain = float(2.0 ** exposure_ev)
    lit = np.array(rgba, dtype=np.float32, copy=True, order="C")
    lit[:, :, :3] = np.clip(lit[:, :, :3] * gain, 0.0, 1.0)
    contrast_value = float(contrast)
    lit[:, :, :3] = np.clip((lit[:, :, :3] - 0.5) * contrast_value + 0.5, 0.0, 1.0)
    lit[:, :, 3] = 1.0
    return lit


def display_source_image(settings: properties.PMSession) -> bpy.types.Image:
    """Return the plate undistortion/background should sample (lit or original)."""
    if (
        settings.view_lighting_applied
        and settings.view_image is not None
        and settings.view_image.name in bpy.data.images
    ):
        return settings.view_image
    if settings.image is None:
        raise ValueError("Load a reference image first")
    return settings.image


def apply_view_lighting(context: bpy.types.Context) -> bpy.types.Image:
    """Bake EV/contrast from the original still into ``*-pm-view.png`` and activate it."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    # Always bake from the solver source still — never from a previous view plate.
    scene.ensure_session_image(settings)
    if settings.image is None:
        raise ValueError("Load a reference image first")
    if scene._is_derived_display_image(settings, settings.image):
        raise ValueError(
            "Reference image pointer points at a view/undistorted plate — "
            "re-open the original still"
        )
    if not settings.image_path:
        raise ValueError("Reference image has no file path to write a view plate beside")

    source = _image_pixels_top_left(settings.image)
    lit = apply_exposure_contrast(source, settings.view_exposure, settings.view_contrast)
    height, width = lit.shape[:2]

    plate_key = _plate_key(settings)
    resolved_path = str(
        Path(default_view_path(settings.image_path, plate_key)).expanduser().resolve()
    )
    # Per-match datablock name — shared source stills must not share one view plate.
    image_name = f"{plate_key}.pm-view"
    existing = bpy.data.images.get(image_name)
    if existing is not None and tuple(existing.size) != (width, height):
        bpy.data.images.remove(existing)
        existing = None
    created_new = existing is None
    output_image = existing or bpy.data.images.new(
        image_name,
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    output_image.alpha_mode = "STRAIGHT"
    try:
        output_image.colorspace_settings.name = "sRGB"
    except TypeError:
        pass
    # Baked display plate must not pick up the scene view transform on top of EV.
    if hasattr(output_image, "use_view_as_render"):
        output_image.use_view_as_render = False
    _write_image_pixels(output_image, lit)
    output_image.filepath_raw = resolved_path
    output_image.file_format = "PNG"
    try:
        output_image.save()
    except Exception:
        if created_new and output_image.users == 0:
            bpy.data.images.remove(output_image)
        raise
    try:
        output_image.pack()
    except RuntimeError:
        pass

    # Keep the plate alive across match switches even if no background references it.
    output_image.use_fake_user = True
    settings.view_image = output_image
    settings.view_path = resolved_path
    settings.view_lighting_applied = True
    settings.view_baked_exposure = float(settings.view_exposure)
    settings.view_baked_contrast = float(settings.view_contrast)

    # Undistorted cache must follow the same re-lit plate when active.
    if settings.view_undistorted and abs(settings.division_lambda) > 1.0e-8:
        generate_undistorted_plate(context)
    else:
        # Drop a stale undistorted cache so the next generate uses the lit plate.
        if settings.undistorted_image is not None:
            cached_undistorted = settings.undistorted_image
            settings.undistorted_image = None
            settings.undistorted_path = ""
            settings.undistorted_width = 0
            settings.undistorted_height = 0
            settings.undistorted_offset_x = 0.0
            settings.undistorted_offset_y = 0.0
            if cached_undistorted.users == 0:
                bpy.data.images.remove(cached_undistorted)
        scene.refresh_background_projection(context)

    settings.status = (
        f"View lighting applied ({settings.view_exposure:+.2f} EV, "
        f"contrast {settings.view_contrast:.2f})"
    )
    return output_image


def _view_plate_is_alive(image: bpy.types.Image | None) -> bool:
    """Return whether a view-plate Image pointer still resolves in bpy.data."""
    if image is None:
        return False
    try:
        name = image.name
    except ReferenceError:
        return False
    return name in bpy.data.images


def _load_view_plate_from_path(
    settings: properties.PMSession,
    expected_name: str,
) -> bpy.types.Image | None:
    """Reload a previously baked view plate from ``view_path`` when the pointer died."""
    if not settings.view_path:
        return None
    try:
        absolute_path = str(
            Path(bpy.path.abspath(settings.view_path)).expanduser().resolve()
        )
    except Exception:
        return None
    if not Path(absolute_path).is_file():
        return None
    loaded = bpy.data.images.load(absolute_path, check_existing=True)
    # Prefer the stable per-match datablock name even if Blender reused another ID.
    if loaded.name != expected_name:
        existing = bpy.data.images.get(expected_name)
        if existing is not None and existing != loaded:
            loaded = existing
        else:
            loaded.name = expected_name
    if hasattr(loaded, "use_view_as_render"):
        loaded.use_view_as_render = False
    loaded.use_fake_user = True
    return loaded


def ensure_match_view_plate(context: bpy.types.Context) -> None:
    """Make sure this match owns a private view plate for its baked lighting.

    Matches that share one source still used to share one ``*.pm-view`` image, so
    Apply on match B quietly overwrote match A's exposure. On activate, restore or
    rebuild when the plate pointer is missing or still uses the old shared name.
    """
    settings = properties.active_session(context)
    if settings is None or not settings.view_lighting_applied:
        return
    expected_name = f"{_plate_key(settings)}.pm-view"
    view_image = settings.view_image if _view_plate_is_alive(settings.view_image) else None

    if view_image is None:
        view_image = _load_view_plate_from_path(settings, expected_name)
        if view_image is not None:
            settings.view_image = view_image

    if view_image is not None and view_image.name == expected_name:
        if hasattr(view_image, "use_view_as_render"):
            view_image.use_view_as_render = False
        view_image.use_fake_user = True
        settings.view_image = view_image
        return

    # Missing, legacy shared name, or renamed — rebake a private per-match plate.
    apply_view_lighting(context)


def reset_view_lighting(context: bpy.types.Context) -> None:
    """Stop using the view plate and restore the original still (and undistorted)."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")

    cached = settings.view_image
    settings.view_lighting_applied = False
    settings.view_image = None
    settings.view_path = ""
    settings.view_exposure = 0.0
    settings.view_contrast = 1.0
    settings.view_baked_exposure = 0.0
    settings.view_baked_contrast = 1.0
    if cached is not None and cached.users == 0:
        bpy.data.images.remove(cached)

    if settings.view_undistorted and abs(settings.division_lambda) > 1.0e-8:
        generate_undistorted_plate(context)
    else:
        if settings.undistorted_image is not None:
            scene.invalidate_undistorted_cache(settings)
        scene.refresh_background_projection(context)
    settings.status = "View lighting reset to original"


def _write_image_pixels(image: bpy.types.Image, top_left_rgba: np.ndarray) -> None:
    """Upload top-left RGBA into a Blender image (bottom-up pixel buffer)."""
    flat = np.ascontiguousarray(np.flipud(top_left_rgba), dtype=np.float32).reshape(-1)
    image.pixels.foreach_set(flat)
    image.update()


def generate_undistorted_plate(
    context: bpy.types.Context,
    output_path: str | None = None,
) -> bpy.types.Image:
    """Generate, save, and activate the undistorted camera background."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    if settings.image is None:
        raise ValueError("Load a reference image first")
    if abs(settings.division_lambda) < 1.0e-8:
        raise ValueError("No lens distortion is currently estimated")

    source_image = display_source_image(settings)
    width = int(source_image.size[0])
    height = int(source_image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Reference image has no readable pixel dimensions")

    # Keep RNA sizes aligned with the original still (solver / overlay space).
    if settings.image is not None:
        settings.image_width = int(settings.image.size[0])
        settings.image_height = int(settings.image.size[1])

    calibration = scene.calibration_from_settings(settings)
    calibration.intrinsics.image_width = width
    calibration.intrinsics.image_height = height
    source = _image_pixels_top_left(source_image)
    canvas = compute_canvas(
        width,
        height,
        calibration.intrinsics,
        settings.division_lambda,
    )
    remapped = remap_rgba(
        source,
        calibration.intrinsics,
        settings.division_lambda,
        canvas,
    )
    output_width, output_height, offset_x, offset_y = canvas
    if float(np.mean(remapped[:, :, 3])) < 0.01:
        raise ValueError("Undistorted plate is empty (near-zero alpha)")

    image_name = f"{_plate_key(settings)}.undistorted"
    existing = bpy.data.images.get(image_name)
    if existing is not None and tuple(existing.size) != (output_width, output_height):
        bpy.data.images.remove(existing)
        existing = None
    created_new = existing is None
    output_image = existing or bpy.data.images.new(
        image_name,
        width=output_width,
        height=output_height,
        alpha=True,
        float_buffer=False,
    )
    output_image.alpha_mode = "STRAIGHT"
    # Blender 5.1 clears pixel RGB if colorspace is changed after foreach_set.
    try:
        output_image.colorspace_settings.name = "sRGB"
    except TypeError:
        pass
    if hasattr(output_image, "use_view_as_render"):
        output_image.use_view_as_render = False
    _write_image_pixels(output_image, remapped)

    resolved_path = str(
        Path(
            output_path
            or default_output_path(settings.image_path, _plate_key(settings))
        ).expanduser().resolve()
    )
    if Path(resolved_path).suffix.lower() != ".png":
        resolved_path = str(Path(resolved_path).with_suffix(".png"))
    output_image.filepath_raw = resolved_path
    output_image.file_format = "PNG"
    try:
        # GENERATED images save from the in-memory buffer.
        output_image.save()
    except Exception:
        if created_new and output_image.users == 0:
            bpy.data.images.remove(output_image)
        raise

    try:
        output_image.pack()
    except RuntimeError:
        pass

    settings.undistorted_image = output_image
    settings.undistorted_path = resolved_path
    settings.undistorted_width = output_width
    settings.undistorted_height = output_height
    settings.undistorted_offset_x = offset_x
    settings.undistorted_offset_y = offset_y
    settings.view_undistorted = True
    scene.refresh_background_projection(context)
    # Path is noisy in the sidebar footer — console is enough.
    print(f"Perspective Match: undistorted plate saved → {resolved_path}")
    return output_image


def sync_undistorted_plate_after_refine(context: bpy.types.Context) -> None:
    """Rebuild the undistorted plate when viewing one; else leave the source plate.

    Call after refine so FOV/PP changes regenerate the plate instead of leaving
    a blank/reset background. Does not re-estimate λ — that is button-only.

    The plate depends on intrinsics and λ only. Orientation-only VP refines keep
    a valid cache — skip the expensive remap + PNG write in that case.
    """
    settings = properties.active_session(context)
    if settings is None:
        return
    if (
        settings.view_undistorted
        and abs(settings.division_lambda) > 1.0e-8
        and not settings.lambda_saturated
    ):
        # Cache still present ⇒ K/λ unchanged (invalidate clears the pointer).
        if settings.undistorted_image is not None:
            return
        try:
            generate_undistorted_plate(context)
        except Exception as error:
            settings.status = f"Distortion estimated; plate failed: {error}"
            print(f"Perspective Match: undistorted plate failed: {error}")
        return
    # Viewing undistorted but λ is gone/unusable — fall back to the source still.
    if settings.view_undistorted:
        scene.invalidate_undistorted_cache(settings)
        scene.refresh_background_projection(context)


def revert_to_original_plate(context: bpy.types.Context) -> None:
    """Clear λ, drop undistorted cache, re-solve, and show the source plate."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    settings.division_lambda = 0.0
    settings.lambda_saturated = False
    scene.invalidate_undistorted_cache(settings)
    line_bundles = scene.line_bundles_from_settings(settings)
    lock_focal = bool(settings.lock_focal or settings.vp_mode == "1")
    if core.can_solve_orientation(
        line_bundles,
        lock_focal=lock_focal,
        vp_mode=settings.vp_mode,
    ):
        scene.refine_match(context)
    else:
        scene.apply_manual_fov(context)
    scene.refresh_background_projection(context)
    settings.status = "Viewing original image"


def use_original_plate(context: bpy.types.Context) -> None:
    """Clear λ and restore the original plate + camera."""
    revert_to_original_plate(context)


def estimate_distortion(context: bpy.types.Context) -> None:
    """Estimate λ once from VP lines and show the undistorted plate.

    Subsequent VP-line refines keep the stored λ; press again to re-fit.
    """
    settings = properties.active_session(context)
    if settings is None or settings.image is None:
        raise ValueError("Load a reference image first")
    # Keep Manual FOV if set — λ is estimated at the locked focal.
    line_bundles = scene.line_bundles_from_settings(settings)
    lock_focal = bool(settings.lock_focal or settings.vp_mode == "1")
    if not core.can_solve_orientation(
        line_bundles,
        lock_focal=lock_focal,
        vp_mode=settings.vp_mode,
    ):
        tip = (
            "draw VP lines, then Apply Manual FOV / Auto from VPs"
            if lock_focal
            else "draw VP lines, then Auto from VPs"
        )
        raise ValueError(f"Need enough VP lines first — {tip}")
    scene.refine_match(context, estimate_distortion=True)
    if (
        abs(settings.division_lambda) > 1.0e-8
        and not settings.lambda_saturated
    ):
        try:
            generate_undistorted_plate(context)
        except Exception as error:
            settings.status = f"Distortion estimated; plate failed: {error}"
            print(f"Perspective Match: undistorted plate failed: {error}")
            raise
        return
    if settings.lambda_saturated:
        settings.status = "Estimate saturated; pinhole retained"
        return
    settings.status = (
        "Need ≥3 concurrent segments on one axis to estimate λ"
    )


def set_undistorted_view(context: bpy.types.Context, enabled: bool) -> None:
    """Switch camera background between source/view plate and cached undistorted plate."""
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Create or activate a match camera first")
    if enabled and settings.undistorted_image is None:
        generate_undistorted_plate(context)
        return
    settings.view_undistorted = enabled
    scene.refresh_background_projection(context)
    if enabled:
        settings.status = "Viewing undistorted plate"
    elif settings.view_lighting_applied:
        settings.status = "Viewing lit plate"
    else:
        settings.status = "Viewing original image"
