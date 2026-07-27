"""Self-contained NumPy image remapping for the division lens model."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

from . import core, scene


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


def default_output_path(source_path: str) -> str:
    """Return the conventional cached PNG path beside the source image."""
    path = Path(source_path)
    return str(path.with_name(f"{path.stem}.undistorted.png"))


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
    settings = context.scene.match_perspective
    if settings.image is None:
        raise ValueError("Load a reference image first")
    if abs(settings.division_lambda) < 1.0e-8:
        raise ValueError("No lens distortion is currently estimated")

    source_image = settings.image
    width = int(source_image.size[0])
    height = int(source_image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Reference image has no readable pixel dimensions")

    # Keep RNA sizes aligned with the buffer we actually remap.
    settings.image_width = width
    settings.image_height = height

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

    image_name = f"{source_image.name}.undistorted"
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
    _write_image_pixels(output_image, remapped)

    resolved_path = str(
        Path(output_path or default_output_path(settings.image_path)).expanduser().resolve()
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
    settings.status = f"Undistorted plate saved: {Path(resolved_path).name}"
    return output_image


def set_undistorted_view(context: bpy.types.Context, enabled: bool) -> None:
    """Switch camera background between source and cached undistorted plate."""
    settings = context.scene.match_perspective
    if enabled and settings.undistorted_image is None:
        generate_undistorted_plate(context)
        return
    settings.view_undistorted = enabled
    scene.refresh_background_projection(context)
    settings.status = "Viewing undistorted plate" if enabled else "Viewing original image"
