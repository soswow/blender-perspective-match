"""Snap a landmark click onto the centre of a nearby AprilTag-like quad.

Needs OpenCV (upscaled adaptive threshold + filled contour). The centre is the
intersection of the quadrilateral's diagonals. The sidebar checkbox is hidden
when the bundled wheel is missing, same as Find AprilTags / Detect VP Lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Crop radii so tiny and mid-size tags both fit.
SEARCH_RADII = (32, 48, 80)
UPSCALE = 3
ADAPTIVE_C = 5
MIN_CONTRAST = 8.0
MIN_SIDE_PX = 3.0
# Recover the rest of a split inner body (~original pixels around the seed).
INNER_EXPAND_PX = 4
# Ignore near-black padding when growing; interior holes are filled by the hull.
MIN_RECLASS_VALUE = 40


@dataclass(frozen=True)
class TagSnapResult:
    """Snapped landmark in the same pixel space as the input gray image."""

    center_xy: tuple[float, float]
    corners_xy: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    mean_shift_px: float
    contrast: float


def _as_uint8_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize HxW image to uint8 luminance."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError("Tag snap expects a single-channel grayscale image")
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float64, copy=False)
    peak = float(np.max(values))
    if peak <= 1.5:
        values = values * 255.0
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _cv2_module():
    """Return the OpenCV module when the bundled wheel is loaded, else None."""
    try:
        from .opencv import capabilities

        return capabilities().module
    except Exception:
        return None


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def projective_quad_center(points_xy: np.ndarray) -> np.ndarray:
    """Return the intersection of a quadrilateral's corner diagonals."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    first_direction = points[2] - points[0]
    second_direction = points[3] - points[1]
    denominator = _cross_2d(first_direction, second_direction)
    if abs(denominator) < 1.0e-12:
        return points.mean(axis=0)
    distance = _cross_2d(points[1] - points[0], second_direction) / denominator
    return points[0] + distance * first_direction


def _order_corners(corners: np.ndarray) -> np.ndarray:
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    return corners[np.argsort(angles)]


def _quad_is_valid(corners: np.ndarray) -> bool:
    if corners.shape[0] != 4:
        return False
    ordered = _order_corners(corners)
    sides = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    if float(np.min(sides)) < MIN_SIDE_PX:
        return False
    signs = []
    for index in range(4):
        edge = ordered[(index + 1) % 4] - ordered[index]
        nxt = ordered[(index + 2) % 4] - ordered[(index + 1) % 4]
        signs.append(_cross_2d(edge, nxt))
    if any(abs(value) < 1.0e-8 for value in signs):
        return False
    if not (
        all(value > 0.0 for value in signs) or all(value < 0.0 for value in signs)
    ):
        return False
    return True


def _point_in_convex_quad(
    point: np.ndarray, corners: np.ndarray, *, margin: float = 3.0
) -> bool:
    ordered = _order_corners(corners)
    distances = []
    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1.0e-9:
            continue
        distances.append(_cross_2d(edge, point - start) / length)
    if not distances:
        return False
    return all(value >= -margin for value in distances) or all(
        value <= margin for value in distances
    )


def _fill_holes(cv2, blob: np.ndarray) -> np.ndarray:
    """Fill holes that do not touch the mask border."""
    inverted = 1 - blob
    flooded = inverted.copy()
    cv2.floodFill(flooded, None, (0, 0), 2)
    filled = blob.copy()
    filled[flooded == 0] = 1
    return filled


def _label_at_or_near(labels: np.ndarray, x: int, y: int, max_radius: int) -> int:
    height, width = labels.shape
    if 0 <= y < height and 0 <= x < width and labels[y, x] != 0:
        return int(labels[y, x])
    for radius in range(1, max_radius + 1):
        y0 = max(0, y - radius)
        x0 = max(0, x - radius)
        y1 = min(height, y + radius + 1)
        x1 = min(width, x + radius + 1)
        nearby = labels[y0:y1, x0:x1]
        nonzero = nearby[nearby > 0]
        if nonzero.size:
            return int(nonzero.flat[0])
    return 0


def _convex_blob(cv2, blob: np.ndarray) -> np.ndarray:
    """Fill the convex hull of the largest external contour."""
    contours, _unused = cv2.findContours(
        blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return blob
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(blob)
    cv2.fillConvexPoly(filled, cv2.convexHull(contour), 1)
    return filled


def _complete_inner_body(cv2, equalized: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Fill the inner black body around a dark seed, stopping at the quiet zone."""
    seed = _convex_blob(cv2, blob)
    if int(seed.sum()) < 8:
        return blob
    radius = max(int(INNER_EXPAND_PX * UPSCALE), 3) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    expanded = cv2.dilate(seed, kernel, iterations=1)
    ring = (expanded > 0) & (seed == 0)
    if not np.any(ring):
        return seed
    interior_mean = float(equalized[seed > 0].mean())
    ring_bright = float(np.percentile(equalized[ring], 70.0))
    threshold = 0.5 * (interior_mean + ring_bright)
    kept = (
        (expanded > 0)
        & (equalized <= threshold)
        & (equalized >= MIN_RECLASS_VALUE)
    ).astype(np.uint8)
    _count, labels = cv2.connectedComponents(kept)
    overlap = labels[seed > 0]
    overlap = overlap[overlap > 0]
    if overlap.size == 0:
        return seed
    label = int(np.bincount(overlap).argmax())
    return _convex_blob(cv2, (labels == label).astype(np.uint8))


def _approx_quad(cv2, blob: np.ndarray) -> np.ndarray | None:
    contours, _unused = cv2.findContours(
        blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter < 8.0:
        return None
    for fraction in np.linspace(0.01, 0.12, 20):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approx) == 4:
            return np.asarray(approx, dtype=np.float64).reshape(4, 2)
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return np.asarray(box, dtype=np.float64).reshape(4, 2)


def snap_click_to_tag_center(
    gray: np.ndarray,
    click_xy: tuple[float, float],
) -> TagSnapResult | None:
    """Refine a click onto the diagonal-intersection centre of a nearby tag.

    Returns None when OpenCV is missing or no dark quadrilateral is found.
    """
    cv2 = _cv2_module()
    if cv2 is None:
        return None
    click = np.array(click_xy, dtype=np.float64)
    if not np.isfinite(click).all():
        return None
    gray_u8 = _as_uint8_gray(gray)
    if min(gray_u8.shape) < 8:
        return None
    height, width = gray_u8.shape
    click_x, click_y = float(click[0]), float(click[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best: tuple[float, TagSnapResult] | None = None
    scale = UPSCALE
    for radius in SEARCH_RADII:
        x0 = int(np.clip(np.floor(click_x - radius), 0, width - 1))
        y0 = int(np.clip(np.floor(click_y - radius), 0, height - 1))
        x1 = int(np.clip(np.ceil(click_x + radius) + 1, 1, width))
        y1 = int(np.clip(np.ceil(click_y + radius) + 1, 1, height))
        roi = gray_u8[y0:y1, x0:x1]
        if min(roi.shape) < 8:
            continue
        big = cv2.resize(
            roi,
            (roi.shape[1] * scale, roi.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        if min(big.shape) >= 8:
            equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(
                big
            )
        else:
            equalized = big
        block = max(11, (min(equalized.shape) // 6) | 1)
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            equalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            ADAPTIVE_C,
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        local_x = int((click_x - x0) * scale)
        local_y = int((click_y - y0) * scale)
        _count, labels = cv2.connectedComponents(binary)
        label = _label_at_or_near(labels, local_x, local_y, 10 * scale)
        if label == 0:
            continue
        blob = _complete_inner_body(
            cv2,
            equalized,
            _fill_holes(cv2, (labels == label).astype(np.uint8)),
        )
        area = int(blob.sum())
        if area < 16 * scale * scale or area > 0.7 * blob.size:
            continue
        quad = _approx_quad(cv2, blob)
        if quad is None:
            continue
        corners = _order_corners(
            quad / float(scale) + np.array((x0, y0), dtype=np.float64)
        )
        if not _quad_is_valid(corners):
            continue
        if not _point_in_convex_quad(click, corners, margin=3.0):
            continue
        small = (
            cv2.resize(
                blob, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
        if int(small.sum()) < 8:
            continue
        dilated = cv2.dilate(small.astype(np.uint8), np.ones((3, 3), np.uint8))
        ring = dilated.astype(bool) & ~small
        if not np.any(ring):
            continue
        contrast = float(roi[ring].mean()) - float(roi[small].mean())
        if contrast < MIN_CONTRAST:
            continue
        area_px = area / float(scale * scale)
        # Prefer a complete inner body over a high-contrast fragment of the bits.
        score = contrast * max(area_px, 1.0) ** 0.5
        center = projective_quad_center(corners)
        result = TagSnapResult(
            center_xy=(float(center[0]), float(center[1])),
            corners_xy=tuple((float(p[0]), float(p[1])) for p in corners),
            mean_shift_px=float(np.linalg.norm(center - click)),
            contrast=contrast / 255.0,
        )
        if best is None or score > best[0]:
            best = (score, result)
    return None if best is None else best[1]


def snap_point_in_session(
    settings,
    click_xy: tuple[float, float],
) -> TagSnapResult | None:
    """Snap a storage-space click using the active match plate."""
    from .. import core, scene
    from . import line_snap

    gray, space = line_snap.gray_for_session(settings)
    if space == "display":
        display = scene._storage_to_display(settings, click_xy[0], click_xy[1])[:2]
        snapped = snap_click_to_tag_center(gray, display)
        if snapped is None:
            return None
        storage = scene._display_to_storage(
            settings, snapped.center_xy[0], snapped.center_xy[1]
        )
        storage_corners = tuple(
            scene._display_to_storage(settings, corner[0], corner[1])
            for corner in snapped.corners_xy
        )
        return TagSnapResult(
            center_xy=storage,
            corners_xy=storage_corners,
            mean_shift_px=snapped.mean_shift_px,
            contrast=snapped.contrast,
        )

    snapped = snap_click_to_tag_center(gray, click_xy)
    if snapped is None:
        return None
    return _correct_center_for_lens(snapped, settings, core)


def _correct_center_for_lens(snapped: TagSnapResult, settings, core) -> TagSnapResult:
    """Recompute the centre in ideal space when the still is distorted."""
    division_lambda = float(getattr(settings, "division_lambda", 0.0))
    brown_conrady = tuple(getattr(settings, "brown_conrady", ()))
    if not core.has_lens_distortion(
        division_lambda,
        brown_conrady,
        threshold=1.0e-15,
    ):
        return snapped
    fx = max(float(getattr(settings, "fx", 0.0)), 1.0e-6)
    fy = max(float(getattr(settings, "fy", 0.0)), 1.0e-6)
    cx = float(getattr(settings, "cx", 0.0))
    cy = float(getattr(settings, "cy", 0.0))
    ideal_corners = core.undistort_points(
        np.asarray(snapped.corners_xy, dtype=np.float64),
        fx,
        fy,
        cx,
        cy,
        division_lambda,
        brown_conrady,
    )
    ideal_center = projective_quad_center(ideal_corners)
    storage_center = core.distort_points(
        ideal_center.reshape(1, 2),
        fx,
        fy,
        cx,
        cy,
        division_lambda,
        brown_conrady,
    )[0]
    return TagSnapResult(
        center_xy=(float(storage_center[0]), float(storage_center[1])),
        corners_xy=snapped.corners_xy,
        mean_shift_px=snapped.mean_shift_px,
        contrast=snapped.contrast,
    )
