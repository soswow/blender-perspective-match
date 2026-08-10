"""Snap a drawn VP segment onto a nearby image edge or thin line.

Fits a straight feature along the whole seed length: step edges (dark↔light),
dark mid-lines (grout / painted strokes), and bright mid-lines. Pure NumPy so
unit tests do not need OpenCV; Blender session helpers load pixels separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Half-width of the perpendicular search band (pixels).
DEFAULT_SEARCH_RADIUS = 6
# Spacing of sample centres along the seed segment.
DEFAULT_SAMPLE_SPACING = 2.0
# Reject fits that rotate more than this from the hand-drawn direction.
DEFAULT_MAX_ANGLE_DEGREES = 10.0
# Fraction of samples that must land on a clear peak.
DEFAULT_MIN_INLIER_FRACTION = 0.55
# Peak score must beat this relative to local profile contrast.
DEFAULT_MIN_RELATIVE_SCORE = 0.12


@dataclass(frozen=True)
class SnapResult:
    """Snapped endpoints in the same pixel space as the input gray image."""

    point_a: tuple[float, float]
    point_b: tuple[float, float]
    kind: str
    confidence: float
    mean_shift_px: float


def _as_float_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize HxW image to float64 luminance in roughly [0, 1]."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError("Snap expects a single-channel grayscale image")
    values = array.astype(np.float64, copy=False)
    peak = float(np.max(values))
    if peak > 1.5:
        values = values / 255.0
    return values


def _bilinear(gray: np.ndarray, x: float, y: float) -> float:
    """Sample with bilinear interpolation; clamp to image bounds."""
    height, width = gray.shape
    if width < 2 or height < 2:
        return float(gray[0, 0]) if gray.size else 0.0
    x_clamped = min(width - 1.001, max(0.0, x))
    y_clamped = min(height - 1.001, max(0.0, y))
    x0 = int(np.floor(x_clamped))
    y0 = int(np.floor(y_clamped))
    x1 = x0 + 1
    y1 = y0 + 1
    tx = x_clamped - x0
    ty = y_clamped - y0
    top = gray[y0, x0] * (1.0 - tx) + gray[y0, x1] * tx
    bottom = gray[y1, x0] * (1.0 - tx) + gray[y1, x1] * tx
    return float(top * (1.0 - ty) + bottom * ty)


def _unit_direction(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (direction, normal, length) or None if the segment is degenerate."""
    delta = point_b - point_a
    length = float(np.linalg.norm(delta))
    if length < 1.0e-6:
        return None
    direction = delta / length
    normal = np.array((-direction[1], direction[0]), dtype=np.float64)
    return direction, normal, length


def _parabolic_offset(scores: np.ndarray, peak_index: int) -> float:
    """Sub-pixel peak via a 3-point parabola; falls back to the integer index."""
    if peak_index <= 0 or peak_index >= len(scores) - 1:
        return float(peak_index)
    left = float(scores[peak_index - 1])
    mid = float(scores[peak_index])
    right = float(scores[peak_index + 1])
    denominator = left - 2.0 * mid + right
    if abs(denominator) < 1.0e-12:
        return float(peak_index)
    delta = 0.5 * (left - right) / denominator
    return float(peak_index) + float(np.clip(delta, -0.5, 0.5))


def _profile_scores(profile: np.ndarray) -> dict[str, np.ndarray]:
    """Score each offset for edge / dark-line / bright-line features."""
    # Discrete Laplacian: positive at dark valleys, negative at bright ridges.
    laplacian = np.convolve(profile, np.array([1.0, -2.0, 1.0]), mode="same")
    gradient = np.abs(np.gradient(profile))
    # Suppress window borders — incomplete neighbourhoods are noisy.
    if len(profile) >= 5:
        gradient[0] = gradient[-1] = 0.0
        laplacian[0] = laplacian[-1] = 0.0
        gradient[1] = gradient[-2] = 0.0
        laplacian[1] = laplacian[-2] = 0.0
    return {
        "edge": gradient,
        "dark_line": laplacian,
        "bright_line": -laplacian,
    }


def _best_offset(
    profile: np.ndarray,
    scores: np.ndarray,
    *,
    kind: str,
    search_radius: int,
    min_relative_score: float,
) -> tuple[float, float] | None:
    """Return (offset_px, peak_score) if the peak is strong enough for ``kind``."""
    if scores.size == 0:
        return None
    peak_index = int(np.argmax(scores))
    peak_score = float(scores[peak_index])
    score_range = float(np.ptp(scores))
    if score_range < 1.0e-8 or peak_score <= 0.0:
        return None
    if peak_score < min_relative_score * score_range:
        return None
    # Dark/bright mid-lines must be real intensity extrema (not step-edge lobes).
    if 0 < peak_index < len(profile) - 1:
        centre = float(profile[peak_index])
        left = float(profile[peak_index - 1])
        right = float(profile[peak_index + 1])
        if kind == "dark_line" and not (centre < left and centre < right):
            return None
        if kind == "bright_line" and not (centre > left and centre > right):
            return None
    offset = _parabolic_offset(scores, peak_index) - float(search_radius)
    return offset, peak_score


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """PCA line through points → (origin, unit direction)."""
    if points.shape[0] < 2:
        return None
    origin = points.mean(axis=0)
    centered = points - origin
    try:
        _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = right[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-12:
        return None
    return origin, direction / norm


def _project_point(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Orthogonal projection onto an infinite line."""
    return origin + float(np.dot(point - origin, direction)) * direction


def _angle_degrees(direction_a: np.ndarray, direction_b: np.ndarray) -> float:
    """Smallest angle between two directions (0…90°)."""
    cosine = abs(float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosine)))


def snap_segment_to_feature(
    gray: np.ndarray,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    sample_spacing: float = DEFAULT_SAMPLE_SPACING,
    max_angle_degrees: float = DEFAULT_MAX_ANGLE_DEGREES,
    min_inlier_fraction: float = DEFAULT_MIN_INLIER_FRACTION,
    min_relative_score: float = DEFAULT_MIN_RELATIVE_SCORE,
) -> SnapResult | None:
    """Refine a seed segment onto the strongest nearby straight feature.

    Samples perpendicular intensity profiles along the whole segment, votes for
    edge / dark-line / bright-line, fits a line through the winning peaks, and
    projects both endpoints onto that fit. Returns None when confidence is low.
    """
    image = _as_float_gray(gray)
    seed_a = np.array(point_a, dtype=np.float64)
    seed_b = np.array(point_b, dtype=np.float64)
    oriented = _unit_direction(seed_a, seed_b)
    if oriented is None:
        return None
    direction, normal, length = oriented
    if length < 8.0:
        return None
    radius = max(int(search_radius), 2)
    sample_count = max(5, int(length / max(sample_spacing, 0.5)) + 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)

    mode_points: dict[str, list[np.ndarray]] = {
        "edge": [],
        "dark_line": [],
        "bright_line": [],
    }
    mode_scores: dict[str, list[float]] = {
        "edge": [],
        "dark_line": [],
        "bright_line": [],
    }
    mode_shifts: dict[str, list[float]] = {
        "edge": [],
        "dark_line": [],
        "bright_line": [],
    }

    for sample_index in range(sample_count):
        t = sample_index / (sample_count - 1)
        centre = seed_a + t * (seed_b - seed_a)
        profile = np.empty(offsets.shape[0], dtype=np.float64)
        for index, offset in enumerate(offsets):
            sample = centre + offset * normal
            profile[index] = _bilinear(image, float(sample[0]), float(sample[1]))
        scored = _profile_scores(profile)
        for kind, scores in scored.items():
            best = _best_offset(
                profile,
                scores,
                kind=kind,
                search_radius=radius,
                min_relative_score=min_relative_score,
            )
            if best is None:
                continue
            shift, peak_score = best
            mode_points[kind].append(centre + shift * normal)
            mode_scores[kind].append(peak_score)
            mode_shifts[kind].append(abs(shift))

    best_result: SnapResult | None = None
    best_rank = -1.0
    min_inliers = max(3, int(np.ceil(min_inlier_fraction * sample_count)))

    for kind, points in mode_points.items():
        if len(points) < min_inliers:
            continue
        fitted = _fit_line(np.asarray(points, dtype=np.float64))
        if fitted is None:
            continue
        origin, fit_direction = fitted
        # Keep orientation close to what the user intended.
        if _angle_degrees(direction, fit_direction) > max_angle_degrees:
            continue
        # Align fit direction with the seed so endpoint order is preserved.
        if float(np.dot(fit_direction, direction)) < 0.0:
            fit_direction = -fit_direction
        snapped_a = _project_point(seed_a, origin, fit_direction)
        snapped_b = _project_point(seed_b, origin, fit_direction)
        mean_score = float(np.mean(mode_scores[kind]))
        mean_shift = float(np.mean(mode_shifts[kind]))
        # Prefer strong peaks and denser inlier sets; light penalty on large moves.
        rank = mean_score * (len(points) / sample_count) - 0.01 * mean_shift
        if rank <= best_rank:
            continue
        best_rank = rank
        best_result = SnapResult(
            point_a=(float(snapped_a[0]), float(snapped_a[1])),
            point_b=(float(snapped_b[0]), float(snapped_b[1])),
            kind=kind,
            confidence=mean_score,
            mean_shift_px=mean_shift,
        )

    return best_result


def gray_from_blender_image(image) -> np.ndarray:
    """Convert a Blender Image to float grayscale (top-left origin)."""
    from . import distortion

    rgba = distortion._image_pixels_top_left(image)
    # Rec. 601 luminance in linear-ish float space (good enough for edges).
    return (
        0.299 * rgba[:, :, 0]
        + 0.587 * rgba[:, :, 1]
        + 0.114 * rgba[:, :, 2]
    ).astype(np.float64)


def gray_from_path(image_path: str) -> np.ndarray | None:
    """Load grayscale from disk via OpenCV when available; else None."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray_u8.astype(np.float64) / 255.0


def gray_for_session(settings) -> tuple[np.ndarray, str]:
    """Return (gray, space) for the plate the user is drawing on.

    ``space`` is ``\"storage\"`` (source pixels) or ``\"display\"`` (undistorted
    plate pixels). Callers must convert endpoints to that space before snap and
    convert results back to storage afterwards.
    """
    if (
        getattr(settings, "view_undistorted", False)
        and getattr(settings, "undistorted_image", None) is not None
    ):
        return gray_from_blender_image(settings.undistorted_image), "display"

    image_path = getattr(settings, "image_path", "") or ""
    if image_path:
        from_disk = gray_from_path(image_path)
        if from_disk is not None:
            return from_disk, "storage"

    image = getattr(settings, "image", None)
    if image is None:
        raise ValueError("Active match has no reference image")
    # Prefer the lit view plate when it matches source size (same geometry).
    view_image = getattr(settings, "view_image", None)
    if (
        view_image is not None
        and getattr(settings, "view_lighting_applied", False)
        and int(view_image.size[0]) == int(image.size[0])
        and int(view_image.size[1]) == int(image.size[1])
    ):
        return gray_from_blender_image(view_image), "storage"
    return gray_from_blender_image(image), "storage"


def snap_segment_in_session(
    settings,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    **kwargs,
) -> SnapResult | None:
    """Snap storage-space endpoints using the active match plate."""
    from . import scene

    gray, space = gray_for_session(settings)
    if space == "display":
        display_a = scene._storage_to_display(settings, point_a[0], point_a[1])[:2]
        display_b = scene._storage_to_display(settings, point_b[0], point_b[1])[:2]
        snapped = snap_segment_to_feature(gray, display_a, display_b, **kwargs)
        if snapped is None:
            return None
        storage_a = scene._display_to_storage(
            settings, snapped.point_a[0], snapped.point_a[1]
        )
        storage_b = scene._display_to_storage(
            settings, snapped.point_b[0], snapped.point_b[1]
        )
        return SnapResult(
            point_a=storage_a,
            point_b=storage_b,
            kind=snapped.kind,
            confidence=snapped.confidence,
            mean_shift_px=snapped.mean_shift_px,
        )
    return snap_segment_to_feature(gray, point_a, point_b, **kwargs)


def kind_label(kind: str) -> str:
    """Short UI label for a snap kind."""
    if kind == "edge":
        return "edge"
    if kind == "dark_line":
        return "dark line"
    if kind == "bright_line":
        return "bright line"
    return kind
