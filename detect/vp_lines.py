"""Automatic vanishing-point line detection for 3-point perspective.

Pipeline: LSD segments (downscaled) → RANSAC VP clusters (plus a dedicated
near-horizontal pass for depth edges) → pick an orthogonal triad that is well
separated in direction space → assign X / Y / Z with the convention green=left,
red=right, blue=up/down. Writes the same ``PMLineSegment`` RNA the Draw /
Edit tool uses (source-image pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import numpy as np

from .. import core
from ..core import AxisId, LineSegment

# Drop short LSD fragments (noise / texture). Measured in full-res pixels.
# Tuned further by ``sensitivity`` (0 = strict, 1 = very sensitive).
_MIN_LENGTH_FRAC = 0.06
_MIN_LENGTH_PX = 48.0
_LENGTH_PERCENTILE_FLOOR = 45.0
_MAX_CANDIDATES = 160
# Run LSD on a downscaled plate, then map endpoints back to full resolution.
_DETECT_MAX_SIDE = 1280
# Default LSD knobs (OpenCV); remapped by sensitivity.
_LSD_QUANT_DEFAULT = 2.0
_LSD_DENSITY_DEFAULT = 0.7
# RANSAC inlier threshold: perpendicular distance of the VP to the infinite line.
_VP_RESIDUAL_PX = 4.5
_RANSAC_ITERATIONS = 100
_RANSAC_POOL = 72
_MAX_CLUSTERS = 6
# Keep a handful of strongest, well-spread members per axis for editing.
_MAX_SEGMENTS_PER_AXIS = 6
# Reject a candidate whose midpoint is closer than this fraction of the diagonal
# to an already-picked segment (near-duplicate uprights / mullions).
_MIN_SEGMENT_SEPARATION_FRAC = 0.07
# Minimum angle at the vanishing point between picked rays (degrees).
# Far VPs (typical uprights) use a lower floor via ``_adaptive_min_angle``.
_MIN_SEGMENT_ANGLE_DEGREES = 6.0
_MIN_SEGMENT_ANGLE_FAR_DEGREES = 2.5
# Prefer focals in a plausible range vs image width.
_FOCAL_MIN_FRAC = 0.25
_FOCAL_MAX_FRAC = 4.0
# Weight for pairwise VP angular separation when ranking triads (radians).
_SEPARATION_WEIGHT = 6.0
# Prefer triads that mix upright + flatter bundles (depth edges are often short).
_UPRIGHT_SPREAD_WEIGHT = 25.0
# Near-2VP still counts in 3-point: reward an X/Y VP that lands on the plate.
_INSIDE_HORIZONTAL_VP_BONUS = 12.0
# Orientation gates when turning a mixed RANSAC cluster into an axis bundle.
# Depth/green: keep flat floor / moulding strokes (door rails are steeper ~0.35+).
_DEPTH_AXIS_UPRIGHT_MAX = 0.32
# Across/red: no image-orientation gate — X strokes can look nearly vertical while
# still converging to the across VP (hallway wall edges). Residual filters outliers.
_ACROSS_AXIS_UPRIGHT_MAX = 1.0
_VERTICAL_AXIS_UPRIGHT_MIN = 0.55
# Segments below this uprightness feed a dedicated depth-VP RANSAC pass.
_HORIZONTAL_UPRIGHT_MAX = 0.35
# Keep more across strokes when the cluster is rich (far wall edges help FOV).
_MAX_SEGMENTS_ACROSS = 10
_MAX_SEGMENTS_DEPTH = 8
# Default Edge Sensitivity slider value (fallback when RNA is missing).
_DEFAULT_SENSITIVITY = 0.7


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _lerp(low: float, high: float, amount: float) -> float:
    return low + (high - low) * _clamp01(amount)


@dataclass(frozen=True)
class EdgeDetectSettings:
    """Resolved knobs for one LSD pass (from the Edge Sensitivity slider)."""

    sensitivity: float
    quant: float
    density_th: float
    min_length_frac: float
    min_length_px: float
    length_percentile_floor: float
    max_candidates: int
    clahe_clip: float
    clahe_blend: float


def edge_detect_settings(sensitivity: float = _DEFAULT_SENSITIVITY) -> EdgeDetectSettings:
    """Map a 0–1 sensitivity knob onto LSD + filter parameters.

    Higher sensitivity lowers gradient/density thresholds, softens length cuts,
    and applies CLAHE so faint architectural edges survive.
    """
    amount = _clamp01(sensitivity)
    return EdgeDetectSettings(
        sensitivity=amount,
        # Lower quant ⇒ weaker gradients still seed a line region.
        quant=_lerp(2.6, 0.55, amount),
        density_th=_lerp(0.85, 0.32, amount),
        min_length_frac=_lerp(0.085, 0.028, amount),
        min_length_px=_lerp(64.0, 24.0, amount),
        length_percentile_floor=_lerp(55.0, 15.0, amount),
        max_candidates=int(round(_lerp(100.0, 300.0, amount))),
        # CLAHE only ramps in above ~0.2 so "Low" stays close to raw LSD.
        clahe_clip=_lerp(0.0, 3.5, max(0.0, (amount - 0.2) / 0.8)),
        clahe_blend=_lerp(0.0, 1.0, max(0.0, (amount - 0.2) / 0.8)),
    )


@dataclass(frozen=True)
class VpCluster:
    """One vanishing-point hypothesis with its supporting segments."""

    vanishing: np.ndarray
    segments: tuple[LineSegment, ...]
    support_length: float


@dataclass(frozen=True)
class DetectVpLinesResult:
    """Counts written after applying automatic VP line detection."""

    candidates: int
    clusters: int
    counts: dict[str, int]


@dataclass(frozen=True)
class DetectVpLinesOutcome:
    """Full detect payload for the UI thread (bundles + debug edges)."""

    bundles: dict[AxisId, list[LineSegment]]
    candidates: tuple[LineSegment, ...]
    result: DetectVpLinesResult


class VpLineDependencyError(RuntimeError):
    """Raised when OpenCV line detection is unavailable."""


def _import_cv2():
    """Import OpenCV with LSD; raise VpLineDependencyError if missing."""
    from .opencv import capabilities

    caps = capabilities()
    if caps.module is None or not caps.line_segment_detector:
        detail = caps.error or "OpenCV with Line Segment Detector is not available"
        raise VpLineDependencyError(
            "VP line detection needs the bundled OpenCV wheel. "
            "Run ./scripts/fetch-wheels.sh, then disable and re-enable Perspective Match "
            f"(or install a fresh platform zip from ./scripts/build-extension.sh). ({detail})"
        )
    return caps.module


def _min_segment_length(
    width: int,
    height: int,
    settings: EdgeDetectSettings | None = None,
) -> float:
    resolved = settings or edge_detect_settings()
    return max(
        resolved.min_length_px,
        resolved.min_length_frac * float(min(width, height)),
    )


def _enhance_gray_for_detection(
    gray: np.ndarray,
    settings: EdgeDetectSettings,
    cv2_module,
) -> np.ndarray:
    """Optionally lift local contrast so faint edges reach LSD."""
    if settings.clahe_blend <= 1.0e-6 or settings.clahe_clip <= 1.0e-6:
        return gray
    clahe = cv2_module.createCLAHE(
        clipLimit=float(settings.clahe_clip),
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)
    blend = float(settings.clahe_blend)
    return cv2_module.addWeighted(enhanced, blend, gray, 1.0 - blend, 0)


def default_vp_detect_debug_path(source_path: str, match_key: str = "") -> str:
    """Sibling debug plate: ``<stem>[-match]-pm-vp-edges.png``."""
    path = Path(source_path)
    token = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in match_key
    ).strip("_")[:48]
    if token:
        return str(path.with_name(f"{path.stem}-{token}-pm-vp-edges.png"))
    return str(path.with_name(f"{path.stem}-pm-vp-edges.png"))


def _prepare_detection_gray(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Optionally downscale for LSD; returns (work_gray, full_from_work_scale)."""
    height, width = gray.shape
    max_side = max(width, height)
    if max_side <= _DETECT_MAX_SIDE:
        return gray, 1.0
    scale = _DETECT_MAX_SIDE / float(max_side)
    work_width = max(1, int(round(width * scale)))
    work_height = max(1, int(round(height * scale)))
    cv2 = _import_cv2()
    work = cv2.resize(gray, (work_width, work_height), interpolation=cv2.INTER_AREA)
    return work, 1.0 / scale


def detect_line_segments(
    gray: np.ndarray,
    *,
    sensitivity: float = _DEFAULT_SENSITIVITY,
) -> list[LineSegment]:
    """Run LSD and return longer segments in full-resolution source pixels."""
    cv2 = _import_cv2()
    if gray.ndim != 2:
        raise ValueError("Detection expects a single-channel grayscale image")
    detect_settings = edge_detect_settings(sensitivity)
    full_height, full_width = gray.shape
    work, to_full = _prepare_detection_gray(gray)
    work = _enhance_gray_for_detection(work, detect_settings, cv2)

    # LSD_REFINE_NONE is much faster; length + RANSAC filter quality afterward.
    # quant / density_th track the Edge Sensitivity slider (lower ⇒ more hits).
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_NONE,
        0.8,
        0.6,
        detect_settings.quant,
        22.5,
        0.0,
        detect_settings.density_th,
        1024,
    )
    detected = detector.detect(work)
    raw_lines = detected[0] if isinstance(detected, tuple) else detected
    if raw_lines is None or len(raw_lines) == 0:
        return []

    min_length_full = _min_segment_length(full_width, full_height, detect_settings)
    min_length_work = min_length_full / to_full
    segments: list[LineSegment] = []
    for row in raw_lines:
        x1, y1, x2, y2 = (float(value) for value in np.asarray(row).reshape(-1)[:4])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length_work:
            continue
        segments.append(
            LineSegment(
                x1 * to_full,
                y1 * to_full,
                x2 * to_full,
                y2 * to_full,
            )
        )

    if not segments:
        return []

    # Drop the shortest survivors so texture grit does not enter clustering.
    lengths = np.array([core.segment_length(segment) for segment in segments])
    percentile_floor = float(
        np.percentile(lengths, detect_settings.length_percentile_floor)
    )
    length_floor = max(min_length_full, percentile_floor)
    segments = [
        segment
        for segment in segments
        if core.segment_length(segment) >= length_floor
    ]
    segments.sort(key=core.segment_length, reverse=True)
    return segments[: detect_settings.max_candidates]


def render_debug_rgba(
    width: int,
    height: int,
    segments: list[LineSegment] | tuple[LineSegment, ...],
) -> np.ndarray:
    """Black plate with white strokes (float RGBA, top-left origin)."""
    cv2 = _import_cv2()
    canvas = np.zeros((height, width), dtype=np.uint8)
    for segment in segments:
        cv2.line(
            canvas,
            (int(round(segment.x1)), int(round(segment.y1))),
            (int(round(segment.x2)), int(round(segment.y2))),
            255,
            1,
            lineType=cv2.LINE_AA,
        )
    luma = canvas.astype(np.float32) / 255.0
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = luma
    rgba[:, :, 1] = luma
    rgba[:, :, 2] = luma
    rgba[:, :, 3] = 1.0
    return rgba


def _homogeneous_vp(vanishing: np.ndarray) -> np.ndarray:
    """Normalize a vanishing point for residual tests."""
    vanishing = np.asarray(vanishing, dtype=np.float64).reshape(3)
    if abs(float(vanishing[2])) < 1.0e-10:
        direction = vanishing[:2]
        length = float(np.linalg.norm(direction))
        if length < 1.0e-12:
            return vanishing
        return np.array(
            [direction[0] / length, direction[1] / length, 0.0],
            dtype=np.float64,
        )
    return vanishing / vanishing[2]


def segment_vp_residual(segment: LineSegment, vanishing: np.ndarray) -> float:
    """Perpendicular distance (px) of the VP from the infinite line, or angular proxy."""
    line = core._line_homogeneous(segment)
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        dx = segment.x2 - segment.x1
        dy = segment.y2 - segment.y1
        length = float(np.hypot(dx, dy))
        if length < 1.0e-9:
            return 1.0e9
        direction = vanishing_h[:2]
        cross = abs(dx * direction[1] - dy * direction[0]) / length
        return float(cross * max(length * 0.25, 8.0))
    return float(abs(line @ vanishing_h))


def _precompute_lines(
    segments: list[LineSegment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Homogeneous lines, lengths, and direction vectors for vectorized residuals."""
    lines = np.stack([core._line_homogeneous(segment) for segment in segments], axis=0)
    lengths = np.array(
        [max(core.segment_length(segment), 1.0e-6) for segment in segments],
        dtype=np.float64,
    )
    directions = np.array(
        [
            [
                (segment.x2 - segment.x1) / max(core.segment_length(segment), 1.0e-6),
                (segment.y2 - segment.y1) / max(core.segment_length(segment), 1.0e-6),
            ]
            for segment in segments
        ],
        dtype=np.float64,
    )
    return lines, lengths, directions


def _residuals_px(
    lines: np.ndarray,
    lengths: np.ndarray,
    directions: np.ndarray,
    vanishing: np.ndarray,
) -> np.ndarray:
    """Vectorized per-segment residuals to a vanishing point."""
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        direction = vanishing_h[:2]
        cross = np.abs(
            directions[:, 0] * direction[1] - directions[:, 1] * direction[0]
        )
        return cross * np.maximum(lengths * 0.25, 8.0)
    return np.abs(lines @ vanishing_h)


def _intersect_lines(first: LineSegment, second: LineSegment) -> np.ndarray | None:
    """Homogeneous intersection of two infinite lines."""
    vanishing = np.cross(
        core._line_homogeneous(first),
        core._line_homogeneous(second),
    )
    if float(np.linalg.norm(vanishing)) < 1.0e-12:
        return None
    return _homogeneous_vp(vanishing)


def _vp_unit_direction(
    vanishing: np.ndarray,
    cx: float,
    cy: float,
    focal: float,
) -> np.ndarray:
    """Camera-ray direction for a VP (proxy focal; used for separation only)."""
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        direction = np.array(
            [vanishing_h[0], vanishing_h[1], 0.0],
            dtype=np.float64,
        )
    else:
        direction = np.array(
            [
                (vanishing_h[0] - cx) / max(focal, 1.0),
                (vanishing_h[1] - cy) / max(focal, 1.0),
                1.0,
            ],
            dtype=np.float64,
        )
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / norm


def _angular_separation_radians(
    first: np.ndarray,
    second: np.ndarray,
    cx: float,
    cy: float,
    focal: float,
) -> float:
    direction_a = _vp_unit_direction(first, cx, cy, focal)
    direction_b = _vp_unit_direction(second, cx, cy, focal)
    return float(
        np.arccos(float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0)))
    )


def _triad_separation_radians(
    vanishing_points: list[np.ndarray],
    cx: float,
    cy: float,
    image_width: float,
) -> float:
    """Sum of pairwise VP angles (larger ⇒ better spread)."""
    focal = max(image_width, 1.0)
    total = 0.0
    for index_a, index_b in combinations(range(len(vanishing_points)), 2):
        total += _angular_separation_radians(
            vanishing_points[index_a],
            vanishing_points[index_b],
            cx,
            cy,
            focal,
        )
    return total


def _fit_cluster(
    segments: list[LineSegment],
    residual_px: float,
) -> VpCluster | None:
    """Fit a Huber VP and keep inliers; require ≥2 members."""
    if len(segments) < 2:
        return None
    vanishing = core.vanishing_point_from_lines(segments)
    if vanishing is None:
        return None
    lines, lengths, directions = _precompute_lines(segments)
    residuals = _residuals_px(lines, lengths, directions, vanishing)
    inliers = [
        segment
        for segment, residual in zip(segments, residuals)
        if float(residual) <= residual_px
    ]
    if len(inliers) < 2:
        return None
    refined = core.vanishing_point_from_lines(inliers)
    if refined is None:
        refined = vanishing
    support = float(sum(core.segment_length(segment) for segment in inliers))
    return VpCluster(
        vanishing=_homogeneous_vp(refined),
        segments=tuple(inliers),
        support_length=support,
    )


def _ransac_cluster(
    segments: list[LineSegment],
    *,
    residual_px: float,
    iterations: int,
    rng: np.random.Generator,
    avoid_vanishings: list[np.ndarray] | None = None,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> VpCluster | None:
    """Find the strongest VP cluster among the remaining segments."""
    if len(segments) < 2:
        return None

    pool_size = min(len(segments), _RANSAC_POOL)
    weights = np.array(
        [core.segment_length(segment) for segment in segments[:pool_size]],
        dtype=np.float64,
    )
    weights = weights / float(np.sum(weights))
    lines, lengths, directions = _precompute_lines(segments)
    cx = 0.5 * image_width
    cy = 0.5 * image_height
    focal = max(image_width, 1.0)
    avoid = avoid_vanishings or []

    best: VpCluster | None = None
    best_score = -1.0e18
    for _ in range(iterations):
        indices = rng.choice(pool_size, size=2, replace=False, p=weights)
        candidate = _intersect_lines(segments[indices[0]], segments[indices[1]])
        if candidate is None:
            continue
        # Skip hypotheses that sit too close to an already-accepted VP.
        # Near-2VP / slight lean: two far VPs can be angularly close yet still
        # distinct left-vs-right convergence points — keep those.
        if avoid:
            conflicts = False
            for other in avoid:
                if _vanishings_conflict(
                    candidate,
                    other,
                    cx=cx,
                    cy=cy,
                    focal=focal,
                    image_width=image_width,
                ):
                    conflicts = True
                    break
            if conflicts:
                continue
        residuals = _residuals_px(lines, lengths, directions, candidate)
        inlier_indices = np.flatnonzero(residuals <= residual_px)
        if inlier_indices.size < 2:
            continue
        inliers = [segments[index] for index in inlier_indices.tolist()]
        cluster = _fit_cluster(inliers, residual_px)
        if cluster is None:
            continue
        score = cluster.support_length
        if avoid:
            # Prefer the next VP to land far from earlier clusters.
            score += 40.0 * min(
                _angular_separation_radians(cluster.vanishing, other, cx, cy, focal)
                for other in avoid
            )
        if score > best_score:
            best_score = score
            best = cluster
    return best


def _vp_image_xy(vanishing: np.ndarray) -> tuple[float, float]:
    """Finite image coordinates for a VP; ±inf when the VP is at infinity."""
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        direction = vanishing_h[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-12:
            return 0.0, 1.0e9
        # Encode direction as a point very far along the ray for L/R compares.
        # Preserve sign(dx) so slight left vs right lean still sorts correctly.
        scale = 1.0e9 / norm
        return float(direction[0] * scale), float(direction[1] * scale)
    return float(vanishing_h[0]), float(vanishing_h[1])


def _vp_inside_image(
    vanishing: np.ndarray,
    image_width: float,
    image_height: float,
    *,
    margin_frac: float = 0.0,
) -> bool:
    """True when the VP projects inside the plate (optional normalized margin)."""
    vx, vy = _vp_image_xy(vanishing)
    if not np.isfinite(vx) or not np.isfinite(vy):
        return False
    if abs(vx) >= 1.0e8 or abs(vy) >= 1.0e8:
        return False
    margin_x = margin_frac * image_width
    margin_y = margin_frac * image_height
    return (
        -margin_x <= vx <= image_width + margin_x
        and -margin_y <= vy <= image_height + margin_y
    )


def _cluster_horizontal_lean(cluster: VpCluster) -> float:
    """Signed lean of a bundle: negative ⇒ toward left, positive ⇒ toward right.

    Uses length-weighted mean of (VP.x − midpoint.x). Almost-parallel strokes
    still pick a side from the slight convergence.
    """
    vx, _vy = _vp_image_xy(cluster.vanishing)
    total = 0.0
    weight = 0.0
    for segment in cluster.segments:
        length = core.segment_length(segment)
        if length < 1.0e-9:
            continue
        mid_x = 0.5 * (segment.x1 + segment.x2)
        total += length * (vx - mid_x)
        weight += length
    if weight < 1.0e-9:
        return 0.0
    return float(total / weight)


def _horizontal_vp_sort_key(cluster: VpCluster) -> tuple[float, float]:
    """Sort key left → right: VP x, then segment lean."""
    vx, _vy = _vp_image_xy(cluster.vanishing)
    return (float(vx), _cluster_horizontal_lean(cluster))


def _vanishings_conflict(
    first: np.ndarray,
    second: np.ndarray,
    *,
    cx: float,
    cy: float,
    focal: float,
    image_width: float,
) -> bool:
    """Whether two VP hypotheses are duplicates for RANSAC avoidance."""
    angle = _angular_separation_radians(first, second, cx, cy, focal)
    if angle >= np.radians(12.0):
        return False
    ax, _ay = _vp_image_xy(first)
    bx, _by = _vp_image_xy(second)
    # Opposite sides of the principal meridian ⇒ keep both (near-2VP / lean).
    if (ax - cx) * (bx - cx) < 0.0 and abs(ax - bx) > 0.2 * max(image_width, 1.0):
        return False
    # Both far but clearly different convergence distances along x.
    if abs(ax - bx) > max(image_width, 1.0) and angle >= np.radians(3.0):
        return False
    return True


def _clusters_are_similar(
    first: VpCluster,
    second: VpCluster,
    *,
    cx: float,
    cy: float,
    focal: float,
    min_angle_degrees: float = 8.0,
) -> bool:
    """True when two clusters likely describe the same vanishing direction."""
    # Reuse conflict logic so left/right lean pairs are not collapsed.
    if not _vanishings_conflict(
        first.vanishing,
        second.vanishing,
        cx=cx,
        cy=cy,
        focal=focal,
        image_width=max(abs(cx) * 2.0, 1.0),
    ):
        return False
    angle = _angular_separation_radians(
        first.vanishing, second.vanishing, cx, cy, focal
    )
    return angle < np.radians(min_angle_degrees)


def _pick_vertical_index(
    chosen: list[VpCluster],
    image_width: float,
    image_height: float,
) -> int:
    """Index of the upright (blue/Z) cluster within a triad.

    Prefers high segment uprightness. When scores are close, avoid calling an
    on-plate / near-center VP “vertical” — that is usually an X/Y depth VP in a
    near-2VP view.
    """
    upright_scores = [_cluster_uprightness(item) for item in chosen]
    best = float(max(upright_scores))
    candidates = [
        index
        for index, score in enumerate(upright_scores)
        if score >= best - 0.12
    ]
    cx = 0.5 * image_width
    cy = 0.5 * image_height

    def vertical_key(index: int) -> tuple[float, float, float]:
        cluster = chosen[index]
        vx, vy = _vp_image_xy(cluster.vanishing)
        inside = _vp_inside_image(
            cluster.vanishing, image_width, image_height, margin_frac=0.05
        )
        near_center = (
            abs(vx - cx) < 0.3 * image_width and abs(vy - cy) < 0.3 * image_height
        )
        # Higher is better: uprightness, vertical distance of VP, not on-plate center.
        return (
            upright_scores[index],
            abs(vy - cy) / max(image_height, 1.0),
            0.0 if (inside and near_center) else 1.0,
        )

    return max(candidates, key=vertical_key)


def _expand_depth_cluster_by_refit(
    cluster: VpCluster,
    candidates: list[LineSegment],
    *,
    upright_max: float,
    accept_residual: float,
    max_additions: int = 8,
) -> VpCluster:
    """Grow a depth VP by trying flat candidates with a provisional re-fit.

    A stroke can disagree with the current VP yet fit well once added (and pull
    the VP slightly). Static residual thresholds miss those; bad outliers still
    fail the provisional-fit residual check.
    """
    members = list(cluster.segments)
    member_keys = {
        (round(segment.x1, 1), round(segment.y1, 1), round(segment.x2, 1), round(segment.y2, 1))
        for segment in members
    }
    pool = [
        segment
        for segment in candidates
        if _segment_uprightness(segment) <= upright_max
        and (
            round(segment.x1, 1),
            round(segment.y1, 1),
            round(segment.x2, 1),
            round(segment.y2, 1),
        )
        not in member_keys
    ]
    vanishing = cluster.vanishing
    for _ in range(max_additions):
        best_segment: LineSegment | None = None
        best_residual = 1.0e18
        best_vanishing = vanishing
        for segment in pool:
            trial_members = members + [segment]
            trial_vanishing = core.vanishing_point_from_lines(trial_members)
            if trial_vanishing is None:
                continue
            trial_vanishing = _homogeneous_vp(trial_vanishing)
            residual = segment_vp_residual(segment, trial_vanishing)
            if residual > accept_residual:
                continue
            # Existing inliers must stay reasonable under the new VP.
            existing_residuals = [
                segment_vp_residual(member, trial_vanishing) for member in members
            ]
            if max(existing_residuals) > accept_residual * 1.5:
                continue
            if float(np.median(np.asarray(existing_residuals, dtype=np.float64))) > (
                accept_residual * 0.75
            ):
                continue
            if residual < best_residual:
                best_residual = residual
                best_segment = segment
                best_vanishing = trial_vanishing
        if best_segment is None:
            break
        members.append(best_segment)
        vanishing = best_vanishing
        pool = [segment for segment in pool if segment is not best_segment]
        member_keys.add(
            (
                round(best_segment.x1, 1),
                round(best_segment.y1, 1),
                round(best_segment.x2, 1),
                round(best_segment.y2, 1),
            )
        )
    # Keep every accepted stroke — a tight inlier cull undoes noisy-but-useful
    # LSD depth rails that only fit after the provisional re-fit.
    refined_vanishing = core.vanishing_point_from_lines(members)
    if refined_vanishing is None:
        return cluster
    support = float(sum(core.segment_length(segment) for segment in members))
    return VpCluster(
        vanishing=_homogeneous_vp(refined_vanishing),
        segments=tuple(members),
        support_length=support,
    )


def _ransac_horizontal_cluster(
    segments: list[LineSegment],
    *,
    residual_px: float,
    iterations: int,
    rng: np.random.Generator,
    image_width: float,
    image_height: float,
    avoid_vanishings: list[np.ndarray],
) -> VpCluster | None:
    """RANSAC restricted to flatter segments — recovers hallway depth VPs."""
    horizontal = [
        segment
        for segment in segments
        if _segment_uprightness(segment) <= _HORIZONTAL_UPRIGHT_MAX
    ]
    if len(horizontal) < 2:
        return None
    # Slightly looser residual: short moulding / baseboard hits are noisier.
    cluster = _ransac_cluster(
        horizontal,
        residual_px=max(residual_px, 6.0),
        iterations=max(iterations, 200),
        rng=rng,
        avoid_vanishings=avoid_vanishings,
        image_width=image_width,
        image_height=image_height,
    )
    if cluster is None:
        return None
    return _expand_depth_cluster_by_refit(
        cluster,
        horizontal,
        upright_max=_HORIZONTAL_UPRIGHT_MAX,
        # Far depth VPs + LSD angle noise → large px residuals; ~180+ still rejects
        # wrong baseboards that do not belong on this VP.
        accept_residual=max(residual_px * 12.0, 90.0),
    )


def cluster_vanishing_points(
    segments: list[LineSegment],
    *,
    residual_px: float = _VP_RESIDUAL_PX,
    max_clusters: int = _MAX_CLUSTERS,
    iterations: int = _RANSAC_ITERATIONS,
    seed: int = 0,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[VpCluster]:
    """Greedily peel RANSAC VP clusters from a segment list (longest support first).

    After the main peel, a dedicated pass over near-horizontal segments tries to
    recover a depth VP that length-weighted upright RANSAC often starves out.
    """
    remaining = list(segments)
    clusters: list[VpCluster] = []
    rng = np.random.default_rng(seed)
    width = float(image_width or 1.0)
    height = float(image_height or 1.0)
    cx = 0.5 * width
    cy = 0.5 * height
    focal = max(width, 1.0)
    for _ in range(max_clusters):
        cluster = _ransac_cluster(
            remaining,
            residual_px=residual_px,
            iterations=iterations,
            rng=rng,
            avoid_vanishings=[item.vanishing for item in clusters],
            image_width=width,
            image_height=height,
        )
        if cluster is None:
            break
        clusters.append(cluster)
        used = {
            (round(s.x1, 2), round(s.y1, 2), round(s.x2, 2), round(s.y2, 2))
            for s in cluster.segments
        }
        remaining = [
            segment
            for segment in remaining
            if (
                round(segment.x1, 2),
                round(segment.y1, 2),
                round(segment.x2, 2),
                round(segment.y2, 2),
            )
            not in used
        ]
        if len(remaining) < 2:
            break

    # Depth / green edges are often shorter; force one horizontal VP candidate.
    # Prefer leftovers so we do not re-steal members of a stronger upright cluster;
    # fall back to all flat strokes if leftovers cannot form a cluster.
    horizontal_cluster = _ransac_horizontal_cluster(
        remaining,
        residual_px=residual_px,
        iterations=iterations,
        rng=rng,
        image_width=width,
        image_height=height,
        avoid_vanishings=[item.vanishing for item in clusters],
    )
    if horizontal_cluster is None:
        horizontal_cluster = _ransac_horizontal_cluster(
            segments,
            residual_px=residual_px,
            iterations=iterations,
            rng=rng,
            image_width=width,
            image_height=height,
            avoid_vanishings=[item.vanishing for item in clusters],
        )
    if horizontal_cluster is not None and _cluster_uprightness(horizontal_cluster) > 0.35:
        # Door-rail families are mid-diagonal; do not promote them as depth/green.
        horizontal_cluster = None
    if horizontal_cluster is not None and not any(
        _clusters_are_similar(
            horizontal_cluster, existing, cx=cx, cy=cy, focal=focal
        )
        for existing in clusters
    ):
        if len(clusters) < max_clusters:
            clusters.append(horizontal_cluster)
        else:
            # Prefer keeping a depth VP over yet another upright duplicate.
            weakest_index = min(
                range(len(clusters)),
                key=lambda index: (
                    # Drop low-support uprights first; never drop a flat cluster.
                    0 if _cluster_uprightness(clusters[index]) > 0.6 else 1,
                    clusters[index].support_length,
                ),
            )
            if (
                _cluster_uprightness(clusters[weakest_index]) > 0.6
                and horizontal_cluster.support_length
                >= 0.15 * clusters[weakest_index].support_length
            ):
                clusters[weakest_index] = horizontal_cluster

    clusters.sort(key=lambda item: item.support_length, reverse=True)
    return clusters


def _segment_uprightness(segment: LineSegment) -> float:
    """1.0 = perfectly vertical in the image; 0.0 = horizontal."""
    length = core.segment_length(segment)
    if length < 1.0e-9:
        return 0.0
    return abs(segment.y2 - segment.y1) / length


def _cluster_uprightness(cluster: VpCluster) -> float:
    """Length-weighted uprightness of supporting segments."""
    total = 0.0
    weight = 0.0
    for segment in cluster.segments:
        length = core.segment_length(segment)
        total += length * _segment_uprightness(segment)
        weight += length
    return total / weight if weight > 1.0e-9 else 0.0


def _segment_midpoint(segment: LineSegment) -> np.ndarray:
    return np.array(
        [0.5 * (segment.x1 + segment.x2), 0.5 * (segment.y1 + segment.y2)],
        dtype=np.float64,
    )


def segment_pair_angle_radians(
    first: LineSegment,
    second: LineSegment,
    vanishing: np.ndarray,
    *,
    image_width: int = 1,
    image_height: int = 1,
) -> float:
    """Angular separation of two segments as seen from their shared VP.

    Finite VP: angle between rays VP→midpoint.
    Infinite VP: equivalent angle from lateral midpoint separation (so parallel
    uprights on opposite sides of the frame still count as well spread).
    """
    midpoint_a = _segment_midpoint(first)
    midpoint_b = _segment_midpoint(second)
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        direction = vanishing_h[:2]
        length = float(np.linalg.norm(direction))
        if length < 1.0e-12:
            direction = np.array([0.0, 1.0], dtype=np.float64)
        else:
            direction = direction / length
        lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
        separation = abs(float(np.dot(midpoint_a - midpoint_b, lateral)))
        # Map lateral gap to an angle with the plate size as a depth proxy.
        depth = max(float(np.hypot(image_width, image_height)) * 0.5, 1.0)
        return float(np.arctan2(separation, depth))

    vp_xy = vanishing_h[:2]
    ray_a = midpoint_a - vp_xy
    ray_b = midpoint_b - vp_xy
    norm_a = float(np.linalg.norm(ray_a))
    norm_b = float(np.linalg.norm(ray_b))
    if norm_a < 1.0e-9 or norm_b < 1.0e-9:
        return 0.0
    ray_a = ray_a / norm_a
    ray_b = ray_b / norm_b
    return float(np.arccos(float(np.clip(np.dot(ray_a, ray_b), -1.0, 1.0))))


def _midpoint_distance(first: LineSegment, second: LineSegment) -> float:
    delta = _segment_midpoint(first) - _segment_midpoint(second)
    return float(np.linalg.norm(delta))


def _adaptive_min_angle_radians(
    vanishing: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    base_degrees: float = _MIN_SEGMENT_ANGLE_DEGREES,
    far_degrees: float = _MIN_SEGMENT_ANGLE_FAR_DEGREES,
) -> float:
    """Smaller angular floor when the VP is far (uprights look nearly parallel)."""
    diagonal = max(float(np.hypot(image_width, image_height)), 1.0)
    vanishing_h = _homogeneous_vp(vanishing)
    if abs(float(vanishing_h[2])) < 1.0e-10:
        return float(np.radians(max(far_degrees, 0.5)))
    vp_xy = vanishing_h[:2]
    center = np.array([0.5 * image_width, 0.5 * image_height], dtype=np.float64)
    distance = float(np.linalg.norm(vp_xy - center))
    if distance >= 2.0 * diagonal:
        degrees = far_degrees
    elif distance >= diagonal:
        # Blend between base and far as the VP recedes.
        amount = (distance - diagonal) / max(diagonal, 1.0)
        degrees = _lerp(base_degrees, far_degrees, amount)
    else:
        degrees = base_degrees
    return float(np.radians(max(degrees, 0.5)))


def select_diverse_segments(
    segments: tuple[LineSegment, ...] | list[LineSegment],
    vanishing: np.ndarray,
    *,
    limit: int,
    image_width: int,
    image_height: int,
    min_angle_degrees: float | None = None,
    min_separation_frac: float = _MIN_SEGMENT_SEPARATION_FRAC,
) -> list[LineSegment]:
    """Pick up to ``limit`` long segments that are spread apart at the VP.

    Seeds with the longest stroke, then repeatedly adds the candidate that
    maximizes minimum angle-to-picked (subject to a midpoint separation floor).
    Near-overlapping long edges are skipped — they add little to the VP solve.
    """
    if limit <= 0 or not segments:
        return []

    ordered = sorted(segments, key=core.segment_length, reverse=True)
    if len(ordered) == 1 or limit == 1:
        return ordered[:1]

    diagonal = float(np.hypot(image_width, image_height))
    min_separation_px = max(12.0, min_separation_frac * diagonal)
    if min_angle_degrees is None:
        min_angle = _adaptive_min_angle_radians(
            vanishing, image_width, image_height
        )
    else:
        min_angle = float(np.radians(max(min_angle_degrees, 0.5)))

    picked: list[LineSegment] = [ordered[0]]
    remaining = list(ordered[1:])

    while len(picked) < limit and remaining:
        best_index: int | None = None
        best_score = -1.0
        for index, candidate in enumerate(remaining):
            # Spatial near-duplicates (same mullion / double-detected edge).
            if any(
                _midpoint_distance(candidate, chosen) < min_separation_px
                for chosen in picked
            ):
                continue
            min_angle_to_picked = min(
                segment_pair_angle_radians(
                    candidate,
                    chosen,
                    vanishing,
                    image_width=image_width,
                    image_height=image_height,
                )
                for chosen in picked
            )
            # Prefer angularly far + still reasonably long. Do not hard-reject on
            # angle: far finite VPs make parallel wall edges subtend ~0°, and those
            # are still useful when midpoints are far enough (already gated above).
            length_weight = np.log1p(core.segment_length(candidate))
            min_mid_to_picked = min(
                _midpoint_distance(candidate, chosen) for chosen in picked
            )
            score = float(min_angle_to_picked) * float(length_weight)
            score += 0.5 * (min_mid_to_picked / diagonal)
            # Soft preference: when angle is below the adaptive floor, require a
            # clearly larger midpoint gap so we still favor spread over twins.
            if min_angle_to_picked < min_angle:
                score *= 0.35
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        picked.append(remaining.pop(best_index))

    # Orientation solve needs ≥2 concurrent segments on an axis. If diversity
    # was too strict, backfill with the next-longest non-overlapping strokes.
    min_pair = min(2, limit, len(ordered))
    if len(picked) < min_pair:
        soft_separation = max(6.0, 0.4 * min_separation_px)
        for candidate in ordered:
            if len(picked) >= min_pair:
                break
            if any(
                candidate is chosen
                or (
                    abs(candidate.x1 - chosen.x1) < 0.5
                    and abs(candidate.y1 - chosen.y1) < 0.5
                    and abs(candidate.x2 - chosen.x2) < 0.5
                    and abs(candidate.y2 - chosen.y2) < 0.5
                )
                for chosen in picked
            ):
                continue
            if any(
                _midpoint_distance(candidate, chosen) < soft_separation
                for chosen in picked
            ):
                continue
            picked.append(candidate)

    return picked


def _focal_consistency_score(
    vanishing_points: dict[AxisId, np.ndarray],
    cx: float,
    cy: float,
    image_width: float,
) -> float:
    """Higher is better: more pairwise focals, similar values, plausible range."""
    estimates = core.focal_estimates_by_pair(vanishing_points, cx, cy)
    if not estimates:
        return -1.0e9
    values = np.array(list(estimates.values()), dtype=np.float64)
    mean = float(np.mean(values))
    if mean < _FOCAL_MIN_FRAC * image_width or mean > _FOCAL_MAX_FRAC * image_width:
        return -1.0e6
    relative_std = float(np.std(values) / max(mean, 1.0))
    return float(len(values)) * 10.0 + mean / image_width - 5.0 * relative_std


def _orientation_filtered_segments(
    segments: tuple[LineSegment, ...] | list[LineSegment],
    *,
    role: str,
) -> list[LineSegment]:
    """Keep members whose image orientation matches the axis role.

    ``depth`` (green/Y): flat floor / baseboard / moulding strokes.
    ``across`` (red/X): keep cluster members — X edges may look nearly vertical.
    ``vertical`` (blue/Z): upright strokes.
    """
    if role == "vertical":
        filtered = [
            segment
            for segment in segments
            if _segment_uprightness(segment) >= _VERTICAL_AXIS_UPRIGHT_MIN
        ]
        if len(filtered) >= 2:
            return filtered
        return sorted(segments, key=_segment_uprightness, reverse=True)

    if role == "across":
        # Trust the across VP cluster; residual consensus drops true outliers.
        return list(segments)

    max_upright = _DEPTH_AXIS_UPRIGHT_MAX
    filtered = [
        segment
        for segment in segments
        if _segment_uprightness(segment) <= max_upright
    ]
    if len(filtered) >= 2:
        return filtered
    soft = [
        segment
        for segment in segments
        if _segment_uprightness(segment) <= 0.55
    ]
    if len(soft) >= 2:
        return soft
    return sorted(segments, key=_segment_uprightness)[: max(2, len(segments))]


def _add_ray_extensions(
    picked: list[LineSegment],
    consensus: list[LineSegment],
    vanishing: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    max_extra: int = 2,
    max_angle_degrees: float = 2.5,
    min_mid_frac: float = 0.08,
) -> list[LineSegment]:
    """Append consensus strokes that continue a picked ray toward the VP.

    Hallway across edges often split into stacked uprights that subtend ~0° at a
    finite VP. Diversity may keep only the longest; the shorter continuation is
    still a strong constraint once midpoints are far apart.
    """
    if max_extra <= 0 or not picked or not consensus:
        return picked
    diagonal = float(np.hypot(image_width, image_height))
    min_mid = max(80.0, min_mid_frac * diagonal)
    max_angle = float(np.radians(max_angle_degrees))
    result = list(picked)
    result_keys = {
        (round(segment.x1, 1), round(segment.y1, 1), round(segment.x2, 1), round(segment.y2, 1))
        for segment in result
    }
    added = 0
    for chosen in picked:
        if added >= max_extra:
            break
        best: LineSegment | None = None
        best_mid = 0.0
        for candidate in consensus:
            key = (
                round(candidate.x1, 1),
                round(candidate.y1, 1),
                round(candidate.x2, 1),
                round(candidate.y2, 1),
            )
            if key in result_keys:
                continue
            angle = segment_pair_angle_radians(
                candidate,
                chosen,
                vanishing,
                image_width=image_width,
                image_height=image_height,
            )
            mid = _midpoint_distance(candidate, chosen)
            if angle > max_angle or mid < min_mid:
                continue
            if mid > best_mid:
                best_mid = mid
                best = candidate
        if best is not None:
            result.append(best)
            result_keys.add(
                (
                    round(best.x1, 1),
                    round(best.y1, 1),
                    round(best.x2, 1),
                    round(best.y2, 1),
                )
            )
            added += 1
    return result


def _bundle_segments_for_axis(
    cluster: VpCluster,
    *,
    role: str,
    image_width: int,
    image_height: int,
    limit: int,
) -> list[LineSegment]:
    """Orientation-filter a cluster, refine its VP, drop outliers, pick diverse strokes."""
    filtered = _orientation_filtered_segments(cluster.segments, role=role)
    vanishing = cluster.vanishing
    if len(filtered) >= 2:
        refined = core.vanishing_point_from_lines(filtered)
        if refined is not None:
            vanishing = _homogeneous_vp(refined)
    # Drop members that disagree with the refined VP.
    # Depth: adaptive vs the core inliers so a near-miss (bad baseboard) drops
    # once better rails have locked the VP.
    if role == "depth" and filtered:
        # Far depth VPs amplify tiny LSD angle noise into large px residuals.
        # Keep mid-hall rails (~60–80px) while dropping wrong baseboards (~150px+).
        residual_thr = 85.0
    elif role == "across":
        residual_thr = 6.0
    else:
        residual_thr = 8.0
    consensus = [
        segment
        for segment in filtered
        if segment_vp_residual(segment, vanishing) <= residual_thr
    ]
    if len(consensus) < 2:
        consensus = filtered
    if role == "depth" and len(consensus) <= _MAX_SEGMENTS_DEPTH:
        return sorted(consensus, key=core.segment_length, reverse=True)[
            :_MAX_SEGMENTS_DEPTH
        ]
    pick_limit = limit
    sep_frac = _MIN_SEGMENT_SEPARATION_FRAC
    if role == "across":
        pick_limit = max(limit, _MAX_SEGMENTS_ACROSS)
        # Wall edges toward an across VP are often near-parallel uprights —
        # allow closer midpoints so far-left / far-right strokes both survive.
        sep_frac = 0.035
    picked = select_diverse_segments(
        consensus,
        vanishing,
        limit=pick_limit,
        image_width=image_width,
        image_height=image_height,
        min_separation_frac=sep_frac,
    )
    if role == "across":
        picked = _add_ray_extensions(
            picked,
            consensus,
            vanishing,
            image_width=image_width,
            image_height=image_height,
            max_extra=2,
        )
    return picked


def _label_horizontal_clusters(
    first: VpCluster,
    second: VpCluster,
    *,
    image_width: float,
) -> tuple[VpCluster, VpCluster]:
    """Return (red/X, green/Y) for two non-vertical clusters.

    When the vanishing points are clearly separated in x, color follows:
    green = left VP, red = right VP. Only when both VPs sit near the same
    meridian does flatter→green break the tie.
    """
    ax, _ay = _vp_image_xy(first.vanishing)
    bx, _by = _vp_image_xy(second.vanishing)
    if abs(ax - bx) >= 0.12 * max(float(image_width), 1.0):
        ordered = sorted([first, second], key=_horizontal_vp_sort_key)
        return ordered[1], ordered[0]  # red=right, green=left
    upright_a = _cluster_uprightness(first)
    upright_b = _cluster_uprightness(second)
    if abs(upright_a - upright_b) >= 0.12:
        if upright_a < upright_b:
            return second, first  # red=steeper, green=flatter
        return first, second
    ordered = sorted([first, second], key=_horizontal_vp_sort_key)
    return ordered[1], ordered[0]  # red=right, green=left


def _horizontal_bundle_role(cluster: VpCluster) -> str:
    """Pick orientation filter from the cluster itself, not from its color.

    Green is not always the flat family (depends which side depth sits on);
    use the flat ``depth`` gate only when the strokes are actually flat.
    """
    if _cluster_uprightness(cluster) <= 0.45:
        return "depth"
    return "across"


def assign_axes_three_point(
    clusters: list[VpCluster],
    image_width: int,
    image_height: int,
    *,
    max_segments_per_axis: int = _MAX_SEGMENTS_PER_AXIS,
) -> dict[AxisId, list[LineSegment]]:
    """Pick three clusters and label them x / y / z for 3-point mode.

    ``y`` (UI Z / Blender up) is the most upright bundle. Of the remaining two,
    ``z`` (UI Y / green) is the left-hand vanishing point and ``x`` (UI X / red)
    is the right-hand one when the VPs are separable in x. Orientation filters
    follow each cluster's uprightness (flat → depth gate, steeper → across).
    """
    if len(clusters) < 3:
        raise ValueError(
            f"Need at least 3 vanishing-point clusters, found {len(clusters)}"
        )

    cx = 0.5 * float(image_width)
    cy = 0.5 * float(image_height)
    width = float(image_width)
    height = float(image_height)
    pool = clusters[: min(6, len(clusters))]
    best_score = -1.0e18
    best_bundles: dict[AxisId, list[LineSegment]] | None = None

    for triad in combinations(range(len(pool)), 3):
        chosen = [pool[index] for index in triad]
        upright_scores = [_cluster_uprightness(item) for item in chosen]
        upright_index = _pick_vertical_index(chosen, width, height)
        vertical = chosen[upright_index]
        horizontals = [chosen[i] for i in range(3) if i != upright_index]
        across, depth = _label_horizontal_clusters(
            horizontals[0], horizontals[1], image_width=width
        )
        separation = _triad_separation_radians(
            [item.vanishing for item in chosen],
            cx,
            cy,
            width,
        )
        upright_spread = float(max(upright_scores) - min(upright_scores))

        vanishing_points: dict[AxisId, np.ndarray] = {
            "x": across.vanishing,
            "z": depth.vanishing,
            "y": vertical.vanishing,
        }
        score = _focal_consistency_score(
            vanishing_points, cx, cy, width
        )
        # Soften empty-focal cases: near-2VP can still be the right triad.
        if score <= -1.0e8:
            score = -50.0
        score += _SEPARATION_WEIGHT * separation
        score += _UPRIGHT_SPREAD_WEIGHT * upright_spread
        # Prefer a clearly flatter + steeper horizontal pair.
        score += 20.0 * abs(
            _cluster_uprightness(across) - _cluster_uprightness(depth)
        )
        # Keep valid near-2VP solves (one X/Y VP on the plate) competitive.
        inside_horizontal = sum(
            1
            for item in (across, depth)
            if _vp_inside_image(item.vanishing, width, height, margin_frac=0.02)
        )
        if inside_horizontal:
            score += _INSIDE_HORIZONTAL_VP_BONUS * inside_horizontal
        # Prefer triads with stronger total edge support as a tie-breaker.
        score += 0.001 * (
            across.support_length + depth.support_length + vertical.support_length
        )
        # Soft bonus when green is left of red (requested color convention).
        ax, _ = _vp_image_xy(across.vanishing)
        zx, _ = _vp_image_xy(depth.vanishing)
        if zx < ax:
            score += 8.0
        if score > best_score:
            best_score = score
            red_role = _horizontal_bundle_role(across)
            green_role = _horizontal_bundle_role(depth)
            best_bundles = {
                "x": _bundle_segments_for_axis(
                    across,
                    role=red_role,
                    image_width=image_width,
                    image_height=image_height,
                    limit=max_segments_per_axis,
                ),
                "z": _bundle_segments_for_axis(
                    depth,
                    role=green_role,
                    image_width=image_width,
                    image_height=image_height,
                    limit=max_segments_per_axis,
                ),
                "y": _bundle_segments_for_axis(
                    vertical,
                    role="vertical",
                    image_width=image_width,
                    image_height=image_height,
                    limit=max_segments_per_axis,
                ),
            }

    if best_bundles is None:
        raise ValueError("Could not assign three orthogonal vanishing-point bundles")
    return best_bundles


def detect_vp_line_bundles(
    gray: np.ndarray,
    *,
    seed: int = 0,
    sensitivity: float = _DEFAULT_SENSITIVITY,
) -> DetectVpLinesOutcome:
    """Full detect pipeline on a grayscale plate (source-image pixels)."""
    candidates = detect_line_segments(gray, sensitivity=sensitivity)
    if len(candidates) < 6:
        raise ValueError(
            f"Too few line segments for VP detection ({len(candidates)}); "
            "need a still with clear straight edges"
        )
    height, width = gray.shape
    clusters = cluster_vanishing_points(
        candidates,
        seed=seed,
        image_width=width,
        image_height=height,
    )
    if len(clusters) < 3:
        raise ValueError(
            f"Found only {len(clusters)} vanishing-point cluster(s); "
            "need three distinct directions (3-point perspective)"
        )
    bundles = assign_axes_three_point(clusters, width, height)
    counts = {axis: len(segments) for axis, segments in bundles.items()}
    result = DetectVpLinesResult(
        candidates=len(candidates),
        clusters=len(clusters),
        counts=counts,
    )
    return DetectVpLinesOutcome(
        bundles=bundles,
        candidates=tuple(candidates),
        result=result,
    )


def apply_vp_line_bundles(
    settings,
    bundles: dict[AxisId, list[LineSegment]],
) -> DetectVpLinesResult:
    """Replace session VP lines with detected bundles (clears existing strokes)."""
    settings.lines.clear()
    settings.selected_line_index = -1
    counts = {"x": 0, "y": 0, "z": 0}
    for axis in ("x", "y", "z"):
        for segment in bundles.get(axis, []):
            line = settings.lines.add()
            line.item_id = f"blender-line-{uuid4().hex}"
            line.axis = axis
            line.x1 = float(segment.x1)
            line.y1 = float(segment.y1)
            line.x2 = float(segment.x2)
            line.y2 = float(segment.y2)
            counts[axis] += 1
    return DetectVpLinesResult(candidates=0, clusters=3, counts=counts)


def write_vp_detect_debug_plate(
    context,
    segments: list[LineSegment] | tuple[LineSegment, ...],
) -> object:
    """Build/update the black debug plate and optionally show it as the camera BG."""
    import bpy

    from ..scene import distortion
    from .. import properties, scene

    settings = properties.active_session(context)
    if settings is None or settings.image is None:
        raise ValueError("Load a reference image on the active match first")
    width = int(settings.image_width or settings.image.size[0])
    height = int(settings.image_height or settings.image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Reference image has no readable pixel dimensions")

    rgba = render_debug_rgba(width, height, segments)
    plate_key = distortion._plate_key(settings)
    image_path = getattr(settings, "image_path", "") or ""
    if image_path:
        resolved_path = str(
            Path(default_vp_detect_debug_path(image_path, plate_key))
            .expanduser()
            .resolve()
        )
    else:
        resolved_path = ""

    image_name = f"{plate_key}.pm-vp-edges"
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
    if hasattr(output_image, "use_view_as_render"):
        output_image.use_view_as_render = False
    distortion._write_image_pixels(output_image, rgba)
    if resolved_path:
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

    output_image.use_fake_user = True
    settings.vp_detect_debug_image = output_image
    settings.vp_detect_debug_path = resolved_path
    if settings.view_vp_detect_debug:
        scene.refresh_background_projection(context)
    return output_image


def set_vp_detect_debug_view(context, enabled: bool) -> None:
    """Show or hide the LSD debug plate on the active match camera."""
    from .. import properties, scene

    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Activate a match camera first")
    if enabled and settings.vp_detect_debug_image is None:
        raise ValueError("No auto-detected edges yet")
    settings.view_vp_detect_debug = bool(enabled)
    scene.refresh_background_projection(context)


def detect_edges_for_debug(
    gray: np.ndarray,
    *,
    sensitivity: float = _DEFAULT_SENSITIVITY,
) -> tuple[tuple[LineSegment, ...], np.ndarray]:
    """LSD + length filter only; returns candidates and a black/white debug plate."""
    candidates = detect_line_segments(gray, sensitivity=sensitivity)
    if not candidates:
        raise ValueError(
            "No edges found — need a still with clear straight lines"
        )
    height, width = gray.shape
    rgba = render_debug_rgba(width, height, candidates)
    return tuple(candidates), rgba


def install_debug_rgba_plate(settings, debug_rgba: np.ndarray):
    """Write a pre-rendered debug plate into the session (main thread)."""
    import bpy

    from ..scene import distortion

    height, width = debug_rgba.shape[:2]
    plate_key = distortion._plate_key(settings)
    image_name = f"{plate_key}.pm-vp-edges"
    existing = bpy.data.images.get(image_name)
    if existing is not None and tuple(existing.size) != (width, height):
        bpy.data.images.remove(existing)
        existing = None
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
    if hasattr(output_image, "use_view_as_render"):
        output_image.use_view_as_render = False
    distortion._write_image_pixels(output_image, debug_rgba)
    image_path = getattr(settings, "image_path", "") or ""
    if image_path:
        resolved_path = str(
            Path(default_vp_detect_debug_path(image_path, plate_key))
            .expanduser()
            .resolve()
        )
        output_image.filepath_raw = resolved_path
        output_image.file_format = "PNG"
        try:
            output_image.save()
        except Exception:
            pass
        try:
            output_image.pack()
        except RuntimeError:
            pass
        settings.vp_detect_debug_path = resolved_path
    output_image.use_fake_user = True
    settings.vp_detect_debug_image = output_image
    return output_image


def load_detection_gray(settings) -> np.ndarray:
    """Load the active match still as grayscale uint8 (main-thread safe)."""
    from . import apriltags as apriltag_detect

    cv2 = _import_cv2()
    return apriltag_detect._load_detection_gray(cv2, settings)


def find_and_apply_vp_lines(context) -> DetectVpLinesResult:
    """Detect VP lines on the active match still and write them into the session.

    Prefer the modal operator for interactive use — this stays available for
    scripts / tests and runs synchronously.
    """
    from .. import properties

    settings = properties.active_session(context)
    if settings is None or settings.image is None:
        raise ValueError("Load a reference image on the active match first")
    if str(settings.vp_mode) != "3":
        raise ValueError("Detect VP Lines requires 3-point perspective mode")

    gray = load_detection_gray(settings)
    sensitivity = float(getattr(settings, "vp_detect_sensitivity", _DEFAULT_SENSITIVITY))
    outcome = detect_vp_line_bundles(gray, sensitivity=sensitivity)
    apply_vp_line_bundles(settings, outcome.bundles)
    write_vp_detect_debug_plate(context, outcome.candidates)
    settings.vp_detect_sensitivity_baked = sensitivity
    properties.tag_viewport_redraw(context)
    return DetectVpLinesResult(
        candidates=outcome.result.candidates,
        clusters=outcome.result.clusters,
        counts={
            "x": len(outcome.bundles["x"]),
            "y": len(outcome.bundles["y"]),
            "z": len(outcome.bundles["z"]),
        },
    )
