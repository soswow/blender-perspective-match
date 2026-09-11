"""Shared-plane landmark buckets (axis-aligned or free)."""

from __future__ import annotations

import numpy as np

from .constants import (
    LINE_PLANE_MIN_SINE,
    PLANE_AXIS_ALIGNED_MIN,
    PLANE_AXIS_INDEX,
    PLANE_FREE_MIN,
    PLANE_GROUP_LIMIT,
    PLANE_HARD_SLACK,
    PLANE_RESIDUAL_PX,
)
from .projection import camera_ray_private
from .lines import (
    _finite_segment_from_line_observations,
    _fit_line_fixed_direction,
)
from .types import SimilarityTransform, SyncLineObservation, SyncMatchInput, SyncObservation

PLANE_AXES = frozenset(PLANE_AXIS_INDEX) | {"FREE"}


def normalize_plane_groups(
    groups: list[tuple[str, str, int]] | None,
) -> list[tuple[str, str, int]]:
    """Keep unique landmark ids with a valid axis and bucket 1..10."""
    seen: set[str] = set()
    normalized: list[tuple[str, str, int]] = []
    for item in groups or ():
        if len(item) != 3:
            continue
        landmark_id, axis, group = item
        landmark_id = str(landmark_id or "")
        axis = str(axis or "").upper()
        if not landmark_id or landmark_id in seen or axis not in PLANE_AXES:
            continue
        try:
            bucket = int(group)
        except (TypeError, ValueError):
            continue
        if bucket < 1 or bucket > PLANE_GROUP_LIMIT:
            continue
        seen.add(landmark_id)
        normalized.append((landmark_id, axis, bucket))
    return normalized


def grouped_plane_members(
    groups: list[tuple[str, str, int]] | None,
) -> dict[tuple[str, int], list[str]]:
    """Map ``(axis, bucket)`` to landmark ids in first-seen order."""
    buckets: dict[tuple[str, int], list[str]] = {}
    for landmark_id, axis, group in normalize_plane_groups(groups):
        buckets.setdefault((axis, group), []).append(landmark_id)
    return buckets


def fit_free_plane(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(centroid, unit normal)`` or None when points are degenerate."""
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] < 3:
        return None
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    try:
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular.size < 2 or float(singular[0]) < 1.0e-12:
        return None
    if float(singular[1]) < 1.0e-8 * max(float(singular[0]), 1.0e-12):
        return None
    normal = vt[-1].copy()
    length = float(np.linalg.norm(normal))
    if length < 1.0e-12:
        return None
    normal /= length
    if float(normal[2]) < 0.0:
        normal = -normal
    return centroid, normal


def _plane_spring(plane_slack: float) -> float:
    slack = max(float(plane_slack), 0.0)
    if slack <= 1.0e-12:
        slack = PLANE_HARD_SLACK
    return PLANE_RESIDUAL_PX / slack


def _member_location(
    landmark_id: str,
    landmarks: dict[str, np.ndarray],
    line_points: dict[str, np.ndarray],
) -> tuple[np.ndarray, str] | None:
    point = line_points.get(landmark_id)
    if point is not None:
        return np.asarray(point, dtype=np.float64).reshape(3), "line"
    point = landmarks.get(landmark_id)
    if point is not None:
        return np.asarray(point, dtype=np.float64).reshape(3), "point"
    return None


def _member_offset(
    landmark_id: str,
    kind: str,
    landmark_offset: dict[str, int],
    line_offset: dict[str, int],
) -> int | None:
    if kind == "line":
        return line_offset.get(landmark_id)
    return landmark_offset.get(landmark_id)


def append_plane_residuals(
    residuals: list[float],
    jacobian_rows: list[np.ndarray],
    *,
    column_count: int,
    landmarks: dict[str, np.ndarray],
    line_points: dict[str, np.ndarray],
    landmark_offset: dict[str, int],
    line_offset: dict[str, int],
    plane_groups: list[tuple[str, str, int]] | None,
    plane_slack: float,
    ground_landmark_ids: list[str] | None = None,
) -> None:
    """Soft coplanarity springs for reconstructed members of each bucket."""
    ground_ids = set(ground_landmark_ids or ())
    spring = _plane_spring(plane_slack)
    for (axis, _group), members in grouped_plane_members(plane_groups).items():
        located: list[tuple[str, np.ndarray, str]] = []
        for landmark_id in members:
            found = _member_location(landmark_id, landmarks, line_points)
            if found is None:
                continue
            located.append((landmark_id, found[0], found[1]))
        if axis == "FREE":
            _append_free_plane_residuals(
                residuals,
                jacobian_rows,
                located,
                column_count=column_count,
                landmark_offset=landmark_offset,
                line_offset=line_offset,
                spring=spring,
            )
            continue
        _append_axis_plane_residuals(
            residuals,
            jacobian_rows,
            located,
            axis=axis,
            column_count=column_count,
            landmark_offset=landmark_offset,
            line_offset=line_offset,
            spring=spring,
            ground_ids=ground_ids,
        )


def _append_axis_plane_residuals(
    residuals: list[float],
    jacobian_rows: list[np.ndarray],
    located: list[tuple[str, np.ndarray, str]],
    *,
    axis: str,
    column_count: int,
    landmark_offset: dict[str, int],
    line_offset: dict[str, int],
    spring: float,
    ground_ids: set[str],
) -> None:
    axis_index = PLANE_AXIS_INDEX[axis]
    if len(located) < PLANE_AXIS_ALIGNED_MIN:
        return
    frozen: list[float] = []
    free_items: list[tuple[np.ndarray, int]] = []
    for landmark_id, point, kind in located:
        if axis == "Z" and landmark_id in ground_ids:
            frozen.append(0.0)
            continue
        start = _member_offset(
            landmark_id, kind, landmark_offset, line_offset
        )
        if start is None:
            frozen.append(float(point[axis_index]))
        else:
            free_items.append((point, start))
    if not free_items:
        return
    if frozen:
        reference = float(np.mean(frozen))
        for point, start in free_items:
            residuals.append(spring * (float(point[axis_index]) - reference))
            row = np.zeros(column_count, dtype=np.float64)
            row[start + axis_index] = spring
            jacobian_rows.append(row)
        return
    if len(free_items) < PLANE_AXIS_ALIGNED_MIN:
        return
    _ref_point, ref_start = free_items[0]
    ref_coord = float(_ref_point[axis_index])
    for point, start in free_items[1:]:
        residuals.append(spring * (float(point[axis_index]) - ref_coord))
        row = np.zeros(column_count, dtype=np.float64)
        row[start + axis_index] = spring
        row[ref_start + axis_index] = -spring
        jacobian_rows.append(row)


def _append_free_plane_residuals(
    residuals: list[float],
    jacobian_rows: list[np.ndarray],
    located: list[tuple[str, np.ndarray, str]],
    *,
    column_count: int,
    landmark_offset: dict[str, int],
    line_offset: dict[str, int],
    spring: float,
) -> None:
    if len(located) < PLANE_FREE_MIN:
        return
    fitted = fit_free_plane(np.stack([point for _id, point, _kind in located]))
    if fitted is None:
        return
    centroid, normal = fitted
    for landmark_id, point, kind in located:
        distance = float(np.dot(normal, point - centroid))
        residuals.append(spring * distance)
        row = np.zeros(column_count, dtype=np.float64)
        start = _member_offset(
            landmark_id, kind, landmark_offset, line_offset
        )
        if start is not None:
            row[start : start + 3] = spring * normal
        jacobian_rows.append(row)


def apply_plane_seed(
    landmarks: dict[str, np.ndarray],
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    plane_groups: list[tuple[str, str, int]] | None,
    *,
    known_ids: set[str] | None = None,
    known_line_ids: set[str] | None = None,
    ground_ids: set[str] | None = None,
) -> None:
    """Project free members onto each bucket plane so BA springs start small."""
    pinned = set(known_ids or ()) | set(known_line_ids or ())
    ground_ids = set(ground_ids or ())
    line_points = {
        landmark_id: 0.5 * (segment[0] + segment[1])
        for landmark_id, segment in line_segments.items()
    }
    for (axis, _group), members in grouped_plane_members(plane_groups).items():
        located: list[tuple[str, np.ndarray, str]] = []
        for landmark_id in members:
            found = _member_location(landmark_id, landmarks, line_points)
            if found is None:
                continue
            located.append((landmark_id, found[0], found[1]))
        if axis == "FREE":
            if len(located) < PLANE_FREE_MIN:
                continue
            fitted = fit_free_plane(np.stack([point for _id, point, _k in located]))
            if fitted is None:
                continue
            centroid, normal = fitted
            for landmark_id, point, kind in located:
                # Ground seeds may become fixed metric references in BA.
                # Preserve them here; nonzero Ground Slack can ease them in BA.
                if landmark_id in pinned or landmark_id in ground_ids:
                    continue
                delta = -normal * float(np.dot(normal, point - centroid))
                _shift_member(
                    landmark_id, kind, delta, landmarks, line_segments
                )
            continue
        if len(located) < PLANE_AXIS_ALIGNED_MIN:
            continue
        axis_index = PLANE_AXIS_INDEX[axis]
        if axis == "Z" and any(landmark_id in ground_ids for landmark_id, _p, _k in located):
            reference = 0.0
        else:
            pinned_coords = [
                float(point[axis_index])
                for landmark_id, point, _kind in located
                if landmark_id in pinned
            ]
            if pinned_coords:
                reference = float(np.mean(pinned_coords))
            else:
                reference = float(
                    np.median([point[axis_index] for _id, point, _k in located])
                )
        for landmark_id, point, kind in located:
            if landmark_id in pinned:
                continue
            if axis == "Z" and landmark_id in ground_ids:
                continue
            delta = np.zeros(3, dtype=np.float64)
            delta[axis_index] = reference - float(point[axis_index])
            _shift_member(landmark_id, kind, delta, landmarks, line_segments)


def seed_plane_points(
    landmarks: dict[str, np.ndarray],
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    plane_groups: list[tuple[str, str, int]] | None,
    observations_by_landmark: dict[str, list[SyncObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    *,
    plane_slack: float,
    location_match_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Seed missing one-view points on supported hard planes; exclude Fit Only."""
    if max(float(plane_slack), 0.0) > 1.0e-12:
        return {}
    line_points = {key: 0.5*(ends[0]+ends[1]) for key, ends in line_segments.items()}
    seeds = {}
    for (axis, _bucket), members in grouped_plane_members(plane_groups).items():
        located = [found[0] for key in members
                   if (found := _member_location(key, landmarks, line_points)) is not None]
        # The new point activates the bucket but does not define its seed plane.
        minimum = PLANE_FREE_MIN-1 if axis == "FREE" else PLANE_AXIS_ALIGNED_MIN-1
        if len(located) < minimum:
            continue
        if axis == "FREE":
            fitted = fit_free_plane(np.asarray(located))
            if fitted is None:
                continue
            origin, normal = fitted
        else:
            origin = np.mean(located, axis=0)
            normal = np.zeros(3, dtype=np.float64)
            normal[PLANE_AXIS_INDEX[axis]] = 1.0
        for key in members:
            if key in landmarks or key in line_points:
                continue
            items = [item for item in observations_by_landmark.get(key, ())
                     if item.match_id in similarities and item.match_id in matches
                     and (location_match_ids is None or item.match_id in location_match_ids)]
            # Leave failed multi-view triangulation to its existing diagnostics.
            if len(items) != 1:
                continue
            item = items[0]
            private_origin, private_ray = camera_ray_private(item.u, item.v, matches[item.match_id].calibration)
            pose = similarities[item.match_id]
            camera_origin = pose.transform_point(private_origin)
            ray = pose.rotation @ private_ray
            denominator = float(normal@ray)
            # Reuse the existing angular-separation budget for plane geometry.
            if abs(denominator) < LINE_PLANE_MIN_SINE:
                continue
            distance = float(normal@(origin-camera_origin))/denominator
            candidate = camera_origin + distance*ray
            if distance > 0.0 and np.isfinite(candidate).all():
                seeds[key] = candidate
    return seeds


def _shift_member(
    landmark_id: str,
    kind: str,
    delta: np.ndarray,
    landmarks: dict[str, np.ndarray],
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if kind == "line" and landmark_id in line_segments:
        point_a, point_b = line_segments[landmark_id]
        line_segments[landmark_id] = (point_a + delta, point_b + delta)
        landmarks[landmark_id] = 0.5 * (point_a + point_b) + delta
        return
    if landmark_id in landmarks:
        landmarks[landmark_id] = landmarks[landmark_id] + delta


def active_plane_group_count(
    groups: list[tuple[str, str, int]] | None,
    landmarks: dict[str, np.ndarray],
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> int:
    """How many buckets currently have enough reconstructed members."""
    line_points = {
        landmark_id: 0.5 * (segment[0] + segment[1])
        for landmark_id, segment in (line_segments or {}).items()
    }
    count = 0
    for (axis, _group), members in grouped_plane_members(groups).items():
        located = 0
        for landmark_id in members:
            if _member_location(landmark_id, landmarks, line_points) is None:
                continue
            located += 1
        minimum = PLANE_FREE_MIN if axis == "FREE" else PLANE_AXIS_ALIGNED_MIN
        if located >= minimum:
            count += 1
    return count


def plane_slack_excesses(
    landmarks: dict[str, np.ndarray],
    groups: list[tuple[str, str, int]] | None,
    plane_slack: float,
    *,
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ground_landmark_ids: list[str] | None = None,
    names: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Members farther from their fitted plane than ``plane_slack``."""
    slack = max(float(plane_slack), 0.0)
    if slack <= 1.0e-12:
        return []
    ground_ids = set(ground_landmark_ids or ())
    labels = names or {}
    line_points = {
        landmark_id: 0.5 * (segment[0] + segment[1])
        for landmark_id, segment in (line_segments or {}).items()
    }
    drifted: list[tuple[str, float]] = []
    for (axis, _group), members in grouped_plane_members(groups).items():
        located: list[tuple[str, np.ndarray]] = []
        for landmark_id in members:
            found = _member_location(landmark_id, landmarks, line_points)
            if found is None:
                continue
            located.append((landmark_id, found[0]))
        if axis == "FREE":
            if len(located) < PLANE_FREE_MIN:
                continue
            fitted = fit_free_plane(np.stack([point for _id, point in located]))
            if fitted is None:
                continue
            centroid, normal = fitted
            for landmark_id, point in located:
                distance = abs(float(np.dot(normal, point - centroid)))
                if distance > slack:
                    drifted.append(
                        (labels.get(landmark_id, landmark_id[:8]), distance)
                    )
            continue
        if len(located) < PLANE_AXIS_ALIGNED_MIN:
            continue
        axis_index = PLANE_AXIS_INDEX[axis]
        if axis == "Z" and any(landmark_id in ground_ids for landmark_id, _p in located):
            reference = 0.0
        else:
            reference = float(np.mean([point[axis_index] for _id, point in located]))
        for landmark_id, point in located:
            if axis == "Z" and landmark_id in ground_ids:
                continue
            distance = abs(float(point[axis_index]) - reference)
            if distance > slack:
                drifted.append(
                    (labels.get(landmark_id, landmark_id[:8]), distance)
                )
    drifted.sort(key=lambda item: -item[1])
    return drifted


def _direction_in_plane(direction: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = direction - normal * float(np.dot(direction, normal))
    length = float(np.linalg.norm(projected))
    if length < 1.0e-12:
        return None
    return projected / length


def enforce_plane_line_segments(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    landmarks: dict[str, np.ndarray],
    plane_groups: list[tuple[str, str, int]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    plane_slack: float,
    ground_landmark_ids: list[str] | None = None,
) -> None:
    """Project free line meshes into assigned planes when slack is a hard pin."""
    if max(float(plane_slack), 0.0) > 1.0e-12:
        return
    ground_ids = set(ground_landmark_ids or ())
    line_points = {
        landmark_id: 0.5 * (segment[0] + segment[1])
        for landmark_id, segment in line_segments.items()
    }
    for (axis, _group), members in grouped_plane_members(plane_groups).items():
        located: list[tuple[str, np.ndarray]] = []
        for landmark_id in members:
            found = _member_location(landmark_id, landmarks, line_points)
            if found is None:
                continue
            located.append((landmark_id, found[0]))
        line_ids = [
            landmark_id
            for landmark_id in members
            if landmark_id in line_segments and landmark_id not in known_lines
        ]
        if not line_ids:
            continue
        if axis == "FREE":
            if len(located) < PLANE_FREE_MIN:
                continue
            fitted = fit_free_plane(np.stack([point for _id, point in located]))
            if fitted is None:
                continue
            centroid, normal = fitted
        else:
            if len(located) < PLANE_AXIS_ALIGNED_MIN:
                continue
            axis_index = PLANE_AXIS_INDEX[axis]
            if axis == "Z" and any(
                landmark_id in ground_ids for landmark_id, _p in located
            ):
                reference = 0.0
            else:
                reference = float(
                    np.mean([point[axis_index] for _id, point in located])
                )
            centroid = np.zeros(3, dtype=np.float64)
            centroid[axis_index] = reference
            normal = np.zeros(3, dtype=np.float64)
            normal[axis_index] = 1.0
        for landmark_id in line_ids:
            point_a, point_b = line_segments[landmark_id]
            direction = point_b - point_a
            if float(np.linalg.norm(direction)) < 1.0e-9:
                continue
            unit = _direction_in_plane(direction, normal)
            if unit is None:
                continue
            midpoint = 0.5 * (point_a + point_b)
            midpoint = midpoint - normal * float(np.dot(midpoint - centroid, normal))
            items = line_observations_by_landmark.get(landmark_id, [])
            fitted_line = _fit_line_fixed_direction(
                unit, items, similarities, matches
            )
            if fitted_line is None:
                point, direction = midpoint, unit
            else:
                point, direction = fitted_line
                point = point - normal * float(np.dot(point - centroid, normal))
            segment = _finite_segment_from_line_observations(
                point, direction, items, similarities, matches
            )
            line_segments[landmark_id] = segment
            landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
