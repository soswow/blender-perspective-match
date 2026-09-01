"""Landmark mirror pairs across one shared-world plane."""

from __future__ import annotations

import re

import numpy as np

from .. import geometry as core
from .projection import (
    _intersect_planes_to_line,
    _plane_from_line_observation,
    camera_ray_private,
    triangulate_midpoint,
)
from .types import (
    SimilarityTransform,
    SyncLineObservation,
    SyncMatchInput,
    SyncObservation,
)


def _unit_normal(normal: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of ``normal``."""
    vector = np.asarray(normal, dtype=np.float64).reshape(3)
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def _normalize_plane(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point, unit_normal)`` copies of a plane."""
    return (
        np.asarray(plane_point, dtype=np.float64).reshape(3).copy(),
        _unit_normal(plane_normal),
    )


def _householder(normal: np.ndarray) -> np.ndarray:
    """Reflection matrix ``I - 2 n nᵀ`` for a unit normal."""
    unit = _unit_normal(normal)
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(unit, unit)


_LEFT_RIGHT_SUFFIX = re.compile(r"(left|right)$", re.IGNORECASE)


def _match_side_case(side: str, sample: str) -> str:
    """Return ``side`` with the capitalization of ``sample``."""
    if sample.isupper():
        return side.upper()
    if sample[:1].isupper():
        return side.capitalize()
    return side


def suggested_mirror_partner_name(name: str) -> str | None:
    """If ``name`` ends with left/right, return it with that token flipped."""
    text = str(name).strip()
    match = _LEFT_RIGHT_SUFFIX.search(text)
    if not match:
        return None
    token = match.group(1)
    swapped = "right" if token.casefold() == "left" else "left"
    return text[: match.start()] + _match_side_case(swapped, token)


def reflect_direction(
    direction: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Reflect a direction vector across a plane (Householder)."""
    reflected = _householder(plane_normal) @ np.asarray(direction, dtype=np.float64)
    return reflected / max(float(np.linalg.norm(reflected)), 1.0e-12)


def _align_direction(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``direction`` if it opposes ``reference``."""
    unit = np.asarray(direction, dtype=np.float64)
    if float(np.dot(unit, reference)) < 0.0:
        return -unit
    return unit


def _reflect_plane(
    plane: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Reflect a world plane ``n·x + d = 0`` across the mirror."""
    origin, normal = _normalize_plane(plane_point, plane_normal)
    householder = _householder(normal)
    plane_n = np.asarray(plane[:3], dtype=np.float64)
    reflected_n = householder @ plane_n
    offset = float(plane[3]) + 2.0 * float(np.dot(normal, origin)) * float(
        np.dot(plane_n, normal)
    )
    return np.array(
        (reflected_n[0], reflected_n[1], reflected_n[2], offset),
        dtype=np.float64,
    )


def reflect_point(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Reflect ``point`` across the plane through ``plane_point`` with ``plane_normal``."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    origin, normal = _normalize_plane(plane_point, plane_normal)
    return point - 2.0 * normal * float(np.dot(normal, point - origin))


def _dedupe_mirror_pairs(
    pairs: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Unique undirected pairs, dropping self-links."""
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for landmark_a, landmark_b in pairs or ():
        if not landmark_a or not landmark_b or landmark_a == landmark_b:
            continue
        key = tuple(sorted((landmark_a, landmark_b)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((key[0], key[1]))
    return unique


def _shared_ray(
    observation: SyncObservation,
    similarity: SimilarityTransform,
    calibration: core.Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a private-frame image ray into shared world coordinates."""
    origin_private, direction_private = camera_ray_private(
        observation.u,
        observation.v,
        calibration,
    )
    origin_shared = similarity.transform_point(origin_private)
    direction_shared = similarity.rotation @ direction_private
    direction_shared = direction_shared / max(
        float(np.linalg.norm(direction_shared)),
        1.0e-12,
    )
    return origin_shared, direction_shared


def _posed_observations(
    landmark_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    similarities: dict[str, SimilarityTransform],
) -> list[SyncObservation]:
    """Picks of ``landmark_id`` whose cameras currently have a pose."""
    return [
        observation
        for observation in observations_by_landmark.get(landmark_id, [])
        if observation.match_id in similarities
    ]


def _seed_pair_from_rays(
    observations_a: list[SyncObservation],
    observations_b: list[SyncObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Triangulate A from ray A and the reflection of ray B."""
    if not observations_a or not observations_b:
        return None
    observation_a = observations_a[0]
    observation_b = observations_b[0]
    match_a = matches.get(observation_a.match_id)
    match_b = matches.get(observation_b.match_id)
    similarity_a = similarities.get(observation_a.match_id)
    similarity_b = similarities.get(observation_b.match_id)
    if match_a is None or match_b is None:
        return None
    if similarity_a is None or similarity_b is None:
        return None
    origin_a, direction_a = _shared_ray(
        observation_a, similarity_a, match_a.calibration
    )
    origin_b, direction_b = _shared_ray(
        observation_b, similarity_b, match_b.calibration
    )
    householder = _householder(plane_normal)
    origin_reflected = reflect_point(origin_b, plane_point, plane_normal)
    direction_reflected = householder @ direction_b
    return triangulate_midpoint(
        [origin_a, origin_reflected],
        [direction_a, direction_reflected],
    )


def seed_mirror_landmarks(
    landmarks: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    pairs: list[tuple[str, str]] | None,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> None:
    """Fill missing mirror partners by reflection or two single-view rays."""
    origin, normal = _normalize_plane(plane_point, plane_normal)
    for landmark_a, landmark_b in _dedupe_mirror_pairs(pairs):
        has_a = landmark_a in landmarks
        has_b = landmark_b in landmarks
        if has_a and has_b:
            continue
        if has_a and not has_b:
            landmarks[landmark_b] = reflect_point(
                landmarks[landmark_a], origin, normal
            )
            continue
        if has_b and not has_a:
            landmarks[landmark_a] = reflect_point(
                landmarks[landmark_b], origin, normal
            )
            continue
        observations_a = _posed_observations(
            landmark_a, observations_by_landmark, similarities
        )
        observations_b = _posed_observations(
            landmark_b, observations_by_landmark, similarities
        )
        seeded = _seed_pair_from_rays(
            observations_a,
            observations_b,
            similarities,
            matches,
            origin,
            normal,
        )
        if seeded is None:
            continue
        landmarks[landmark_a] = seeded
        landmarks[landmark_b] = reflect_point(seeded, origin, normal)


def mirror_plane_offset(
    landmarks: dict[str, np.ndarray],
    pairs: list[tuple[str, str]] | None,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> float | None:
    """Mean signed offset of pair midpoints along the plane normal."""
    origin, normal = _normalize_plane(plane_point, plane_normal)
    offsets: list[float] = []
    for landmark_a, landmark_b in _dedupe_mirror_pairs(pairs):
        point_a = landmarks.get(landmark_a)
        point_b = landmarks.get(landmark_b)
        if point_a is None or point_b is None:
            continue
        midpoint = 0.5 * (point_a + point_b)
        offsets.append(float(np.dot(midpoint - origin, normal)))
    if not offsets:
        return None
    return float(np.mean(offsets))


def _posed_line_observations(
    landmark_id: str,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
) -> list[SyncLineObservation]:
    """Line strokes of ``landmark_id`` whose cameras currently have a pose."""
    return [
        observation
        for observation in line_observations_by_landmark.get(landmark_id, [])
        if observation.match_id in similarities
    ]


def _store_mirror_line(
    landmark_id: str,
    point: np.ndarray,
    direction: np.ndarray,
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    landmarks: dict[str, np.ndarray],
    observations: list[SyncLineObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
) -> None:
    from .lines import _finite_segment_from_line_observations

    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    if observations:
        segment = _finite_segment_from_line_observations(
            point, unit, observations, similarities, matches
        )
    else:
        segment = (point - unit, point + unit)
    line_segments[landmark_id] = segment
    landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])


def seed_mirror_line_segments(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    landmarks: dict[str, np.ndarray],
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    pairs: list[tuple[str, str]] | None,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Fill missing mirrored line partners by reflection or two single-view planes."""
    from .lines import _reconstruct_line_from_observations

    origin, normal = _normalize_plane(plane_point, plane_normal)
    known = known_lines or {}
    for landmark_a, landmark_b in _dedupe_mirror_pairs(pairs):
        has_a = landmark_a in line_segments
        has_b = landmark_b in line_segments
        if has_a and has_b:
            continue
        observations_a = _posed_line_observations(
            landmark_a, line_observations_by_landmark, similarities
        )
        observations_b = _posed_line_observations(
            landmark_b, line_observations_by_landmark, similarities
        )
        if has_a and landmark_b not in known:
            point_a, point_b = line_segments[landmark_a]
            direction = point_b - point_a
            if float(np.linalg.norm(direction)) < 1.0e-9:
                continue
            _store_mirror_line(
                landmark_b,
                reflect_point(0.5 * (point_a + point_b), origin, normal),
                reflect_direction(direction, normal),
                line_segments,
                landmarks,
                observations_b,
                similarities,
                matches,
            )
            continue
        if has_b and landmark_a not in known:
            point_a, point_b = line_segments[landmark_b]
            direction = point_b - point_a
            if float(np.linalg.norm(direction)) < 1.0e-9:
                continue
            _store_mirror_line(
                landmark_a,
                reflect_point(0.5 * (point_a + point_b), origin, normal),
                reflect_direction(direction, normal),
                line_segments,
                landmarks,
                observations_a,
                similarities,
                matches,
            )
            continue
        reconstructed_a = _reconstruct_line_from_observations(
            observations_a, similarities, matches
        )
        reconstructed_b = _reconstruct_line_from_observations(
            observations_b, similarities, matches
        )
        if reconstructed_a is not None and landmark_b not in known:
            point, direction = reconstructed_a
            _store_mirror_line(
                landmark_a,
                point,
                direction,
                line_segments,
                landmarks,
                observations_a,
                similarities,
                matches,
            )
            _store_mirror_line(
                landmark_b,
                reflect_point(point, origin, normal),
                reflect_direction(direction, normal),
                line_segments,
                landmarks,
                observations_b,
                similarities,
                matches,
            )
            continue
        if reconstructed_b is not None and landmark_a not in known:
            point, direction = reconstructed_b
            _store_mirror_line(
                landmark_b,
                point,
                direction,
                line_segments,
                landmarks,
                observations_b,
                similarities,
                matches,
            )
            _store_mirror_line(
                landmark_a,
                reflect_point(point, origin, normal),
                reflect_direction(direction, normal),
                line_segments,
                landmarks,
                observations_a,
                similarities,
                matches,
            )
            continue
        if not observations_a or not observations_b:
            continue
        if landmark_a in known or landmark_b in known:
            continue
        match_a = matches.get(observations_a[0].match_id)
        match_b = matches.get(observations_b[0].match_id)
        similarity_a = similarities.get(observations_a[0].match_id)
        similarity_b = similarities.get(observations_b[0].match_id)
        if match_a is None or match_b is None:
            continue
        if similarity_a is None or similarity_b is None:
            continue
        plane_a = _plane_from_line_observation(
            observations_a[0], match_a.calibration, similarity_a
        )
        plane_b = _plane_from_line_observation(
            observations_b[0], match_b.calibration, similarity_b
        )
        if plane_a is None or plane_b is None:
            continue
        seeded = _intersect_planes_to_line(
            plane_a, _reflect_plane(plane_b, origin, normal)
        )
        if seeded is None:
            continue
        point, direction = seeded
        _store_mirror_line(
            landmark_a,
            point,
            direction,
            line_segments,
            landmarks,
            observations_a,
            similarities,
            matches,
        )
        _store_mirror_line(
            landmark_b,
            reflect_point(point, origin, normal),
            reflect_direction(direction, normal),
            line_segments,
            landmarks,
            observations_b,
            similarities,
            matches,
        )


def enforce_mirror_line_segments(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    landmarks: dict[str, np.ndarray],
    pairs: list[tuple[str, str]] | None,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Snap free mirrored edges onto one reflected 3D line pair."""
    from .lines import _fit_line_fixed_direction

    origin, normal = _normalize_plane(plane_point, plane_normal)
    known = known_lines or {}
    for landmark_a, landmark_b in _dedupe_mirror_pairs(pairs):
        if landmark_a not in line_segments or landmark_b not in line_segments:
            continue
        point_a, end_a = line_segments[landmark_a]
        point_b, end_b = line_segments[landmark_b]
        direction_a = end_a - point_a
        direction_b = end_b - point_b
        if (
            float(np.linalg.norm(direction_a)) < 1.0e-9
            or float(np.linalg.norm(direction_b)) < 1.0e-9
        ):
            continue
        direction_a = direction_a / float(np.linalg.norm(direction_a))
        direction_b = direction_b / float(np.linalg.norm(direction_b))
        reflected_dir = _align_direction(
            reflect_direction(direction_a, normal), direction_b
        )
        consensus = reflected_dir + direction_b
        if float(np.linalg.norm(consensus)) < 1.0e-9:
            consensus = direction_b
        consensus = consensus / float(np.linalg.norm(consensus))
        mid_a = 0.5 * (point_a + end_a)
        mid_b = 0.5 * (point_b + end_b)
        reflected_mid = reflect_point(mid_a, origin, normal)
        for landmark_id, seed_point, seed_dir, skip_known in (
            (landmark_a, mid_a, _align_direction(reflect_direction(consensus, normal), direction_a), landmark_a in known),
            (landmark_b, 0.5 * (mid_b + reflected_mid), consensus, landmark_b in known),
        ):
            if skip_known:
                continue
            observations = _posed_line_observations(
                landmark_id, line_observations_by_landmark, similarities
            )
            fitted = None
            if len(observations) >= 2:
                fitted = _fit_line_fixed_direction(
                    seed_dir, observations, similarities, matches
                )
            if fitted is None:
                fitted = (seed_point, seed_dir)
            _store_mirror_line(
                landmark_id,
                fitted[0],
                fitted[1],
                line_segments,
                landmarks,
                observations,
                similarities,
                matches,
            )
