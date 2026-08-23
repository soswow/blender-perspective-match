"""Joint bundle adjustment, residuals, and leave-one-out Diagnose."""

from __future__ import annotations

import numpy as np

from .. import geometry as core
from .constants import (
    GROUND_Z_RESIDUAL_PX,
    OUTLIER_WEIGHT_FACTOR,
    RADIAL_WEIGHT_GAIN,
    SPATIAL_GRID_SIZE,
    SPATIAL_WEIGHT_CLIP,
)
from .projection import (
    _known_line_reprojection_errors,
    _line_observation_reprojection_errors,
    _log_rodrigues,
    _normalized_camera_ray,
    _project_shared_points,
    _rodrigues,
    _skew,
    camera_ray_private,
    project_private_point,
    refine_triangulated_point,
    triangulate_midpoint,
)
from .types import (
    SimilarityTransform,
    SyncLineObservation,
    SyncMatchInput,
    SyncObservation,
    _observation_scale,
)

def _image_radius_norm(
    u_coord: float, v_coord: float, calibration: core.Calibration
) -> float:
    """Distance from the principal point, 1 at the farthest image corner."""
    intrinsics = calibration.intrinsics
    cx_coord = float(intrinsics.cx)
    cy_coord = float(intrinsics.cy)
    width = float(intrinsics.image_width)
    height = float(intrinsics.image_height)
    max_radius = 0.0
    for corner_u, corner_v in (
        (0.0, 0.0),
        (width, 0.0),
        (0.0, height),
        (width, height),
    ):
        max_radius = max(
            max_radius,
            float(np.hypot(corner_u - cx_coord, corner_v - cy_coord)),
        )
    return float(
        np.hypot(float(u_coord) - cx_coord, float(v_coord) - cy_coord)
    ) / max(max_radius, 1.0)


def _image_grid_cell(
    u_coord: float,
    v_coord: float,
    calibration: core.Calibration,
    grid: int,
) -> tuple[int, int]:
    """Row/column in a uniform image grid, clipped to the plate."""
    width = max(float(calibration.intrinsics.image_width), 1.0)
    height = max(float(calibration.intrinsics.image_height), 1.0)
    column = min(grid - 1, max(0, int(float(u_coord) / width * grid)))
    row = min(grid - 1, max(0, int(float(v_coord) / height * grid)))
    return row, column


def _balance_observation_weights(
    observations: list[SyncObservation],
    matches: dict[str, SyncMatchInput],
    *,
    grid: int = SPATIAL_GRID_SIZE,
    clip: float = SPATIAL_WEIGHT_CLIP,
    radial_gain: float = RADIAL_WEIGHT_GAIN,
) -> list[SyncObservation]:
    """Reweight picks so occupied image-grid cells share influence per camera.

    Confidence ratios stay (mean spatial boost is 1). Isolated cells clip so
    one mismatched pick cannot dominate.
    """
    if grid < 1 or not observations:
        return observations
    by_match: dict[str, list[int]] = {}
    for index, observation in enumerate(observations):
        by_match.setdefault(observation.match_id, []).append(index)
    balanced = list(observations)
    clip = max(float(clip), 1.0)
    min_boost = 1.0 / clip
    for match_id, indices in by_match.items():
        match = matches.get(match_id)
        if match is None or len(indices) < 2:
            continue
        calibration = match.calibration
        counts: dict[tuple[int, int], int] = {}
        cells: list[tuple[int, int]] = []
        radii: list[float] = []
        for index in indices:
            observation = observations[index]
            cell = _image_grid_cell(
                observation.u, observation.v, calibration, grid
            )
            cells.append(cell)
            counts[cell] = counts.get(cell, 0) + 1
            radii.append(
                _image_radius_norm(observation.u, observation.v, calibration)
            )
        occupied = len(counts)
        count = len(indices)
        boosts: list[float] = []
        for cell, radius in zip(cells, radii):
            spatial = float(count) / (float(counts[cell]) * float(occupied))
            radial = 1.0 + float(radial_gain) * radius * radius
            boosts.append(min(max(spatial * radial, min_boost), clip))
        mean_boost = float(np.mean(boosts))
        if mean_boost <= 1.0e-12:
            continue
        for index, boost in zip(indices, boosts):
            observation = observations[index]
            balanced[index] = SyncObservation(
                match_id=observation.match_id,
                landmark_id=observation.landmark_id,
                u=observation.u,
                v=observation.v,
                on_ground=observation.on_ground,
                landmark_name=observation.landmark_name,
                weight=float(observation.weight) * (boost / mean_boost),
            )
    return balanced


def _pack_params(
    match_ids: list[str],
    landmark_ids: list[str],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
) -> np.ndarray:
    """Pack similarities (log s, R, t) then landmark positions."""
    values: list[float] = []
    for match_id in match_ids:
        similarity = similarities[match_id]
        values.append(float(np.log(max(similarity.scale, 1.0e-8))))
        values.extend(_log_rodrigues(similarity.rotation).tolist())
        values.extend(similarity.translation.tolist())
    for landmark_id in landmark_ids:
        values.extend(landmarks[landmark_id].tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_params(
    params: np.ndarray,
    match_ids: list[str],
    landmark_ids: list[str],
) -> tuple[dict[str, SimilarityTransform], dict[str, np.ndarray]]:
    similarities: dict[str, SimilarityTransform] = {}
    offset = 0
    for match_id in match_ids:
        similarities[match_id] = SimilarityTransform(
            scale=float(np.exp(params[offset])),
            rotation=_rodrigues(params[offset + 1 : offset + 4]),
            translation=params[offset + 4 : offset + 7].copy(),
        )
        offset += 7
    landmarks: dict[str, np.ndarray] = {}
    for landmark_id in landmark_ids:
        landmarks[landmark_id] = params[offset : offset + 3].copy()
        offset += 3
    return similarities, landmarks


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


def _triangulate_landmarks(
    landmark_ids: list[str],
    observations_by_landmark: dict[str, list[SyncObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    *,
    min_views: int = 2,
) -> dict[str, np.ndarray]:
    """Triangulate landmarks that have enough registered-match rays.

    Observations from matches that are not registered yet are skipped.
    ``min_views=2`` (default) refuses single-ray depth guesses so a later PnP
    cannot pretend to know metric structure from the anchor alone.
    """
    landmarks: dict[str, np.ndarray] = {}
    for landmark_id in landmark_ids:
        registered_observations = [
            observation
            for observation in observations_by_landmark[landmark_id]
            if observation.match_id in similarities
        ]
        if len(registered_observations) < min_views:
            continue
        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        weights: list[float] = []
        for observation in registered_observations:
            origin, direction = _shared_ray(
                observation,
                similarities[observation.match_id],
                matches[observation.match_id].calibration,
            )
            origins.append(origin)
            directions.append(direction)
            weights.append(float(observation.weight))
        point = triangulate_midpoint(origins, directions, weights)
        if point is None:
            continue
        refined = refine_triangulated_point(
            point,
            registered_observations,
            similarities,
            matches,
        )
        landmarks[landmark_id] = point if refined is None else refined
    return landmarks


def _residual_vector(
    params: np.ndarray,
    match_ids: list[str],
    landmark_ids: list[str],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    *,
    weighted: bool = False,
) -> np.ndarray:
    """Geometric reprojection residuals with free landmark positions.

    When ``weighted``, each residual is scaled by sqrt(observation.weight).
    """
    similarities, landmarks = _unpack_params(params, match_ids, landmark_ids)
    similarities[anchor_id] = SimilarityTransform()
    residuals: list[float] = []
    for observation in observations:
        similarity = similarities[observation.match_id]
        point_shared = landmarks[observation.landmark_id]
        point_private = similarity.inverse_point(point_shared)
        projected = project_private_point(
            point_private,
            matches[observation.match_id].calibration,
        )
        scale = _observation_scale(observation) if weighted else 1.0
        if projected is None:
            residuals.extend((scale * 1.0e3, scale * 1.0e3))
            continue
        residuals.append(scale * float(projected[0] - observation.u))
        residuals.append(scale * float(projected[1] - observation.v))
    return np.asarray(residuals, dtype=np.float64)


def _sampson_residuals(
    params: np.ndarray,
    match_ids: list[str],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    pairs_by_match: dict[str, list[tuple[SyncObservation, SyncObservation]]],
) -> np.ndarray:
    """Epipolar Sampson residuals for each non-anchor vs the anchor."""
    similarities = {}
    offset = 0
    for match_id in match_ids:
        rotation_vector = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        similarities[match_id] = SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues(rotation_vector),
            translation=translation.copy(),
        )
        offset += 6
    residuals: list[float] = []
    anchor = matches[anchor_id].calibration
    for match_id in match_ids:
        similarity = similarities[match_id]
        other = matches[match_id].calibration
        rotation_b_shared = other.rotation_w2c @ similarity.rotation.T
        center_b_shared = similarity.transform_point(other.camera_center)
        rotation_rel = rotation_b_shared @ anchor.rotation_w2c.T
        translation_rel = rotation_b_shared @ (anchor.camera_center - center_b_shared)
        essential = _skew(translation_rel) @ rotation_rel
        for anchor_obs, other_obs in pairs_by_match.get(match_id, ()):
            ray_a = _normalized_camera_ray(anchor_obs.u, anchor_obs.v, anchor)
            ray_b = _normalized_camera_ray(other_obs.u, other_obs.v, other)
            residuals.append(_sampson_distance(ray_a, ray_b, essential))
    return np.asarray(residuals, dtype=np.float64)


def _sampson_distance(
    ray_a: np.ndarray,
    ray_b: np.ndarray,
    essential: np.ndarray,
) -> float:
    """Signed Sampson error for one correspondence (normalized rays)."""
    # Use first two components as inhomogeneous coordinates.
    point_a = ray_a / max(abs(float(ray_a[2])), 1.0e-12)
    point_b = ray_b / max(abs(float(ray_b[2])), 1.0e-12)
    ex = essential @ point_a
    etx = essential.T @ point_b
    numerator = float(point_b @ essential @ point_a)
    denominator = ex[0] * ex[0] + ex[1] * ex[1] + etx[0] * etx[0] + etx[1] * etx[1]
    if denominator < 1.0e-16:
        return numerator * 1.0e3
    return numerator / np.sqrt(denominator)


def _numeric_jacobian_sampson(
    params: np.ndarray,
    match_ids: list[str],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    pairs_by_match: dict[str, list[tuple[SyncObservation, SyncObservation]]],
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    base = _sampson_residuals(params, match_ids, anchor_id, matches, pairs_by_match)
    jacobian = np.zeros((base.size, params.size), dtype=np.float64)
    for index in range(params.size):
        perturbed = params.copy()
        delta = step if abs(float(params[index])) < 1.0 else step * abs(float(params[index]))
        perturbed[index] += delta
        sample = _sampson_residuals(
            perturbed,
            match_ids,
            anchor_id,
            matches,
            pairs_by_match,
        )
        jacobian[:, index] = (sample - base) / delta
    return jacobian


def _numeric_jacobian(
    params: np.ndarray,
    match_ids: list[str],
    landmark_ids: list[str],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    base = _residual_vector(
        params,
        match_ids,
        landmark_ids,
        anchor_id,
        matches,
        observations,
    )
    column_count = params.size
    jacobian = np.zeros((base.size, column_count), dtype=np.float64)
    for index in range(column_count):
        perturbed = params.copy()
        delta = step if abs(float(params[index])) < 1.0 else step * abs(float(params[index]))
        perturbed[index] += delta
        sample = _residual_vector(
            perturbed,
            match_ids,
            landmark_ids,
            anchor_id,
            matches,
            observations,
        )
        jacobian[:, index] = (sample - base) / delta
    return jacobian


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Per-residual scale so weighted squares implement Huber loss."""
    if delta <= 0.0 or residuals.size == 0:
        return np.ones(residuals.shape, dtype=np.float64)
    abs_residuals = np.abs(residuals)
    weights = np.ones(residuals.shape, dtype=np.float64)
    large = abs_residuals > delta
    weights[large] = np.sqrt(delta / np.maximum(abs_residuals[large], 1.0e-12))
    return weights


def _cauchy_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Per-residual scale for Cauchy/Lorentzian loss (stronger outlier taper)."""
    if delta <= 0.0 or residuals.size == 0:
        return np.ones(residuals.shape, dtype=np.float64)
    # Cost ~ log(1 + (r/c)^2) ⇒ IRLS weight 1/(1+(r/c)^2); scale residuals by √w.
    ratio = residuals / max(float(delta), 1.0e-12)
    return 1.0 / np.sqrt(1.0 + ratio * ratio)


def _robust_weights(
    residuals: np.ndarray,
    delta: float,
    *,
    kind: str = "huber",
) -> np.ndarray:
    """IRLS residual scales for the joint BA robust kernel."""
    if kind == "cauchy":
        return _cauchy_weights(residuals, delta)
    return _huber_weights(residuals, delta)


def _similarity_param_stride(
    *,
    lock_scale: bool,
    lock_rotation: bool,
    lock_translation: bool = False,
) -> int:
    """Pose parameter count: optional α, optional ω, optional t."""
    return (
        (0 if lock_scale else 1)
        + (0 if lock_rotation else 3)
        + (0 if lock_translation else 3)
    )


def _pack_ba_params(
    free_match_ids: list[str],
    free_landmark_ids: list[str],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
    *,
    lock_scale: bool,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    free_line_ids: list[str] | None = None,
    free_line_points: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pack free-match pose (+ optional log-scale / rotation / translation), free landmarks, free line midpoints."""
    values: list[float] = []
    for match_id in free_match_ids:
        similarity = similarities[match_id]
        if not lock_scale:
            values.append(float(np.log(max(similarity.scale, 1.0e-8))))
        if not lock_rotation:
            values.extend(_log_rodrigues(similarity.rotation).tolist())
        if not lock_translation:
            values.extend(similarity.translation.tolist())
    for landmark_id in free_landmark_ids:
        values.extend(landmarks[landmark_id].tolist())
    for line_id in free_line_ids or []:
        point = (free_line_points or {})[line_id]
        values.extend(np.asarray(point, dtype=np.float64).tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_ba_params(
    params: np.ndarray,
    free_match_ids: list[str],
    free_landmark_ids: list[str],
    *,
    lock_scale: bool,
    fixed_scales: dict[str, float],
    lock_rotation: bool = False,
    fixed_rotations: dict[str, np.ndarray] | None = None,
    lock_translation: bool = False,
    fixed_translations: dict[str, np.ndarray] | None = None,
    free_line_ids: list[str] | None = None,
) -> tuple[
    dict[str, SimilarityTransform],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    similarities: dict[str, SimilarityTransform] = {}
    offset = 0
    stride = _similarity_param_stride(
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    fixed_rotations = fixed_rotations or {}
    fixed_translations = fixed_translations or {}
    for match_id in free_match_ids:
        cursor = offset
        if lock_scale:
            scale = max(float(fixed_scales.get(match_id, 1.0)), 1.0e-8)
        else:
            scale = float(np.exp(params[cursor]))
            cursor += 1
        if lock_rotation:
            fixed = fixed_rotations.get(match_id)
            rotation = (
                np.asarray(fixed, dtype=np.float64).reshape(3, 3).copy()
                if fixed is not None
                else np.eye(3, dtype=np.float64)
            )
        else:
            rotation = _rodrigues(params[cursor : cursor + 3])
            cursor += 3
        if lock_translation:
            fixed_t = fixed_translations.get(match_id)
            translation = (
                np.asarray(fixed_t, dtype=np.float64).reshape(3).copy()
                if fixed_t is not None
                else np.zeros(3, dtype=np.float64)
            )
        else:
            translation = params[cursor : cursor + 3].copy()
        similarities[match_id] = SimilarityTransform(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        offset += stride
    landmarks: dict[str, np.ndarray] = {}
    for landmark_id in free_landmark_ids:
        landmarks[landmark_id] = params[offset : offset + 3].copy()
        offset += 3
    free_line_points: dict[str, np.ndarray] = {}
    for line_id in free_line_ids or []:
        free_line_points[line_id] = params[offset : offset + 3].copy()
        offset += 3
    return similarities, landmarks, free_line_points


def _rodrigues_partials(
    rotation_vector: np.ndarray,
    *,
    step: float = 1.0e-6,
) -> list[np.ndarray]:
    """Finite-difference ∂R/∂ω_i for the absolute Rodrigues chart (3×3 each)."""
    base = _rodrigues(rotation_vector)
    partials: list[np.ndarray] = []
    for index in range(3):
        perturbed = rotation_vector.copy()
        perturbed[index] += step
        partials.append((_rodrigues(perturbed) - base) / step)
    return partials


def _project_private_jacobian(
    point_private: np.ndarray,
    calibration: core.Calibration,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (uv, ∂uv/∂X_private) for a private-world point, or None if behind."""
    camera_point = calibration.rotation_w2c @ (point_private - calibration.camera_center)
    depth = float(camera_point[2])
    if depth <= 1.0e-8:
        return None
    intrinsics = calibration.intrinsics
    ideal = np.array(
        (
            intrinsics.fx * float(camera_point[0]) / depth + intrinsics.cx,
            intrinsics.fy * float(camera_point[1]) / depth + intrinsics.cy,
        ),
        dtype=np.float64,
    )
    # Pinhole Jacobian in camera coordinates, then chain through R_w2c.
    inv_depth = 1.0 / depth
    inv_depth_sq = inv_depth * inv_depth
    d_ideal_d_camera = np.array(
        (
            (
                intrinsics.fx * inv_depth,
                0.0,
                -intrinsics.fx * float(camera_point[0]) * inv_depth_sq,
            ),
            (
                0.0,
                intrinsics.fy * inv_depth,
                -intrinsics.fy * float(camera_point[1]) * inv_depth_sq,
            ),
        ),
        dtype=np.float64,
    )
    d_ideal_d_private = d_ideal_d_camera @ calibration.rotation_w2c
    if not calibration.has_distortion:
        return ideal, d_ideal_d_private
    # Distortion: chain a tiny FD through the 2D ideal → observed map.
    distorted = core.distort_points(
        ideal.reshape(1, 2),
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        calibration.division_lambda,
        calibration.brown_conrady,
    )[0]
    d_dist_d_ideal = np.zeros((2, 2), dtype=np.float64)
    for axis in range(2):
        offset = ideal.copy()
        offset[axis] += 1.0e-4
        sample = core.distort_points(
            offset.reshape(1, 2),
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            calibration.division_lambda,
            calibration.brown_conrady,
        )[0]
        d_dist_d_ideal[:, axis] = (sample - distorted) / 1.0e-4
    return distorted, d_dist_d_ideal @ d_ideal_d_private


def _ba_raw_residuals_and_jacobian(
    params: np.ndarray,
    free_match_ids: list[str],
    free_landmark_ids: list[str],
    fixed_landmarks: dict[str, np.ndarray],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    line_constraints: list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]],
    *,
    lock_scale: bool,
    fixed_scales: dict[str, float],
    lock_rotation: bool = False,
    fixed_rotations: dict[str, np.ndarray] | None = None,
    lock_translation: bool = False,
    fixed_translations: dict[str, np.ndarray] | None = None,
    free_line_ids: list[str] | None = None,
    fixed_line_points: dict[str, np.ndarray] | None = None,
    fixed_line_directions: dict[str, np.ndarray] | None = None,
    ground_landmark_ids: list[str] | None = None,
    ground_slack: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unweighted BA residuals and block-analytic Jacobian."""
    free_line_ids = free_line_ids or []
    fixed_line_points = fixed_line_points or {}
    fixed_line_directions = fixed_line_directions or {}
    fixed_rotations = fixed_rotations or {}
    fixed_translations = fixed_translations or {}
    similarities, free_landmarks, free_line_points = _unpack_ba_params(
        params,
        free_match_ids,
        free_landmark_ids,
        lock_scale=lock_scale,
        fixed_scales=fixed_scales,
        lock_rotation=lock_rotation,
        fixed_rotations=fixed_rotations,
        lock_translation=lock_translation,
        fixed_translations=fixed_translations,
        free_line_ids=free_line_ids,
    )
    similarities[anchor_id] = SimilarityTransform()
    # Both locks omit Empty poses from the parameter vector — keep identity
    # similarities so residual eval can still project through every match.
    if lock_rotation and lock_translation:
        for match_id in matches:
            similarities.setdefault(match_id, SimilarityTransform())
    landmarks = dict(fixed_landmarks)
    landmarks.update(free_landmarks)
    line_points = dict(fixed_line_points)
    line_points.update(free_line_points)

    stride = _similarity_param_stride(
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    match_offset = {
        match_id: index * stride for index, match_id in enumerate(free_match_ids)
    }
    landmark_base = len(free_match_ids) * stride
    landmark_offset = {
        landmark_id: landmark_base + index * 3
        for index, landmark_id in enumerate(free_landmark_ids)
    }
    line_base = landmark_base + 3 * len(free_landmark_ids)
    line_offset = {
        line_id: line_base + index * 3 for index, line_id in enumerate(free_line_ids)
    }
    rotation_partials = (
        {}
        if lock_rotation
        else {
            match_id: _rodrigues_partials(
                _log_rodrigues(similarities[match_id].rotation)
            )
            for match_id in free_match_ids
        }
    )

    residuals: list[float] = []
    jacobian_rows: list[np.ndarray] = []
    column_count = int(params.size)

    for observation in observations:
        point_shared = landmarks.get(observation.landmark_id)
        if point_shared is None:
            continue
        match_id = observation.match_id
        similarity = similarities[match_id]
        calibration = matches[match_id].calibration
        scale = _observation_scale(observation)
        private = similarity.inverse_point(point_shared)
        projected = _project_private_jacobian(private, calibration)
        row_u = np.zeros(column_count, dtype=np.float64)
        row_v = np.zeros(column_count, dtype=np.float64)
        if projected is None:
            residuals.extend((scale * 1.0e3, scale * 1.0e3))
            jacobian_rows.extend((row_u, row_v))
            continue
        uv_coordinate, d_uv_d_private = projected
        residuals.append(scale * float(uv_coordinate[0] - observation.u))
        residuals.append(scale * float(uv_coordinate[1] - observation.v))
        d_uv_d_private = scale * d_uv_d_private

        # Shared → private: X_p = Rᵀ (X - t) / s
        inv_scale = 1.0 / max(float(similarity.scale), 1.0e-12)
        rotation_t = similarity.rotation.T
        d_private_d_shared = inv_scale * rotation_t
        if observation.landmark_id in landmark_offset:
            start = landmark_offset[observation.landmark_id]
            block = d_uv_d_private @ d_private_d_shared
            row_u[start : start + 3] = block[0]
            row_v[start : start + 3] = block[1]
        if match_id in match_offset:
            start = match_offset[match_id]
            offset = start
            if not lock_scale:
                # ∂X_p/∂α = -X_p when s = exp(α)
                d_uv_d_alpha = d_uv_d_private @ (-private)
                row_u[offset] = float(d_uv_d_alpha[0])
                row_v[offset] = float(d_uv_d_alpha[1])
                offset += 1
            if not lock_rotation:
                y_vector = (point_shared - similarity.translation) * inv_scale
                for axis, partial in enumerate(rotation_partials[match_id]):
                    d_private_d_omega = partial.T @ y_vector
                    d_uv = d_uv_d_private @ d_private_d_omega
                    row_u[offset + axis] = float(d_uv[0])
                    row_v[offset + axis] = float(d_uv[1])
                offset += 3
            if not lock_translation:
                d_private_d_translation = -d_private_d_shared
                block = d_uv_d_private @ d_private_d_translation
                row_u[offset : offset + 3] = block[0]
                row_v[offset : offset + 3] = block[1]
        jacobian_rows.extend((row_u, row_v))

    slack = max(float(ground_slack), 0.0)
    if slack > 1.0e-12:
        spring = GROUND_Z_RESIDUAL_PX / slack
        for landmark_id in ground_landmark_ids or ():
            if landmark_id not in landmark_offset:
                continue
            point_shared = landmarks.get(landmark_id)
            if point_shared is None:
                continue
            residuals.append(spring * float(point_shared[2]))
            row_z = np.zeros(column_count, dtype=np.float64)
            start = landmark_offset[landmark_id]
            row_z[start + 2] = spring
            jacobian_rows.append(row_z)

    # Lines: pose FD + free-midpoint FD (directions stay fixed from the seed).
    for landmark_id, _seed_point, direction, observation in line_constraints:
        match_id = observation.match_id
        similarity = similarities[match_id]
        calibration = matches[match_id].calibration
        point = line_points.get(landmark_id, _seed_point)
        direction = fixed_line_directions.get(landmark_id, direction)
        base_errors = _line_observation_reprojection_errors(
            point,
            direction,
            observation,
            calibration,
            similarity,
        )
        residuals.extend(base_errors)
        row_a = np.zeros(column_count, dtype=np.float64)
        row_b = np.zeros(column_count, dtype=np.float64)
        columns: list[int] = []
        if match_id in match_offset:
            start = match_offset[match_id]
            columns.extend(range(start, start + stride))
        if landmark_id in line_offset:
            start = line_offset[landmark_id]
            columns.extend(range(start, start + 3))
        for column in columns:
            perturbed = params.copy()
            delta = (
                1.0e-5
                if abs(float(params[column])) < 1.0
                else 1.0e-5 * abs(float(params[column]))
            )
            perturbed[column] += delta
            trial_sims, _trial_landmarks, trial_lines = _unpack_ba_params(
                perturbed,
                free_match_ids,
                free_landmark_ids,
                lock_scale=lock_scale,
                fixed_scales=fixed_scales,
                lock_rotation=lock_rotation,
                fixed_rotations=fixed_rotations,
                lock_translation=lock_translation,
                fixed_translations=fixed_translations,
                free_line_ids=free_line_ids,
            )
            trial_sims[anchor_id] = SimilarityTransform()
            if lock_rotation and lock_translation:
                for trial_match_id in matches:
                    trial_sims.setdefault(trial_match_id, SimilarityTransform())
            trial_point = trial_lines.get(landmark_id, point)
            sample = _line_observation_reprojection_errors(
                trial_point,
                direction,
                observation,
                calibration,
                trial_sims[match_id],
            )
            row_a[column] = (sample[0] - base_errors[0]) / delta
            row_b[column] = (sample[1] - base_errors[1]) / delta
        jacobian_rows.extend((row_a, row_b))

    residual_array = np.asarray(residuals, dtype=np.float64)
    if not jacobian_rows:
        return residual_array, np.zeros((0, column_count), dtype=np.float64)
    return residual_array, np.vstack(jacobian_rows)


def _ba_residual_vector(
    params: np.ndarray,
    free_match_ids: list[str],
    free_landmark_ids: list[str],
    fixed_landmarks: dict[str, np.ndarray],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    line_constraints: list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]],
    *,
    lock_scale: bool,
    fixed_scales: dict[str, float],
    huber_delta: float,
    lock_rotation: bool = False,
    fixed_rotations: dict[str, np.ndarray] | None = None,
    lock_translation: bool = False,
    fixed_translations: dict[str, np.ndarray] | None = None,
    free_line_ids: list[str] | None = None,
    fixed_line_points: dict[str, np.ndarray] | None = None,
    fixed_line_directions: dict[str, np.ndarray] | None = None,
    ground_landmark_ids: list[str] | None = None,
    ground_slack: float = 0.0,
) -> np.ndarray:
    """Joint reprojection residuals for free poses + free landmarks (+ lines)."""
    residual_array, _jacobian = _ba_raw_residuals_and_jacobian(
        params,
        free_match_ids,
        free_landmark_ids,
        fixed_landmarks,
        anchor_id,
        matches,
        observations,
        line_constraints,
        lock_scale=lock_scale,
        fixed_scales=fixed_scales,
        lock_rotation=lock_rotation,
        fixed_rotations=fixed_rotations,
        lock_translation=lock_translation,
        fixed_translations=fixed_translations,
        free_line_ids=free_line_ids,
        fixed_line_points=fixed_line_points,
        fixed_line_directions=fixed_line_directions,
        ground_landmark_ids=ground_landmark_ids,
        ground_slack=ground_slack,
    )
    return residual_array * _robust_weights(residual_array, huber_delta)


def _jacobian_ba(
    params: np.ndarray,
    residual_kwargs: dict,
) -> np.ndarray:
    """Block-analytic BA Jacobian (robust IRLS weights, matching residual scaling)."""
    huber_delta = float(residual_kwargs["huber_delta"])
    raw_kwargs = {
        key: value
        for key, value in residual_kwargs.items()
        if key != "huber_delta"
    }
    residuals, jacobian = _ba_raw_residuals_and_jacobian(params, **raw_kwargs)
    weights = _robust_weights(residuals, huber_delta)
    return jacobian * weights[:, np.newaxis]


def _collect_ba_line_constraints(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]],
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    connected: set[str],
) -> list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]]:
    """3D lines used as soft constraints while refining poses (+ free midpoints)."""
    constraints: list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]] = []
    for landmark_id, items in line_observations_by_landmark.items():
        if landmark_id in line_segments:
            point_a, point_b = line_segments[landmark_id]
        elif landmark_id in known_lines:
            point_a, point_b = known_lines[landmark_id]
        else:
            continue
        direction = point_b - point_a
        span = float(np.linalg.norm(direction))
        if span < 1.0e-9:
            continue
        direction = direction / span
        point = 0.5 * (point_a + point_b)
        for observation in items:
            if observation.match_id not in connected:
                continue
            constraints.append((landmark_id, point, direction, observation))
    return constraints


def _auto_downweight_outlier_observations(
    observations: list[SyncObservation],
    per_landmark_rmse: dict[str, float],
    *,
    factor: float = OUTLIER_WEIGHT_FACTOR,
    absolute_px: float = 20.0,
    relative_to_median: float = 2.5,
) -> tuple[list[SyncObservation], list[str]]:
    """Clone observations with reduced weight for severe landmark outliers."""
    point_values = [
        rmse
        for landmark_id, rmse in per_landmark_rmse.items()
        if any(item.landmark_id == landmark_id for item in observations)
    ]
    if not point_values:
        return observations, []
    median = float(np.median(point_values))
    threshold = max(float(absolute_px), float(relative_to_median) * max(median, 1.0e-6))
    outlier_ids = {
        landmark_id
        for landmark_id, rmse in per_landmark_rmse.items()
        if rmse > threshold
    }
    if not outlier_ids:
        return observations, []
    adjusted: list[SyncObservation] = []
    for observation in observations:
        if observation.landmark_id in outlier_ids:
            adjusted.append(
                SyncObservation(
                    match_id=observation.match_id,
                    landmark_id=observation.landmark_id,
                    u=observation.u,
                    v=observation.v,
                    on_ground=observation.on_ground,
                    landmark_name=observation.landmark_name,
                    weight=max(float(observation.weight) * factor, 1.0e-3),
                )
            )
        else:
            adjusted.append(observation)
    return adjusted, sorted(outlier_ids)


def _bundle_adjust_registration(
    free_match_ids: list[str],
    free_landmark_ids: list[str],
    fixed_landmarks: dict[str, np.ndarray],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    line_constraints: list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]],
    *,
    known_line_ids: set[str] | None = None,
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    lock_scale: bool = True,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    max_iterations: int = 20,
    huber_delta: float = 6.0,
    max_free_lines: int = 24,
    ground_landmark_ids: list[str] | None = None,
    ground_slack: float = 0.0,
) -> tuple[
    dict[str, SimilarityTransform],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
    bool,
]:
    """Joint LM over free Empty poses, landmarks, and free-line midpoints.

    Pairwise registration seeds the solve; this pass couples every match and
    landmark into one reprojection objective (Huber-weighted). Free line
    midpoints move; directions stay fixed from the seed (parallel enforcement
    can still lock families afterward).
    """
    known_line_ids = known_line_ids or set()
    line_segments = {
        landmark_id: (segment[0].copy(), segment[1].copy())
        for landmark_id, segment in (line_segments or {}).items()
    }
    if not free_match_ids and not free_landmark_ids and not line_constraints:
        return similarities, landmarks, line_segments, False
    if not observations and not line_constraints:
        return similarities, landmarks, line_segments, False

    fixed_scales = {
        match_id: float(similarities[match_id].scale) for match_id in free_match_ids
    }
    # Prefer rigid Empty transforms when every seed is already metric.
    if lock_scale and any(abs(scale - 1.0) > 1.0e-3 for scale in fixed_scales.values()):
        lock_scale = False
    # Locked rotation stays on the cube group (90° axis jumps), not identity.
    if lock_rotation:
        from .pose import _snap_to_axis_aligned_rotation

        for match_id in free_match_ids:
            similarity = similarities[match_id]
            similarities[match_id] = SimilarityTransform(
                scale=float(similarity.scale),
                rotation=_snap_to_axis_aligned_rotation(similarity.rotation),
                translation=np.asarray(similarity.translation, dtype=np.float64).copy(),
            )
        fixed_rotations = {
            match_id: similarities[match_id].rotation.copy()
            for match_id in free_match_ids
        }
    else:
        fixed_rotations = {}
    # Zero translation when translation is locked (private worlds share origin axes).
    fixed_translations = {
        match_id: np.zeros(3, dtype=np.float64) for match_id in free_match_ids
    } if lock_translation else {}
    # Fully locked Empty → no pose DOFs in BA (landmarks / lines only).
    pose_match_ids = (
        []
        if (lock_rotation and lock_translation)
        else list(free_match_ids)
    )
    if lock_rotation and lock_translation:
        for match_id in free_match_ids:
            similarities[match_id] = SimilarityTransform()

    working_landmarks = {
        landmark_id: landmarks[landmark_id].copy()
        for landmark_id in free_landmark_ids
        if landmark_id in landmarks
    }
    if len(working_landmarks) != len(free_landmark_ids):
        free_landmark_ids = [
            landmark_id
            for landmark_id in free_landmark_ids
            if landmark_id in working_landmarks
        ]

    # Free midpoints for non-Known lines that participate in the BA objective.
    free_line_points: dict[str, np.ndarray] = {}
    fixed_line_points: dict[str, np.ndarray] = {}
    fixed_line_directions: dict[str, np.ndarray] = {}
    for landmark_id, point, direction, _observation in line_constraints:
        fixed_line_directions[landmark_id] = np.asarray(direction, dtype=np.float64)
        if landmark_id in known_line_ids:
            fixed_line_points[landmark_id] = np.asarray(point, dtype=np.float64)
        elif landmark_id not in free_line_points:
            free_line_points[landmark_id] = np.asarray(point, dtype=np.float64).copy()
    free_line_ids = sorted(free_line_points)[: max(0, int(max_free_lines))]
    # Overflow free lines stay fixed soft constraints.
    for landmark_id in list(free_line_points):
        if landmark_id not in free_line_ids:
            fixed_line_points[landmark_id] = free_line_points.pop(landmark_id)

    params = _pack_ba_params(
        pose_match_ids,
        free_landmark_ids,
        similarities,
        working_landmarks,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        free_line_ids=free_line_ids,
        free_line_points=free_line_points,
    )
    residual_kwargs = {
        "free_match_ids": pose_match_ids,
        "free_landmark_ids": free_landmark_ids,
        "fixed_landmarks": fixed_landmarks,
        "anchor_id": anchor_id,
        "matches": matches,
        "observations": observations,
        "line_constraints": line_constraints,
        "lock_scale": lock_scale,
        "fixed_scales": fixed_scales,
        "lock_rotation": lock_rotation,
        "fixed_rotations": fixed_rotations,
        "lock_translation": lock_translation,
        "fixed_translations": fixed_translations,
        "huber_delta": huber_delta,
        "free_line_ids": free_line_ids,
        "fixed_line_points": fixed_line_points,
        "fixed_line_directions": fixed_line_directions,
        "ground_landmark_ids": list(ground_landmark_ids or ()),
        "ground_slack": float(ground_slack),
    }
    damping = 1.0e-2
    previous_cost = float("inf")
    # True once BA has a residual vector to evaluate (even if already optimal).
    ran_ba = False
    for _iteration in range(max_iterations):
        residuals = _ba_residual_vector(params, **residual_kwargs)
        if residuals.size == 0:
            break
        ran_ba = True
        cost = float(residuals @ residuals)
        if cost < 1.0e-8:
            break
        if abs(previous_cost - cost) / max(previous_cost, 1.0e-12) < 1.0e-8:
            break
        jacobian = _jacobian_ba(params, residual_kwargs)
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        step_accepted = False
        for _attempt in range(8):
            try:
                delta = np.linalg.solve(
                    gram + damping * np.diag(np.maximum(np.diag(gram), 1.0e-8)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = params + delta
            candidate_residuals = _ba_residual_vector(candidate, **residual_kwargs)
            candidate_cost = float(candidate_residuals @ candidate_residuals)
            if candidate_cost < cost:
                params = candidate
                previous_cost = cost
                damping = max(damping * 0.3, 1.0e-8)
                step_accepted = True
                break
            damping *= 10.0
        if not step_accepted:
            break

    refined_similarities, refined_free, refined_lines = _unpack_ba_params(
        params,
        pose_match_ids,
        free_landmark_ids,
        lock_scale=lock_scale,
        fixed_scales=fixed_scales,
        lock_rotation=lock_rotation,
        fixed_rotations=fixed_rotations,
        lock_translation=lock_translation,
        fixed_translations=fixed_translations,
        free_line_ids=free_line_ids,
    )
    result_similarities = dict(similarities)
    result_similarities.update(refined_similarities)
    result_similarities[anchor_id] = SimilarityTransform()
    result_landmarks = dict(landmarks)
    result_landmarks.update(fixed_landmarks)
    result_landmarks.update(refined_free)
    # Apply refined free midpoints onto finite segments (keep seed length).
    for landmark_id, midpoint in refined_lines.items():
        direction = fixed_line_directions.get(landmark_id)
        if direction is None:
            continue
        if landmark_id in line_segments:
            point_a, point_b = line_segments[landmark_id]
            half = 0.5 * float(np.linalg.norm(point_b - point_a))
        else:
            half = 0.5
        half = max(half, 1.0e-3)
        unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        line_segments[landmark_id] = (
            midpoint - half * unit,
            midpoint + half * unit,
        )
        result_landmarks[landmark_id] = midpoint.copy()
    return result_similarities, result_landmarks, line_segments, ran_ba


def _point_landmark_rmse_snapshot(
    free_match_ids: list[str],
    landmark_ids: list[str],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
) -> dict[str, float]:
    """Unweighted per-landmark RMSE for the current registration."""
    residual_landmark_ids = [
        landmark_id for landmark_id in landmark_ids if landmark_id in landmarks
    ]
    residual_observations = [
        observation
        for observation in observations
        if observation.landmark_id in landmarks
    ]
    if not residual_landmark_ids or not residual_observations:
        return {}
    residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {
                landmark_id: landmarks[landmark_id]
                for landmark_id in residual_landmark_ids
            },
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        matches,
        residual_observations,
        weighted=False,
    )
    per_landmark_sse: dict[str, list[float]] = {
        landmark_id: [] for landmark_id in residual_landmark_ids
    }
    residual_index = 0
    for observation in residual_observations:
        error_u = float(residuals[residual_index])
        error_v = float(residuals[residual_index + 1])
        residual_index += 2
        per_landmark_sse[observation.landmark_id].append(
            error_u * error_u + error_v * error_v
        )
    return {
        landmark_id: float(np.sqrt(np.mean(values)))
        for landmark_id, values in per_landmark_sse.items()
        if values
    }


def _per_match_rmse_snapshot(
    free_match_ids: list[str],
    landmark_ids: list[str],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
    anchor_id: str,
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
) -> dict[str, float]:
    """Unweighted per-match RMSE for the current registration."""
    residual_landmark_ids = [
        landmark_id for landmark_id in landmark_ids if landmark_id in landmarks
    ]
    residual_observations = [
        observation
        for observation in observations
        if observation.landmark_id in landmarks
    ]
    if not residual_landmark_ids or not residual_observations:
        return {}
    residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {
                landmark_id: landmarks[landmark_id]
                for landmark_id in residual_landmark_ids
            },
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        matches,
        residual_observations,
        weighted=False,
    )
    per_match_sse: dict[str, list[float]] = {
        match_id: [] for match_id in list(similarities)
    }
    residual_index = 0
    for observation in residual_observations:
        error_u = float(residuals[residual_index])
        error_v = float(residuals[residual_index + 1])
        residual_index += 2
        per_match_sse.setdefault(observation.match_id, []).append(
            error_u * error_u + error_v * error_v
        )
    return {
        match_id: float(np.sqrt(np.mean(values)))
        for match_id, values in per_match_sse.items()
        if values
    }


def leave_one_out_landmark_report(
    matches: list[SyncMatchInput],
    observations: list[SyncObservation],
    *,
    anchor_id: str,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
    top_k: int = 5,
    baseline: SyncSolveResult | None = None,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    ground_slack: float | None = None,
) -> list[tuple[str, float, float]]:
    """Re-solve without each of the worst landmarks; report RMSE deltas.

    Returns ``(landmark_name, with_rmse, without_rmse)`` sorted by how much
    removing the landmark helps (largest improvement first).
    """
    if baseline is None:
        baseline = solve_landmark_sync(
            matches,
            observations,
            anchor_id=anchor_id,
            known_world=known_world,
            line_observations=line_observations,
            known_lines=known_lines,
            parallel_pairs=parallel_pairs,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            ground_slack=ground_slack,
        )
    if not baseline.per_landmark_rmse_px:
        return []
    names = {
        observation.landmark_id: (observation.landmark_name or observation.landmark_id)
        for observation in observations
        if observation.landmark_name
    }
    ranked = sorted(
        baseline.per_landmark_rmse_px.items(),
        key=lambda item: -item[1],
    )[: max(1, int(top_k))]
    report: list[tuple[str, float, float]] = []
    for landmark_id, with_rmse in ranked:
        filtered = [
            observation
            for observation in observations
            if observation.landmark_id != landmark_id
        ]
        filtered_lines = [
            observation
            for observation in (line_observations or [])
            if observation.landmark_id != landmark_id
        ]
        filtered_known = {
            key: value
            for key, value in (known_world or {}).items()
            if key != landmark_id
        }
        filtered_known_lines = {
            key: value
            for key, value in (known_lines or {}).items()
            if key != landmark_id
        }
        filtered_parallel = [
            pair
            for pair in (parallel_pairs or [])
            if landmark_id not in pair
        ]
        from .solve import solve_landmark_sync

        without = solve_landmark_sync(
            matches,
            filtered,
            anchor_id=anchor_id,
            known_world=filtered_known,
            line_observations=filtered_lines,
            known_lines=filtered_known_lines,
            parallel_pairs=filtered_parallel,
            initial_similarities=baseline.similarities,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            ground_slack=ground_slack,
        )
        without_rmse = (
            float(without.mean_reprojection_px)
            if without.success
            else float("inf")
        )
        report.append(
            (
                names.get(landmark_id, landmark_id[:8]),
                float(with_rmse),
                without_rmse,
            )
        )
    report.sort(
        key=lambda item: (
            -(item[1] - item[2]) if np.isfinite(item[2]) else -item[1]
        )
    )
    return report

