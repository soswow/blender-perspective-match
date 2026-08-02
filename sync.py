"""Multi-match landmark sync: register private worlds into an anchor frame.

Each match keeps its VP solve in a private world. Sync finds a rigid Empty
transform ``X_shared = R X_private + t`` (scale 1) per non-anchor match.

Enough 2D↔2D landmark correspondences recover relative orientation and
baseline *direction* (same idea as SfM). Absolute baseline length vs the
already-metric anchor world is pinned by optional On Ground picks,
known Blender-object 3D positions, or a depth heuristic. Intrinsics stay
frozen here; ``lens_refine`` can adjust focals in an outer loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core


@dataclass
class SimilarityTransform:
    """Maps a match private world into shared (anchor) world: ``s R x + t``."""

    scale: float = 1.0
    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64),
    )
    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )

    def matrix(self) -> np.ndarray:
        """Return a 4×4 homogeneous matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Map a private-world point into shared world."""
        return self.scale * (self.rotation @ point) + self.translation

    def inverse_point(self, point: np.ndarray) -> np.ndarray:
        """Map a shared-world point into private world."""
        scale = max(float(self.scale), 1.0e-12)
        return self.rotation.T @ ((point - self.translation) / scale)


# Relative least-squares weights for pick confidence (UI: High / Normal / Low).
CONFIDENCE_WEIGHTS = {
    "HIGH": 4.0,
    "NORMAL": 1.0,
    "LOW": 0.25,
}


@dataclass
class SyncObservation:
    """One landmark click in one match, in source-image pixels."""

    match_id: str
    landmark_id: str
    u: float
    v: float
    on_ground: bool = False
    landmark_name: str = ""
    # Relative least-squares weight (High=4, Normal=1, Low=0.25).
    weight: float = 1.0


@dataclass
class SyncLineObservation:
    """One 2D segment observation of a shared 3D edge, in source-image pixels."""

    match_id: str
    landmark_id: str
    u1: float
    v1: float
    u2: float
    v2: float
    landmark_name: str = ""
    weight: float = 1.0


def confidence_weight(confidence: str) -> float:
    """Map a UI confidence enum to a sync residual weight."""
    return float(CONFIDENCE_WEIGHTS.get(confidence, 1.0))


def _observation_scale(observation: SyncObservation) -> float:
    """sqrt(weight) so cost uses weight * r^2."""
    return float(np.sqrt(max(float(observation.weight), 1.0e-12)))


def _pair_scale(
    anchor_obs: SyncObservation,
    other_obs: SyncObservation,
) -> float:
    """Correspondence scale from geometric-mean weight of both picks."""
    return float(
        np.sqrt(
            max(float(anchor_obs.weight) * float(other_obs.weight), 1.0e-12)
        )
    )


@dataclass
class SyncMatchInput:
    """Frozen private-frame calibration for one match."""

    match_id: str
    calibration: core.Calibration


@dataclass
class SyncSolveResult:
    """Result of a landmark-graph sync solve."""

    similarities: dict[str, SimilarityTransform]
    landmarks: dict[str, np.ndarray]
    mean_reprojection_px: float
    per_match_rmse_px: dict[str, float]
    per_landmark_rmse_px: dict[str, float]
    message: str
    success: bool = True
    # Finite 3D segments for LINE landmarks (debug mesh viz).
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )


def _rodrigues(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert a 3-vector to a rotation matrix."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64) + _skew(rotation_vector)
    axis = rotation_vector / angle
    cosine = np.cos(angle)
    sine = np.sin(angle)
    skew = _skew(axis)
    return (
        cosine * np.eye(3, dtype=np.float64)
        + sine * skew
        + (1.0 - cosine) * np.outer(axis, axis)
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x_coordinate, y_coordinate, z_coordinate = (float(value) for value in vector)
    return np.array(
        (
            (0.0, -z_coordinate, y_coordinate),
            (z_coordinate, 0.0, -x_coordinate),
            (-y_coordinate, x_coordinate, 0.0),
        ),
        dtype=np.float64,
    )


def _log_rodrigues(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a 3-vector (principal value)."""
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    skew = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    )
    if angle > np.pi - 1.0e-6:
        # Near 180°: extract axis from the symmetric part.
        diagonal = np.clip(np.diag(rotation) + 1.0, 0.0, None)
        axis = np.sqrt(diagonal)
        if abs(float(rotation[0, 1])) + abs(float(rotation[0, 2])) + abs(
            float(rotation[1, 2])
        ) > 1.0e-8:
            axis[0] = abs(float(axis[0])) * np.sign(float(rotation[0, 1]) or float(rotation[0, 2]) or 1.0)
            axis[1] = abs(float(axis[1])) * np.sign(float(rotation[0, 1]) or float(rotation[1, 2]) or 1.0)
            axis[2] = abs(float(axis[2])) * np.sign(float(rotation[0, 2]) or float(rotation[1, 2]) or 1.0)
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-10:
            return np.array((np.pi, 0.0, 0.0), dtype=np.float64)
        return axis / norm * angle
    return skew * (angle / (2.0 * np.sin(angle)))


def project_private_point(
    point_private: np.ndarray,
    calibration: core.Calibration,
) -> np.ndarray | None:
    """Project a private-world point to distorted source pixels, or None if behind."""
    camera_point = calibration.rotation_w2c @ (point_private - calibration.camera_center)
    depth = float(camera_point[2])
    if depth <= 1.0e-8:
        return None
    intrinsics = calibration.intrinsics
    ideal = np.array(
        (
            (
                intrinsics.fx * float(camera_point[0]) / depth + intrinsics.cx,
                intrinsics.fy * float(camera_point[1]) / depth + intrinsics.cy,
            ),
        ),
        dtype=np.float64,
    )
    distorted = core.distort_points(
        ideal,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        calibration.division_lambda,
    )[0]
    return distorted


def camera_ray_private(
    u_coordinate: float,
    v_coordinate: float,
    calibration: core.Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, unit direction) of an image ray in the private world."""
    ideal = core.undistort_points(
        np.array([[u_coordinate, v_coordinate]], dtype=np.float64),
        calibration.intrinsics.fx,
        calibration.intrinsics.fy,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
        calibration.division_lambda,
    )[0]
    direction_camera = core.pixel_ray(
        float(ideal[0]),
        float(ideal[1]),
        calibration.intrinsics,
    )
    direction_world = calibration.rotation_w2c.T @ direction_camera
    direction_world = direction_world / max(float(np.linalg.norm(direction_world)), 1.0e-12)
    return calibration.camera_center.copy(), direction_world


def triangulate_midpoint(
    origins: list[np.ndarray],
    directions: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray | None:
    """Triangulate a point as the least-squares midpoint of skew rays.

    Optional ``weights`` pull the result toward higher-confidence rays.
    """
    if len(origins) < 2:
        return None
    # Solve sum w (I - d d^T)(X - o) = 0 → A X = b
    accumulator = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for index, (origin, direction) in enumerate(zip(origins, directions)):
        weight = 1.0 if weights is None else max(float(weights[index]), 1.0e-12)
        direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        projector = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
        accumulator += weight * projector
        rhs += weight * (projector @ origin)
    try:
        point = np.linalg.solve(accumulator, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _image_line_homogeneous(u1: float, v1: float, u2: float, v2: float) -> np.ndarray | None:
    """Homogeneous line coefficients through two image points."""
    point_a = np.array((u1, v1, 1.0), dtype=np.float64)
    point_b = np.array((u2, v2, 1.0), dtype=np.float64)
    line = np.cross(point_a, point_b)
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1.0e-12:
        return None
    return line / norm


def _point_to_image_line_distance(
    u: float,
    v: float,
    line: np.ndarray,
) -> float:
    """Perpendicular pixel distance from (u,v) to a normalized homogeneous line."""
    return abs(float(line[0] * u + line[1] * v + line[2]))


def _plane_from_line_observation(
    observation: SyncLineObservation,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> np.ndarray | None:
    """Back-project a 2D segment to a plane in shared world: π=[n; d] with n·x + d = 0."""
    origin_private, direction_a = camera_ray_private(
        observation.u1, observation.v1, calibration
    )
    _origin_b, direction_b = camera_ray_private(
        observation.u2, observation.v2, calibration
    )
    origin = similarity.transform_point(origin_private)
    dir_a = similarity.rotation @ direction_a
    dir_b = similarity.rotation @ direction_b
    normal = np.cross(dir_a, dir_b)
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-12:
        return None
    normal = normal / norm
    return np.array(
        (normal[0], normal[1], normal[2], -float(np.dot(normal, origin))),
        dtype=np.float64,
    )


def _intersect_planes_to_line(
    plane_a: np.ndarray,
    plane_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (point_on_line, direction) for the intersection of two planes."""
    normal_a = plane_a[:3]
    normal_b = plane_b[:3]
    direction = np.cross(normal_a, normal_b)
    span = float(np.linalg.norm(direction))
    if span < 1.0e-10:
        return None
    direction = direction / span
    # Solve n_a·x = -d_a, n_b·x = -d_b, prefer point nearest the origin in the plane.
    matrix = np.stack([normal_a, normal_b, direction])
    rhs = np.array((-plane_a[3], -plane_b[3], 0.0), dtype=np.float64)
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def _project_world_line_to_image(
    point: np.ndarray,
    direction: np.ndarray,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> np.ndarray | None:
    """Project an infinite 3D line into a match still as a homogeneous image line."""
    # Two distant samples along the line, mapped into the match private frame.
    samples = (point - 2.0 * direction, point + 2.0 * direction)
    projected: list[tuple[float, float]] = []
    for sample in samples:
        private = similarity.inverse_point(sample)
        image_point = project_private_point(private, calibration)
        if image_point is None:
            continue
        projected.append((float(image_point[0]), float(image_point[1])))
    if len(projected) < 2:
        return None
    return _image_line_homogeneous(
        projected[0][0],
        projected[0][1],
        projected[1][0],
        projected[1][1],
    )


def _line_observation_reprojection_errors(
    point: np.ndarray,
    direction: np.ndarray,
    observation: SyncLineObservation,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> list[float]:
    """Pixel distances from the observed segment endpoints to the projected 3D line."""
    projected = _project_world_line_to_image(
        point, direction, calibration, similarity
    )
    if projected is None:
        return [1.0e3, 1.0e3]
    scale = float(np.sqrt(max(float(observation.weight), 1.0e-12)))
    return [
        scale
        * _point_to_image_line_distance(observation.u1, observation.v1, projected),
        scale
        * _point_to_image_line_distance(observation.u2, observation.v2, projected),
    ]


def _known_line_reprojection_errors(
    point_a: np.ndarray,
    point_b: np.ndarray,
    observation: SyncLineObservation,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> list[float]:
    """Pixel distances from Known 3D line endpoints projected onto the observed 2D line."""
    observed = _image_line_homogeneous(
        observation.u1, observation.v1, observation.u2, observation.v2
    )
    if observed is None:
        return [1.0e3, 1.0e3]
    scale = float(np.sqrt(max(float(observation.weight), 1.0e-12)))
    errors: list[float] = []
    for point in (point_a, point_b):
        private = similarity.inverse_point(point)
        projected = project_private_point(private, calibration)
        if projected is None:
            errors.append(scale * 1.0e3)
            continue
        errors.append(
            scale
            * _point_to_image_line_distance(
                float(projected[0]), float(projected[1]), observed
            )
        )
    return errors


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
        landmarks[landmark_id] = point
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


def _connected_match_ids(
    anchor_id: str,
    observations: list[SyncObservation],
    *,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> set[str]:
    """Matches reachable from the anchor through shared landmarks.

    Known-world landmarks (Blender Empties) also bridge: a pick in any match
    links that match to the anchor even without an anchor 2D observation.
    """
    landmark_to_matches: dict[str, set[str]] = {}
    for observation in observations:
        landmark_to_matches.setdefault(observation.landmark_id, set()).add(
            observation.match_id,
        )
    for observation in line_observations or ():
        landmark_to_matches.setdefault(observation.landmark_id, set()).add(
            observation.match_id,
        )
    if known_world:
        for landmark_id in known_world:
            landmark_to_matches.setdefault(landmark_id, set()).add(anchor_id)
    if known_lines:
        for landmark_id in known_lines:
            landmark_to_matches.setdefault(landmark_id, set()).add(anchor_id)
    adjacency: dict[str, set[str]] = {}
    for match_ids in landmark_to_matches.values():
        for match_id in match_ids:
            adjacency.setdefault(match_id, set()).update(match_ids - {match_id})
    reached = {anchor_id}
    queue = [anchor_id]
    while queue:
        current = queue.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _reconstruct_line_from_observations(
    observations: list[SyncLineObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Intersect back-projected planes from ≥2 registered views into a 3D line."""
    planes: list[np.ndarray] = []
    for observation in observations:
        similarity = similarities.get(observation.match_id)
        match = matches.get(observation.match_id)
        if similarity is None or match is None:
            continue
        plane = _plane_from_line_observation(
            observation, match.calibration, similarity
        )
        if plane is not None:
            planes.append(plane)
    if len(planes) < 2:
        return None
    return _intersect_planes_to_line(planes[0], planes[1])


def _closest_point_on_line_to_ray(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray:
    """Closest point on an infinite 3D line to a 3D ray (midpoint of skew gap)."""
    direction = line_direction / max(float(np.linalg.norm(line_direction)), 1.0e-12)
    ray = ray_direction / max(float(np.linalg.norm(ray_direction)), 1.0e-12)
    cross = np.cross(direction, ray)
    denom = float(np.dot(cross, cross))
    offset = ray_origin - line_point
    if denom < 1.0e-12:
        # Nearly parallel — fall back to projection of ray origin onto the line.
        return line_point + float(np.dot(offset, direction)) * direction
    # From skew-line formula: t = ((o2-o1) × d2) · (d1 × d2) / ||d1 × d2||^2
    parameter = float(np.dot(np.cross(offset, ray), cross) / denom)
    return line_point + parameter * direction


def _finite_segment_from_line_observations(
    point: np.ndarray,
    direction: np.ndarray,
    observations: list[SyncLineObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
) -> tuple[np.ndarray, np.ndarray]:
    """Project drawn 2D segment ends onto the 3D line for a visible mesh edge."""
    samples: list[np.ndarray] = []
    for observation in observations:
        similarity = similarities.get(observation.match_id)
        match = matches.get(observation.match_id)
        if similarity is None or match is None:
            continue
        for u_coord, v_coord in (
            (observation.u1, observation.v1),
            (observation.u2, observation.v2),
        ):
            origin_private, direction_private = camera_ray_private(
                u_coord, v_coord, match.calibration
            )
            origin = similarity.transform_point(origin_private)
            ray = similarity.rotation @ direction_private
            samples.append(
                _closest_point_on_line_to_ray(point, direction, origin, ray)
            )
    if len(samples) < 2:
        unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        return point - unit, point + unit
    # Extent along the line axis from all endpoint projections.
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    parameters = [float(np.dot(sample - point, unit)) for sample in samples]
    return point + min(parameters) * unit, point + max(parameters) * unit


def _direction_from_vanishing_point(
    vanishing: np.ndarray,
    calibration: core.Calibration,
) -> np.ndarray | None:
    """Map an image vanishing point to a unit direction in the private world."""
    vector = np.asarray(vanishing, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(vector)):
        return None
    intrinsics = calibration.intrinsics
    fx = max(float(intrinsics.fx), 1.0e-12)
    fy = max(float(intrinsics.fy), 1.0e-12)
    inverse_k = np.array(
        (
            (1.0 / fx, 0.0, -intrinsics.cx / fx),
            (0.0, 1.0 / fy, -intrinsics.cy / fy),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    direction_camera = inverse_k @ vector
    norm = float(np.linalg.norm(direction_camera))
    if norm < 1.0e-12:
        return None
    direction_camera = direction_camera / norm
    direction_world = calibration.rotation_w2c.T @ direction_camera
    norm = float(np.linalg.norm(direction_world))
    if norm < 1.0e-12:
        return None
    return direction_world / norm


def _vanishing_point_from_line_pair(
    observation_a: SyncLineObservation,
    observation_b: SyncLineObservation,
) -> np.ndarray | None:
    """Intersect two image lines to a homogeneous vanishing point."""
    line_a = _image_line_homogeneous(
        observation_a.u1, observation_a.v1, observation_a.u2, observation_a.v2
    )
    line_b = _image_line_homogeneous(
        observation_b.u1, observation_b.v1, observation_b.u2, observation_b.v2
    )
    if line_a is None or line_b is None:
        return None
    vanishing = np.cross(line_a, line_b)
    if float(np.linalg.norm(vanishing)) < 1.0e-12:
        return None
    return vanishing


def _parallel_pair_rotation_error(
    observation_a_anchor: SyncLineObservation,
    observation_b_anchor: SyncLineObservation,
    observation_a_other: SyncLineObservation,
    observation_b_other: SyncLineObservation,
    anchor: core.Calibration,
    other: core.Calibration,
    similarity: SimilarityTransform,
) -> float | None:
    """Sin of angle between parallel-family directions under a candidate Empty pose."""
    vanishing_anchor = _vanishing_point_from_line_pair(
        observation_a_anchor, observation_b_anchor
    )
    vanishing_other = _vanishing_point_from_line_pair(
        observation_a_other, observation_b_other
    )
    if vanishing_anchor is None or vanishing_other is None:
        return None
    direction_anchor = _direction_from_vanishing_point(vanishing_anchor, anchor)
    direction_other_private = _direction_from_vanishing_point(
        vanishing_other, other
    )
    if direction_anchor is None or direction_other_private is None:
        return None
    direction_other = similarity.rotation @ direction_other_private
    direction_other = direction_other / max(
        float(np.linalg.norm(direction_other)), 1.0e-12
    )
    return min(
        float(np.linalg.norm(np.cross(direction_anchor, direction_other))),
        float(np.linalg.norm(np.cross(direction_anchor, -direction_other))),
    )


def _parallel_direction_error(
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """Sin of angle between two directions (flip-invariant)."""
    unit_a = direction_a / max(float(np.linalg.norm(direction_a)), 1.0e-12)
    unit_b = direction_b / max(float(np.linalg.norm(direction_b)), 1.0e-12)
    return min(
        float(np.linalg.norm(np.cross(unit_a, unit_b))),
        float(np.linalg.norm(np.cross(unit_a, -unit_b))),
    )


def _parallel_landmark_groups(
    parallel_pairs: list[tuple[str, str]] | None,
) -> list[list[str]]:
    """Connected components of pairwise Is-Parallel-To links."""
    adjacency: dict[str, set[str]] = {}
    for landmark_a, landmark_b in parallel_pairs or ():
        adjacency.setdefault(landmark_a, set()).add(landmark_b)
        adjacency.setdefault(landmark_b, set()).add(landmark_a)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for landmark_id in sorted(adjacency):
        if landmark_id in seen:
            continue
        queue = [landmark_id]
        seen.add(landmark_id)
        group: list[str] = []
        while queue:
            current = queue.pop()
            group.append(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        groups.append(sorted(group))
    return groups


def _orthonormal_basis_perpendicular(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit axes spanning the plane perpendicular to direction."""
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    helper = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(unit, helper))) > 0.9:
        helper = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    axis_a = np.cross(unit, helper)
    axis_a = axis_a / max(float(np.linalg.norm(axis_a)), 1.0e-12)
    axis_b = np.cross(unit, axis_a)
    axis_b = axis_b / max(float(np.linalg.norm(axis_b)), 1.0e-12)
    return axis_a, axis_b


def _fit_line_fixed_direction(
    direction: np.ndarray,
    observations: list[SyncLineObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares line with fixed direction against back-projected observation planes."""
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    planes: list[np.ndarray] = []
    for observation in observations:
        similarity = similarities.get(observation.match_id)
        match = matches.get(observation.match_id)
        if similarity is None or match is None:
            continue
        plane = _plane_from_line_observation(
            observation, match.calibration, similarity
        )
        if plane is not None:
            planes.append(plane)
    if not planes:
        return None
    axis_a, axis_b = _orthonormal_basis_perpendicular(unit)
    # p = a e1 + b e2; each plane contributes n·p + d ≈ 0.
    gram = np.zeros((2, 2), dtype=np.float64)
    rhs = np.zeros(2, dtype=np.float64)
    for plane in planes:
        normal = plane[:3]
        row = np.array(
            (float(np.dot(normal, axis_a)), float(np.dot(normal, axis_b))),
            dtype=np.float64,
        )
        gram += np.outer(row, row)
        rhs -= float(plane[3]) * row
    try:
        coefficients = np.linalg.solve(gram + 1.0e-9 * np.eye(2), rhs)
    except np.linalg.LinAlgError:
        return None
    point = coefficients[0] * axis_a + coefficients[1] * axis_b
    return point, unit


def _consensus_parallel_direction(
    directions: list[np.ndarray],
    weights: list[float],
) -> np.ndarray | None:
    """Weighted average of unit directions with flip-invariant alignment."""
    if not directions:
        return None
    reference = directions[0] / max(float(np.linalg.norm(directions[0])), 1.0e-12)
    accumulator = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for direction, weight in zip(directions, weights):
        unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        if float(np.dot(unit, reference)) < 0.0:
            unit = -unit
        scale = max(float(weight), 1.0e-6)
        accumulator += scale * unit
        total_weight += scale
    if total_weight < 1.0e-12:
        return None
    consensus = accumulator / total_weight
    norm = float(np.linalg.norm(consensus))
    if norm < 1.0e-12:
        return None
    return consensus / norm


def _enforce_parallel_line_segments(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    landmarks: dict[str, np.ndarray],
    parallel_pairs: list[tuple[str, str]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Force free line meshes in a parallel family to share one 3D direction.

    Camera pose stays frozen; Known 3D edges anchor the shared direction when
    present. This is what fixes a visually skewed free edge after a good pose.
    """
    for group in _parallel_landmark_groups(parallel_pairs):
        members = [landmark_id for landmark_id in group if landmark_id in line_segments]
        if len(members) < 2:
            continue
        directions: list[np.ndarray] = []
        weights: list[float] = []
        known_direction: np.ndarray | None = None
        for landmark_id in members:
            point_a, point_b = line_segments[landmark_id]
            direction = point_b - point_a
            if float(np.linalg.norm(direction)) < 1.0e-9:
                continue
            if landmark_id in known_lines:
                known_direction = direction
                # Known metric edges dominate the family direction.
                directions = [direction]
                weights = [1.0e6]
                break
            directions.append(direction)
            # Prefer edges that already reproject better (inverse soft RMSE proxy).
            items = line_observations_by_landmark.get(landmark_id, [])
            unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
            point = 0.5 * (point_a + point_b)
            errors: list[float] = []
            for observation in items:
                similarity = similarities.get(observation.match_id)
                match = matches.get(observation.match_id)
                if similarity is None or match is None:
                    continue
                errors.extend(
                    _line_observation_reprojection_errors(
                        point, unit, observation, match.calibration, similarity
                    )
                )
            mean_error = (
                float(np.sqrt(np.mean(np.square(errors)))) if errors else 20.0
            )
            weights.append(1.0 / max(mean_error, 1.0))
        if known_direction is not None:
            consensus = known_direction / max(
                float(np.linalg.norm(known_direction)), 1.0e-12
            )
        else:
            consensus = _consensus_parallel_direction(directions, weights)
        if consensus is None:
            continue
        for landmark_id in members:
            if landmark_id in known_lines:
                continue
            items = line_observations_by_landmark.get(landmark_id, [])
            fitted = _fit_line_fixed_direction(
                consensus, items, similarities, matches
            )
            if fitted is None:
                # Fall back: keep midpoint, swap direction only.
                point_a, point_b = line_segments[landmark_id]
                fitted = (0.5 * (point_a + point_b), consensus)
            point, direction = fitted
            segment = _finite_segment_from_line_observations(
                point, direction, items, similarities, matches
            )
            line_segments[landmark_id] = segment
            landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])


def _parallel_vp_specs_for_match_pair(
    anchor_id: str,
    other_id: str,
    parallel_pairs: list[tuple[str, str]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] | None,
) -> list[
    tuple[
        SyncLineObservation,
        SyncLineObservation,
        SyncLineObservation,
        SyncLineObservation,
    ]
]:
    """Collect parallel VP observation quads observed in both matches."""
    specs: list[
        tuple[
            SyncLineObservation,
            SyncLineObservation,
            SyncLineObservation,
            SyncLineObservation,
        ]
    ] = []
    line_by_lm = line_observations_by_landmark or {}
    for landmark_a, landmark_b in parallel_pairs or ():
        by_a = {item.match_id: item for item in line_by_lm.get(landmark_a, [])}
        by_b = {item.match_id: item for item in line_by_lm.get(landmark_b, [])}
        if (
            anchor_id in by_a
            and anchor_id in by_b
            and other_id in by_a
            and other_id in by_b
        ):
            specs.append(
                (
                    by_a[anchor_id],
                    by_b[anchor_id],
                    by_a[other_id],
                    by_b[other_id],
                )
            )
    return specs


def _estimate_rigid_from_rays(
    anchor_id: str,
    other_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
) -> SimilarityTransform:
    """Warm-start rigid sync from the essential matrix between two matches."""
    pairs: list[tuple[SyncObservation, SyncObservation]] = []
    for items in observations_by_landmark.values():
        by_match = {item.match_id: item for item in items}
        if anchor_id in by_match and other_id in by_match:
            pairs.append((by_match[anchor_id], by_match[other_id]))
    if len(pairs) < 5:
        return SimilarityTransform()

    rays_anchor = []
    rays_other = []
    for anchor_obs, other_obs in pairs:
        rays_anchor.append(
            _normalized_camera_ray(anchor_obs.u, anchor_obs.v, matches[anchor_id].calibration)
        )
        rays_other.append(
            _normalized_camera_ray(other_obs.u, other_obs.v, matches[other_id].calibration)
        )
    essential = _essential_eight_point(np.stack(rays_anchor), np.stack(rays_other))
    if essential is None:
        return SimilarityTransform()

    candidates = _decompose_essential(essential)
    anchor = matches[anchor_id].calibration
    other = matches[other_id].calibration
    best: SimilarityTransform | None = None
    best_score = -1

    for rotation_rel, translation_dir in candidates:
        # R_rel maps anchor-camera coords → other-camera coords (OpenCV x2^T E x1).
        # R_b_shared = R_rel @ R_a
        rotation_b_shared = rotation_rel @ anchor.rotation_w2c
        # R_b_shared = R_b_priv @ R_sim.T  ⇒  R_sim = R_b_priv.T? wait:
        # R_b_shared = R_priv @ R_sim.T ⇒ R_sim.T = R_priv.T @ R_b_shared
        # ⇒ R_sim = R_b_shared.T @ R_priv
        rotation_sim = rotation_b_shared.T @ other.rotation_w2c

        # t_rel ~ R_b_shared (C_a - C_b_shared); solve C_b_shared via scale search.
        for sign in (1.0, -1.0):
            direction = sign * translation_dir
            for scale in _baseline_scale_candidates(
                pairs,
                anchor,
                other,
                rotation_sim,
                rotation_b_shared,
                direction,
            ):
                center_b_shared = anchor.camera_center - scale * (
                    rotation_b_shared.T @ direction
                )
                translation_sim = center_b_shared - rotation_sim @ other.camera_center
                candidate = SimilarityTransform(
                    scale=1.0,
                    rotation=rotation_sim,
                    translation=translation_sim,
                )
                score = _cheirality_score(
                    pairs,
                    candidate,
                    matches,
                    anchor_id,
                    other_id,
                )
                if score > best_score:
                    best_score = score
                    best = candidate

    return best if best is not None else SimilarityTransform()


def _normalized_camera_ray(
    u_coordinate: float,
    v_coordinate: float,
    calibration: core.Calibration,
) -> np.ndarray:
    """Return a unit ray in the OpenCV camera frame for an image pixel."""
    ideal = core.undistort_points(
        np.array([[u_coordinate, v_coordinate]], dtype=np.float64),
        calibration.intrinsics.fx,
        calibration.intrinsics.fy,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
        calibration.division_lambda,
    )[0]
    return core.pixel_ray(float(ideal[0]), float(ideal[1]), calibration.intrinsics)


def _essential_eight_point(
    rays_a: np.ndarray,
    rays_b: np.ndarray,
) -> np.ndarray | None:
    """Estimate essential matrix from normalized camera rays (x_b^T E x_a = 0)."""
    if len(rays_a) < 5:
        return None
    design = []
    for ray_a, ray_b in zip(rays_a, rays_b):
        design.append(
            [
                ray_b[0] * ray_a[0],
                ray_b[0] * ray_a[1],
                ray_b[0] * ray_a[2],
                ray_b[1] * ray_a[0],
                ray_b[1] * ray_a[1],
                ray_b[1] * ray_a[2],
                ray_b[2] * ray_a[0],
                ray_b[2] * ray_a[1],
                ray_b[2] * ray_a[2],
            ]
        )
    design_matrix = np.asarray(design, dtype=np.float64)
    _, _, vt_matrix = np.linalg.svd(design_matrix)
    essential = vt_matrix[-1].reshape(3, 3)
    u_matrix, singular, vt_matrix = np.linalg.svd(essential)
    singular = np.array([(singular[0] + singular[1]) * 0.5, (singular[0] + singular[1]) * 0.5, 0.0])
    return u_matrix @ np.diag(singular) @ vt_matrix


def _decompose_essential(
    essential: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the four (R, t_unit) candidates from an essential matrix."""
    u_matrix, _, vt_matrix = np.linalg.svd(essential)
    if np.linalg.det(u_matrix) < 0:
        u_matrix *= -1.0
    if np.linalg.det(vt_matrix) < 0:
        vt_matrix *= -1.0
    twist = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    rotation_1 = u_matrix @ twist @ vt_matrix
    rotation_2 = u_matrix @ twist.T @ vt_matrix
    translation = u_matrix[:, 2]
    translation = translation / max(float(np.linalg.norm(translation)), 1.0e-12)
    return [
        (rotation_1, translation),
        (rotation_1, -translation),
        (rotation_2, translation),
        (rotation_2, -translation),
    ]


def _baseline_scale_candidates(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    rotation_sim: np.ndarray,
    rotation_b_shared: np.ndarray,
    translation_dir: np.ndarray,
) -> list[float]:
    """Propose baseline lengths from triangulating the first few pairs."""
    scales: list[float] = []
    for anchor_obs, other_obs in pairs[:4]:
        # Build shared rays for a hypothesized unit baseline, then rescale.
        # C_b = C_a - α R_b^T t_dir; use α=1 to get a direction, then fix depth.
        center_b = anchor.camera_center - (rotation_b_shared.T @ translation_dir)
        translation_sim = center_b - rotation_sim @ other.camera_center
        trial = SimilarityTransform(
            scale=1.0,
            rotation=rotation_sim,
            translation=translation_sim,
        )
        origin_a, direction_a = camera_ray_private(
            anchor_obs.u,
            anchor_obs.v,
            anchor,
        )
        origin_b_priv, direction_b_priv = camera_ray_private(
            other_obs.u,
            other_obs.v,
            other,
        )
        origin_b = trial.transform_point(origin_b_priv)
        direction_b = trial.rotation @ direction_b_priv
        point = triangulate_midpoint([origin_a, origin_b], [direction_a, direction_b])
        if point is None:
            continue
        # Recover α such that C_b(α) and triangulation stay consistent-ish:
        # distance from anchor camera to point projected on baseline axis.
        offset = point - anchor.camera_center
        axis = rotation_b_shared.T @ translation_dir
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1.0e-10:
            continue
        # Fallback geometric baseline: use distance between cameras from a
        # depth estimate along the anchor ray.
        depth = float(np.dot(offset, direction_a))
        if depth > 0.2:
            scales.append(depth)
    if not scales:
        return [1.0, 2.0, 5.0]
    median = float(np.median(scales))
    return [median * factor for factor in (0.35, 0.7, 1.0, 1.5, 2.5)]


def _cheirality_score(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    candidate: SimilarityTransform,
    matches: dict[str, SyncMatchInput],
    anchor_id: str,
    other_id: str,
) -> int:
    """Count landmarks that triangulate in front of both cameras."""
    score = 0
    for anchor_obs, other_obs in pairs:
        origin_a, direction_a = camera_ray_private(
            anchor_obs.u,
            anchor_obs.v,
            matches[anchor_id].calibration,
        )
        origin_b_priv, direction_b_priv = camera_ray_private(
            other_obs.u,
            other_obs.v,
            matches[other_id].calibration,
        )
        origin_b = candidate.transform_point(origin_b_priv)
        direction_b = candidate.rotation @ direction_b_priv
        point = triangulate_midpoint([origin_a, origin_b], [direction_a, direction_b])
        if point is None:
            continue
        if float(np.dot(point - origin_a, direction_a)) <= 0.0:
            continue
        if float(np.dot(point - origin_b, direction_b)) <= 0.0:
            continue
        # Also require positive depth in each camera frame.
        private_a = point  # anchor Sim = I
        private_b = candidate.inverse_point(point)
        cam_a = matches[anchor_id].calibration.rotation_w2c @ (
            private_a - matches[anchor_id].calibration.camera_center
        )
        cam_b = matches[other_id].calibration.rotation_w2c @ (
            private_b - matches[other_id].calibration.camera_center
        )
        if float(cam_a[2]) > 1.0e-6 and float(cam_b[2]) > 1.0e-6:
            score += 1
    return score


def _umeyama(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool = True,
) -> SimilarityTransform | None:
    """Similarity (or rigid) mapping source points onto target points."""
    if len(source) < 2 or len(source) != len(target):
        return None
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centered_source = source - mean_source
    centered_target = target - mean_target
    variance_source = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    if variance_source < 1.0e-12:
        return None
    covariance = (centered_source.T @ centered_target) / float(len(source))
    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    reflection = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0.0:
        reflection[-1, -1] = -1.0
    rotation = vt_matrix.T @ reflection @ u_matrix.T
    if with_scale:
        scale = float(np.sum(singular * np.diag(reflection)) / variance_source)
        if scale <= 1.0e-8:
            return None
    else:
        scale = 1.0
    translation = mean_target - scale * (rotation @ mean_source)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def _kabsch_rigid(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Best rigid transform mapping source points onto target points."""
    result = _umeyama(source, target, with_scale=False)
    return result if result is not None else SimilarityTransform()


def _similarity_from_camera_pose(
    calibration: core.Calibration,
    rotation_w2c_shared: np.ndarray,
    center_shared: np.ndarray,
) -> SimilarityTransform:
    """Build Empty similarity (s=1) that realizes a shared camera pose."""
    # R_w2c_shared = R_priv @ R_sim.T  ⇒  R_sim = R_w2c_shared.T @ R_priv
    rotation_sim = rotation_w2c_shared.T @ calibration.rotation_w2c
    # Orthonormalize in case of numeric drift.
    u_matrix, _, vt_matrix = np.linalg.svd(rotation_sim)
    rotation_sim = u_matrix @ vt_matrix
    if np.linalg.det(rotation_sim) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation_sim = u_matrix @ vt_matrix
    translation = center_shared - rotation_sim @ calibration.camera_center
    return SimilarityTransform(scale=1.0, rotation=rotation_sim, translation=translation)


def _pnp_similarity(
    match_id: str,
    points_shared: np.ndarray,
    points_image: np.ndarray,
    calibration: core.Calibration,
    *,
    initial: SimilarityTransform | None = None,
    max_iterations: int = 40,
    weights: np.ndarray | None = None,
) -> SimilarityTransform | None:
    """Solve Empty rigid transform (s=1) from shared 3D ↔ image 2D correspondences."""
    if len(points_shared) < 3:
        return None
    seed = initial or SimilarityTransform()
    params = np.concatenate(
        [_log_rodrigues(seed.rotation), seed.translation]
    )
    if weights is None:
        point_weights = np.ones(len(points_shared), dtype=np.float64)
    else:
        point_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if point_weights.size != len(points_shared):
            point_weights = np.ones(len(points_shared), dtype=np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues(values[:3]),
            translation=values[3:6].copy(),
        )
        errors: list[float] = []
        for point_shared, image_point, weight in zip(
            points_shared, points_image, point_weights
        ):
            scale = float(np.sqrt(max(float(weight), 1.0e-12)))
            point_private = similarity.inverse_point(point_shared)
            projected = project_private_point(point_private, calibration)
            if projected is None:
                errors.extend((scale * 1.0e3, scale * 1.0e3))
                continue
            errors.append(scale * float(projected[0] - image_point[0]))
            errors.append(scale * float(projected[1] - image_point[1]))
        return np.asarray(errors, dtype=np.float64)

    damping = 1.0e-3
    previous_cost = float("inf")
    for _iteration in range(max_iterations):
        residuals = residual(params)
        cost = float(residuals @ residuals)
        if cost < 1.0e-8:
            break
        if abs(previous_cost - cost) / max(previous_cost, 1.0e-12) < 1.0e-9:
            break
        jacobian = np.zeros((residuals.size, params.size), dtype=np.float64)
        for index in range(params.size):
            perturbed = params.copy()
            step = 1.0e-5
            perturbed[index] += step
            jacobian[:, index] = (residual(perturbed) - residuals) / step
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        improved = False
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
            candidate_cost = float(residual(candidate) @ residual(candidate))
            if candidate_cost < cost:
                params = candidate
                previous_cost = cost
                damping = max(damping * 0.3, 1.0e-8)
                improved = True
                break
            damping *= 10.0
        if not improved:
            break

    return SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues(params[:3]),
        translation=params[3:6].copy(),
    )


def _skew_lines_distance(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """Distance between two skew lines (0 when they intersect)."""
    normal = np.cross(direction_a, direction_b)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-10:
        offset = origin_a - origin_b
        return float(np.linalg.norm(np.cross(offset, direction_b)))
    return abs(float(np.dot(origin_a - origin_b, normal))) / normal_norm


def _collect_pnp_correspondences(
    match_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    landmarks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Shared 3D ↔ image 2D pairs for PnP on one match."""
    points_shared: list[np.ndarray] = []
    points_image: list[np.ndarray] = []
    for landmark_id, items in observations_by_landmark.items():
        if landmark_id not in landmarks:
            continue
        for observation in items:
            if observation.match_id != match_id:
                continue
            points_shared.append(landmarks[landmark_id])
            points_image.append(
                np.array((observation.u, observation.v), dtype=np.float64)
            )
    if len(points_shared) < 3:
        return None
    return np.stack(points_shared), np.stack(points_image)


def _anchor_ground_landmarks(
    observations_by_landmark: dict[str, list[SyncObservation]],
    anchor_id: str,
    anchor: core.Calibration,
) -> dict[str, np.ndarray]:
    """Recover metric 3D for On Ground landmarks via ray∩Z=0 in the anchor."""
    landmarks: dict[str, np.ndarray] = {}
    for landmark_id, items in observations_by_landmark.items():
        by_match = {item.match_id: item for item in items}
        observation = by_match.get(anchor_id)
        if observation is None or not observation.on_ground:
            continue
        hit = core.ground_hit((observation.u, observation.v), anchor)
        if hit is not None:
            landmarks[landmark_id] = hit
    return landmarks


def _metric_landmarks(
    observations_by_landmark: dict[str, list[SyncObservation]],
    anchor_id: str,
    anchor: core.Calibration,
    known_world: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Fixed shared-world 3D from known objects and/or On Ground raycasts."""
    landmarks = dict(known_world or {})
    # Ground fills gaps; known Empties win when both are set.
    for landmark_id, point in _anchor_ground_landmarks(
        observations_by_landmark, anchor_id, anchor
    ).items():
        landmarks.setdefault(landmark_id, point)
    return landmarks


def _metric_pnp_correspondences(
    other_id: str,
    metric: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[float]]:
    """Shared 3D ↔ other-image 2D for metric landmarks."""
    points_shared: list[np.ndarray] = []
    points_image: list[np.ndarray] = []
    landmark_ids: list[str] = []
    weights: list[float] = []
    for landmark_id, point in metric.items():
        items = observations_by_landmark.get(landmark_id)
        if not items:
            continue
        by_match = {item.match_id: item for item in items}
        observation = by_match.get(other_id)
        if observation is None:
            continue
        points_shared.append(point)
        points_image.append(
            np.array((observation.u, observation.v), dtype=np.float64)
        )
        landmark_ids.append(landmark_id)
        weights.append(float(observation.weight))
    return points_shared, points_image, landmark_ids, weights


def _points_collinear_3d(
    points: list[np.ndarray],
    *,
    relative_tolerance: float = 0.02,
) -> bool:
    """Return True when 3D control points lie nearly on one line."""
    if len(points) < 3:
        return True
    stacked = np.stack(points)
    centered = stacked - stacked.mean(axis=0)
    _u_matrix, singular, _vt_matrix = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2:
        return True
    span = float(singular[0])
    if span < 1.0e-8:
        return True
    return float(singular[1]) < relative_tolerance * span


def _refine_rigid_from_rays(
    seed: SimilarityTransform,
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    *,
    max_iterations: int = 60,
) -> SimilarityTransform:
    """Levenberg–Marquardt on Empty (R, t) minimizing ray–ray distances."""
    params = np.concatenate([_log_rodrigues(seed.rotation), seed.translation])

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues(values[:3]),
            translation=values[3:6].copy(),
        )
        errors: list[float] = []
        for anchor_obs, other_obs in pairs:
            scale = _pair_scale(anchor_obs, other_obs)
            origin_a, direction_a = camera_ray_private(
                anchor_obs.u, anchor_obs.v, anchor
            )
            origin_b_priv, direction_b_priv = camera_ray_private(
                other_obs.u, other_obs.v, other
            )
            origin_b = similarity.transform_point(origin_b_priv)
            direction_b = similarity.rotation @ direction_b_priv
            direction_b = direction_b / max(
                float(np.linalg.norm(direction_b)), 1.0e-12
            )
            errors.append(
                scale
                * _skew_lines_distance(origin_a, direction_a, origin_b, direction_b)
            )
        return np.asarray(errors, dtype=np.float64)

    damping = 1.0e-2
    previous_cost = float("inf")
    for _iteration in range(max_iterations):
        residuals = residual(params)
        cost = float(residuals @ residuals)
        if cost < 1.0e-14:
            break
        if abs(previous_cost - cost) / max(previous_cost, 1.0e-12) < 1.0e-10:
            break
        jacobian = np.zeros((residuals.size, params.size), dtype=np.float64)
        for index in range(params.size):
            perturbed = params.copy()
            step = 1.0e-5
            perturbed[index] += step
            jacobian[:, index] = (residual(perturbed) - residuals) / step
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        improved = False
        for _attempt in range(10):
            try:
                delta = np.linalg.solve(
                    gram + damping * np.diag(np.maximum(np.diag(gram), 1.0e-8)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = params + delta
            candidate_cost = float(residual(candidate) @ residual(candidate))
            if candidate_cost < cost:
                params = candidate
                previous_cost = cost
                damping = max(damping * 0.3, 1.0e-8)
                improved = True
                break
            damping *= 10.0
        if not improved:
            break

    return SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues(params[:3]),
        translation=params[3:6].copy(),
    )


def _apply_depth_heuristic_scale(
    similarity: SimilarityTransform,
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
) -> SimilarityTransform:
    """Scale baseline around the anchor so median depth ≈ ||C_anchor||."""
    depths: list[float] = []
    for anchor_obs, other_obs in pairs:
        origin_a, direction_a = camera_ray_private(
            anchor_obs.u, anchor_obs.v, anchor
        )
        origin_b_priv, direction_b_priv = camera_ray_private(
            other_obs.u, other_obs.v, other
        )
        origin_b = similarity.transform_point(origin_b_priv)
        direction_b = similarity.rotation @ direction_b_priv
        point = triangulate_midpoint(
            [origin_a, origin_b],
            [direction_a, direction_b],
            [float(anchor_obs.weight), float(other_obs.weight)],
        )
        if point is None:
            continue
        depth = float((anchor.rotation_w2c @ (point - anchor.camera_center))[2])
        if depth > 1.0e-6:
            depths.append(depth)
    if not depths:
        return similarity
    target = float(np.linalg.norm(anchor.camera_center))
    if target < 0.2:
        target = 5.0
    median_depth = float(np.median(depths))
    if median_depth < 1.0e-6:
        return similarity
    factor = target / median_depth
    center_b = similarity.transform_point(other.camera_center)
    new_center = anchor.camera_center + factor * (center_b - anchor.camera_center)
    translation = new_center - similarity.rotation @ other.camera_center
    return SimilarityTransform(
        scale=1.0,
        rotation=similarity.rotation,
        translation=translation,
    )


def _image_points_collinear(
    points: np.ndarray,
    *,
    min_cross_spread_px: float = 8.0,
) -> bool:
    """Return True when 2D picks lack a second principal direction (nearly a line)."""
    if len(points) < 3:
        return True
    centered = points - points.mean(axis=0)
    # 2×2 covariance via SVD of the centered Nx2 matrix.
    _u_matrix, singular, _vt_matrix = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2:
        return True
    return float(singular[1]) < min_cross_spread_px


def _correspondence_geometry_issue(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    other_id: str,
) -> str | None:
    """Describe degenerate 2D layouts that make relative pose unsolvable."""
    points_anchor = np.array(
        [(anchor_obs.u, anchor_obs.v) for anchor_obs, _other in pairs],
        dtype=np.float64,
    )
    points_other = np.array(
        [(other_obs.u, other_obs.v) for _anchor, other_obs in pairs],
        dtype=np.float64,
    )
    if _image_points_collinear(points_anchor):
        return (
            f"Landmarks are nearly collinear in the anchor image — "
            f"spread picks in 2D (not along one line) before syncing {other_id}"
        )
    if _image_points_collinear(points_other):
        return (
            f"Landmarks are nearly collinear in '{other_id}' — "
            "spread picks in 2D (not along one line)"
        )
    return None


def _landmark_names(
    observations_by_landmark: dict[str, list[SyncObservation]],
) -> dict[str, str]:
    """Prefer user-facing landmark names for diagnostics."""
    names: dict[str, str] = {}
    for landmark_id, items in observations_by_landmark.items():
        for item in items:
            if item.landmark_name:
                names[landmark_id] = item.landmark_name
                break
        names.setdefault(landmark_id, landmark_id[:8])
    return names


def _format_worst_landmarks(
    per_landmark_rmse: dict[str, float],
    names: dict[str, str],
    *,
    limit: int = 3,
) -> str:
    """Compact 'Worst: A 80px, B 40px' suffix for status strings."""
    if not per_landmark_rmse:
        return ""
    ranked = sorted(per_landmark_rmse.items(), key=lambda item: -item[1])[:limit]
    bits = [
        f"{names.get(landmark_id, landmark_id[:8])} {rmse:.0f}px"
        for landmark_id, rmse in ranked
    ]
    return "Worst: " + ", ".join(bits)


def _per_landmark_rmse_for_similarity(
    similarity: SimilarityTransform,
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    points_shared: list[np.ndarray],
    points_image: list[np.ndarray],
    known_landmark_ids: list[str] | None = None,
) -> dict[str, float]:
    """Mean reprojection RMSE per landmark for a candidate Empty pose."""
    per_sse: dict[str, list[float]] = {}
    for anchor_obs, other_obs in pairs:
        landmark_id = anchor_obs.landmark_id
        origin_a, direction_a = camera_ray_private(
            anchor_obs.u, anchor_obs.v, anchor
        )
        origin_b_priv, direction_b_priv = camera_ray_private(
            other_obs.u, other_obs.v, other
        )
        origin_b = similarity.transform_point(origin_b_priv)
        direction_b = similarity.rotation @ direction_b_priv
        point = triangulate_midpoint(
            [origin_a, origin_b],
            [direction_a, direction_b],
            [float(anchor_obs.weight), float(other_obs.weight)],
        )
        if point is None:
            per_sse.setdefault(landmark_id, []).extend((1.0e6, 1.0e6))
            continue
        projected_a = project_private_point(point, anchor)
        projected_b = project_private_point(
            similarity.inverse_point(point), other
        )
        if projected_a is None or projected_b is None:
            per_sse.setdefault(landmark_id, []).extend((1.0e6, 1.0e6))
            continue
        err_a = float(
            np.hypot(projected_a[0] - anchor_obs.u, projected_a[1] - anchor_obs.v)
        )
        err_b = float(
            np.hypot(projected_b[0] - other_obs.u, projected_b[1] - other_obs.v)
        )
        per_sse.setdefault(landmark_id, []).extend((err_a * err_a, err_b * err_b))

    known_ids = known_landmark_ids or []
    for index, (point_shared, image_point) in enumerate(
        zip(points_shared, points_image)
    ):
        landmark_id = (
            known_ids[index] if index < len(known_ids) else f"known{index}"
        )
        projected = project_private_point(
            similarity.inverse_point(point_shared), other
        )
        if projected is None:
            per_sse.setdefault(landmark_id, []).append(1.0e6)
            continue
        err = float(
            np.hypot(projected[0] - image_point[0], projected[1] - image_point[1])
        )
        per_sse.setdefault(landmark_id, []).append(err * err)

    return {
        landmark_id: float(np.sqrt(np.mean(values)))
        for landmark_id, values in per_sse.items()
        if values
    }


def _reprojection_errors_for_similarity(
    similarity: SimilarityTransform,
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    points_shared: list[np.ndarray],
    points_image: list[np.ndarray],
    *,
    point_weights: list[float] | None = None,
    weighted: bool = False,
) -> list[float]:
    """Pixel errors for 2D↔2D triangulations and known-3D projections.

    When ``weighted``, each error is scaled by sqrt(weight) for candidate scoring.
    """
    errors: list[float] = []
    if point_weights is None:
        known_weights = [1.0] * len(points_shared)
    else:
        known_weights = list(point_weights)
        if len(known_weights) != len(points_shared):
            known_weights = [1.0] * len(points_shared)
    for anchor_obs, other_obs in pairs:
        origin_a, direction_a = camera_ray_private(
            anchor_obs.u, anchor_obs.v, anchor
        )
        origin_b_priv, direction_b_priv = camera_ray_private(
            other_obs.u, other_obs.v, other
        )
        origin_b = similarity.transform_point(origin_b_priv)
        direction_b = similarity.rotation @ direction_b_priv
        point = triangulate_midpoint(
            [origin_a, origin_b],
            [direction_a, direction_b],
            [float(anchor_obs.weight), float(other_obs.weight)],
        )
        pair_scale = _pair_scale(anchor_obs, other_obs) if weighted else 1.0
        if point is None:
            errors.extend((pair_scale * 1.0e3, pair_scale * 1.0e3))
            continue
        projected_a = project_private_point(point, anchor)
        projected_b = project_private_point(
            similarity.inverse_point(point), other
        )
        if projected_a is None or projected_b is None:
            errors.extend((pair_scale * 1.0e3, pair_scale * 1.0e3))
            continue
        scale_a = _observation_scale(anchor_obs) if weighted else 1.0
        scale_b = _observation_scale(other_obs) if weighted else 1.0
        errors.append(
            scale_a
            * float(np.hypot(projected_a[0] - anchor_obs.u, projected_a[1] - anchor_obs.v))
        )
        errors.append(
            scale_b
            * float(np.hypot(projected_b[0] - other_obs.u, projected_b[1] - other_obs.v))
        )
    for point_shared, image_point, weight in zip(
        points_shared, points_image, known_weights
    ):
        scale = float(np.sqrt(max(float(weight), 1.0e-12))) if weighted else 1.0
        projected = project_private_point(
            similarity.inverse_point(point_shared), other
        )
        if projected is None:
            errors.append(scale * 1.0e3)
            continue
        errors.append(
            scale
            * float(np.hypot(projected[0] - image_point[0], projected[1] - image_point[1]))
        )
    return errors


def _solve_relative_from_pairs(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
) -> SimilarityTransform | None:
    """Multi-start ray-distance LM for Empty (R, t) from 2D↔2D pairs only."""
    if len(pairs) < 5:
        return None
    seeds: list[SimilarityTransform] = [
        SimilarityTransform(),
        SimilarityTransform(
            scale=1.0,
            rotation=np.eye(3),
            translation=anchor.camera_center - other.camera_center,
        ),
    ]
    if len(pairs) >= 8:
        rays_a = np.stack(
            [_normalized_camera_ray(a.u, a.v, anchor) for a, _b in pairs]
        )
        rays_b = np.stack(
            [_normalized_camera_ray(b.u, b.v, other) for _a, b in pairs]
        )
        essential = _essential_eight_point(rays_a, rays_b)
        if essential is not None:
            for rotation_rel, translation_hat in _decompose_essential(essential):
                rotation_b = rotation_rel @ anchor.rotation_w2c
                rotation_sim = rotation_b.T @ other.rotation_w2c
                u_matrix, _, vt_matrix = np.linalg.svd(rotation_sim)
                rotation_sim = u_matrix @ vt_matrix
                if np.linalg.det(rotation_sim) < 0.0:
                    u_matrix[:, -1] *= -1.0
                    rotation_sim = u_matrix @ vt_matrix
                baseline = float(np.linalg.norm(anchor.camera_center)) or 5.0
                center_b = anchor.camera_center - baseline * (
                    rotation_b.T @ translation_hat
                )
                translation = center_b - rotation_sim @ other.camera_center
                seeds.append(
                    SimilarityTransform(
                        scale=1.0,
                        rotation=rotation_sim,
                        translation=translation,
                    )
                )
    for yaw in (-0.6, -0.3, 0.3, 0.6, 0.9, -0.9):
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        rotation_yaw = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        seeds.append(
            SimilarityTransform(
                scale=1.0,
                rotation=rotation_yaw,
                translation=anchor.camera_center
                - rotation_yaw @ other.camera_center,
            )
        )

    best: SimilarityTransform | None = None
    best_key: tuple[float, float, float] | None = None
    for seed in seeds:
        refined = _refine_rigid_from_rays(seed, pairs, anchor, other)
        center_b = refined.transform_point(other.camera_center)
        baseline = float(np.linalg.norm(center_b - anchor.camera_center))
        if baseline < 0.2:
            continue
        cost = 0.0
        cheirality = 0
        reprojection_errors: list[float] = []
        for anchor_obs, other_obs in pairs:
            origin_a, direction_a = camera_ray_private(
                anchor_obs.u, anchor_obs.v, anchor
            )
            origin_b_priv, direction_b_priv = camera_ray_private(
                other_obs.u, other_obs.v, other
            )
            origin_b = refined.transform_point(origin_b_priv)
            direction_b = refined.rotation @ direction_b_priv
            direction_b = direction_b / max(
                float(np.linalg.norm(direction_b)), 1.0e-12
            )
            cost += _skew_lines_distance(
                origin_a, direction_a, origin_b, direction_b
            ) ** 2
            point = triangulate_midpoint(
                [origin_a, origin_b],
                [direction_a, direction_b],
            )
            if point is None:
                continue
            depth_a = float(np.dot(point - origin_a, direction_a))
            depth_b = float(np.dot(point - origin_b, direction_b))
            if depth_a <= 0.2 or depth_b <= 0.2:
                continue
            cheirality += 1
            projected_a = project_private_point(point, anchor)
            projected_b = project_private_point(
                refined.inverse_point(point), other
            )
            if projected_a is None or projected_b is None:
                continue
            reprojection_errors.append(
                float(np.hypot(projected_a[0] - anchor_obs.u, projected_a[1] - anchor_obs.v))
            )
            reprojection_errors.append(
                float(np.hypot(projected_b[0] - other_obs.u, projected_b[1] - other_obs.v))
            )
        if cheirality < max(4, (len(pairs) * 3) // 4):
            continue
        if not reprojection_errors:
            continue
        mean_reproj = float(np.mean(reprojection_errors))
        if mean_reproj > 15.0:
            continue
        key = (-float(cheirality), mean_reproj, cost)
        if best_key is None or key < best_key:
            best_key = key
            best = refined
    return best


def _apply_known_baseline_scale(
    similarity: SimilarityTransform,
    points_shared: list[np.ndarray],
    points_image: list[np.ndarray],
    other: core.Calibration,
    anchor: core.Calibration,
) -> SimilarityTransform:
    """Keep R fixed; search baseline length so Known 3D reproject in the other still."""
    if not points_shared:
        return similarity
    rotation = similarity.rotation
    center_old = similarity.transform_point(other.camera_center)
    offset = center_old - anchor.camera_center
    offset_norm = float(np.linalg.norm(offset))
    if offset_norm < 1.0e-8:
        return similarity
    direction = offset / offset_norm

    best_alpha = offset_norm
    best_cost = float("inf")
    for factor in np.linspace(0.2, 5.0, 49):
        alpha = offset_norm * float(factor)
        center_b = anchor.camera_center + alpha * direction
        translation = center_b - rotation @ other.camera_center
        trial = SimilarityTransform(
            scale=1.0, rotation=rotation, translation=translation
        )
        errors: list[float] = []
        for point_shared, image_point in zip(points_shared, points_image):
            projected = project_private_point(
                trial.inverse_point(point_shared), other
            )
            if projected is None:
                errors.append(1.0e3)
                continue
            errors.append(
                float(
                    np.hypot(
                        projected[0] - image_point[0],
                        projected[1] - image_point[1],
                    )
                )
            )
        cost = float(np.mean(np.square(errors)))
        if cost < best_cost:
            best_cost = cost
            best_alpha = alpha

    center_b = anchor.camera_center + best_alpha * direction
    return SimilarityTransform(
        scale=1.0,
        rotation=rotation,
        translation=center_b - rotation @ other.camera_center,
    )


def _refine_rigid_mixed(
    seed: SimilarityTransform,
    free_pairs: list[tuple[SyncObservation, SyncObservation]],
    points_shared: list[np.ndarray],
    points_image: list[np.ndarray],
    anchor: core.Calibration,
    other: core.Calibration,
    *,
    max_iterations: int = 80,
    point_weights: list[float] | None = None,
    known_line_constraints: list[
        tuple[np.ndarray, np.ndarray, SyncLineObservation]
    ]
    | None = None,
    parallel_vp_constraints: list[
        tuple[
            SyncLineObservation,
            SyncLineObservation,
            SyncLineObservation,
            SyncLineObservation,
        ]
    ]
    | None = None,
    parallel_weight: float = 0.0,
) -> SimilarityTransform:
    """LM on Empty (R,t): free 2D↔2D ray gaps + Known 3D point/line reprojection."""
    params = np.concatenate([_log_rodrigues(seed.rotation), seed.translation])
    # Ray distance (scene units) → rough pixels using focal length.
    ray_to_px = float(max(other.intrinsics.fx, other.intrinsics.fy)) / 5.0
    if point_weights is None:
        known_weights = [1.0] * len(points_shared)
    else:
        known_weights = list(point_weights)
        if len(known_weights) != len(points_shared):
            known_weights = [1.0] * len(points_shared)
    line_constraints = known_line_constraints or []
    parallel_constraints = parallel_vp_constraints or []

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = SimilarityTransform(
            scale=1.0,
            rotation=_rodrigues(values[:3]),
            translation=values[3:6].copy(),
        )
        errors: list[float] = []
        for anchor_obs, other_obs in free_pairs:
            scale = _pair_scale(anchor_obs, other_obs)
            origin_a, direction_a = camera_ray_private(
                anchor_obs.u, anchor_obs.v, anchor
            )
            origin_b_priv, direction_b_priv = camera_ray_private(
                other_obs.u, other_obs.v, other
            )
            origin_b = similarity.transform_point(origin_b_priv)
            direction_b = similarity.rotation @ direction_b_priv
            direction_b = direction_b / max(
                float(np.linalg.norm(direction_b)), 1.0e-12
            )
            errors.append(
                scale
                * ray_to_px
                * _skew_lines_distance(origin_a, direction_a, origin_b, direction_b)
            )
        for point_shared, image_point, weight in zip(
            points_shared, points_image, known_weights
        ):
            scale = float(np.sqrt(max(float(weight), 1.0e-12)))
            projected = project_private_point(
                similarity.inverse_point(point_shared), other
            )
            if projected is None:
                errors.extend((scale * 1.0e3, scale * 1.0e3))
                continue
            errors.append(scale * float(projected[0] - image_point[0]))
            errors.append(scale * float(projected[1] - image_point[1]))
        for point_a, point_b, line_obs in line_constraints:
            errors.extend(
                _known_line_reprojection_errors(
                    point_a, point_b, line_obs, other, similarity
                )
            )
        # Soft parallel: weight is ~pixels at sin(angle)=1; keep << typical misfits.
        if parallel_weight > 0.0:
            for (
                obs_a_anchor,
                obs_b_anchor,
                obs_a_other,
                obs_b_other,
            ) in parallel_constraints:
                parallel_error = _parallel_pair_rotation_error(
                    obs_a_anchor,
                    obs_b_anchor,
                    obs_a_other,
                    obs_b_other,
                    anchor,
                    other,
                    similarity,
                )
                if parallel_error is not None:
                    errors.append(float(parallel_weight) * parallel_error)
        return np.asarray(errors, dtype=np.float64)

    damping = 1.0e-2
    previous_cost = float("inf")
    for _iteration in range(max_iterations):
        residuals = residual(params)
        if residuals.size == 0:
            break
        cost = float(residuals @ residuals)
        if cost < 1.0e-10:
            break
        if abs(previous_cost - cost) / max(previous_cost, 1.0e-12) < 1.0e-9:
            break
        jacobian = np.zeros((residuals.size, params.size), dtype=np.float64)
        for index in range(params.size):
            perturbed = params.copy()
            step = 1.0e-5
            perturbed[index] += step
            jacobian[:, index] = (residual(perturbed) - residuals) / step
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        improved = False
        for _attempt in range(10):
            try:
                delta = np.linalg.solve(
                    gram + damping * np.diag(np.maximum(np.diag(gram), 1.0e-8)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = params + delta
            candidate_cost = float(residual(candidate) @ residual(candidate))
            if candidate_cost < cost:
                params = candidate
                previous_cost = cost
                damping = max(damping * 0.3, 1.0e-8)
                improved = True
                break
            damping *= 10.0
        if not improved:
            break

    return SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues(params[:3]),
        translation=params[3:6].copy(),
    )



def _mixed_pose_seeds(
    anchor: core.Calibration,
    other: core.Calibration,
) -> list[SimilarityTransform]:
    """Yaw / identity seeds for joint free-pair + Known 3D registration."""
    seeds: list[SimilarityTransform] = [
        SimilarityTransform(),
        SimilarityTransform(
            scale=1.0,
            rotation=np.eye(3),
            translation=anchor.camera_center - other.camera_center,
        ),
    ]
    for yaw in (-1.2, -0.8, -0.4, 0.4, 0.8, 1.2):
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        rotation_yaw = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        seeds.append(
            SimilarityTransform(
                scale=1.0,
                rotation=rotation_yaw,
                translation=anchor.camera_center
                - rotation_yaw @ other.camera_center,
            )
        )
    return seeds


def _relative_pose_from_correspondences(
    anchor_id: str,
    other_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    known_world: dict[str, np.ndarray] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]]
    | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
) -> tuple[SimilarityTransform | None, str]:
    """Register other from free 2D↔2D and/or Known 3D points/lines.

    Known 3D landmarks are *not* used as photo↔photo pairs (auto-projected
    anchor picks can disagree with the photo feature). They only provide metric
    3D↔2D in the other still. Free point landmarks provide 2D↔2D. Known 3D
    lines use 2D segments in the other still. Free 2D↔2D lines need ≥3 stills
    (handled after two matches are registered).
    """
    known_ids = set(known_world or {})
    free_pairs: list[tuple[SyncObservation, SyncObservation]] = []
    for landmark_id, items in observations_by_landmark.items():
        by_match = {item.match_id: item for item in items}
        if anchor_id not in by_match or other_id not in by_match:
            continue
        pair = (by_match[anchor_id], by_match[other_id])
        if landmark_id not in known_ids:
            free_pairs.append(pair)

    anchor = matches[anchor_id].calibration
    other = matches[other_id].calibration
    metric = _metric_landmarks(
        observations_by_landmark, anchor_id, anchor, known_world
    )
    points_shared, points_image, known_landmark_ids, known_weights = (
        _metric_pnp_correspondences(other_id, metric, observations_by_landmark)
    )
    known_line_constraints: list[
        tuple[np.ndarray, np.ndarray, SyncLineObservation]
    ] = []
    for landmark_id, (point_a, point_b) in (known_lines or {}).items():
        items = (line_observations_by_landmark or {}).get(landmark_id, [])
        by_match = {item.match_id: item for item in items}
        observation = by_match.get(other_id)
        if observation is None:
            continue
        known_line_constraints.append(
            (
                np.asarray(point_a, dtype=np.float64).reshape(3),
                np.asarray(point_b, dtype=np.float64).reshape(3),
                observation,
            )
        )
    metric_collinear = (
        len(points_shared) >= 3 and _points_collinear_3d(points_shared)
    )
    has_metric_pnp = len(points_shared) >= 3 and not metric_collinear
    has_known_lines = len(known_line_constraints) >= 1
    has_pnl = len(known_line_constraints) >= 3
    free_geometry_issue = (
        _correspondence_geometry_issue(free_pairs, other_id)
        if len(free_pairs) >= 5
        else None
    )
    free_pairs_ok = len(free_pairs) >= 5 and free_geometry_issue is None

    can_pairs_only = free_pairs_ok
    can_joint = len(free_pairs) >= 2 and (
        len(points_shared) >= 1 or has_known_lines
    )
    can_pnp_only = has_metric_pnp
    can_pnl_only = has_pnl

    if not (can_pairs_only or can_joint or can_pnp_only or can_pnl_only):
        if free_geometry_issue:
            return None, free_geometry_issue
        if len(free_pairs) < 2 and metric_collinear and not has_known_lines:
            return (
                None,
                f"Not enough constraints for '{other_id}'. Known 3D points lie on "
                "one line — add ordinary 2D picks, off-line Known 3D, or ≥3 "
                "Known 3D line edges",
            )
        if (
            len(free_pairs) < 5
            and len(points_shared) == 0
            and not has_known_lines
        ):
            return (
                None,
                f"'{other_id}' needs ≥5 point landmarks in both stills "
                f"(currently {len(free_pairs)}), or Known 3D lines / points",
            )
        if len(free_pairs) < 2 and not can_pnl_only:
            return (
                None,
                f"'{other_id}' needs ≥2 ordinary point landmarks in both stills "
                "plus Known 3D, or ≥3 Known 3D line edges drawn in this still",
            )
        return (
            None,
            f"Could not register '{other_id}' — add spread-out 2D point picks, "
            "Known 3D points, or ≥3 Known 3D line landmarks",
        )

    candidates: list[SimilarityTransform] = []
    mixed_kwargs = {
        "point_weights": known_weights,
        "known_line_constraints": known_line_constraints,
    }

    if can_pairs_only:
        relative = _solve_relative_from_pairs(free_pairs, anchor, other)
        if relative is not None:
            if points_shared or known_line_constraints:
                if points_shared:
                    candidates.append(
                        _apply_known_baseline_scale(
                            relative, points_shared, points_image, other, anchor
                        )
                    )
                else:
                    candidates.append(relative)
                if has_metric_pnp:
                    scaled = _pnp_similarity(
                        other_id,
                        np.stack(points_shared),
                        np.stack(points_image),
                        other,
                        initial=relative,
                        weights=np.asarray(known_weights, dtype=np.float64),
                    )
                    if scaled is not None:
                        candidates.append(
                            _refine_rigid_mixed(
                                scaled,
                                free_pairs,
                                points_shared,
                                points_image,
                                anchor,
                                other,
                                **mixed_kwargs,
                            )
                        )
                if known_line_constraints:
                    candidates.append(
                        _refine_rigid_mixed(
                            relative,
                            free_pairs,
                            points_shared,
                            points_image,
                            anchor,
                            other,
                            **mixed_kwargs,
                        )
                    )
            else:
                candidates.append(
                    _apply_depth_heuristic_scale(relative, free_pairs, anchor, other)
                )

    if can_joint:
        for seed in _mixed_pose_seeds(anchor, other):
            candidates.append(
                _refine_rigid_mixed(
                    seed,
                    free_pairs,
                    points_shared,
                    points_image,
                    anchor,
                    other,
                    **mixed_kwargs,
                )
            )
        if can_pairs_only:
            relative = _solve_relative_from_pairs(free_pairs, anchor, other)
            if relative is not None:
                candidates.append(
                    _refine_rigid_mixed(
                        relative,
                        free_pairs,
                        points_shared,
                        points_image,
                        anchor,
                        other,
                        **mixed_kwargs,
                    )
                )

    if can_pnp_only:
        solved = _pnp_similarity(
            other_id,
            np.stack(points_shared),
            np.stack(points_image),
            other,
            weights=np.asarray(known_weights, dtype=np.float64),
        )
        if solved is not None:
            if free_pairs or known_line_constraints:
                solved = _refine_rigid_mixed(
                    solved,
                    free_pairs,
                    points_shared,
                    points_image,
                    anchor,
                    other,
                    **mixed_kwargs,
                )
            candidates.append(solved)

    if can_pnl_only:
        for seed in _mixed_pose_seeds(anchor, other):
            candidates.append(
                _refine_rigid_mixed(
                    seed,
                    free_pairs,
                    points_shared,
                    points_image,
                    anchor,
                    other,
                    **mixed_kwargs,
                )
            )

    if not candidates:
        return (
            None,
            f"Could not lock a pose for '{other_id}' — check 2D picks and Known 3D",
        )

    best: SimilarityTransform | None = None
    best_rmse = float("inf")
    for candidate in candidates:
        errors = _reprojection_errors_for_similarity(
            candidate,
            free_pairs,
            anchor,
            other,
            points_shared,
            points_image,
            point_weights=known_weights,
            weighted=True,
        )
        for point_a, point_b, line_obs in known_line_constraints:
            errors.extend(
                _known_line_reprojection_errors(
                    point_a, point_b, line_obs, other, candidate
                )
            )
        if not errors:
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if rmse < best_rmse:
            best_rmse = rmse
            best = candidate

    if best is None or best_rmse > 40.0:
        detail = (
            f"Sync for '{other_id}' failed (reproj ~"
            f"{best_rmse if best is not None else float('nan'):.0f} px). "
        )
        if best is not None:
            per_landmark = _per_landmark_rmse_for_similarity(
                best,
                free_pairs,
                anchor,
                other,
                points_shared,
                points_image,
                known_landmark_ids,
            )
            worst = _format_worst_landmarks(
                per_landmark,
                _landmark_names(observations_by_landmark),
            )
            if worst:
                detail += worst + ". "
        detail += (
            "Re-pick the worst landmarks; if many are high, re-check VP/FOV "
            "on each match"
        )
        return None, detail

    # Soft parallel nudge after a geometry-locked pose — never let parallel
    # pick the pose (it was overpowering point RMSE by ~focal scaling).
    parallel_specs = _parallel_vp_specs_for_match_pair(
        anchor_id,
        other_id,
        parallel_pairs,
        line_observations_by_landmark,
    )
    if parallel_specs:
        refined = _refine_rigid_mixed(
            best,
            free_pairs,
            points_shared,
            points_image,
            anchor,
            other,
            point_weights=known_weights,
            known_line_constraints=known_line_constraints,
            parallel_vp_constraints=parallel_specs,
            parallel_weight=12.0,
        )
        refined_errors = _reprojection_errors_for_similarity(
            refined,
            free_pairs,
            anchor,
            other,
            points_shared,
            points_image,
            point_weights=known_weights,
            weighted=True,
        )
        for point_a, point_b, line_obs in known_line_constraints:
            refined_errors.extend(
                _known_line_reprojection_errors(
                    point_a, point_b, line_obs, other, refined
                )
            )
        if refined_errors:
            refined_rmse = float(np.sqrt(np.mean(np.square(refined_errors))))
            # Keep parallel refine only when it does not wreck point fit.
            if refined_rmse <= best_rmse + 5.0:
                best = refined
    return best, ""



def _register_from_relative_pose(
    anchor_id: str,
    free_match_ids: list[str],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    known_world: dict[str, np.ndarray] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]]
    | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, SimilarityTransform] | None, str]:
    """Register free matches vs anchor, then bridge via triangulated landmarks."""
    similarities: dict[str, SimilarityTransform] = {
        anchor_id: SimilarityTransform(),
    }
    pending = set(free_match_ids)
    failure_details: list[str] = []

    for match_id in sorted(list(pending)):
        solved, detail = _relative_pose_from_correspondences(
            anchor_id,
            match_id,
            observations_by_landmark,
            matches,
            known_world,
            known_lines=known_lines,
            line_observations_by_landmark=line_observations_by_landmark,
            parallel_pairs=parallel_pairs,
        )
        if solved is None:
            if detail:
                failure_details.append(detail)
            continue
        similarities[match_id] = solved
        pending.discard(match_id)

    max_passes = len(pending) + 1
    for _pass in range(max_passes):
        if not pending:
            break
        progressed = False
        landmarks = _triangulate_landmarks(
            sorted(observations_by_landmark.keys()),
            observations_by_landmark,
            similarities,
            matches,
        )
        landmarks.update(
            _metric_landmarks(
                observations_by_landmark,
                anchor_id,
                matches[anchor_id].calibration,
                known_world,
            )
        )
        for match_id in sorted(list(pending)):
            collected = _collect_pnp_correspondences(
                match_id,
                observations_by_landmark,
                landmarks,
            )
            line_constraints: list[
                tuple[np.ndarray, np.ndarray, SyncLineObservation]
            ] = []
            for landmark_id, (point_a, point_b) in (known_lines or {}).items():
                items = (line_observations_by_landmark or {}).get(landmark_id, [])
                by_match = {item.match_id: item for item in items}
                observation = by_match.get(match_id)
                if observation is not None:
                    line_constraints.append((point_a, point_b, observation))
            for landmark_id, items in (line_observations_by_landmark or {}).items():
                if known_lines and landmark_id in known_lines:
                    continue
                by_match = {item.match_id: item for item in items}
                if match_id not in by_match:
                    continue
                registered = [
                    item for item in items if item.match_id in similarities
                ]
                if len(registered) < 2:
                    continue
                reconstructed = _reconstruct_line_from_observations(
                    registered, similarities, matches
                )
                if reconstructed is None:
                    continue
                point, direction = reconstructed
                line_constraints.append(
                    (
                        point - direction,
                        point + direction,
                        by_match[match_id],
                    )
                )
            solved = None
            if collected is not None:
                points_shared, points_image = collected
                solved = _pnp_similarity(
                    match_id,
                    points_shared,
                    points_image,
                    matches[match_id].calibration,
                )
                if solved is not None and line_constraints:
                    solved = _refine_rigid_mixed(
                        solved,
                        [],
                        list(points_shared),
                        list(points_image),
                        matches[anchor_id].calibration,
                        matches[match_id].calibration,
                        known_line_constraints=line_constraints,
                    )
            elif len(line_constraints) >= 3:
                for seed in _mixed_pose_seeds(
                    matches[anchor_id].calibration,
                    matches[match_id].calibration,
                ):
                    candidate = _refine_rigid_mixed(
                        seed,
                        [],
                        [],
                        [],
                        matches[anchor_id].calibration,
                        matches[match_id].calibration,
                        known_line_constraints=line_constraints,
                    )
                    errors: list[float] = []
                    for point_a, point_b, line_obs in line_constraints:
                        errors.extend(
                            _known_line_reprojection_errors(
                                point_a,
                                point_b,
                                line_obs,
                                matches[match_id].calibration,
                                candidate,
                            )
                        )
                    if errors and float(np.sqrt(np.mean(np.square(errors)))) < 40.0:
                        solved = candidate
                        break
            if solved is None:
                continue
            similarities[match_id] = solved
            pending.discard(match_id)
            progressed = True
        if not progressed:
            break

    if pending:
        pending_list = ", ".join(f"'{name}'" for name in sorted(pending))
        if failure_details:
            return None, failure_details[0]
        return (
            None,
            f"Could not register {pending_list} — need ≥5 well-spread 2D "
            "landmarks shared with the anchor, or ≥3 Known 3D / On Ground picks",
        )
    return similarities, ""



def solve_landmark_sync(
    matches: list[SyncMatchInput],
    observations: list[SyncObservation],
    *,
    anchor_id: str,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
) -> SyncSolveResult:
    """Register non-anchor matches from 2D correspondences and/or known 3D.

    Enough multi-view 2D picks determine relative orientation and translation
    *direction*. Absolute baseline scale vs the anchor world is pinned by
    Known 3D Blender objects, On Ground picks, Known 3D lines, or a depth
    heuristic. Free 2D↔2D line landmarks help once ≥3 stills share an edge.
    Intrinsics stay frozen.
    """
    known_world = {
        landmark_id: np.asarray(point, dtype=np.float64).reshape(3)
        for landmark_id, point in (known_world or {}).items()
    }
    known_lines = {
        landmark_id: (
            np.asarray(pair[0], dtype=np.float64).reshape(3),
            np.asarray(pair[1], dtype=np.float64).reshape(3),
        )
        for landmark_id, pair in (known_lines or {}).items()
    }
    match_map = {item.match_id: item for item in matches}
    identity_result = {item.match_id: SimilarityTransform() for item in matches}
    if anchor_id not in match_map:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="Anchor match is missing",
            success=False,
        )

    valid_observations = [
        observation
        for observation in observations
        if observation.match_id in match_map
    ]
    valid_line_observations = [
        observation
        for observation in (line_observations or [])
        if observation.match_id in match_map
    ]
    observations_by_landmark: dict[str, list[SyncObservation]] = {}
    for observation in valid_observations:
        observations_by_landmark.setdefault(observation.landmark_id, []).append(
            observation,
        )
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] = {}
    for observation in valid_line_observations:
        line_observations_by_landmark.setdefault(
            observation.landmark_id, []
        ).append(observation)

    multi_ids = {
        landmark_id
        for landmark_id, items in observations_by_landmark.items()
        if len({item.match_id for item in items}) >= 2
    }
    known_observed_ids = {
        landmark_id
        for landmark_id in known_world
        if landmark_id in observations_by_landmark
    }
    ground_observed_ids = {
        landmark_id
        for landmark_id, items in observations_by_landmark.items()
        if any(item.on_ground for item in items)
    }
    known_line_metric_ids = {
        landmark_id
        for landmark_id in known_lines
        if any(
            item.match_id != anchor_id
            for item in line_observations_by_landmark.get(landmark_id, [])
        )
    }
    # Free lines seen in ≥3 stills can constrain after two matches register.
    free_line_multi_ids = {
        landmark_id
        for landmark_id, items in line_observations_by_landmark.items()
        if landmark_id not in known_lines
        and len({item.match_id for item in items}) >= 3
    }
    metric_ids = known_observed_ids | ground_observed_ids | known_line_metric_ids
    usable_ids = multi_ids | known_observed_ids | ground_observed_ids
    if (
        len(multi_ids) < 5
        and len(metric_ids) < 3
        and len(known_line_metric_ids) < 3
        and not free_line_multi_ids
    ):
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message=(
                "Need ≥5 point landmarks in two+ matches, ≥3 Known 3D points / "
                "On Ground / Known 3D lines, or line landmarks shared across ≥3 stills"
            ),
            success=False,
        )
    if not usable_ids and not known_line_metric_ids and not free_line_multi_ids:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="No usable landmarks for sync",
            success=False,
        )

    usable_observations = [
        observation
        for observation in valid_observations
        if observation.landmark_id in usable_ids
    ]
    connected = _connected_match_ids(
        anchor_id,
        usable_observations,
        known_world=known_world,
        line_observations=valid_line_observations,
        known_lines=known_lines,
    )
    free_match_ids = sorted(
        match_id for match_id in connected if match_id != anchor_id
    )
    if not free_match_ids:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="No non-anchor matches are connected through landmarks",
            success=False,
        )

    usable_observations = [
        observation
        for observation in usable_observations
        if observation.match_id in connected
    ]
    observations_by_landmark = {}
    for observation in usable_observations:
        observations_by_landmark.setdefault(observation.landmark_id, []).append(
            observation,
        )
    line_observations_by_landmark = {
        landmark_id: [
            item for item in items if item.match_id in connected
        ]
        for landmark_id, items in line_observations_by_landmark.items()
    }
    landmark_ids = sorted(observations_by_landmark.keys())

    similarities, failure_detail = _register_from_relative_pose(
        anchor_id,
        free_match_ids,
        observations_by_landmark,
        match_map,
        known_world,
        known_lines=known_lines,
        line_observations_by_landmark=line_observations_by_landmark,
        parallel_pairs=parallel_pairs,
    )
    if similarities is None:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message=failure_detail
            or (
                "Could not register every match — need ≥5 well-spread 2D "
                "landmarks or ≥3 Known 3D / On Ground / Known 3D line picks"
            ),
            success=False,
        )

    for match_id in match_map:
        similarities.setdefault(match_id, SimilarityTransform())
    similarities[anchor_id] = SimilarityTransform()

    landmarks = _triangulate_landmarks(
        landmark_ids,
        observations_by_landmark,
        similarities,
        match_map,
    )
    # Prefer fixed metric positions over triangulated estimates.
    landmarks.update(
        _metric_landmarks(
            observations_by_landmark,
            anchor_id,
            match_map[anchor_id].calibration,
            known_world,
        )
    )
    # Keep known points even if they lack multi-view triangulation.
    for landmark_id, point in known_world.items():
        landmarks[landmark_id] = point
    # Line landmarks: midpoint + finite segment for viewport mesh viz.
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for landmark_id, (point_a, point_b) in known_lines.items():
        landmarks[landmark_id] = 0.5 * (point_a + point_b)
        line_segments[landmark_id] = (point_a.copy(), point_b.copy())
    for landmark_id, items in line_observations_by_landmark.items():
        if landmark_id in line_segments:
            continue
        reconstructed = _reconstruct_line_from_observations(
            items, similarities, match_map
        )
        if reconstructed is None:
            continue
        point, direction = reconstructed
        segment = _finite_segment_from_line_observations(
            point, direction, items, similarities, match_map
        )
        landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
        line_segments[landmark_id] = segment

    # Parallel families: lock free-line mesh directions (pose stays as solved).
    _enforce_parallel_line_segments(
        line_segments,
        landmarks,
        parallel_pairs,
        line_observations_by_landmark,
        similarities,
        match_map,
        known_lines,
    )

    residual_landmark_ids = [
        landmark_id for landmark_id in landmark_ids if landmark_id in landmarks
    ]
    residual_observations = [
        observation
        for observation in usable_observations
        if observation.landmark_id in landmarks
    ]
    residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {landmark_id: landmarks[landmark_id] for landmark_id in residual_landmark_ids},
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        match_map,
        residual_observations,
        weighted=False,
    )
    weighted_residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {landmark_id: landmarks[landmark_id] for landmark_id in residual_landmark_ids},
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        match_map,
        residual_observations,
        weighted=True,
    )
    per_match_sse: dict[str, list[float]] = {match_id: [] for match_id in connected}
    per_landmark_sse: dict[str, list[float]] = {
        landmark_id: [] for landmark_id in residual_landmark_ids
    }
    residual_index = 0
    for observation in residual_observations:
        error_u = float(residuals[residual_index])
        error_v = float(residuals[residual_index + 1])
        residual_index += 2
        squared = error_u * error_u + error_v * error_v
        per_match_sse[observation.match_id].append(squared)
        per_landmark_sse[observation.landmark_id].append(squared)

    # Pose quality = point residuals only. Line px (esp. after Parallel lock)
    # is diagnostic and must not reject a good camera solve.
    point_sse = [value for values in per_match_sse.values() for value in values]
    weighted_point_sse = (
        list(np.square(weighted_residuals)) if weighted_residuals.size else []
    )

    parallel_landmark_ids: set[str] = set()
    for landmark_a, landmark_b in parallel_pairs or ():
        parallel_landmark_ids.add(landmark_a)
        parallel_landmark_ids.add(landmark_b)

    line_error_values: list[float] = []
    weighted_line_error_values: list[float] = []
    per_line_sse: dict[str, list[float]] = {}
    for landmark_id, items in line_observations_by_landmark.items():
        if landmark_id in line_segments:
            point_a, point_b = line_segments[landmark_id]
            direction = point_b - point_a
            span = float(np.linalg.norm(direction))
            if span < 1.0e-9:
                continue
            direction = direction / span
            point = 0.5 * (point_a + point_b)
        elif landmark_id in known_lines:
            point_a, point_b = known_lines[landmark_id]
            direction = point_b - point_a
            span = float(np.linalg.norm(direction))
            if span < 1.0e-9:
                continue
            direction = direction / span
            point = point_a
        else:
            reconstructed = _reconstruct_line_from_observations(
                items, similarities, match_map
            )
            if reconstructed is None:
                continue
            point, direction = reconstructed
        for observation in items:
            similarity = similarities.get(observation.match_id)
            match = match_map.get(observation.match_id)
            if similarity is None or match is None:
                continue
            raw = _line_observation_reprojection_errors(
                point,
                direction,
                SyncLineObservation(
                    match_id=observation.match_id,
                    landmark_id=observation.landmark_id,
                    u1=observation.u1,
                    v1=observation.v1,
                    u2=observation.u2,
                    v2=observation.v2,
                    weight=1.0,
                ),
                match.calibration,
                similarity,
            )
            weighted = _line_observation_reprojection_errors(
                point,
                direction,
                observation,
                match.calibration,
                similarity,
            )
            for value in raw:
                squared = value * value
                line_error_values.append(squared)
                per_line_sse.setdefault(landmark_id, []).append(squared)
                # Keep lines visible in per-landmark Diagnose, not in pose reject.
                per_landmark_sse.setdefault(observation.landmark_id, []).append(squared)
                per_match_sse.setdefault(observation.match_id, []).append(squared)
            weighted_line_error_values.extend(value * value for value in weighted)

    # Parallel pairs: report residual angle after enforcement (should be ~0°).
    parallel_angles_deg: list[float] = []
    for landmark_a, landmark_b in parallel_pairs or ():
        direction_a = None
        direction_b = None
        if landmark_a in line_segments:
            point_a, point_b = line_segments[landmark_a]
            direction_a = point_b - point_a
        if landmark_b in line_segments:
            point_a, point_b = line_segments[landmark_b]
            direction_b = point_b - point_a
        if direction_a is None or direction_b is None:
            continue
        parallel_error = _parallel_direction_error(direction_a, direction_b)
        parallel_angles_deg.append(float(np.degrees(np.arcsin(min(parallel_error, 1.0)))))

    def _rmse(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.sqrt(np.mean(values)))

    per_match_rmse = {
        match_id: _rmse(values) for match_id, values in per_match_sse.items()
    }
    per_landmark_rmse = {
        landmark_id: _rmse(values) for landmark_id, values in per_landmark_sse.items()
    }
    per_line_rmse = {
        landmark_id: _rmse(values) for landmark_id, values in per_line_sse.items()
    }
    mean_rmse = _rmse(point_sse) if point_sse else _rmse(line_error_values)
    mean_weighted_rmse = (
        _rmse(weighted_point_sse) if weighted_point_sse else mean_rmse
    )
    names = _landmark_names(observations_by_landmark)
    for landmark_id, items in line_observations_by_landmark.items():
        for item in items:
            if item.landmark_name:
                names[landmark_id] = item.landmark_name
                break
        names.setdefault(landmark_id, landmark_id[:8])
    if mean_weighted_rmse > 40.0 and point_sse:
        # Worst among points — line Parallel miss is not a pose failure.
        point_only_rmse = {
            landmark_id: rmse
            for landmark_id, rmse in per_landmark_rmse.items()
            if landmark_id in residual_landmark_ids
        }
        worst = _format_worst_landmarks(point_only_rmse, names)
        hint = (
            "Re-pick the worst landmarks on both stills. "
            "If several ordinary landmarks are all high, FOV/VP may be off."
        )
        message = (
            f"Sync rejected (reproj {mean_rmse:.0f} px, "
            f"weighted {mean_weighted_rmse:.0f} px)."
        )
        if worst:
            message += f" {worst}."
        message += f" {hint}"
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=mean_rmse,
            per_match_rmse_px=per_match_rmse,
            per_landmark_rmse_px=per_landmark_rmse,
            message=message,
            success=False,
        )
    disconnected = sorted(set(match_map) - connected)
    known_count = sum(1 for landmark_id in known_world if landmark_id in landmarks)
    known_line_count = len(known_lines)
    ground_count = sum(
        1
        for landmark_id, items in observations_by_landmark.items()
        if any(item.on_ground for item in items) and landmark_id not in known_world
    )
    free_line_count = sum(
        1
        for landmark_id in line_observations_by_landmark
        if landmark_id not in known_lines
    )
    message = (
        f"Synced {len(free_match_ids)} match(es) · {len(landmarks)} landmarks · "
        f"RMSE {mean_rmse:.2f} px"
    )
    scale_bits = []
    if known_count:
        scale_bits.append(f"{known_count} known 3D")
    if known_line_count:
        scale_bits.append(f"{known_line_count} known lines")
    if ground_count:
        scale_bits.append(f"{ground_count} ground")
    if free_line_count:
        scale_bits.append(f"{free_line_count} free lines")
    if parallel_pairs:
        scale_bits.append(f"{len(parallel_pairs)} parallel")
    if scale_bits:
        message += " · scale from " + " + ".join(scale_bits)
    else:
        message += " · scale from depth heuristic"
    if parallel_angles_deg:
        mean_angle = float(np.mean(parallel_angles_deg))
        message += f" · parallel Δ {mean_angle:.1f}°"
        if mean_angle > 1.0:
            message += " (could not lock family — redraw / Known 3D)"
        else:
            message += " (direction locked)"
    if disconnected:
        message += f" · skipped {len(disconnected)} disconnected"
    # Soft warn: accepted but likely inaccurate picks or intrinsics.
    if mean_rmse > 8.0:
        point_only_rmse = {
            landmark_id: rmse
            for landmark_id, rmse in per_landmark_rmse.items()
            if landmark_id in residual_landmark_ids
        }
        worst = _format_worst_landmarks(point_only_rmse, names)
        message += " · WARN high error"
        if worst:
            message += f" ({worst})"
        message += " — check picks / FOV"
    # Line miss after Parallel is expected when drawings disagree with the lock.
    high_parallel_lines = [
        (names.get(landmark_id, landmark_id), rmse)
        for landmark_id, rmse in per_line_rmse.items()
        if landmark_id in parallel_landmark_ids and rmse > 20.0
    ]
    high_parallel_lines.sort(key=lambda item: item[1], reverse=True)
    if high_parallel_lines:
        bits = ", ".join(
            f"{name} {rmse:.0f}px" for name, rmse in high_parallel_lines[:3]
        )
        message += (
            f" · parallel line miss ({bits}) — 2D drawings vs locked 3D direction"
        )

    return SyncSolveResult(
        similarities=similarities,
        landmarks=landmarks,
        mean_reprojection_px=mean_rmse,
        per_match_rmse_px=per_match_rmse,
        per_landmark_rmse_px=per_landmark_rmse,
        message=message,
        success=True,
        line_segments=line_segments,
    )
