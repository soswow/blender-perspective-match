"""Camera projection, triangulation, and image-line geometry."""

from __future__ import annotations

import numpy as np

from .. import geometry as core
from .constants import (
    ACCEPT_RMSE_PX,
    TRIANGULATION_ANGLE_WEIGHT_FLOOR,
    TRIANGULATION_GN_STEPS,
    TRIANGULATION_PARALLEL_COSINE,
)
from .types import (
    SimilarityTransform,
    SyncLineObservation,
    SyncMatchInput,
    SyncObservation,
    _observation_scale,
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
        calibration.brown_conrady,
    )[0]
    return distorted


def _project_shared_points(
    points_shared: np.ndarray,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a batch of shared-world points through one match.

    Lens refinement evaluates the same landmark set thousands of times while
    building numerical Jacobians. Keeping this path in NumPy avoids one Python
    call and several temporary arrays per landmark per evaluation.
    """
    points = np.asarray(points_shared, dtype=np.float64).reshape(-1, 3)
    private = (
        (points - similarity.translation) / max(float(similarity.scale), 1.0e-12)
    ) @ similarity.rotation
    camera = (private - calibration.camera_center) @ calibration.rotation_w2c.T
    depths = camera[:, 2]
    valid = depths > 1.0e-8
    safe_depths = np.where(valid, depths, 1.0)
    intrinsics = calibration.intrinsics
    ideal = np.column_stack(
        (
            intrinsics.fx * camera[:, 0] / safe_depths + intrinsics.cx,
            intrinsics.fy * camera[:, 1] / safe_depths + intrinsics.cy,
        )
    )
    projected = core.distort_points(
        ideal,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        calibration.division_lambda,
        calibration.brown_conrady,
    )
    return projected, valid


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
        calibration.brown_conrady,
    )[0]
    direction_camera = core.pixel_ray(
        float(ideal[0]),
        float(ideal[1]),
        calibration.intrinsics,
    )
    direction_world = calibration.rotation_w2c.T @ direction_camera
    direction_world = direction_world / max(float(np.linalg.norm(direction_world)), 1.0e-12)
    return calibration.camera_center.copy(), direction_world


def _unit_directions(directions: list[np.ndarray]) -> list[np.ndarray]:
    """Return unit-length copies of ray directions."""
    units: list[np.ndarray] = []
    for direction in directions:
        vector = np.asarray(direction, dtype=np.float64).reshape(3)
        units.append(vector / max(float(np.linalg.norm(vector)), 1.0e-12))
    return units


def _triangulation_angle_weights(directions: list[np.ndarray]) -> list[float]:
    """Weight each ray by sin² of its largest stereo angle, shared among near-parallel views."""
    units = _unit_directions(directions)
    weights: list[float] = []
    for index, direction in enumerate(units):
        best = TRIANGULATION_ANGLE_WEIGHT_FLOOR
        similar = 1
        for other_index, other in enumerate(units):
            if other_index == index:
                continue
            cosine = float(np.clip(np.dot(direction, other), -1.0, 1.0))
            sine_squared = 1.0 - cosine * cosine
            if sine_squared > best:
                best = sine_squared
            if cosine > TRIANGULATION_PARALLEL_COSINE:
                similar += 1
        weights.append(best / similar)
    return weights


def _linear_ray_midpoint(
    origins: list[np.ndarray],
    directions: list[np.ndarray],
    weights: list[float],
) -> np.ndarray | None:
    """Least-squares midpoint of skew rays (no cheirality)."""
    if len(origins) < 2:
        return None
    accumulator = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction, weight in zip(origins, directions, weights):
        scale = max(float(weight), 1.0e-12)
        projector = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
        accumulator += scale * projector
        rhs += scale * (projector @ origin)
    try:
        point = np.linalg.solve(accumulator, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _ray_is_in_front(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> bool:
    """True when ``point`` lies in front of the ray origin along ``direction``."""
    return float(np.dot(point - origin, direction)) > 1.0e-8


def triangulate_midpoint(
    origins: list[np.ndarray],
    directions: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray | None:
    """Triangulate a point as the least-squares midpoint of skew rays.

    Optional ``weights`` pull the result toward higher-confidence rays.
    Near-parallel extra views are downweighted by triangulation angle.
    Views that put the point behind the camera are dropped; ≥2 must remain.
    """
    if len(origins) < 2:
        return None
    units = _unit_directions(directions)
    angle_weights = _triangulation_angle_weights(units)
    observation_weights = (
        [1.0] * len(origins) if weights is None else [float(value) for value in weights]
    )
    combined = [
        max(obs, 1.0e-12) * angle
        for obs, angle in zip(observation_weights, angle_weights)
    ]
    point = _linear_ray_midpoint(origins, units, combined)
    if point is None:
        return None
    keep = [
        index
        for index, (origin, direction) in enumerate(zip(origins, units))
        if _ray_is_in_front(point, origin, direction)
    ]
    if len(keep) < 2:
        return None
    if len(keep) < len(origins):
        point = _linear_ray_midpoint(
            [origins[index] for index in keep],
            [units[index] for index in keep],
            [combined[index] for index in keep],
        )
        if point is None:
            return None
        keep = [
            index
            for index in keep
            if _ray_is_in_front(
                point, origins[index], units[index]
            )
        ]
        if len(keep) < 2:
            return None
    return point


def refine_triangulated_point(
    point: np.ndarray,
    observations: list[SyncObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    *,
    steps: int | None = None,
) -> np.ndarray | None:
    """Gauss–Newton on pixel reprojection; drops views that put the point behind."""
    xyz = np.asarray(point, dtype=np.float64).reshape(3).copy()
    iteration_count = TRIANGULATION_GN_STEPS if steps is None else int(steps)

    def project_observation(observation: SyncObservation, candidate: np.ndarray):
        private = similarities[observation.match_id].inverse_point(candidate)
        return project_private_point(
            private, matches[observation.match_id].calibration
        )

    def in_front(observation: SyncObservation, candidate: np.ndarray) -> bool:
        return project_observation(observation, candidate) is not None

    kept = [item for item in observations if in_front(item, xyz)]
    if len(kept) < 2:
        return None

    def residual_px(observation: SyncObservation, candidate: np.ndarray) -> float | None:
        projected = project_observation(observation, candidate)
        if projected is None:
            return None
        return float(
            np.hypot(
                float(projected[0]) - float(observation.u),
                float(projected[1]) - float(observation.v),
            )
        )

    def inliers(candidate: np.ndarray) -> list[SyncObservation]:
        selected: list[SyncObservation] = []
        for item in observations:
            error = residual_px(item, candidate)
            if error is not None and error <= ACCEPT_RMSE_PX:
                selected.append(item)
        return selected if len(selected) >= 2 else [
            item for item in observations if in_front(item, candidate)
        ]

    best = xyz.copy()
    best_cost = float("inf")
    for _ in range(max(iteration_count, 0)):
        kept = inliers(xyz)
        if len(kept) < 2:
            break
        residuals: list[float] = []
        jacobian_rows: list[list[float]] = []
        for observation in kept:
            projected = project_observation(observation, xyz)
            if projected is None:
                continue
            scale = float(np.sqrt(max(float(observation.weight), 1.0e-12)))
            residuals.extend(
                (
                    scale * (float(projected[0]) - float(observation.u)),
                    scale * (float(projected[1]) - float(observation.v)),
                )
            )
            row_u: list[float] = []
            row_v: list[float] = []
            for axis in range(3):
                offset = np.zeros(3, dtype=np.float64)
                offset[axis] = 1.0e-5
                shifted = project_observation(observation, xyz + offset)
                if shifted is None:
                    row_u.append(0.0)
                    row_v.append(0.0)
                    continue
                row_u.append(
                    scale * (float(shifted[0]) - float(projected[0])) / 1.0e-5
                )
                row_v.append(
                    scale * (float(shifted[1]) - float(projected[1])) / 1.0e-5
                )
            jacobian_rows.append(row_u)
            jacobian_rows.append(row_v)
        if len(residuals) < 4:
            break
        residual_vector = np.asarray(residuals, dtype=np.float64)
        jacobian = np.asarray(jacobian_rows, dtype=np.float64)
        cost = float(np.dot(residual_vector, residual_vector))
        if cost <= best_cost:
            best_cost = cost
            best = xyz.copy()
        try:
            update, *_rest = np.linalg.lstsq(jacobian, residual_vector, rcond=None)
        except np.linalg.LinAlgError:
            break
        xyz = xyz - update
        if float(np.linalg.norm(update)) < 1.0e-9:
            break
    if sum(1 for item in observations if in_front(item, best)) >= 2:
        return best
    if sum(1 for item in observations if in_front(item, xyz)) >= 2:
        return xyz
    return None


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
        calibration.brown_conrady,
    )[0]
    return core.pixel_ray(float(ideal[0]), float(ideal[1]), calibration.intrinsics)


def _normalized_image_point(
    observation: SyncObservation,
    calibration: core.Calibration,
) -> np.ndarray:
    """Undistorted normalized pinhole coordinate with homogeneous z=1."""
    ray = _normalized_camera_ray(observation.u, observation.v, calibration)
    if abs(float(ray[2])) < 1.0e-12:
        return np.array((ray[0], ray[1], 1.0), dtype=np.float64)
    return np.array(
        (ray[0] / ray[2], ray[1] / ray[2], 1.0),
        dtype=np.float64,
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
    _u_matrix, singular, _vt_matrix = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2:
        return True
    return float(singular[1]) < min_cross_spread_px


