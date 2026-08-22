"""Calibrated ground-plane initialization from On Ground homographies."""

from __future__ import annotations

from itertools import combinations, islice

import numpy as np

from .. import geometry as core
from .projection import camera_ray_private, _image_points_collinear, _normalized_image_point
from .types import (
    GroundPlaneInitialization,
    SyncMatchInput,
    SyncObservation,
    _GroundHomographyCandidate,
)

def _normalize_homography_points(
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hartley-normalize 2D points and return (points, transform)."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = np.sqrt(2.0) / max(mean_distance, 1.0e-12)
    transform = np.array(
        (
            (scale, 0.0, -scale * center[0]),
            (0.0, scale, -scale * center[1]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def _fit_homography_dlt(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray | None:
    """Fit normalized DLT homography mapping A image points to B."""
    source = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if len(source) < 4 or len(source) != len(target):
        return None
    if _image_points_collinear(source, min_cross_spread_px=1.0e-5):
        return None
    if _image_points_collinear(target, min_cross_spread_px=1.0e-5):
        return None
    source_n, transform_a = _normalize_homography_points(source)
    target_n, transform_b = _normalize_homography_points(target)
    rows: list[list[float]] = []
    for (x_coord, y_coord), (u_coord, v_coord) in zip(source_n, target_n):
        rows.append(
            [
                -x_coord,
                -y_coord,
                -1.0,
                0.0,
                0.0,
                0.0,
                u_coord * x_coord,
                u_coord * y_coord,
                u_coord,
            ]
        )
        rows.append(
            [
                0.0,
                0.0,
                0.0,
                -x_coord,
                -y_coord,
                -1.0,
                v_coord * x_coord,
                v_coord * y_coord,
                v_coord,
            ]
        )
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        return None
    _u_matrix, _singular, vt_matrix = np.linalg.svd(design)
    normalized_h = vt_matrix[-1].reshape(3, 3)
    try:
        homography = np.linalg.inv(transform_b) @ normalized_h @ transform_a
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(homography))
    return homography / norm if norm > 1.0e-12 else None


def _homography_transfer_errors(
    homography: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    focal_a: float,
    focal_b: float,
) -> np.ndarray:
    """Symmetric transfer error in approximate pixels."""
    source = np.column_stack((points_a, np.ones(len(points_a))))
    target = np.column_stack((points_b, np.ones(len(points_b))))
    mapped_b = (homography @ source.T).T
    valid_b = np.abs(mapped_b[:, 2]) > 1.0e-12
    mapped_b[:, :2] /= np.where(valid_b, mapped_b[:, 2], 1.0)[:, None]
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full(len(points_a), float("inf"), dtype=np.float64)
    mapped_a = (inverse @ target.T).T
    valid_a = np.abs(mapped_a[:, 2]) > 1.0e-12
    mapped_a[:, :2] /= np.where(valid_a, mapped_a[:, 2], 1.0)[:, None]
    forward = np.linalg.norm(mapped_b[:, :2] - points_b, axis=1) * focal_b
    backward = np.linalg.norm(mapped_a[:, :2] - points_a, axis=1) * focal_a
    errors = np.sqrt(0.5 * (forward * forward + backward * backward))
    errors[~(valid_a & valid_b)] = float("inf")
    return errors


def _fit_ground_homography(
    points_a: np.ndarray,
    points_b: np.ndarray,
    focal_a: float,
    focal_b: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Deterministically fit a small robust homography and return its inliers."""
    count = len(points_a)
    if count < 4 or count != len(points_b):
        return None
    subsets: list[tuple[int, ...]] = [tuple(range(count))]
    if count > 4:
        # Ground-marker sets are normally small. Cap deterministic four-point
        # hypotheses so accidental large sets cannot become combinatorial.
        subsets.extend(islice(combinations(range(count), 4), 96))
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for subset in subsets:
        homography = _fit_homography_dlt(points_a[list(subset)], points_b[list(subset)])
        if homography is None:
            continue
        errors = _homography_transfer_errors(
            homography,
            points_a,
            points_b,
            focal_a,
            focal_b,
        )
        inliers = np.isfinite(errors) & (errors <= 5.0)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 4:
            continue
        median = float(np.median(errors[inliers]))
        candidate = (inlier_count, -median, homography, inliers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    _count, _negative_median, _homography, inliers = best
    refined = _fit_homography_dlt(points_a[inliers], points_b[inliers])
    return (refined, inliers) if refined is not None else None


def _plane_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed orthonormal basis tangent to ``normal``."""
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
    reference = (
        np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(normal[0])) < 0.85
        else np.array((0.0, 1.0, 0.0), dtype=np.float64)
    )
    tangent_a = reference - normal * float(np.dot(reference, normal))
    tangent_a /= max(float(np.linalg.norm(tangent_a)), 1.0e-12)
    tangent_b = np.cross(normal, tangent_a)
    tangent_b /= max(float(np.linalg.norm(tangent_b)), 1.0e-12)
    return tangent_a, tangent_b


def _rotation_and_translation_from_plane_normal(
    homography: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover H = R + u n^T for one homography normal candidate."""
    tangent_a, tangent_b = _plane_tangent_basis(normal)
    rotated_a = homography @ tangent_a
    rotated_a /= max(float(np.linalg.norm(rotated_a)), 1.0e-12)
    rotated_b = homography @ tangent_b
    rotated_b -= rotated_a * float(np.dot(rotated_a, rotated_b))
    rotated_b /= max(float(np.linalg.norm(rotated_b)), 1.0e-12)
    rotated_normal = np.cross(rotated_a, rotated_b)
    rotation = np.column_stack((rotated_a, rotated_b, rotated_normal)) @ np.column_stack(
        (tangent_a, tangent_b, normal)
    ).T
    translation_over_distance = (homography - rotation) @ normal
    return rotation, translation_over_distance


def _ground_homography_candidates(
    homography: np.ndarray,
    rays_a: np.ndarray,
    rays_b: np.ndarray,
) -> list[_GroundHomographyCandidate]:
    """Decompose a calibrated homography into its physical plane candidates."""
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    _u_matrix, singular, vt_matrix = np.linalg.svd(matrix)
    middle = float(singular[1])
    if middle < 1.0e-12:
        return []
    matrix = matrix / middle
    if float(np.linalg.det(matrix)) < 0.0:
        matrix = -matrix
    _u_matrix, singular, vt_matrix = np.linalg.svd(matrix)
    d1, _d2, d3 = (float(value) for value in singular)
    denominator = d1 * d1 - d3 * d3
    strength = max(d1 - 1.0, 1.0 - d3)
    if denominator < 1.0e-10 or strength < 1.0e-4:
        # Pure rotation (or negligible baseline) contains no plane-normal cue.
        return []
    x1 = np.sqrt(max((d1 * d1 - 1.0) / denominator, 0.0))
    x3 = np.sqrt(max((1.0 - d3 * d3) / denominator, 0.0))
    right_vectors = vt_matrix.T
    candidates: list[_GroundHomographyCandidate] = []
    for local_normal in (
        np.array((x1, 0.0, x3), dtype=np.float64),
        np.array((-x1, 0.0, x3), dtype=np.float64),
    ):
        normal = right_vectors @ local_normal
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        # Choose the sign whose unit-distance plane lies in front of the anchor.
        if float(np.median(rays_a @ normal)) < 0.0:
            normal = -normal
        rotation, translation = _rotation_and_translation_from_plane_normal(
            matrix, normal
        )
        positive = 0
        for ray_a, ray_b in zip(rays_a, rays_b):
            denominator_a = float(np.dot(normal, ray_a))
            if denominator_a <= 1.0e-9:
                continue
            point_a = ray_a / denominator_a
            point_b = rotation @ point_a + translation
            if float(point_b[2]) <= 1.0e-9:
                continue
            predicted = point_b / point_b[2]
            observed = ray_b / max(float(ray_b[2]), 1.0e-12)
            if float(np.linalg.norm(predicted[:2] - observed[:2])) <= 0.02:
                positive += 1
        normal_b = rotation @ normal
        distance_ratio = 1.0 + float(np.dot(normal_b, translation))
        if distance_ratio <= 1.0e-4:
            continue
        candidates.append(
            _GroundHomographyCandidate(
                normal_camera=normal,
                relative_rotation=rotation,
                plane_distance_ratio=distance_ratio,
                positive_depth_count=positive,
                point_count=len(rays_a),
                strength=strength,
            )
        )
    return candidates


def estimate_anchor_ground_plane(
    matches: list[SyncMatchInput],
    observations: list[SyncObservation],
    *,
    anchor_id: str,
) -> GroundPlaneInitialization | None:
    """Infer anchor ground normal and relative rotations from planar picks.

    All inputs are calibrated. Four non-collinear On Ground correspondences are
    sufficient for one pair; additional matches disambiguate the two valid
    calibrated-homography decompositions by normal consensus.
    """
    match_map = {item.match_id: item for item in matches}
    anchor_match = match_map.get(anchor_id)
    if anchor_match is None:
        return None
    by_landmark: dict[str, dict[str, SyncObservation]] = {}
    for observation in observations:
        if not observation.on_ground or observation.match_id not in match_map:
            continue
        by_landmark.setdefault(observation.landmark_id, {})[
            observation.match_id
        ] = observation

    groups: dict[str, list[_GroundHomographyCandidate]] = {}
    for other_id, other_match in match_map.items():
        if other_id == anchor_id:
            continue
        pairs = [
            (items[anchor_id], items[other_id])
            for items in by_landmark.values()
            if anchor_id in items and other_id in items
        ]
        if len(pairs) < 4:
            continue
        rays_a = np.stack(
            [
                _normalized_image_point(item_a, anchor_match.calibration)
                for item_a, _item_b in pairs
            ]
        )
        rays_b = np.stack(
            [
                _normalized_image_point(item_b, other_match.calibration)
                for _item_a, item_b in pairs
            ]
        )
        points_a = rays_a[:, :2]
        points_b = rays_b[:, :2]
        focal_a = 0.5 * (
            anchor_match.calibration.intrinsics.fx
            + anchor_match.calibration.intrinsics.fy
        )
        focal_b = 0.5 * (
            other_match.calibration.intrinsics.fx
            + other_match.calibration.intrinsics.fy
        )
        fitted = _fit_ground_homography(
            points_a,
            points_b,
            focal_a,
            focal_b,
        )
        if fitted is None:
            continue
        homography, inliers = fitted
        candidates = _ground_homography_candidates(
            homography,
            rays_a[inliers],
            rays_b[inliers],
        )
        required_positive = max(4, int(np.count_nonzero(inliers)) - 1)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.positive_depth_count >= required_positive
        ]
        if candidates:
            groups[other_id] = candidates
    if len(groups) < 2:
        # One calibrated homography has two physically valid plane solutions.
        # A second translated supporting view resolves that ambiguity because
        # only the real anchor-plane normal is common to both decompositions.
        return None

    # Pick the anchor-normal hypothesis that has one close candidate in the
    # largest number of other views. The false homography solution changes with
    # baseline; the physical ground normal is common to every pair.
    best_selection: (
        tuple[float, list[tuple[str, _GroundHomographyCandidate]]] | None
    ) = None
    for seed in [candidate for items in groups.values() for candidate in items]:
        selected: list[tuple[str, _GroundHomographyCandidate]] = []
        score = 0.0
        for match_id, candidates in groups.items():
            closest = max(
                candidates,
                key=lambda item: float(np.dot(seed.normal_camera, item.normal_camera)),
            )
            alignment = float(np.dot(seed.normal_camera, closest.normal_camera))
            if alignment < np.cos(np.radians(25.0)):
                continue
            selected.append((match_id, closest))
            support = closest.positive_depth_count / max(closest.point_count, 1)
            score += closest.strength * alignment**4 * (0.5 + 0.5 * support)
        ranked = (score + 10.0 * len(selected), selected)
        if best_selection is None or ranked[0] > best_selection[0]:
            best_selection = ranked
    if best_selection is None or len(best_selection[1]) < 2:
        return None

    selected = best_selection[1]
    weights = np.array(
        [max(candidate.strength, 1.0e-6) for _match_id, candidate in selected],
        dtype=np.float64,
    )
    normals = np.stack(
        [candidate.normal_camera for _match_id, candidate in selected]
    )
    normal = np.average(normals, axis=0, weights=weights)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    deviations = [
        np.degrees(
            np.arccos(
                np.clip(float(np.dot(normal, candidate.normal_camera)), -1.0, 1.0)
            )
        )
        for _match_id, candidate in selected
    ]
    return GroundPlaneInitialization(
        plane_normal_camera=normal,
        relative_rotations={
            match_id: candidate.relative_rotation
            for match_id, candidate in selected
        },
        plane_distance_ratios={
            match_id: candidate.plane_distance_ratio
            for match_id, candidate in selected
        },
        supporting_match_ids=[match_id for match_id, _candidate in selected],
        mean_normal_deviation_degrees=float(np.mean(deviations)),
    )


def rotation_from_ground_normal(
    up_camera: np.ndarray,
    *,
    reference_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Build world-to-camera axes from vertical, choosing deterministic yaw."""
    up = np.asarray(up_camera, dtype=np.float64).reshape(3)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    reference = (
        np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)[:, 0]
        if reference_rotation is not None
        else np.array((1.0, 0.0, 0.0), dtype=np.float64)
    )
    axis_x = reference - up * float(np.dot(reference, up))
    if float(np.linalg.norm(axis_x)) < 1.0e-6:
        axis_x, _unused = _plane_tangent_basis(up)
    axis_x /= max(float(np.linalg.norm(axis_x)), 1.0e-12)
    axis_y = np.cross(up, axis_x)
    axis_y /= max(float(np.linalg.norm(axis_y)), 1.0e-12)
    return np.column_stack((axis_x, axis_y, up))

