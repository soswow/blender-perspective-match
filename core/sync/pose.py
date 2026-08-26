"""Pairwise pose: essential, PnP, IPPE, and graph registration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
import math
import os
import threading

import numpy as np

from .. import geometry as core
from .ba import (
    _residual_vector,
    _sampson_distance,
    _triangulate_landmarks,
)
from .constants import (
    ACCEPT_RMSE_PX,
    GROUND_PLANE_Z_FRACTION,
    LOG_SCALE_CLIP,
    STRETCHED_PIXEL_RATIO,
)
from .ground import (
    _fit_homography_dlt,
    _homography_transfer_errors,
    _normalize_homography_points,
    _plane_tangent_basis,
)
from .lines import (
    _axis_line_constraints_for_match,
    _axis_line_rotation_error,
    _parallel_pair_rotation_error,
    _parallel_vp_specs_for_match_pair,
    _reconstruct_line_from_observations,
)
from .projection import (
    _known_line_reprojection_errors,
    _line_observation_reprojection_errors,
    _log_rodrigues,
    _normalized_camera_ray,
    _image_points_collinear,
    _project_shared_points,
    _rodrigues,
    camera_ray_private,
    project_private_point,
    triangulate_midpoint,
)
from .types import (
    SimilarityTransform,
    SyncLineObservation,
    SyncMatchInput,
    SyncObservation,
    _compose_similarities,
    _inverse_similarity,
    _observation_scale,
    _pair_scale,
)

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
    *,
    scale: float = 1.0,
) -> SimilarityTransform:
    """Build Empty similarity that realizes a shared camera pose."""
    # R_w2c_shared = R_priv @ R_sim.T  ⇒  R_sim = R_w2c_shared.T @ R_priv
    rotation_sim = rotation_w2c_shared.T @ calibration.rotation_w2c
    # Orthonormalize in case of numeric drift.
    u_matrix, _, vt_matrix = np.linalg.svd(rotation_sim)
    rotation_sim = u_matrix @ vt_matrix
    if np.linalg.det(rotation_sim) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation_sim = u_matrix @ vt_matrix
    scale = max(float(scale), 1.0e-8)
    translation = center_shared - scale * (rotation_sim @ calibration.camera_center)
    return SimilarityTransform(
        scale=scale, rotation=rotation_sim, translation=translation
    )


def _copy_similarity(similarity: SimilarityTransform) -> SimilarityTransform:
    """Independent copy so later BA cannot mutate a cached pose."""
    return SimilarityTransform(
        scale=float(similarity.scale),
        rotation=np.asarray(similarity.rotation, dtype=np.float64).copy(),
        translation=np.asarray(similarity.translation, dtype=np.float64).copy(),
    )


def _rounded_floats(values, digits: int = 8) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in np.asarray(values).ravel())


def _calibration_key(calibration: core.Calibration) -> tuple:
    """Private-frame K + pose; origin / FOV changes must miss the cache."""
    intrinsics = calibration.intrinsics
    return (
        round(float(intrinsics.fx), 6),
        round(float(intrinsics.fy), 6),
        round(float(intrinsics.cx), 6),
        round(float(intrinsics.cy), 6),
        int(intrinsics.image_width),
        int(intrinsics.image_height),
        round(float(calibration.division_lambda), 8),
        _rounded_floats(calibration.brown_conrady),
        _rounded_floats(calibration.camera_center),
        _rounded_floats(calibration.rotation_w2c),
    )


def _observation_point_bit(item: SyncObservation) -> tuple:
    return (
        item.match_id,
        item.landmark_id,
        round(float(item.u), 4),
        round(float(item.v), 4),
        round(float(item.weight), 6),
        bool(item.on_ground),
    )


def _observation_line_bit(item: SyncLineObservation) -> tuple:
    return (
        item.match_id,
        item.landmark_id,
        round(float(item.u1), 4),
        round(float(item.v1), 4),
        round(float(item.u2), 4),
        round(float(item.v2), 4),
        round(float(item.weight), 6),
    )


def _pair_correspondence_key(
    anchor_id: str,
    other_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    known_world: dict[str, np.ndarray] | None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] | None,
    parallel_pairs: list[tuple[str, str]] | None,
    *,
    lock_rotation: bool,
    lock_translation: bool,
) -> tuple:
    """Fingerprint the two-view inputs that determine relative pose.

    A re-pick on a third still does not invalidate this pair. Known 3D seen
    only in ``other`` is included because PnP uses it.
    """
    pair_ids = {anchor_id, other_id}
    involved: set[str] = set()
    point_bits: list[tuple] = []
    line_bits: list[tuple] = []
    for landmark_id, items in observations_by_landmark.items():
        by_match = {item.match_id: item for item in items}
        in_pair = pair_ids.intersection(by_match)
        used = False
        if anchor_id in by_match and other_id in by_match:
            used = True
        elif known_world and landmark_id in known_world and other_id in by_match:
            used = True
        if not used:
            continue
        involved.add(landmark_id)
        for match_id in in_pair:
            point_bits.append(_observation_point_bit(by_match[match_id]))
        if known_world and landmark_id in known_world:
            point_bits.append(
                ("known3d", landmark_id, *_rounded_floats(known_world[landmark_id]))
            )
    for landmark_id, items in (line_observations_by_landmark or {}).items():
        by_match = {item.match_id: item for item in items}
        in_pair = pair_ids.intersection(by_match)
        used = False
        if in_pair:
            used = True
        elif known_lines and landmark_id in known_lines and other_id in by_match:
            used = True
        if not used:
            continue
        involved.add(landmark_id)
        for match_id in in_pair:
            line_bits.append(_observation_line_bit(by_match[match_id]))
        if known_lines and landmark_id in known_lines:
            point_a, point_b = known_lines[landmark_id]
            line_bits.append(
                (
                    "known3d",
                    landmark_id,
                    *_rounded_floats(point_a),
                    *_rounded_floats(point_b),
                )
            )
    parallel_bits = tuple(
        sorted(
            pair
            for pair in (parallel_pairs or [])
            if pair[0] in involved or pair[1] in involved
        )
    )
    return (
        anchor_id,
        other_id,
        bool(lock_rotation),
        bool(lock_translation),
        tuple(sorted(point_bits)),
        tuple(sorted(line_bits)),
        _calibration_key(matches[anchor_id].calibration),
        _calibration_key(matches[other_id].calibration),
        parallel_bits,
    )


def _pairs_only_key(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    *,
    lock_rotation: bool,
    lock_translation: bool,
) -> tuple:
    bits = tuple(
        sorted(
            (
                getattr(anchor_obs, "landmark_id", ""),
                round(float(anchor_obs.u), 4),
                round(float(anchor_obs.v), 4),
                round(float(anchor_obs.weight), 6),
                round(float(other_obs.u), 4),
                round(float(other_obs.v), 4),
                round(float(other_obs.weight), 6),
            )
            for anchor_obs, other_obs in pairs
        )
    )
    return (
        bits,
        _calibration_key(anchor),
        _calibration_key(other),
        bool(lock_rotation),
        bool(lock_translation),
    )


_PAIR_CACHE: dict[tuple, object] = {}
_PAIR_CACHE_LOCK = threading.Lock()
_CACHE_MISS = object()


def clear_registration_cache() -> None:
    """Drop cached pairwise poses (Clear Sync / tests)."""
    with _PAIR_CACHE_LOCK:
        _PAIR_CACHE.clear()


def _cached_similarity(value: SimilarityTransform | None) -> SimilarityTransform | None:
    if value is None:
        return None
    return _copy_similarity(value)


def _pair_worker_count(job_count: int) -> int:
    """One worker per independent pair, capped at the host CPU count."""
    if job_count <= 1:
        return 1
    return min(int(job_count), os.cpu_count() or 1)


def _map_pair_jobs(func, items: list):
    """Run independent pair solves; keep input order."""
    if not items:
        return []
    if len(items) == 1:
        return [func(items[0])]
    workers = _pair_worker_count(len(items))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items))


def _is_collapsed_scale(scale: float) -> bool:
    """True when log-scale sat on the LM clip floor (Empty would vanish)."""
    if float(scale) <= 0.0:
        return True
    return abs(math.log(float(scale))) >= LOG_SCALE_CLIP - 0.5


def _metric_scale_similarity(
    similarity: SimilarityTransform,
    calibration: core.Calibration,
) -> SimilarityTransform:
    """Keep the shared camera; rewrite a collapsed Empty scale as rigid s=1."""
    if not _is_collapsed_scale(similarity.scale):
        return similarity
    center_shared = similarity.transform_point(calibration.camera_center)
    rotation_w2c_shared = calibration.rotation_w2c @ similarity.rotation.T
    return _similarity_from_camera_pose(
        calibration,
        rotation_w2c_shared,
        center_shared,
        scale=1.0,
    )



@lru_cache(maxsize=1)
def _axis_aligned_rotations() -> tuple[np.ndarray, ...]:
    """The 24 proper rotations that map world axes onto world axes."""
    matrices: list[np.ndarray] = []
    for perm in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        inversions = int(perm[0] > perm[1]) + int(perm[0] > perm[2]) + int(perm[1] > perm[2])
        perm_sign = 1.0 if inversions % 2 == 0 else -1.0
        for sign_x in (1.0, -1.0):
            for sign_y in (1.0, -1.0):
                sign_z = perm_sign * sign_x * sign_y
                matrix = np.zeros((3, 3), dtype=np.float64)
                matrix[perm[0], 0] = sign_x
                matrix[perm[1], 1] = sign_y
                matrix[perm[2], 2] = sign_z
                matrices.append(matrix)
    return tuple(matrices)


def _snap_to_axis_aligned_rotation(rotation: np.ndarray) -> np.ndarray:
    """Nearest 90° axis permutation (preserves handedness)."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    best = _axis_aligned_rotations()[0]
    best_score = float("-inf")
    for candidate in _axis_aligned_rotations():
        score = float(np.tensordot(candidate, rotation))
        if score > best_score:
            best_score = score
            best = candidate
    return best.copy()


def _axis_aligned_pose_seeds(
    anchor: core.Calibration,
    other: core.Calibration,
    scale_guesses: list[float],
) -> list[SimilarityTransform]:
    """Empty poses whose rotation is a 90° world-axis jump."""
    seeds: list[SimilarityTransform] = []
    for scale in scale_guesses:
        for rotation in _axis_aligned_rotations():
            seeds.append(
                SimilarityTransform(
                    scale=scale,
                    rotation=rotation.copy(),
                    translation=anchor.camera_center
                    - scale * (rotation @ other.camera_center),
                )
            )
    return seeds


def _apply_pose_locks(
    seed: SimilarityTransform,
    *,
    lock_scale: bool = False,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> SimilarityTransform:
    """Force locked scale/translation to identity; snap locked rotation to 90°."""
    scale = 1.0 if lock_scale else float(seed.scale)
    if lock_rotation:
        rotation = _snap_to_axis_aligned_rotation(seed.rotation)
    else:
        rotation = np.asarray(seed.rotation, dtype=np.float64).reshape(3, 3).copy()
    translation = (
        np.zeros(3, dtype=np.float64)
        if lock_translation
        else np.asarray(seed.translation, dtype=np.float64).reshape(3).copy()
    )
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def _pack_similarity_pose(
    similarity: SimilarityTransform,
    *,
    lock_scale: bool = False,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> np.ndarray:
    """Pack pose params — layout mirrors lock_scale / lock_rotation / lock_translation."""
    parts: list[np.ndarray] = []
    if not lock_scale:
        parts.append(
            np.asarray(
                [float(np.log(max(float(similarity.scale), 1.0e-8)))],
                dtype=np.float64,
            )
        )
    if not lock_rotation:
        parts.append(_log_rodrigues(similarity.rotation))
    if not lock_translation:
        parts.append(np.asarray(similarity.translation, dtype=np.float64).reshape(3))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def _unpack_similarity_pose(
    params: np.ndarray,
    *,
    lock_scale: bool = False,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    fixed_scale: float = 1.0,
    fixed_rotation: np.ndarray | None = None,
    fixed_translation: np.ndarray | None = None,
) -> SimilarityTransform:
    """Unpack pose params into a similarity."""
    offset = 0
    if lock_scale:
        scale = max(float(fixed_scale), 1.0e-8)
    else:
        # Least-squares may briefly probe extreme log-scales while initializing
        # a planar solve. Keep those trial values finite without constraining
        # any physically useful scene scale.
        scale = float(
            np.exp(
                np.clip(float(params[0]), -LOG_SCALE_CLIP, LOG_SCALE_CLIP)
            )
        )
        offset = 1
    if lock_rotation:
        rotation = (
            np.asarray(fixed_rotation, dtype=np.float64).reshape(3, 3).copy()
            if fixed_rotation is not None
            else np.eye(3, dtype=np.float64)
        )
    else:
        rotation = _rodrigues(params[offset : offset + 3])
        offset += 3
    if lock_translation:
        translation = (
            np.asarray(fixed_translation, dtype=np.float64).reshape(3).copy()
            if fixed_translation is not None
            else np.zeros(3, dtype=np.float64)
        )
    else:
        translation = params[offset : offset + 3].copy()
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def _square_pixel_intrinsics_if_stretched(calibration: core.Calibration) -> bool:
    """Set fy=fx when they differ enough that K was likely aspect-stretched.

    Copying a portrait locked K onto a landscape still used to scale fx and fy
    by independent width/height ratios. That K is not a Euclidean pinhole and
    plane-homography pose cannot lock.
    """
    fx = float(calibration.intrinsics.fx)
    fy = float(calibration.intrinsics.fy)
    if abs(fx - fy) <= STRETCHED_PIXEL_RATIO * max(abs(fx), abs(fy), 1.0):
        return False
    calibration.intrinsics.fy = fx
    return True


def _coplanar_inlier_indices(
    points: np.ndarray,
    *,
    relative_thickness: float = 0.05,
) -> np.ndarray | None:
    """Indices of a non-collinear coplanar subset, or None."""
    stacked = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = len(stacked)
    if count < 4:
        return None
    centered = stacked - stacked.mean(axis=0)
    _u_matrix, singular, _vt_matrix = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2:
        return None
    span = float(singular[0])
    if span < 1.0e-8 or float(singular[1]) < 0.02 * span:
        return None
    if singular.size >= 3 and float(singular[2]) <= relative_thickness * span:
        return np.arange(count)
    thickness = relative_thickness * span
    best_mask: np.ndarray | None = None
    best_count = 3
    for index_a, index_b, index_c in combinations(range(count), 3):
        origin = stacked[index_a]
        normal = np.cross(
            stacked[index_b] - origin,
            stacked[index_c] - origin,
        )
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-10:
            continue
        distances = np.abs((stacked - origin) @ (normal / norm))
        mask = distances <= thickness
        inlier_count = int(np.count_nonzero(mask))
        if inlier_count > best_count:
            best_count = inlier_count
            best_mask = mask
    if best_mask is None or best_count < 4:
        return None
    return np.flatnonzero(best_mask)


def _rotation_mapping_vector_to_z(vector: np.ndarray) -> np.ndarray:
    """Rotation that maps ``vector`` onto +Z (IPPE ``rotateVec2ZAxis``)."""
    axis = np.asarray(vector, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    ax_coord, ay_coord, cosine = float(axis[0]), float(axis[1]), float(axis[2])
    if abs(1.0 + cosine) < 1.0e-15:
        return np.diag((1.0, 1.0, -1.0)).astype(np.float64)
    scale = 1.0 / (1.0 + cosine)
    return np.array(
        (
            (-ax_coord * ax_coord * scale + 1.0, -ax_coord * ay_coord * scale, -ax_coord),
            (-ax_coord * ay_coord * scale, -ay_coord * ay_coord * scale + 1.0, -ay_coord),
            (ax_coord, ay_coord, 1.0 - (ax_coord * ax_coord + ay_coord * ay_coord) * scale),
        ),
        dtype=np.float64,
    )


def _ippe_rotations_from_homography(homography: np.ndarray) -> list[np.ndarray]:
    """Two plane-to-camera rotations from H (Collins IPPE; OpenCV calib3d/ippe.cpp)."""
    scale = float(homography[2, 2])
    if abs(scale) < 1.0e-12:
        return []
    matrix = homography / scale
    jacobian_00 = matrix[0, 0] - matrix[2, 0] * matrix[0, 2]
    jacobian_01 = matrix[0, 1] - matrix[2, 1] * matrix[0, 2]
    jacobian_10 = matrix[1, 0] - matrix[2, 0] * matrix[1, 2]
    jacobian_11 = matrix[1, 1] - matrix[2, 1] * matrix[1, 2]
    p_coord = float(matrix[0, 2])
    q_coord = float(matrix[1, 2])
    rotate_v = _rotation_mapping_vector_to_z(
        np.array((p_coord, q_coord, 1.0), dtype=np.float64)
    ).T
    b00 = rotate_v[0, 0] - p_coord * rotate_v[2, 0]
    b01 = rotate_v[0, 1] - p_coord * rotate_v[2, 1]
    b10 = rotate_v[1, 0] - q_coord * rotate_v[2, 0]
    b11 = rotate_v[1, 1] - q_coord * rotate_v[2, 1]
    determinant = b00 * b11 - b01 * b10
    if abs(float(determinant)) < 1.0e-12:
        return []
    inverse = 1.0 / determinant
    binv00, binv01 = inverse * b11, -inverse * b01
    binv10, binv11 = -inverse * b10, inverse * b00
    a00 = binv00 * jacobian_00 + binv01 * jacobian_10
    a01 = binv00 * jacobian_01 + binv01 * jacobian_11
    a10 = binv10 * jacobian_00 + binv11 * jacobian_10
    a11 = binv10 * jacobian_01 + binv11 * jacobian_11
    ata00 = a00 * a00 + a01 * a01
    ata01 = a00 * a10 + a01 * a11
    ata11 = a10 * a10 + a11 * a11
    gamma_sq = 0.5 * (
        ata00 + ata11 + math.sqrt(max((ata00 - ata11) ** 2 + 4.0 * ata01 * ata01, 0.0))
    )
    if gamma_sq < 1.0e-16:
        return []
    gamma = math.sqrt(gamma_sq)
    r00, r01 = a00 / gamma, a01 / gamma
    r10, r11 = a10 / gamma, a11 / gamma
    b0 = math.sqrt(max(1.0 - r00 * r00 - r10 * r10, 0.0))
    b1 = math.sqrt(max(1.0 - r01 * r01 - r11 * r11, 0.0))
    if (-r00 * r01 - r10 * r11) < 0.0:
        b1 = -b1
    det_r = r00 * r11 - r01 * r10
    rtilde_a = np.array(
        (
            (r00, r01, b1 * r10 - b0 * r11),
            (r10, r11, b0 * r01 - b1 * r00),
            (b0, b1, det_r),
        ),
        dtype=np.float64,
    )
    rtilde_b = np.array(
        (
            (r00, r01, b0 * r11 - b1 * r10),
            (r10, r11, b1 * r00 - b0 * r01),
            (-b0, -b1, det_r),
        ),
        dtype=np.float64,
    )
    return [rotate_v @ rtilde_a, rotate_v @ rtilde_b]


def _ippe_translation(
    rotation: np.ndarray,
    plane_xy: np.ndarray,
    normalized_xy: np.ndarray,
) -> np.ndarray:
    """Least-squares camera translation for a known rotation and Z=0 points."""
    rows: list[tuple[float, float, float]] = []
    rhs: list[float] = []
    for (x_object, y_object), (x_image, y_image) in zip(plane_xy, normalized_xy):
        rotated = rotation @ np.array((x_object, y_object, 0.0), dtype=np.float64)
        rows.append((1.0, 0.0, -float(x_image)))
        rhs.append(float(x_image) * float(rotated[2]) - float(rotated[0]))
        rows.append((0.0, 1.0, -float(y_image)))
        rhs.append(float(y_image) * float(rotated[2]) - float(rotated[1]))
    translation, _, _, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(rhs, dtype=np.float64),
        rcond=None,
    )
    return translation


def _decompose_plane_homography(
    homography: np.ndarray,
    plane_xy: np.ndarray,
    normalized_xy: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plane-to-camera poses from H: [X, Y, 1] on Z=0 → λ [x, y, 1]."""
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for rotation in _ippe_rotations_from_homography(homography):
        translation = _ippe_translation(rotation, plane_xy, normalized_xy)
        if np.all(np.isfinite(translation)):
            poses.append((rotation, translation))
    # Gram-Schmidt on H's first two columns (Zhang) plus LS translation.
    # IPPE's Jacobian path can miss when the plane origin is far from the PP
    # in a high-focal still; this recovers the same H as a pinhole pose.
    for sign in (1.0, -1.0):
        matrix = sign * np.asarray(homography, dtype=np.float64).reshape(3, 3)
        column_a = matrix[:, 0]
        column_b = matrix[:, 1]
        norm_a = float(np.linalg.norm(column_a))
        if norm_a < 1.0e-12:
            continue
        axis_a = column_a / norm_a
        column_b = column_b - axis_a * float(np.dot(axis_a, column_b))
        norm_b = float(np.linalg.norm(column_b))
        if norm_b < 1.0e-12:
            continue
        axis_b = column_b / norm_b
        axis_c = np.cross(axis_a, axis_b)
        rotation = np.column_stack((axis_a, axis_b, axis_c))
        if np.linalg.det(rotation) < 0.0:
            axis_c = -axis_c
            rotation = np.column_stack((axis_a, axis_b, axis_c))
        translation = _ippe_translation(rotation, plane_xy, normalized_xy)
        if np.all(np.isfinite(translation)):
            poses.append((rotation, translation))
    return poses


def _planar_homography_similarities(
    points_shared: np.ndarray,
    points_image: np.ndarray,
    calibration: core.Calibration,
    *,
    weights: np.ndarray | None = None,
) -> list[SimilarityTransform]:
    """Closed-form pose from coplanar 3D↔2D, then a short rigid PnP polish.

    Generic PnP LM from yaw seeds misses cameras that look along the plane
    normal (image ≈ ground). A plane homography is well-posed there.
    """
    shared = np.asarray(points_shared, dtype=np.float64).reshape(-1, 3)
    image = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
    if len(shared) != len(image) or len(shared) < 4:
        return []
    inliers = _coplanar_inlier_indices(shared)
    if inliers is None:
        return []
    plane_points = shared[inliers]
    plane_image = image[inliers]
    centroid = plane_points.mean(axis=0)
    centered = plane_points - centroid
    _u_matrix, _singular, vt_matrix = np.linalg.svd(centered, full_matrices=False)
    normal = vt_matrix[-1]
    if float(normal[2]) < 0.0:
        normal = -normal
    tangent_a, tangent_b = _plane_tangent_basis(normal)
    plane_xy = np.column_stack((centered @ tangent_a, centered @ tangent_b))
    normalized: list[np.ndarray] = []
    keep: list[int] = []
    for index, (u_coord, v_coord) in enumerate(plane_image):
        ray = _normalized_camera_ray(float(u_coord), float(v_coord), calibration)
        depth = float(ray[2])
        if abs(depth) < 1.0e-12:
            continue
        normalized.append(np.array((ray[0] / depth, ray[1] / depth), dtype=np.float64))
        keep.append(index)
    if len(keep) < 4:
        return []
    homography = _fit_homography_dlt(plane_xy[keep], np.stack(normalized))
    if homography is None:
        return []
    plane_from_shared = np.column_stack((tangent_a, tangent_b, normal)).T
    similarities: list[SimilarityTransform] = []
    seen: list[np.ndarray] = []
    for rotation_plane, translation_plane in _decompose_plane_homography(
        homography,
        plane_xy[keep],
        np.stack(normalized),
    ):
        rotation_w2c = rotation_plane @ plane_from_shared
        translation_shared = translation_plane - rotation_w2c @ centroid
        camera_points = (shared - (-rotation_w2c.T @ translation_shared)) @ rotation_w2c.T
        if float(np.median(camera_points[:, 2])) <= 1.0e-6:
            continue
        center_shared = -rotation_w2c.T @ translation_shared
        seed = _similarity_from_camera_pose(calibration, rotation_w2c, center_shared)
        duplicate = False
        for previous in seen:
            if float(np.linalg.norm(seed.translation - previous)) < 1.0e-6:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(seed.translation.copy())
        similarities.append(seed)
        polished = _pnp_similarity(
            "planar",
            shared,
            image,
            calibration,
            initial=seed,
            weights=weights,
            lock_scale=True,
        )
        if polished is not None:
            similarities.append(polished)
    return similarities


def _pnp_similarity(
    match_id: str,
    points_shared: np.ndarray,
    points_image: np.ndarray,
    calibration: core.Calibration,
    *,
    initial: SimilarityTransform | None = None,
    max_iterations: int = 40,
    weights: np.ndarray | None = None,
    lock_scale: bool = True,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> SimilarityTransform | None:
    """Solve Empty pose from shared 3D ↔ image 2D correspondences.

    ``lock_scale=True`` (default) keeps ``s=1`` — preferred when private worlds
    already share metric units. Free scale is a fallback when rigid PnP fails.
    ``lock_rotation=True`` keeps R on a 90° axis jump; ``lock_translation=True``
    keeps ``t=0``.
    """
    if len(points_shared) < 3:
        return None
    if lock_rotation and not lock_translation and initial is None:
        best: SimilarityTransform | None = None
        best_cost = float("inf")
        shared_array = np.asarray(points_shared, dtype=np.float64).reshape(-1, 3)
        image_array = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
        for rotation in _axis_aligned_rotations():
            candidate = _pnp_similarity(
                match_id,
                points_shared,
                points_image,
                calibration,
                initial=SimilarityTransform(rotation=rotation.copy()),
                max_iterations=max_iterations,
                weights=weights,
                lock_scale=lock_scale,
                lock_rotation=True,
                lock_translation=False,
            )
            if candidate is None:
                continue
            projected, valid = _project_shared_points(
                shared_array, calibration, candidate
            )
            errors = projected - image_array
            errors[~valid] = 1.0e3
            cost = float(np.mean(errors * errors))
            if cost < best_cost:
                best_cost = cost
                best = candidate
        return best
    seed = _apply_pose_locks(
        initial or SimilarityTransform(),
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    if lock_rotation and lock_translation:
        return seed
    params = _pack_similarity_pose(
        seed,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    fixed_scale = float(seed.scale)
    fixed_rotation = seed.rotation.copy()
    fixed_translation = seed.translation.copy()
    if weights is None:
        point_weights = np.ones(len(points_shared), dtype=np.float64)
    else:
        point_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if point_weights.size != len(points_shared):
            point_weights = np.ones(len(points_shared), dtype=np.float64)

    shared_array = np.asarray(points_shared, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
    residual_scales = np.sqrt(np.maximum(point_weights, 1.0e-12))[:, None]

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = _unpack_similarity_pose(
            values,
            lock_scale=lock_scale,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            fixed_scale=fixed_scale,
            fixed_rotation=fixed_rotation,
            fixed_translation=fixed_translation,
        )
        projected, valid = _project_shared_points(
            shared_array, calibration, similarity
        )
        errors = residual_scales * (projected - image_array)
        errors[~valid] = residual_scales[~valid] * 1.0e3
        return errors.reshape(-1)

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

    return _unpack_similarity_pose(
        params,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        fixed_scale=fixed_scale,
        fixed_rotation=fixed_rotation,
        fixed_translation=fixed_translation,
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


def _ground_metric_agrees(
    metric_point: np.ndarray,
    triangulated: np.ndarray,
) -> bool:
    """True when an On Ground raycast matches multi-view triangulation.

    Off-plane points marked On Ground raycast to a different XY on Z=0 than
    the triangulated 3D point. Those must not pin BA.
    """
    metric = np.asarray(metric_point, dtype=np.float64).reshape(3)
    triangulated_point = np.asarray(triangulated, dtype=np.float64).reshape(3)
    delta = float(np.linalg.norm(metric - triangulated_point))
    scale = max(
        float(np.linalg.norm(metric)),
        float(np.linalg.norm(triangulated_point)),
        1.0e-3,
    )
    z_delta = abs(float(triangulated_point[2] - metric[2]))
    return delta <= GROUND_PLANE_Z_FRACTION * scale and z_delta <= GROUND_PLANE_Z_FRACTION * scale


def _consistent_metric_landmarks(
    metric: dict[str, np.ndarray],
    triangulated: dict[str, np.ndarray],
    known_world_ids: set[str],
    *,
    ground_slack: float = 0.0,
) -> dict[str, np.ndarray]:
    """Keep Known 3D always; keep On Ground only when triangulation agrees.

    ``ground_slack > 0`` leaves On Ground free so joint BA can pull Z toward
    the plane with a spring instead of pinning the Z=0 raycast.
    """
    consistent: dict[str, np.ndarray] = {}
    slack = max(float(ground_slack), 0.0)
    for landmark_id, point in metric.items():
        if landmark_id in known_world_ids:
            consistent[landmark_id] = point
            continue
        if slack > 1.0e-12:
            continue
        triangulated_point = triangulated.get(landmark_id)
        if triangulated_point is None or _ground_metric_agrees(
            point, triangulated_point
        ):
            consistent[landmark_id] = point
    return consistent


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
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> SimilarityTransform:
    """Levenberg–Marquardt on Empty (R, t) minimizing ray–ray distances."""
    seed = _apply_pose_locks(
        seed,
        lock_scale=True,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    if lock_rotation and lock_translation:
        return seed
    params = _pack_similarity_pose(
        seed,
        lock_scale=True,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    fixed_rotation = seed.rotation.copy()
    fixed_translation = seed.translation.copy()
    anchor_directions: list[np.ndarray] = []
    other_directions: list[np.ndarray] = []
    pair_scales: list[float] = []
    for anchor_obs, other_obs in pairs:
        _origin_a, direction_a = camera_ray_private(
            anchor_obs.u, anchor_obs.v, anchor
        )
        _origin_b, direction_b = camera_ray_private(
            other_obs.u, other_obs.v, other
        )
        anchor_directions.append(direction_a)
        other_directions.append(direction_b)
        pair_scales.append(_pair_scale(anchor_obs, other_obs))
    anchor_direction_array = np.stack(anchor_directions)
    other_direction_array = np.stack(other_directions)
    pair_scale_array = np.asarray(pair_scales, dtype=np.float64)
    origin_a = anchor.camera_center
    origin_b_private = other.camera_center

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = _unpack_similarity_pose(
            values,
            lock_scale=True,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            fixed_scale=1.0,
            fixed_rotation=fixed_rotation,
            fixed_translation=fixed_translation,
        )
        origin_b = similarity.transform_point(origin_b_private)
        direction_b = other_direction_array @ similarity.rotation.T
        normals = np.cross(anchor_direction_array, direction_b)
        normal_norms = np.linalg.norm(normals, axis=1)
        offset = origin_a - origin_b
        distances = np.empty(len(pairs), dtype=np.float64)
        regular = normal_norms >= 1.0e-10
        distances[regular] = (
            np.abs(normals[regular] @ offset) / normal_norms[regular]
        )
        if np.any(~regular):
            distances[~regular] = np.linalg.norm(
                np.cross(offset, direction_b[~regular]),
                axis=1,
            )
        return pair_scale_array * distances

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

    return _unpack_similarity_pose(
        params,
        lock_scale=True,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        fixed_scale=1.0,
        fixed_rotation=fixed_rotation,
        fixed_translation=fixed_translation,
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
    *,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    use_pose_cache: bool = False,
) -> SimilarityTransform | None:
    """Multi-start ray-distance LM for Empty (R, t) from 2D↔2D pairs only."""
    if use_pose_cache:
        cache_key = (
            "pairs",
            _pairs_only_key(
                pairs,
                anchor,
                other,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
            ),
        )
        with _PAIR_CACHE_LOCK:
            hit = _PAIR_CACHE.get(cache_key, _CACHE_MISS)
            if hit is not _CACHE_MISS:
                return _cached_similarity(hit)  # type: ignore[arg-type]
        solved = _compute_relative_from_pairs(
            pairs,
            anchor,
            other,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
        )
        with _PAIR_CACHE_LOCK:
            _PAIR_CACHE[cache_key] = _cached_similarity(solved)
        return solved
    return _compute_relative_from_pairs(
        pairs,
        anchor,
        other,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )


def _compute_relative_from_pairs(
    pairs: list[tuple[SyncObservation, SyncObservation]],
    anchor: core.Calibration,
    other: core.Calibration,
    *,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> SimilarityTransform | None:
    """Uncached 2D↔2D relative pose (see ``_solve_relative_from_pairs``)."""
    if len(pairs) < 5:
        return None
    if lock_rotation and lock_translation:
        return SimilarityTransform()
    seeds: list[SimilarityTransform] = []

    def add_rotation_seeds(rotation: np.ndarray) -> None:
        """Seed both baseline signs from the epipolar nullspace for one R."""
        directions_a = []
        directions_b = []
        pair_scales = []
        for anchor_obs, other_obs in pairs:
            _origin_a, direction_a = camera_ray_private(
                anchor_obs.u, anchor_obs.v, anchor
            )
            _origin_b, direction_b = camera_ray_private(
                other_obs.u, other_obs.v, other
            )
            directions_a.append(direction_a)
            directions_b.append(rotation @ direction_b)
            pair_scales.append(_pair_scale(anchor_obs, other_obs))
        normals = np.cross(np.stack(directions_a), np.stack(directions_b))
        normals *= np.asarray(pair_scales, dtype=np.float64)[:, None]
        _u_matrix, _singular, vt_matrix = np.linalg.svd(
            normals, full_matrices=False
        )
        baseline_direction = vt_matrix[-1]
        norm = float(np.linalg.norm(baseline_direction))
        if norm < 1.0e-10:
            return
        baseline_direction /= norm
        # Two-view translation has no absolute scale. Start far enough from
        # the zero-baseline minimum that ray-distance LM can refine R and t.
        baseline = max(float(np.linalg.norm(anchor.camera_center)), 5.0)
        for sign in (1.0, -1.0):
            center_b = (
                anchor.camera_center + sign * baseline * baseline_direction
            )
            seeds.append(
                SimilarityTransform(
                    scale=1.0,
                    rotation=rotation.copy(),
                    translation=center_b - rotation @ other.camera_center,
                )
            )

    # Match-local VP/world axes may differ by any proper 90° permutation.
    # These 24 seeds also cover the opposite camera hemisphere and 180° yaw.
    for rotation in _axis_aligned_rotations():
        add_rotation_seeds(rotation)

    # Locked rotation: keep the axis-aligned rotations above; skip essential.
    if not lock_rotation:
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

    best: SimilarityTransform | None = None
    best_key: tuple[float, float, float] | None = None
    for seed in seeds:
        refined = _refine_rigid_from_rays(
            seed, pairs, anchor, other,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
        )
        center_b = refined.transform_point(other.camera_center)
        baseline = float(np.linalg.norm(center_b - anchor.camera_center))
        # Absolute baseline length is unobservable from 2D↔2D pairs. Reject
        # only numerical collapse; a Blender-unit threshold can discard an
        # otherwise exact solution merely because LM chose a smaller scale.
        if baseline < 1.0e-8:
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
            if depth_a <= 1.0e-6 * baseline or depth_b <= 1.0e-6 * baseline:
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
        key = (-float(cheirality), mean_reproj, cost / (baseline * baseline))
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
    empty_scale = max(float(similarity.scale), 1.0e-8)
    for factor in np.linspace(0.2, 5.0, 49):
        alpha = offset_norm * float(factor)
        center_b = anchor.camera_center + alpha * direction
        translation = center_b - empty_scale * (rotation @ other.camera_center)
        trial = SimilarityTransform(
            scale=empty_scale,
            rotation=rotation,
            translation=translation,
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
        scale=float(similarity.scale),
        rotation=rotation,
        translation=center_b - float(similarity.scale) * (rotation @ other.camera_center),
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
    axis_line_constraints: list[tuple[np.ndarray, SyncLineObservation]]
    | None = None,
    parallel_weight: float = 0.0,
    lock_scale: bool = True,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> SimilarityTransform:
    """LM on Empty pose: free 2D↔2D ray gaps + Known 3D point/line reprojection."""
    seed = _apply_pose_locks(
        seed,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    if lock_rotation and lock_translation:
        return seed
    params = _pack_similarity_pose(
        seed,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )
    fixed_scale = float(seed.scale)
    fixed_rotation = seed.rotation.copy()
    fixed_translation = seed.translation.copy()
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
    axis_constraints = axis_line_constraints or []
    anchor_directions: list[np.ndarray] = []
    other_directions: list[np.ndarray] = []
    pair_scales: list[float] = []
    for anchor_obs, other_obs in free_pairs:
        _origin_a, direction_a = camera_ray_private(
            anchor_obs.u, anchor_obs.v, anchor
        )
        _origin_b, direction_b = camera_ray_private(
            other_obs.u, other_obs.v, other
        )
        anchor_directions.append(direction_a)
        other_directions.append(direction_b)
        pair_scales.append(_pair_scale(anchor_obs, other_obs))
    anchor_direction_array = (
        np.stack(anchor_directions)
        if anchor_directions
        else np.empty((0, 3), dtype=np.float64)
    )
    other_direction_array = (
        np.stack(other_directions)
        if other_directions
        else np.empty((0, 3), dtype=np.float64)
    )
    pair_scale_array = np.asarray(pair_scales, dtype=np.float64)
    origin_a = anchor.camera_center
    origin_b_private = other.camera_center
    shared_array = np.asarray(points_shared, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
    known_scale_array = np.sqrt(
        np.maximum(np.asarray(known_weights, dtype=np.float64), 1.0e-12)
    )[:, None]

    def residual(values: np.ndarray) -> np.ndarray:
        similarity = _unpack_similarity_pose(
            values,
            lock_scale=lock_scale,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            fixed_scale=fixed_scale,
            fixed_rotation=fixed_rotation,
            fixed_translation=fixed_translation,
        )
        errors: list[float] = []
        if len(free_pairs):
            origin_b = similarity.transform_point(origin_b_private)
            direction_b = other_direction_array @ similarity.rotation.T
            normals = np.cross(anchor_direction_array, direction_b)
            normal_norms = np.linalg.norm(normals, axis=1)
            offset = origin_a - origin_b
            distances = np.empty(len(free_pairs), dtype=np.float64)
            regular = normal_norms >= 1.0e-10
            distances[regular] = (
                np.abs(normals[regular] @ offset) / normal_norms[regular]
            )
            if np.any(~regular):
                distances[~regular] = np.linalg.norm(
                    np.cross(offset, direction_b[~regular]),
                    axis=1,
                )
            errors.extend(
                (pair_scale_array * ray_to_px * distances).tolist()
            )
        if len(shared_array):
            projected, valid = _project_shared_points(
                shared_array, other, similarity
            )
            point_errors = known_scale_array * (projected - image_array)
            point_errors[~valid] = known_scale_array[~valid] * 1.0e3
            errors.extend(point_errors.reshape(-1).tolist())
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
            for direction, observation in axis_constraints:
                axis_error = _axis_line_rotation_error(
                    direction, observation, other, similarity
                )
                if axis_error is not None:
                    errors.append(float(parallel_weight) * axis_error)
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

    return _unpack_similarity_pose(
        params,
        lock_scale=lock_scale,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        fixed_scale=fixed_scale,
        fixed_rotation=fixed_rotation,
        fixed_translation=fixed_translation,
    )



def _mixed_pose_seeds(
    anchor: core.Calibration,
    other: core.Calibration,
    *,
    include_scale_grid: bool = False,
    dense_yaw: bool = False,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> list[SimilarityTransform]:
    """Yaw seeds (and optional scale grid) for Known 3D / mixed registration."""
    scale_guesses = [1.0]
    if include_scale_grid:
        anchor_norm = float(np.linalg.norm(anchor.camera_center))
        other_norm = float(np.linalg.norm(other.camera_center))
        if other_norm > 1.0e-6:
            ratio = max(anchor_norm / other_norm, 1.0e-3)
            for factor in (0.5, 1.0, 2.0):
                guess = ratio * factor
                if all(
                    abs(guess - existing) > 0.05 * max(guess, existing)
                    for existing in scale_guesses
                ):
                    scale_guesses.append(guess)
            for guess in (0.15, 0.25, 0.5, 2.0, 4.0):
                if all(
                    abs(guess - existing) > 0.05 * max(guess, existing)
                    for existing in scale_guesses
                ):
                    scale_guesses.append(guess)

    # Both locks: identity Empty. Rotation-only lock: 90° axis jumps.
    if lock_rotation and lock_translation:
        return [SimilarityTransform(scale=scale) for scale in scale_guesses]
    if lock_rotation:
        return _axis_aligned_pose_seeds(anchor, other, scale_guesses)
    if lock_translation:
        seeds = [SimilarityTransform(scale=scale) for scale in scale_guesses]
        for yaw in (-0.8, -0.4, 0.4, 0.8):
            cosine = float(np.cos(yaw))
            sine = float(np.sin(yaw))
            rotation_yaw = np.array(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            for scale in scale_guesses:
                seeds.append(
                    SimilarityTransform(
                        scale=scale,
                        rotation=rotation_yaw,
                        translation=np.zeros(3, dtype=np.float64),
                    )
                )
        return seeds

    # Dense yaw (incl. ±90°/180°) for pure Known 3D PnP; keep the smaller set
    # elsewhere so line-only registration does not fall into wrong basins.
    if dense_yaw:
        yaw_angles = (
            -np.pi,
            -2.5,
            -2.0,
            -0.5 * np.pi,
            -1.2,
            -0.8,
            -0.4,
            0.0,
            0.4,
            0.8,
            1.2,
            0.5 * np.pi,
            2.0,
            2.5,
            np.pi,
        )
    else:
        yaw_angles = (-1.2, -0.8, -0.4, 0.4, 0.8, 1.2)

    seeds: list[SimilarityTransform] = [
        SimilarityTransform(),
        SimilarityTransform(
            scale=1.0,
            rotation=np.eye(3),
            translation=anchor.camera_center - other.camera_center,
        ),
    ]
    for scale in scale_guesses:
        for yaw in yaw_angles:
            if abs(yaw) < 1.0e-12 and abs(scale - 1.0) < 1.0e-12:
                continue
            cosine = float(np.cos(yaw))
            sine = float(np.sin(yaw))
            rotation_yaw = np.array(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            seeds.append(
                SimilarityTransform(
                    scale=scale,
                    rotation=rotation_yaw,
                    translation=anchor.camera_center
                    - scale * (rotation_yaw @ other.camera_center),
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
    initial_similarity: SimilarityTransform | None = None,
    *,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    use_pose_cache: bool = False,
) -> tuple[SimilarityTransform | None, str]:
    """Register other from free 2D↔2D and/or Known 3D points/lines.

    Known 3D landmarks are *not* used as photo↔photo pairs (auto-projected
    anchor picks can disagree with the photo feature). They only provide metric
    3D↔2D in the other still. Free point landmarks provide 2D↔2D. Known 3D
    lines use 2D segments in the other still. Free 2D↔2D lines need ≥3 stills
    (handled after two matches are registered).
    """
    if use_pose_cache:
        cache_key = (
            "rel",
            _pair_correspondence_key(
                anchor_id,
                other_id,
                observations_by_landmark,
                matches,
                known_world,
                known_lines,
                line_observations_by_landmark,
                parallel_pairs,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
            ),
        )
        with _PAIR_CACHE_LOCK:
            hit = _PAIR_CACHE.get(cache_key, _CACHE_MISS)
            if hit is not _CACHE_MISS:
                solved, detail = hit  # type: ignore[misc]
                return _cached_similarity(solved), detail
        solved, detail = _compute_relative_pose_from_correspondences(
            anchor_id,
            other_id,
            observations_by_landmark,
            matches,
            known_world,
            known_lines=known_lines,
            line_observations_by_landmark=line_observations_by_landmark,
            parallel_pairs=parallel_pairs,
            initial_similarity=initial_similarity,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
        )
        with _PAIR_CACHE_LOCK:
            _PAIR_CACHE[cache_key] = (_cached_similarity(solved), detail)
        return solved, detail
    return _compute_relative_pose_from_correspondences(
        anchor_id,
        other_id,
        observations_by_landmark,
        matches,
        known_world,
        known_lines=known_lines,
        line_observations_by_landmark=line_observations_by_landmark,
        parallel_pairs=parallel_pairs,
        initial_similarity=initial_similarity,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
    )


def _compute_relative_pose_from_correspondences(
    anchor_id: str,
    other_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    known_world: dict[str, np.ndarray] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]]
    | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
    initial_similarity: SimilarityTransform | None = None,
    *,
    lock_rotation: bool = False,
    lock_translation: bool = False,
) -> tuple[SimilarityTransform | None, str]:
    """Uncached pairwise register (see ``_relative_pose_from_correspondences``)."""
    if lock_rotation and lock_translation:
        return SimilarityTransform(), ""
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
    _square_pixel_intrinsics_if_stretched(other)
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

    # Lens trials only perturb one focal at a time, so the previous accepted
    # metric pose is normally in the same basin. Try one local mixed solve
    # before repeating the expensive global yaw/essential multi-start search.
    # Pure 2D↔2D problems keep the global path because their baseline scale is
    # unobservable and a local ray-distance solve can collapse translation.
    if initial_similarity is not None and (
        points_shared or known_line_constraints
    ):
        lock_warm_scale = abs(float(initial_similarity.scale) - 1.0) < 1.0e-6
        warm_seed = initial_similarity
        if lock_rotation or lock_translation:
            warm_seed = _apply_pose_locks(
                warm_seed,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
            )
        warm = _refine_rigid_mixed(
            warm_seed,
            free_pairs,
            points_shared,
            points_image,
            anchor,
            other,
            point_weights=known_weights,
            known_line_constraints=known_line_constraints,
            lock_scale=lock_warm_scale,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
        )
        warm_errors = _reprojection_errors_for_similarity(
            warm,
            free_pairs,
            anchor,
            other,
            points_shared,
            points_image,
            point_weights=known_weights,
            weighted=True,
        )
        for point_a, point_b, line_obs in known_line_constraints:
            warm_errors.extend(
                _known_line_reprojection_errors(
                    point_a,
                    point_b,
                    line_obs,
                    other,
                    warm,
                )
            )
        if warm_errors:
            warm_rmse = float(np.sqrt(np.mean(np.square(warm_errors))))
            if warm_rmse <= ACCEPT_RMSE_PX:
                return warm, ""

    candidates: list[SimilarityTransform] = []
    mixed_kwargs = {
        "point_weights": known_weights,
        "known_line_constraints": known_line_constraints,
        "lock_rotation": lock_rotation,
        "lock_translation": lock_translation,
    }

    if has_metric_pnp and not (lock_rotation and lock_translation):
        for seed in _planar_homography_similarities(
            np.stack(points_shared),
            np.stack(points_image),
            other,
            weights=np.asarray(known_weights, dtype=np.float64),
        ):
            locked = _apply_pose_locks(
                seed,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
            )
            candidates.append(locked)
            if free_pairs or known_line_constraints:
                candidates.append(
                    _refine_rigid_mixed(
                        locked,
                        free_pairs,
                        points_shared,
                        points_image,
                        anchor,
                        other,
                        **mixed_kwargs,
                    )
                )

    relative: SimilarityTransform | None = None
    if can_pairs_only:
        relative = _solve_relative_from_pairs(free_pairs, anchor, other, lock_rotation=lock_rotation, lock_translation=lock_translation)
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
                        lock_rotation=lock_rotation,
                        lock_translation=lock_translation,
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
        for seed in _mixed_pose_seeds(anchor, other, lock_rotation=lock_rotation, lock_translation=lock_translation):
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
        # Multi-start: pure Known 3D has no 2D↔2D essential seed — try yaw grid.
        # Rigid first (s=1); free-scale only if that cannot lock a pose.
        for allow_scale in (False, True):
            best_pnp: SimilarityTransform | None = None
            best_pnp_rmse = float("inf")
            seeds = _mixed_pose_seeds(
                anchor,
                other,
                include_scale_grid=allow_scale,
                dense_yaw=True,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
            )
            for seed in seeds:
                solved = _pnp_similarity(
                    other_id,
                    np.stack(points_shared),
                    np.stack(points_image),
                    other,
                    initial=seed,
                    weights=np.asarray(known_weights, dtype=np.float64),
                    lock_scale=not allow_scale,
                    lock_rotation=lock_rotation,
                    lock_translation=lock_translation,
                )
                if solved is None:
                    continue
                if free_pairs or known_line_constraints:
                    solved = _refine_rigid_mixed(
                        solved,
                        free_pairs,
                        points_shared,
                        points_image,
                        anchor,
                        other,
                        lock_scale=not allow_scale,
                        **mixed_kwargs,
                    )
                errors = _reprojection_errors_for_similarity(
                    solved,
                    free_pairs,
                    anchor,
                    other,
                    points_shared,
                    points_image,
                    point_weights=known_weights,
                    weighted=True,
                )
                if not errors:
                    continue
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                if rmse < best_pnp_rmse:
                    best_pnp_rmse = rmse
                    best_pnp = solved
                if best_pnp_rmse < 8.0:
                    break
            if best_pnp is not None and best_pnp_rmse <= ACCEPT_RMSE_PX:
                candidates.append(best_pnp)
                break
            if best_pnp is not None:
                candidates.append(best_pnp)

    if can_pnl_only:
        for seed in _mixed_pose_seeds(anchor, other, lock_rotation=lock_rotation, lock_translation=lock_translation):
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

    # On Ground / Known 3D can poison a mixed score when those 3D points are
    # wrong. Keep a pure 2D↔2D pose if it still locks.
    if (best is None or best_rmse > ACCEPT_RMSE_PX) and relative is not None and free_pairs_ok:
        pairs_pose = _apply_depth_heuristic_scale(relative, free_pairs, anchor, other)
        pair_errors = _reprojection_errors_for_similarity(
            pairs_pose,
            free_pairs,
            anchor,
            other,
            [],
            [],
            weighted=True,
        )
        if pair_errors:
            pairs_rmse = float(np.sqrt(np.mean(np.square(pair_errors))))
            if pairs_rmse <= ACCEPT_RMSE_PX:
                best = pairs_pose
                best_rmse = pairs_rmse

    if best is None or best_rmse > ACCEPT_RMSE_PX:
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
    axis_specs = _axis_line_constraints_for_match(
        other_id,
        parallel_pairs,
        line_observations_by_landmark,
    )
    if (parallel_specs or axis_specs) and not lock_rotation and not lock_translation:
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
            axis_line_constraints=axis_specs,
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
    return _metric_scale_similarity(best, other), ""


def _point_pairs_between_matches(
    reference_id: str,
    other_id: str,
    observations_by_landmark: dict[str, list[SyncObservation]],
    *,
    excluded_landmark_ids: set[str] | None = None,
) -> list[tuple[SyncObservation, SyncObservation]]:
    """Return ordinary 2D↔2D pairs ordered reference then other."""
    excluded = excluded_landmark_ids or set()
    pairs: list[tuple[SyncObservation, SyncObservation]] = []
    for landmark_id, items in observations_by_landmark.items():
        if landmark_id in excluded:
            continue
        by_match = {item.match_id: item for item in items}
        if reference_id in by_match and other_id in by_match:
            pairs.append((by_match[reference_id], by_match[other_id]))
    return pairs


def _registration_candidate_rmse(
    match_id: str,
    candidate: SimilarityTransform,
    registered: dict[str, SimilarityTransform],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    landmarks: dict[str, np.ndarray],
    *,
    known_world_ids: set[str] | None = None,
) -> float:
    """Score one shared-frame pose against every currently registered view."""
    errors: list[float] = []
    for reference_id, reference_similarity in registered.items():
        pairs = _point_pairs_between_matches(
            reference_id,
            match_id,
            observations_by_landmark,
            excluded_landmark_ids=known_world_ids,
        )
        if not pairs:
            continue
        relative = _compose_similarities(
            _inverse_similarity(reference_similarity),
            candidate,
        )
        errors.extend(
            _reprojection_errors_for_similarity(
                relative,
                pairs,
                matches[reference_id].calibration,
                matches[match_id].calibration,
                [],
                [],
                weighted=True,
            )
        )

    # Multi-view points enforce one shared 3D location and therefore resolve
    # the scale that an isolated two-view relative pose cannot observe.
    for landmark_id, point in landmarks.items():
        items = observations_by_landmark.get(landmark_id, [])
        observation = next(
            (item for item in items if item.match_id == match_id),
            None,
        )
        if observation is None:
            continue
        projected = project_private_point(
            candidate.inverse_point(point),
            matches[match_id].calibration,
        )
        scale = _observation_scale(observation)
        if projected is None:
            errors.append(scale * 1.0e3)
            continue
        errors.append(
            scale
            * float(
                np.hypot(
                    projected[0] - observation.u,
                    projected[1] - observation.v,
                )
            )
        )
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def _registration_strong_pair_rmse(
    match_id: str,
    candidate: SimilarityTransform,
    registered: dict[str, SimilarityTransform],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    *,
    known_world_ids: set[str] | None = None,
) -> float:
    """Score against registered views that independently constrain pose."""
    errors: list[float] = []
    for reference_id, reference_similarity in registered.items():
        pairs = _point_pairs_between_matches(
            reference_id,
            match_id,
            observations_by_landmark,
            excluded_landmark_ids=known_world_ids,
        )
        if len(pairs) < 5:
            continue
        relative = _compose_similarities(
            _inverse_similarity(reference_similarity),
            candidate,
        )
        errors.extend(
            _reprojection_errors_for_similarity(
                relative,
                pairs,
                matches[reference_id].calibration,
                matches[match_id].calibration,
                [],
                [],
                weighted=True,
            )
        )
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def _select_registration_candidate(
    candidates: list[tuple[float, float, SimilarityTransform]],
    *,
    pair_branch_slack_px: float = 5.0,
    acceptance_px: float = ACCEPT_RMSE_PX,
) -> SimilarityTransform | None:
    """Choose graph scale/refinement without leaving the best pairwise branch.

    Candidate tuples are ``(strong_pair_rmse, graph_rmse, similarity)``.
    Pairwise reprojection identifies the relative-pose hemisphere but cannot
    observe baseline scale. Whole-graph reprojection therefore breaks ties
    among candidates that remain close to the best strong-pair solution.
    """
    if not candidates:
        return None
    strong_candidates = [item for item in candidates if np.isfinite(item[0])]
    if strong_candidates:
        best_pair_rmse = min(item[0] for item in strong_candidates)
        if best_pair_rmse > acceptance_px:
            return None
        branch_limit = min(
            acceptance_px,
            best_pair_rmse + max(float(pair_branch_slack_px), 0.0),
        )
        same_branch = [
            item for item in strong_candidates if item[0] <= branch_limit
        ]
        return min(same_branch, key=lambda item: item[1])[2]

    selected = min(candidates, key=lambda item: item[1])
    return selected[2] if selected[1] <= acceptance_px else None


def _iter_bridge_pair_inputs(
    match_ids: list[str],
    registered_ids: list[str],
    observations_by_landmark: dict[str, list[SyncObservation]],
    known_world_ids: set[str] | None,
) -> list[tuple[str, str, list[tuple[SyncObservation, SyncObservation]]]]:
    """Independent (reference, pending) pairs with enough 2D overlap to bridge."""
    jobs: list[tuple[str, str, list[tuple[SyncObservation, SyncObservation]]]] = []
    for match_id in match_ids:
        for reference_id in registered_ids:
            if reference_id == match_id:
                continue
            pairs = _point_pairs_between_matches(
                reference_id,
                match_id,
                observations_by_landmark,
                excluded_landmark_ids=known_world_ids,
            )
            if len(pairs) < 5:
                continue
            if _correspondence_geometry_issue(pairs, match_id) is not None:
                continue
            jobs.append((reference_id, match_id, pairs))
    return jobs


def _bridge_pose_candidates(
    match_id: str,
    registered: dict[str, SimilarityTransform],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    landmarks: dict[str, np.ndarray],
    *,
    known_world_ids: set[str] | None = None,
    use_pose_cache: bool = False,
    relatives: dict[str, SimilarityTransform | None] | None = None,
) -> list[SimilarityTransform]:
    """Register a pending match through any solved view with 5+ shared picks."""
    candidates: list[SimilarityTransform] = []
    for reference_id, reference_similarity in registered.items():
        pairs = _point_pairs_between_matches(
            reference_id,
            match_id,
            observations_by_landmark,
            excluded_landmark_ids=known_world_ids,
        )
        if len(pairs) < 5:
            continue
        if _correspondence_geometry_issue(pairs, match_id) is not None:
            continue
        if relatives is not None:
            relative = relatives.get(reference_id)
        else:
            relative = _solve_relative_from_pairs(
                pairs,
                matches[reference_id].calibration,
                matches[match_id].calibration,
                use_pose_cache=use_pose_cache,
            )
        if relative is None:
            continue
        relative_candidates = [relative]

        points_reference: list[np.ndarray] = []
        points_image: list[np.ndarray] = []
        point_weights: list[float] = []
        inverse_reference = _inverse_similarity(reference_similarity)
        for landmark_id, point in landmarks.items():
            observation = next(
                (
                    item
                    for item in observations_by_landmark.get(landmark_id, [])
                    if item.match_id == match_id
                ),
                None,
            )
            if observation is None:
                continue
            points_reference.append(inverse_reference.transform_point(point))
            points_image.append(
                np.array((observation.u, observation.v), dtype=np.float64)
            )
            point_weights.append(float(observation.weight))
        if points_reference:
            scaled = _apply_known_baseline_scale(
                relative,
                points_reference,
                points_image,
                matches[match_id].calibration,
                matches[reference_id].calibration,
            )
            relative_candidates.append(scaled)
            relative_candidates.append(
                _refine_rigid_mixed(
                    scaled,
                    pairs,
                    points_reference,
                    points_image,
                    matches[reference_id].calibration,
                    matches[match_id].calibration,
                    point_weights=point_weights,
                )
            )

        candidates.extend(
            _compose_similarities(reference_similarity, item)
            for item in relative_candidates
        )
    return candidates


def _image_spread_px(points: np.ndarray) -> float:
    """Second singular value of centered 2D picks (cross-spread, pixels)."""
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    _u_matrix, singular, _vt_matrix = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2:
        return 0.0
    return float(singular[1])


def _pair_geometry_score(
    pairs: list[tuple[SyncObservation, SyncObservation]],
) -> tuple[float, float, int] | None:
    """Return ``(disparity, min_spread, overlap)`` when the pair can register."""
    if len(pairs) < 5:
        return None
    if _correspondence_geometry_issue(pairs, "_") is not None:
        return None
    points_a = np.array(
        [(obs_a.u, obs_a.v) for obs_a, _obs_b in pairs],
        dtype=np.float64,
    )
    points_b = np.array(
        [(obs_b.u, obs_b.v) for _obs_a, obs_b in pairs],
        dtype=np.float64,
    )
    spread = min(_image_spread_px(points_a), _image_spread_px(points_b))
    deltas = points_b - points_a
    magnitudes = np.hypot(deltas[:, 0], deltas[:, 1])
    # Lower-percentile flow so a few mismatched picks cannot look like a wide baseline.
    disparity = float(np.percentile(magnitudes, 40))
    return (disparity, spread, len(pairs))


def _pair_reprojection_rmse(
    relative: SimilarityTransform,
    pairs: list[tuple[SyncObservation, SyncObservation]],
    calibration_a: core.Calibration,
    calibration_b: core.Calibration,
) -> float:
    """Weighted pair RMSE for a solved 2D↔2D relative pose."""
    errors = _reprojection_errors_for_similarity(
        relative,
        pairs,
        calibration_a,
        calibration_b,
        [],
        [],
        weighted=True,
    )
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def _iter_pair_edges(
    match_ids: list[str],
    observations_by_landmark: dict[str, list[SyncObservation]],
    known_world_ids: set[str],
) -> list[
    tuple[tuple[float, float, int], str, str, list[tuple[SyncObservation, SyncObservation]]]
]:
    """Undirected well-spread 2D edges, strongest geometry first."""
    edges: list[
        tuple[
            tuple[float, float, int],
            str,
            str,
            list[tuple[SyncObservation, SyncObservation]],
        ]
    ] = []
    for id_a, id_b in combinations(sorted(match_ids), 2):
        pairs = _point_pairs_between_matches(
            id_a,
            id_b,
            observations_by_landmark,
            excluded_landmark_ids=known_world_ids,
        )
        score = _pair_geometry_score(pairs)
        if score is None:
            continue
        edges.append((score, id_a, id_b, pairs))
    edges.sort(key=lambda item: item[0], reverse=True)
    return edges


def _connected_component(adjacency: dict[str, list[str]], start_id: str) -> set[str]:
    """Undirected BFS component containing ``start_id``."""
    component: set[str] = set()
    stack = [start_id]
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        stack.extend(adjacency.get(node, []))
    return component


def _seed_from_strongest_pair(
    anchor_id: str,
    pending: set[str],
    similarities: dict[str, SimilarityTransform],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    known_world_ids: set[str],
    failure_details: list[str],
    *,
    use_pose_cache: bool,
    solve_vs_anchor,
) -> None:
    """Register one camera from the strongest pair via the usual vs-anchor solver.

    A non-anchor strongest pair still enters through the member that best
    connects to the anchor. Later cameras join via easiest-next bridges so a
    wide-baseline pair is not discarded just because both stills also see the
    anchor.
    """
    if len(similarities) > 1:
        return
    match_ids = [anchor_id, *pending]
    edges = _iter_pair_edges(match_ids, observations_by_landmark, known_world_ids)
    if not edges:
        return
    adjacency: dict[str, list[str]] = {match_id: [] for match_id in match_ids}
    for _score, id_a, id_b, _pairs in edges:
        adjacency[id_a].append(id_b)
        adjacency[id_b].append(id_a)
    component = _connected_component(adjacency, anchor_id)
    component_edges = [
        item
        for item in edges
        if item[1] in component and item[2] in component
    ]
    if not component_edges:
        return
    scored_edges: list[
        tuple[float, tuple[float, float, int], str, str]
    ] = []
    for geometry, id_a, id_b, pairs in component_edges[:4]:
        relative = _solve_relative_from_pairs(
            pairs,
            matches[id_a].calibration,
            matches[id_b].calibration,
            use_pose_cache=use_pose_cache,
        )
        rmse = float("inf")
        if relative is not None:
            rmse = _pair_reprojection_rmse(
                relative,
                pairs,
                matches[id_a].calibration,
                matches[id_b].calibration,
            )
        scored_edges.append((rmse, geometry, id_a, id_b))
    accepted = [
        item for item in scored_edges if item[0] <= ACCEPT_RMSE_PX
    ]
    pool = accepted if accepted else scored_edges
    pool.sort(
        key=lambda item: (item[0], -item[1][0], -item[1][1], -item[1][2])
    )
    _rmse, _geometry, id_a, id_b = pool[0]
    if anchor_id in (id_a, id_b):
        first_id = id_b if id_a == anchor_id else id_a
    else:
        vs_a = _pair_geometry_score(
            _point_pairs_between_matches(
                anchor_id,
                id_a,
                observations_by_landmark,
                excluded_landmark_ids=known_world_ids,
            )
        )
        vs_b = _pair_geometry_score(
            _point_pairs_between_matches(
                anchor_id,
                id_b,
                observations_by_landmark,
                excluded_landmark_ids=known_world_ids,
            )
        )
        if vs_a is None and vs_b is None:
            first_id = None
            seen = {anchor_id}
            queue = list(adjacency.get(anchor_id, []))
            while queue:
                node = queue.pop(0)
                if node in seen:
                    continue
                seen.add(node)
                if node in (id_a, id_b):
                    first_id = node
                    break
                queue.extend(adjacency.get(node, []))
            if first_id is None:
                return
        elif vs_a is None:
            first_id = id_b
        elif vs_b is None:
            first_id = id_a
        else:
            first_id = id_a if vs_a >= vs_b else id_b
    match_id, solved, detail = solve_vs_anchor(first_id)
    if solved is None:
        if detail:
            failure_details.append(detail)
        return
    similarities[match_id] = solved
    pending.discard(match_id)


def _easiest_pending_order(
    pending_ids: list[str],
    registered_ids: list[str],
    relatives_by_match: dict[str, dict[str, SimilarityTransform | None]],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    landmarks: dict[str, np.ndarray],
    known_world_ids: set[str],
) -> list[str]:
    """Pending cameras easiest to add next (never alphabetical except ties)."""
    keys: list[tuple] = []
    for match_id in pending_ids:
        overlap_ids: set[str] = set()
        best_rmse = float("inf")
        best_geometry = (0.0, 0.0, 0)
        for reference_id in registered_ids:
            pairs = _point_pairs_between_matches(
                reference_id,
                match_id,
                observations_by_landmark,
                excluded_landmark_ids=known_world_ids,
            )
            for obs_a, _obs_b in pairs:
                overlap_ids.add(obs_a.landmark_id)
            geometry = _pair_geometry_score(pairs)
            relative = relatives_by_match.get(match_id, {}).get(reference_id)
            rmse = float("inf")
            if relative is not None and geometry is not None:
                rmse = _pair_reprojection_rmse(
                    relative,
                    pairs,
                    matches[reference_id].calibration,
                    matches[match_id].calibration,
                )
            if geometry is not None and (
                rmse < best_rmse
                or (rmse == best_rmse and geometry > best_geometry)
            ):
                best_rmse = rmse
                best_geometry = geometry
        collected = _collect_pnp_correspondences(
            match_id, observations_by_landmark, landmarks
        )
        pnp_count = 0 if collected is None else len(collected[0])
        overlap = max(len(overlap_ids), pnp_count)
        disparity, spread, count = best_geometry
        keys.append(
            (-overlap, best_rmse, -disparity, -spread, -count, match_id)
        )
    keys.sort()
    return [item[-1] for item in keys]


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
    initial_similarities: dict[str, SimilarityTransform] | None = None,
    *,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    use_pose_cache: bool = False,
) -> tuple[dict[str, SimilarityTransform] | None, str]:
    """Register free matches vs anchor, then bridge via triangulated landmarks."""
    similarities: dict[str, SimilarityTransform] = {
        anchor_id: SimilarityTransform(),
    }
    if lock_rotation and lock_translation:
        for match_id in free_match_ids:
            similarities[match_id] = SimilarityTransform()
        return similarities, ""
    pending = set(free_match_ids)
    failure_details: list[str] = []
    known_world_ids = set(known_world or {})
    # Diagnose leave-one-out and lens refine already have a locked graph.
    # Reuse those poses so we skip the expensive multi-start pairwise search.
    for match_id in list(pending):
        seed = (initial_similarities or {}).get(match_id)
        if seed is None:
            continue
        similarities[match_id] = _copy_similarity(seed)
        pending.discard(match_id)
    anchor_metric = _metric_landmarks(
        observations_by_landmark,
        anchor_id,
        matches[anchor_id].calibration,
        known_world,
    )

    def strong_anchor_support(match_id: str) -> bool:
        """Avoid accepting a minimal direct-anchor pose before graph bridges."""
        pairs = _point_pairs_between_matches(
            anchor_id,
            match_id,
            observations_by_landmark,
            excluded_landmark_ids=known_world_ids,
        )
        metric_count = sum(
            1
            for landmark_id in anchor_metric
            if any(
                item.match_id == match_id
                for item in observations_by_landmark.get(landmark_id, [])
            )
        )
        known_line_count = sum(
            1
            for landmark_id in (known_lines or {})
            if any(
                item.match_id == match_id
                for item in (line_observations_by_landmark or {}).get(
                    landmark_id, []
                )
            )
        )
        return bool(
            len(pairs) >= 5
            or metric_count >= 3
            or known_line_count >= 3
            or match_id in (initial_similarities or {})
            or lock_rotation
            or lock_translation
        )

    def solve_vs_anchor(match_id: str) -> tuple[str, SimilarityTransform | None, str]:
        solved, detail = _relative_pose_from_correspondences(
            anchor_id,
            match_id,
            observations_by_landmark,
            matches,
            known_world,
            known_lines=known_lines,
            line_observations_by_landmark=line_observations_by_landmark,
            parallel_pairs=parallel_pairs,
            initial_similarity=(initial_similarities or {}).get(match_id),
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            use_pose_cache=use_pose_cache,
        )
        return match_id, solved, detail

    if lock_rotation or lock_translation:
        direct_ids = [
            match_id
            for match_id in sorted(pending)
            if strong_anchor_support(match_id)
        ]
        for match_id, solved, detail in _map_pair_jobs(solve_vs_anchor, direct_ids):
            if solved is None:
                if detail:
                    failure_details.append(detail)
                continue
            similarities[match_id] = solved
            pending.discard(match_id)
    else:
        _seed_from_strongest_pair(
            anchor_id,
            pending,
            similarities,
            observations_by_landmark,
            matches,
            known_world_ids,
            failure_details,
            use_pose_cache=use_pose_cache,
            solve_vs_anchor=solve_vs_anchor,
        )

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
        pending_now = list(pending)
        relatives_by_match: dict[str, dict[str, SimilarityTransform | None]] = {
            match_id: {} for match_id in pending_now
        }
        if not lock_rotation and not lock_translation and pending_now:

            def solve_bridge(
                job: tuple[
                    str,
                    str,
                    list[tuple[SyncObservation, SyncObservation]],
                ],
            ) -> tuple[str, str, SimilarityTransform | None]:
                reference_id, match_id, pairs = job
                relative = _solve_relative_from_pairs(
                    pairs,
                    matches[reference_id].calibration,
                    matches[match_id].calibration,
                    use_pose_cache=use_pose_cache,
                )
                return reference_id, match_id, relative

            for reference_id, match_id, relative in _map_pair_jobs(
                solve_bridge,
                _iter_bridge_pair_inputs(
                    pending_now,
                    list(similarities.keys()),
                    observations_by_landmark,
                    known_world_ids,
                ),
            ):
                relatives_by_match.setdefault(match_id, {})[reference_id] = relative

        for match_id in _easiest_pending_order(
            pending_now,
            list(similarities.keys()),
            relatives_by_match,
            observations_by_landmark,
            matches,
            landmarks,
            known_world_ids,
        ):
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
            pose_candidates: list[SimilarityTransform] = []
            if strong_anchor_support(match_id):
                _match_id, vs_solved, vs_detail = solve_vs_anchor(match_id)
                if vs_solved is not None:
                    solved = vs_solved
                elif vs_detail:
                    failure_details.append(vs_detail)
            if solved is None and collected is not None:
                points_shared, points_image = collected
                pnp_candidate = _pnp_similarity(
                    match_id,
                    points_shared,
                    points_image,
                    matches[match_id].calibration,
                    lock_rotation=lock_rotation,
                    lock_translation=lock_translation,
                )
                if pnp_candidate is not None and line_constraints:
                    pnp_candidate = _refine_rigid_mixed(
                        pnp_candidate,
                        [],
                        list(points_shared),
                        list(points_image),
                        matches[anchor_id].calibration,
                        matches[match_id].calibration,
                        known_line_constraints=line_constraints,
                        lock_rotation=lock_rotation,
                        lock_translation=lock_translation,
                    )
                if pnp_candidate is not None:
                    pose_candidates.append(pnp_candidate)

            if solved is None and not lock_rotation and not lock_translation:
                pose_candidates.extend(
                    _bridge_pose_candidates(
                        match_id,
                        similarities,
                        observations_by_landmark,
                        matches,
                        landmarks,
                        known_world_ids=known_world_ids,
                        use_pose_cache=use_pose_cache,
                        relatives=relatives_by_match.get(match_id, {}),
                    )
                )

            if solved is None:
                ranked_candidates: list[
                    tuple[float, float, SimilarityTransform]
                ] = []
                for candidate in pose_candidates:
                    graph_rmse = _registration_candidate_rmse(
                        match_id,
                        candidate,
                        similarities,
                        observations_by_landmark,
                        matches,
                        landmarks,
                        known_world_ids=known_world_ids,
                    )
                    strong_pair_rmse = _registration_strong_pair_rmse(
                        match_id,
                        candidate,
                        similarities,
                        observations_by_landmark,
                        matches,
                        known_world_ids=known_world_ids,
                    )
                    ranked_candidates.append(
                        (strong_pair_rmse, graph_rmse, candidate)
                    )
                selected = _select_registration_candidate(ranked_candidates)
                if selected is not None:
                    solved = selected

            if solved is None and len(line_constraints) >= 3:
                for seed in _mixed_pose_seeds(
                    matches[anchor_id].calibration,
                    matches[match_id].calibration,
                    lock_rotation=lock_rotation,
                    lock_translation=lock_translation,
                ):
                    candidate = _refine_rigid_mixed(
                        seed,
                        [],
                        [],
                        [],
                        matches[anchor_id].calibration,
                        matches[match_id].calibration,
                        known_line_constraints=line_constraints,
                        lock_rotation=lock_rotation,
                        lock_translation=lock_translation,
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
                    if errors and float(np.sqrt(np.mean(np.square(errors)))) < ACCEPT_RMSE_PX:
                        solved = candidate
                        break
            if solved is None:
                continue
            similarities[match_id] = solved
            pending.discard(match_id)
            progressed = True
            break
        if not progressed:
            # Pairwise bridges already ran for every pending still. Retry the
            # mixed vs-anchor solver (On Ground / Known 3D) for cameras that
            # still have no pose — including those with a non-anchor pair.
            fallback_ids = [
                match_id
                for match_id in pending
                if strong_anchor_support(match_id)
            ]
            for match_id, solved, detail in _map_pair_jobs(
                solve_vs_anchor, fallback_ids
            ):
                if solved is None:
                    if detail:
                        failure_details.append(detail)
                    continue
                similarities[match_id] = solved
                pending.discard(match_id)
                progressed = True
            if not progressed:
                break

    if pending:
        pending_list = ", ".join(f"'{name}'" for name in sorted(pending))
        detail = (
            failure_details[0]
            if failure_details
            else (
                f"Could not register {pending_list} — need ≥5 well-spread 2D "
                "landmarks shared with the anchor, or ≥3 Known 3D / On Ground picks"
            )
        )
        if len(similarities) <= 1:
            return None, detail
        return similarities, detail
    return similarities, ""
