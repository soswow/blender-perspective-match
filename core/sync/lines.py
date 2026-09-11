"""Free and Known 3D line landmarks, including Is-Parallel-To."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .. import geometry as core
from .constants import (
    LINE_FIXED_ANCHOR_MIN,
    LINE_PLANE_MIN_SINE,
    LINE_RECONSTRUCT_TRUNCATE_PX,
    PLANE_HARD_SLACK,
    WORLD_AXIS_DIRECTIONS,
)
from .projection import (
    _image_line_homogeneous,
    _intersect_planes_to_line,
    _line_observation_reprojection_errors,
    _plane_from_line_observation,
    camera_ray_private,
    project_private_point,
)
from .types import SimilarityTransform, SyncLineObservation, SyncMatchInput

def _line_anchor_match_ids(
    observations: list[SyncLineObservation],
    fixed_match_ids: set[str] | None,
    *,
    minimum: int = LINE_FIXED_ANCHOR_MIN,
) -> set[str] | None:
    """Pose-locked cameras that see this line, when there are enough to anchor 3D."""
    if not fixed_match_ids:
        return None
    anchored = {
        observation.match_id
        for observation in observations
        if observation.match_id in fixed_match_ids
    }
    if len(anchored) >= minimum:
        return anchored
    return None


def _line_observation_rms_px(
    point: np.ndarray,
    direction: np.ndarray,
    observation: SyncLineObservation,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> float:
    """Unweighted endpoint RMS of one 2D stroke vs a 3D line."""
    stroke = replace(observation, weight=1.0)
    errors = _line_observation_reprojection_errors(
        point, direction, stroke, calibration, similarity
    )
    return float(np.sqrt(0.5 * (errors[0] * errors[0] + errors[1] * errors[1])))


def _reconstruct_line_from_observations(
    observations: list[SyncLineObservation],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    *,
    prefer_match_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Intersect back-projected planes from ≥2 registered views into a 3D line.

    Uses the pair of interpretation planes with the lowest truncated RMS across
    all strokes (near-parallel pairs skipped). ``prefer_match_ids`` is only a
    tie-break; helper extent still uses locked views separately.
    """
    preferred = prefer_match_ids or set()
    records: list[
        tuple[SyncLineObservation, np.ndarray, core.Calibration, SimilarityTransform]
    ] = []
    for observation in observations:
        similarity = similarities.get(observation.match_id)
        match = matches.get(observation.match_id)
        if similarity is None or match is None:
            continue
        plane = _plane_from_line_observation(
            observation, match.calibration, similarity
        )
        if plane is not None:
            records.append((observation, plane, match.calibration, similarity))
    if len(records) < 2:
        return None

    min_sine = float(LINE_PLANE_MIN_SINE)
    truncate = float(LINE_RECONSTRUCT_TRUNCATE_PX)
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for index_a, (obs_a, plane_a, _cal_a, _sim_a) in enumerate(records):
        for obs_b, plane_b, _cal_b, _sim_b in records[index_a + 1 :]:
            sine = float(np.linalg.norm(np.cross(plane_a[:3], plane_b[:3])))
            if sine < min_sine:
                continue
            reconstructed = _intersect_planes_to_line(plane_a, plane_b)
            if reconstructed is None:
                continue
            point, direction = reconstructed
            errors = [
                _line_observation_rms_px(
                    point, direction, observation, calibration, similarity
                )
                for observation, _plane, calibration, similarity in records
            ]
            truncated = [min(error, truncate) for error in errors]
            score = float(np.mean(truncated))
            locked_hits = int(obs_a.match_id in preferred) + int(
                obs_b.match_id in preferred
            )
            key = (score, -locked_hits, -sine)
            if best is None or key < (best[0], best[1], best[2]):
                best = (score, -locked_hits, -sine, point, direction)

    if best is None:
        return _intersect_planes_to_line(records[0][1], records[1][1])

    _score, _locked, _sine, point, direction = best
    return point, direction


def line_support_angles(
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]],
    observations_by_landmark: dict[str, list[SyncLineObservation]],
    similarities: dict[str, SimilarityTransform],
    matches: dict[str, SyncMatchInput],
    *,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    mirror_pairs: list[tuple[str, str]] | None = None,
    mirror_normal: np.ndarray | None = None,
    support_planes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Best supporting plane separation for free lines; not a confidence interval."""
    known_ids = set(known_lines or {})
    partners: dict[str, str] = {}
    if mirror_normal is not None:
        for a, b in mirror_pairs or ():
            partners[a], partners[b] = b, a
    normals_by_id: dict[str, list[np.ndarray]] = {}
    for landmark_id, ends in line_segments.items():
        point_a, point_b = ends
        direction = point_b-point_a
        normals = []
        for observation in observations_by_landmark.get(landmark_id, []):
            match = matches.get(observation.match_id)
            similarity = similarities.get(observation.match_id)
            if match is None or similarity is None:
                continue
            # A contradicting stroke must not make an unsupported line look strong.
            residual = _line_observation_rms_px(point_a, direction, observation, match.calibration, similarity)
            if not np.isfinite(residual) or residual > LINE_RECONSTRUCT_TRUNCATE_PX:
                continue
            plane = _plane_from_line_observation(observation, match.calibration, similarity)
            if plane is not None and np.isfinite(plane).all():
                normals.append(plane[:3])
        supported = (support_planes or {}).get(landmark_id)
        if supported is not None:
            origin, normal = supported
            distances = [abs(float(normal@(point-origin))) for point in (point_a,point_b)]
            if max(distances) <= PLANE_HARD_SLACK:
                normals.append(normal)
        normals_by_id[landmark_id] = normals
    reflection = None
    if mirror_normal is not None:
        unit = np.asarray(mirror_normal, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
        reflection = np.eye(3)-2*np.outer(unit,unit)
    angles = {}
    for landmark_id in line_segments:
        partner = partners.get(landmark_id)
        if landmark_id in known_ids or partner in known_ids:
            continue  # Known geometry, including its reflection, supplies 3D directly.
        normals = list(normals_by_id.get(landmark_id, []))
        if reflection is not None and partner:
            normals += [reflection @ n for n in normals_by_id.get(partner, [])]
        sine = max((float(np.linalg.norm(np.cross(a,b))) for i,a in enumerate(normals)
                    for b in normals[i+1:]), default=0.0)
        angles[landmark_id] = float(np.degrees(np.arcsin(np.clip(sine,0,1))))
    return angles


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
    *,
    extent_match_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project drawn 2D segment ends onto the 3D line for a visible mesh edge."""
    pool = observations
    if extent_match_ids:
        filtered = [
            observation
            for observation in observations
            if observation.match_id in extent_match_ids
        ]
        if len(filtered) >= 2:
            pool = filtered
    samples: list[np.ndarray] = []
    for observation in pool:
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


def _world_axis_in_group(group: list[str]) -> np.ndarray | None:
    """Fixed shared-world direction named by a parallel-family graph node."""
    for item_id in group:
        direction = WORLD_AXIS_DIRECTIONS.get(item_id)
        if direction is not None:
            return direction
    return None


def _axis_line_constraints_for_match(
    match_id: str,
    parallel_pairs: list[tuple[str, str]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] | None,
) -> list[tuple[np.ndarray, SyncLineObservation]]:
    """Observed Lines whose parallel family is fixed to a shared-world axis."""
    constraints: list[tuple[np.ndarray, SyncLineObservation]] = []
    line_by_lm = line_observations_by_landmark or {}
    for group in _parallel_landmark_groups(parallel_pairs):
        axis = _world_axis_in_group(group)
        if axis is None:
            continue
        for landmark_id in group:
            if landmark_id in WORLD_AXIS_DIRECTIONS:
                continue
            observation = next(
                (
                    item
                    for item in line_by_lm.get(landmark_id, [])
                    if item.match_id == match_id
                ),
                None,
            )
            if observation is not None:
                constraints.append((axis, observation))
    return constraints


def _axis_line_rotation_error(
    direction: np.ndarray,
    observation: SyncLineObservation,
    calibration: core.Calibration,
    similarity: SimilarityTransform,
) -> float | None:
    """Sine of angle by which a world direction misses an observed image line."""
    plane = _plane_from_line_observation(observation, calibration, similarity)
    if plane is None:
        return None
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    return abs(float(np.dot(plane[:3], unit)))


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
    *,
    prefer_match_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares line with fixed direction against back-projected observation planes.

    Uses every posed stroke. Near-parallel duplicate planes are dropped (unlocked
    views kept first) so a locked near-duplicate cannot pull depth off a recovered
    still. ``prefer_match_ids`` only orders that duplicate choice.
    """
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    records: list[tuple[SyncLineObservation, np.ndarray]] = []
    for observation in observations:
        similarity = similarities.get(observation.match_id)
        match = matches.get(observation.match_id)
        if similarity is None or match is None:
            continue
        plane = _plane_from_line_observation(
            observation, match.calibration, similarity
        )
        if plane is not None:
            records.append((observation, plane))
    if prefer_match_ids:
        records.sort(key=lambda item: item[0].match_id in prefer_match_ids)
    min_sine = float(LINE_PLANE_MIN_SINE)
    planes: list[np.ndarray] = []
    for _observation, plane in records:
        normal = plane[:3]
        if any(
            float(np.linalg.norm(np.cross(normal, other[:3]))) < min_sine
            for other in planes
        ):
            continue
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
        axis_direction = _world_axis_in_group(group)
        if len(members) < 1 or (len(members) < 2 and axis_direction is None):
            continue
        directions: list[np.ndarray] = []
        weights: list[float] = []
        known_direction: np.ndarray | None = axis_direction
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
