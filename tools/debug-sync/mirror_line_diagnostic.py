"""Pure geometry measurements for mirrored finite segments and infinite lines."""

from __future__ import annotations

import numpy as np


def mirror_line_diagnostic(
    left_segment, right_segment, plane_point, plane_normal, *, coordinate_origin=None,
):
    """Compare published segments as infinite mirror lines and finite helpers."""
    left = np.asarray(left_segment, dtype=float).reshape(2, 3)
    right = np.asarray(right_segment, dtype=float).reshape(2, 3)
    origin = np.zeros(3) if coordinate_origin is None else np.asarray(
        coordinate_origin, dtype=float).reshape(3)
    plane_point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    reflected = left - 2.0 * ((left - plane_point) @ normal)[:, None] * normal
    source_direction = left[1] - left[0]
    source_direction /= max(float(np.linalg.norm(source_direction)), 1.0e-12)
    left_direction = reflected[1] - reflected[0]
    right_direction = right[1] - right[0]
    left_direction /= max(float(np.linalg.norm(left_direction)), 1.0e-12)
    right_direction /= max(float(np.linalg.norm(right_direction)), 1.0e-12)
    if float(left_direction @ right_direction) < 0.0:
        reflected = reflected[::-1]
        left_direction = -left_direction

    def canonical(point, direction):
        relative = point - origin
        return origin + relative - float(relative @ direction) * direction

    source_midpoint = np.mean(left, axis=0)
    reflected_midpoint = np.mean(reflected, axis=0)
    right_midpoint = np.mean(right, axis=0)
    source_witness = canonical(source_midpoint, source_direction)
    left_witness = source_witness - 2.0 * normal * float(
        normal @ (source_witness - plane_point)
    )
    right_witness = canonical(right_midpoint, right_direction)
    position_vector = np.cross(right_direction, left_witness - right_witness)
    direction_vector = np.cross(right_direction, left_direction)
    direction_sine = float(np.linalg.norm(direction_vector))

    endpoint_delta = right - reflected
    endpoint_along = endpoint_delta @ right_direction
    endpoint_perpendicular = endpoint_delta - endpoint_along[:, None] * right_direction
    midpoint_delta = np.mean(endpoint_delta, axis=0)
    midpoint_along = float(midpoint_delta @ right_direction)
    midpoint_perpendicular = midpoint_delta - midpoint_along * right_direction

    axes = np.stack((left_direction, -right_direction), axis=1)
    parameters = np.linalg.lstsq(
        axes, right_midpoint - reflected_midpoint, rcond=None,
    )[0]
    closest_left = reflected_midpoint + float(parameters[0]) * left_direction
    closest_right = right_midpoint + float(parameters[1]) * right_direction

    return dict(
        left_segment=left.tolist(),
        reflected_left_segment=reflected.tolist(),
        right_segment=right.tolist(),
        left_length=float(np.linalg.norm(left[1] - left[0])),
        right_length=float(np.linalg.norm(right[1] - right[0])),
        scoring_reference_origin=origin.tolist(),
        scoring_left_witness=left_witness.tolist(),
        scoring_right_witness=right_witness.tolist(),
        scoring_witness_distance_from_left_midpoint=float(np.linalg.norm(
            left_witness - reflected_midpoint)),
        scoring_witness_distance_from_right_midpoint=float(np.linalg.norm(
            right_witness - right_midpoint)),
        scorer_position_gap=float(np.linalg.norm(position_vector)),
        infinite_direction_sine=direction_sine,
        infinite_direction_angle_deg=float(np.degrees(np.arcsin(np.clip(
            direction_sine, 0.0, 1.0)))),
        closest_approach_gap=float(np.linalg.norm(closest_right - closest_left)),
        closest_approach_distance_from_left_midpoint=abs(float(parameters[0])),
        closest_approach_distance_from_right_midpoint=abs(float(parameters[1])),
        endpoint_delta=endpoint_delta.tolist(),
        endpoint_along=endpoint_along.tolist(),
        endpoint_perpendicular=endpoint_perpendicular.tolist(),
        endpoint_perpendicular_norm=[float(np.linalg.norm(value))
                                     for value in endpoint_perpendicular],
        midpoint_delta=midpoint_delta.tolist(),
        midpoint_along=midpoint_along,
        midpoint_perpendicular=midpoint_perpendicular.tolist(),
        midpoint_perpendicular_norm=float(np.linalg.norm(midpoint_perpendicular)),
    )
