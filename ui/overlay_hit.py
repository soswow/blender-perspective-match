"""Screen-space hit testing for landmark overlay picks (no bpy)."""

from __future__ import annotations

import math


def distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Nearest distance from a point to a finite segment."""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared = delta_x * delta_x + delta_y * delta_y
    if squared < 1.0e-8:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ratio = max(0.0, min(1.0, ratio / squared))
    closest_x = start[0] + delta_x * ratio
    closest_y = start[1] + delta_y * ratio
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def nearest_landmark_hit(
    mouse: tuple[float, float],
    landmarks: list[
        tuple[int, str, tuple[float, float], tuple[float, float] | None]
    ],
    *,
    point_radius: float,
    line_radius: float,
) -> int:
    """Return the collection index of the closest overlay hit, or -1.

    Each item is ``(index, kind, point_a, point_b)``. ``kind`` is ``POINT`` or
    ``LINE``; ``point_b`` is required for lines and ignored for points.
    """
    best_index = -1
    best_distance = float("inf")
    for index, kind, point_a, point_b in landmarks:
        hit_distance: float | None = None
        if kind == "LINE" and point_b is not None:
            segment = distance_to_segment(mouse, point_a, point_b)
            end_a = math.hypot(mouse[0] - point_a[0], mouse[1] - point_a[1])
            end_b = math.hypot(mouse[0] - point_b[0], mouse[1] - point_b[1])
            if end_a <= point_radius:
                hit_distance = end_a
            if end_b <= point_radius and (
                hit_distance is None or end_b < hit_distance
            ):
                hit_distance = end_b
            if segment <= line_radius and (
                hit_distance is None or segment < hit_distance
            ):
                hit_distance = segment
        else:
            distance = math.hypot(mouse[0] - point_a[0], mouse[1] - point_a[1])
            if distance <= point_radius:
                hit_distance = distance
        if hit_distance is not None and hit_distance < best_distance:
            best_index = index
            best_distance = hit_distance
    return best_index
