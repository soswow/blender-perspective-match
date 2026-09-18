"""Lines whose geometry is exactly defined by two point landmarks."""

from __future__ import annotations

import numpy as np


MIN_DERIVED_LINE_SPAN = 1.0e-9


def normalize_derived_lines(derived_lines) -> list[tuple[str, str, str]]:
    """Validate and copy ``(line_id, endpoint_a_id, endpoint_b_id)`` records."""
    result: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in derived_lines or ():
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 3
            or any(not isinstance(value, str) or not value for value in item)
        ):
            raise ValueError("From Points line references are invalid")
        line_id, endpoint_a, endpoint_b = item
        if line_id in seen:
            raise ValueError("From Points line is defined more than once")
        if line_id in {endpoint_a, endpoint_b} or endpoint_a == endpoint_b:
            raise ValueError("From Points line needs two different point landmarks")
        seen.add(line_id)
        result.append((line_id, endpoint_a, endpoint_b))
    return result


def validate_derived_lines(derived_lines, point_ids, *, other_line_ids=()):
    """Return normalized records after validating point and line identities."""
    result = normalize_derived_lines(derived_lines)
    points = set(point_ids)
    other_lines = set(other_line_ids)
    for line_id, endpoint_a, endpoint_b in result:
        if endpoint_a not in points or endpoint_b not in points:
            raise ValueError("From Points line references a missing point landmark")
        if line_id in points or line_id in other_lines:
            raise ValueError("From Points line conflicts with point or drawn line geometry")
    return result


def derived_line_geometry(points, derived_lines, *, allow_coincident=False):
    """Materialize endpoint segments and normalized infinite-line geometry."""
    segments = {}
    geometry = {}
    for line_id, endpoint_a, endpoint_b in normalize_derived_lines(derived_lines):
        try:
            first = np.asarray(points[endpoint_a], dtype=np.float64).reshape(3)
            second = np.asarray(points[endpoint_b], dtype=np.float64).reshape(3)
        except KeyError as exc:
            raise ValueError("From Points line endpoint has no fitted geometry") from exc
        if not np.isfinite((first, second)).all():
            raise ValueError("From Points line endpoint geometry is invalid")
        direction = second - first
        span = float(np.linalg.norm(direction))
        if span < MIN_DERIVED_LINE_SPAN:
            if allow_coincident:
                continue
            raise ValueError("From Points line endpoints are coincident")
        segments[line_id] = (first.copy(), second.copy())
        geometry[line_id] = (0.5 * (first + second), direction / span)
    return segments, geometry
