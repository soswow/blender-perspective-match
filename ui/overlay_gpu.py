"""Screen-space TRS for reused overlay GPU batches (no bpy).

Blender 5.1 GPUVertBuf is static: attr_fill after the first draw raises
"Can't fill, static buffer already in use". Overlay therefore keeps a few
unit-geometry batches and places them with gpu.matrix (T then R then S, so
the vertex is scaled, rotated, then translated).
"""

from __future__ import annotations

import math

_MIN_LENGTH = 1.0e-8


def segment_trs(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float] | None:
    """Translate/rotate/scale mapping unit X (0→1) onto the pixel segment."""
    delta_x = x1 - x0
    delta_y = y1 - y0
    length = math.hypot(delta_x, delta_y)
    if length < _MIN_LENGTH:
        return None
    return x0, y0, math.atan2(delta_y, delta_x), length


def rect_trs(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float] | None:
    """Translate/scale mapping the unit square onto an axis-aligned pixel rect."""
    width = x1 - x0
    height = y1 - y0
    if abs(width) < _MIN_LENGTH or abs(height) < _MIN_LENGTH:
        return None
    return x0, y0, width, height


def circle_trs(
    center_x: float, center_y: float, radius: float
) -> tuple[float, float, float] | None:
    """Translate/uniform-scale mapping the unit circle onto a pixel circle."""
    if radius < _MIN_LENGTH:
        return None
    return center_x, center_y, radius


def apply_segment_trs(
    trs: tuple[float, float, float, float], x: float, y: float = 0.0
) -> tuple[float, float]:
    """Apply T·R·S from `segment_trs` to a unit-line vertex."""
    origin_x, origin_y, angle, length = trs
    scaled_x = x * length
    scaled_y = y
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        origin_x + cosine * scaled_x - sine * scaled_y,
        origin_y + sine * scaled_x + cosine * scaled_y,
    )


def apply_rect_trs(
    trs: tuple[float, float, float, float], x: float, y: float
) -> tuple[float, float]:
    origin_x, origin_y, width, height = trs
    return origin_x + x * width, origin_y + y * height


def apply_circle_trs(
    trs: tuple[float, float, float], x: float, y: float
) -> tuple[float, float]:
    origin_x, origin_y, radius = trs
    return origin_x + x * radius, origin_y + y * radius
