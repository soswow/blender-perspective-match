"""GPU overlay for VP guides, handles, origin, and landmarks."""

from __future__ import annotations

import math

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .. import core, properties, scene

AXIS_COLORS = {
    # Match Blender's default axis gizmo colors (X red, Y green, Z blue).
    "x": (0.96, 0.26, 0.26, 1.0),
    "y": (0.26, 0.50, 0.96, 1.0),
    "z": (0.26, 0.80, 0.30, 1.0),
}

# Ideal-space VPs farther than this many image diagonals are treated as parallel.
_MAX_VP_DIAGONALS = 8.0
# When a VP is too far / at infinity, still draw a capped guide this many diagonals long.
_PARALLEL_GUIDE_DIAGONALS = 4.0

# Screen-space dash pattern for VP continuations and the horizon.
_GUIDE_DASH_LENGTH = 9.0
_GUIDE_GAP_LENGTH = 7.0
_HORIZON_SLOT_LENGTH = 8.0
_GUIDE_LINE_THICKNESS = 0.9
# Residual labels sit near the preferred endpoint, offset off the stroke.
_ERROR_LABEL_INSET = 0.12
_ERROR_LABEL_OFFSET_PX = 8.0
_ERROR_LABEL_FONT_SIZE = 11.0
# Landmark name labels sit beside the pick / segment midpoint.
_LANDMARK_LABEL_OFFSET_PX = 10.0
_LANDMARK_LABEL_FONT_SIZE = 12.0

# Modal interact chrome — hard to miss while a draw / pick tool owns LMB.
_MODE_BANNER_HEIGHT = 34.0
_MODE_BANNER_FONT_SIZE = 14.0
_MODE_FRAME_THICKNESS = 4.0
_MODE_CHROME = {
    "LINE": {
        "title": "VP Lines",
        "hint": "drag to draw · click to select · Esc exits",
        "color": (0.95, 0.55, 0.12, 0.92),
    },
    "ORIGIN": {
        "title": "Pick Origin",
        "hint": "click a ground point · Esc exits",
        "color": (0.98, 0.78, 0.18, 0.92),
    },
    "PP": {
        "title": "Principal Point",
        "hint": "drag the violet crosshair · Esc exits",
        "color": (0.72, 0.35, 1.0, 0.92),
    },
    "LANDMARK": {
        "title": "Landmark",
        "hint": "click / drag on the plate · Esc exits",
        "color": (0.15, 0.85, 0.85, 0.92),
    },
}

_draw_handle = None
_preview: dict[str, object] = {
    "kind": "",
    "start": None,
    "end": None,
    "area_pointer": 0,
}

# Recompute Huber VP fits only when line / intrinsics RNA changes.
_vp_solve_cache: dict[str, object] = {
    "key": None,
    "vanishing_points": {},
    "overlay_vanishing_points": {},
    "ideal_endpoints": (),
}

# While set, polyline draws accumulate into one GPU batch per color/width.
_active_line_batcher: "_LineBatcher | None" = None


def ui_scale() -> float:
    """Blender UI/DPI scale for POST_PIXEL sizes (≈2 on Retina, 1 on standard)."""
    try:
        scale = float(bpy.context.preferences.system.ui_scale)
    except Exception:
        return 1.0
    return scale if scale > 1.0e-6 else 1.0


def _s(value: float) -> float:
    """Scale a logical screen-pixel size into current display pixels."""
    return value * ui_scale()


class _LineBatcher:
    """Collect screen-space line segments and flush as few GPU draws as possible."""

    def __init__(self) -> None:
        self._groups: dict[tuple, list[tuple[float, float, float]]] = {}

    def add_segment(
        self,
        point_a: Vector,
        point_b: Vector,
        color,
        thickness: float,
    ) -> None:
        key = (color, float(thickness))
        bucket = self._groups.get(key)
        if bucket is None:
            bucket = []
            self._groups[key] = bucket
        bucket.append(_as_pos3(point_a))
        bucket.append(_as_pos3(point_b))

    def flush(self) -> None:
        if not self._groups:
            return
        shader = _line_shader()
        viewport = gpu.state.viewport_get()[2:]
        scale = ui_scale()
        shader.bind()
        shader.uniform_float("viewportSize", viewport)
        for (color, thickness), positions in self._groups.items():
            if len(positions) < 2:
                continue
            batch = batch_for_shader(shader, "LINES", {"pos": positions})
            # Thickness is authored in logical px; Retina needs ui_scale.
            shader.uniform_float("lineWidth", thickness * scale)
            shader.uniform_float("color", color)
            batch.draw(shader)
        self._groups.clear()


def set_preview(
    context: bpy.types.Context,
    kind: str,
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> None:
    """Set transient modal draft geometry in source-image coordinates."""
    _preview["kind"] = kind
    _preview["start"] = start
    _preview["end"] = end
    _preview["area_pointer"] = context.area.as_pointer() if context.area else 0
    properties.tag_viewport_redraw(context)


def clear_preview(context: bpy.types.Context | None = None) -> None:
    """Clear transient modal draft geometry."""
    _preview["kind"] = ""
    _preview["start"] = None
    _preview["end"] = None
    _preview["area_pointer"] = 0
    properties.tag_viewport_redraw(context)


def interact_mode_label(mode: str) -> str:
    """Short title for an active Draw / Pick tool mode."""
    chrome = _MODE_CHROME.get(mode)
    return chrome["title"] if chrome is not None else "Tool"


def _fill_shader():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _line_shader():
    return gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")


def _with_alpha(color, alpha: float):
    return color[0], color[1], color[2], max(0.0, min(1.0, color[3] * alpha))


def _as_pos3(point: Vector) -> tuple[float, float, float]:
    """POST_PIXEL polyline shaders expect 3D positions with z = 0."""
    return float(point.x), float(point.y), 0.0


def _draw_line(
    point_a: Vector | None,
    point_b: Vector | None,
    color,
    thickness: float = 1.5,
) -> None:
    """Draw an antialiased thick stroke with Blender's polyline shader."""
    if point_a is None or point_b is None:
        return
    if _active_line_batcher is not None:
        _active_line_batcher.add_segment(point_a, point_b, color, thickness)
        return
    shader = _line_shader()
    batch = batch_for_shader(
        shader,
        "LINES",
        {"pos": [_as_pos3(point_a), _as_pos3(point_b)]},
    )
    shader.bind()
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", _s(thickness))
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_dashed_polyline(
    points: list[Vector],
    thickness: float,
    pattern,
    *,
    slot_lengths: list[float] | None = None,
) -> None:
    """Stroke a screen-space polyline with a repeating color pattern (None = gap)."""
    if len(points) < 2 or not pattern:
        return
    # Dash/gap lengths are logical px; scale so Retina dashes match 1× displays.
    scale = ui_scale()
    lengths = [
        float(length) * scale
        for length in (slot_lengths or [_GUIDE_DASH_LENGTH] * len(pattern))
    ]
    if len(lengths) != len(pattern):
        raise ValueError("dash pattern and slot_lengths must match")
    period = float(sum(max(value, 0.0) for value in lengths))
    if period < 1.0e-6:
        color = next((entry for entry in pattern if entry is not None), None)
        if color is None:
            return
        for index in range(len(points) - 1):
            _draw_line(points[index], points[index + 1], color, thickness)
        return

    # Prefixed slot ends so we can map distance → pattern index quickly.
    slot_ends = []
    running = 0.0
    for length in lengths:
        running += max(float(length), 0.0)
        slot_ends.append(running)

    def color_at(distance: float):
        position = distance % period
        for index, end in enumerate(slot_ends):
            if position < end or index == len(slot_ends) - 1:
                remaining = end - position
                return pattern[index], max(remaining, 1.0e-8)
        return pattern[-1], 1.0e-8

    distance_along = 0.0
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        edge = end - start
        edge_length = float(edge.length)
        if edge_length < 1.0e-8:
            continue
        direction = edge / edge_length
        traveled = 0.0
        while traveled < edge_length - 1.0e-8:
            color, remaining = color_at(distance_along)
            step = min(remaining, edge_length - traveled)
            if color is not None and step > 1.0e-8:
                _draw_line(
                    start + direction * traveled,
                    start + direction * (traveled + step),
                    color,
                    thickness,
                )
            traveled += step
            distance_along += step


def _append_clipped_chain(
    chains: list[list[Vector]],
    current: list[Vector],
    point_a: Vector,
    point_b: Vector,
) -> list[Vector]:
    """Extend or restart a polyline chain from one clipped screen segment."""
    if not current:
        return [point_a, point_b]
    if (current[-1] - point_a).length > 1.0e-3:
        if len(current) >= 2:
            chains.append(current)
        return [point_a, point_b]
    current.append(point_b)
    return current


def _ideal_screen_chains(
    context: bpy.types.Context,
    point_a: tuple[float, float] | np.ndarray,
    point_b: tuple[float, float] | np.ndarray,
    *,
    samples: int,
    clip_bounds: tuple[float, float, float, float],
    frame_bounds: tuple[float, float, float, float] | None = None,
) -> list[list[Vector]]:
    """Map an ideal segment to clipped screen-space polylines."""
    settings = properties.active_session(context)
    if settings is None:
        return []
    first = np.asarray(point_a, dtype=np.float64)
    second = np.asarray(point_b, dtype=np.float64)
    # Straight in ideal space when λ≈0 — two samples are enough.
    sample_count = 2 if abs(settings.division_lambda) < 1.0e-12 else max(3, samples)
    chains: list[list[Vector]] = []
    current: list[Vector] = []
    previous = None
    for ratio in np.linspace(0.0, 1.0, sample_count):
        point = first * (1.0 - ratio) + second * ratio
        screen = scene.ideal_to_region(
            context,
            float(point[0]),
            float(point[1]),
            bounds=frame_bounds,
        )
        if previous is not None and screen is not None:
            clipped = _clip_segment_to_bounds(previous, screen, clip_bounds)
            if clipped is None:
                if len(current) >= 2:
                    chains.append(current)
                current = []
            else:
                current = _append_clipped_chain(chains, current, clipped[0], clipped[1])
        elif screen is None and current:
            if len(current) >= 2:
                chains.append(current)
            current = []
        previous = screen
    if len(current) >= 2:
        chains.append(current)
    return chains


def _region_bounds(context: bpy.types.Context) -> tuple[float, float, float, float] | None:
    """Return the 3D View region rectangle in POST_PIXEL coordinates."""
    region = context.region
    if region is None:
        return None
    return 0.0, float(region.width), 0.0, float(region.height)


def _image_diagonal(settings) -> float:
    return float(np.hypot(settings.image_width, settings.image_height))


def _finite_vanishing_xy(vanishing: np.ndarray) -> np.ndarray | None:
    """Return ideal-pixel VP coordinates, or None when the VP is at infinity."""
    if abs(float(vanishing[2])) < 1.0e-10:
        return None
    return vanishing[:2] / vanishing[2]


def _vanishing_within_draw_limit(settings, ideal_xy: np.ndarray) -> bool:
    """Reject near-parallel VPs that would stretch guides across the whole viewport."""
    center = np.array((settings.image_width * 0.5, settings.image_height * 0.5), dtype=np.float64)
    return float(np.linalg.norm(ideal_xy - center)) <= _image_diagonal(settings) * _MAX_VP_DIAGONALS


def _point_in_bounds(
    point: Vector,
    bounds: tuple[float, float, float, float],
) -> bool:
    left, right, bottom, top = bounds
    return left <= point.x <= right and bottom <= point.y <= top


def _draw_ideal_segment(
    context: bpy.types.Context,
    point_a: tuple[float, float] | np.ndarray,
    point_b: tuple[float, float] | np.ndarray,
    color,
    thickness: float,
    *,
    samples: int = 20,
    clip_bounds: tuple[float, float, float, float] | None = None,
    frame_bounds: tuple[float, float, float, float] | None = None,
    dash_pattern=None,
    dash_slot_lengths: list[float] | None = None,
) -> None:
    """Draw an ideal-space line, curving it on the original distorted plate."""
    settings = properties.active_session(context)
    if settings is None:
        return
    bounds = clip_bounds if clip_bounds is not None else frame_bounds
    if bounds is None:
        bounds = scene.camera_frame_bounds(context)
    if bounds is None:
        return
    plate_bounds = frame_bounds if frame_bounds is not None else bounds
    chains = _ideal_screen_chains(
        context,
        point_a,
        point_b,
        samples=samples,
        clip_bounds=bounds,
        frame_bounds=plate_bounds,
    )
    if not dash_pattern:
        for chain in chains:
            for index in range(len(chain) - 1):
                _draw_line(chain[index], chain[index + 1], color, thickness)
        return
    for chain in chains:
        _draw_dashed_polyline(
            chain,
            thickness,
            dash_pattern,
            slot_lengths=dash_slot_lengths,
        )


def _guide_sample_count(settings) -> int:
    """Dense enough for curved λ≠0 guides; cheap when the plate is linear."""
    if abs(settings.division_lambda) < 1.0e-12:
        return 2
    return 24


def _draw_ideal_guide_line(
    context: bpy.types.Context,
    ideal_point: np.ndarray,
    ideal_direction: np.ndarray,
    color,
    thickness: float,
    *,
    target_xy: np.ndarray | None = None,
    frame_bounds: tuple[float, float, float, float] | None = None,
    dash_pattern=None,
    dash_slot_lengths: list[float] | None = None,
) -> None:
    """Draw a VP guide; reach a finite VP when nearby, otherwise use a parallel cap."""
    direction = np.asarray(ideal_direction, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1.0e-12:
        return
    direction /= direction_length
    settings = properties.active_session(context)
    if settings is None:
        return
    region_bounds = _region_bounds(context)
    if region_bounds is None:
        return
    plate_bounds = frame_bounds if frame_bounds is not None else scene.camera_frame_bounds(context)
    image_center = np.array((settings.image_width * 0.5, settings.image_height * 0.5))
    point = np.asarray(ideal_point, dtype=np.float64)
    anchor = point + direction * float(np.dot(image_center - point, direction))
    diagonal = _image_diagonal(settings)
    plate_extent = diagonal * 2.0

    if target_xy is not None and _vanishing_within_draw_limit(settings, target_xy):
        # Cover the plate on the far side, then stop exactly at the VP.
        to_target = float(np.dot(target_xy - anchor, direction))
        far_extent = max(plate_extent, abs(to_target))
        start = anchor - direction * far_extent
        end = np.asarray(target_xy, dtype=np.float64)
        # Keep winding stable so dense sampling stays evenly spaced.
        if to_target < 0.0:
            start, end = end, anchor + direction * far_extent
    else:
        extent = diagonal * _PARALLEL_GUIDE_DIAGONALS
        start = anchor - direction * extent
        end = anchor + direction * extent

    pattern = dash_pattern if dash_pattern is not None else (color, None)
    slots = dash_slot_lengths
    if slots is None and dash_pattern is None:
        slots = [_GUIDE_DASH_LENGTH, _GUIDE_GAP_LENGTH]
    _draw_ideal_segment(
        context,
        start,
        end,
        color,
        thickness,
        samples=_guide_sample_count(settings),
        clip_bounds=region_bounds,
        frame_bounds=plate_bounds,
        dash_pattern=pattern,
        dash_slot_lengths=slots,
    )


def _draw_circle(shader, center: Vector | None, radius: float, color, *, filled: bool) -> None:
    if center is None:
        return
    radius = _s(radius)
    if filled:
        vertices = [tuple(center)]
        primitive = "TRI_FAN"
        count = 33
    else:
        vertices = []
        primitive = "LINE_LOOP"
        count = 32
    for index in range(count):
        angle = index / 32.0 * math.tau
        vertices.append((center.x + radius * math.cos(angle), center.y + radius * math.sin(angle)))
    batch = batch_for_shader(shader, primitive, {"pos": vertices})
    shader.bind()
    shader.uniform_float("color", color)
    if not filled:
        # LINE_LOOP width is framebuffer pixels; scale so outlines match 1× displays.
        gpu.state.line_width_set(max(1.0, _s(1.25)))
    batch.draw(shader)
    if not filled:
        gpu.state.line_width_set(1.0)


def _clip_segment_to_bounds(
    point_a: Vector,
    point_b: Vector,
    bounds: tuple[float, float, float, float],
) -> tuple[Vector, Vector] | None:
    """Clip a finite 2D segment to a rectangular pixel border."""
    left, right, bottom, top = bounds
    delta = point_b - point_a
    entering = 0.0
    leaving = 1.0
    for p_value, q_value in (
        (-delta.x, point_a.x - left),
        (delta.x, right - point_a.x),
        (-delta.y, point_a.y - bottom),
        (delta.y, top - point_a.y),
    ):
        if abs(p_value) < 1.0e-12:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            entering = max(entering, ratio)
        else:
            leaving = min(leaving, ratio)
        if entering > leaving:
            return None
    return point_a + delta * entering, point_a + delta * leaving


def _draw_crosshair(shader, center: Vector | None, color, radius: float = 8.0) -> None:
    if center is None:
        return
    # Radius stays logical for _draw_circle; arm extension is screen px too.
    arm = _s(radius + 4.0)
    _draw_circle(shader, center, radius, color, filled=False)
    _draw_line(center + Vector((-arm, 0.0)), center + Vector((arm, 0.0)), color)
    _draw_line(center + Vector((0.0, -arm)), center + Vector((0.0, arm)), color)


def _error_label_anchor(point_a: Vector, point_b: Vector) -> Vector:
    """Prefer the right end on horizontal strokes, the top end on vertical ones."""
    delta_x = point_b.x - point_a.x
    delta_y = point_b.y - point_a.y
    if abs(delta_x) >= abs(delta_y):
        preferred = point_a if point_a.x >= point_b.x else point_b
        other = point_b if preferred is point_a else point_a
    else:
        preferred = point_a if point_a.y >= point_b.y else point_b
        other = point_b if preferred is point_a else point_a
    # Sit slightly inset from the preferred endpoint so handles stay clear.
    anchor = preferred.lerp(other, _ERROR_LABEL_INSET)
    tangent = other - preferred
    length = float(tangent.length)
    offset = _s(_ERROR_LABEL_OFFSET_PX)
    if length < 1.0e-6:
        return preferred + Vector((0.0, offset))
    tangent /= length
    # Screen-upward offset keeps the number "above" / beside the stroke.
    normal = Vector((-tangent.y, tangent.x))
    if normal.y < 0.0:
        normal = -normal
    return anchor + normal * offset


def _draw_error_label(position: Vector, text: str, opacity: float) -> None:
    """Draw a small shadowed residual number in POST_PIXEL space."""
    _draw_overlay_label(
        position,
        text,
        opacity,
        font_size=_ERROR_LABEL_FONT_SIZE,
        align="center",
    )


def _draw_overlay_label(
    position: Vector,
    text: str,
    opacity: float,
    *,
    font_size: float,
    align: str = "center",
) -> None:
    """Draw shadowed POST_PIXEL text; align is center or left of position."""
    if not text:
        return
    font_id = 0
    # blf.size is logical UI points; multiply so Retina text matches 1× displays.
    blf.size(font_id, _s(font_size))
    width, height = blf.dimensions(font_id, text)
    if align == "left":
        x = float(position.x)
    else:
        x = float(position.x) - width * 0.5
    blf.position(
        font_id,
        x,
        float(position.y) - height * 0.35,
        0.0,
    )
    blf.color(font_id, 1.0, 1.0, 1.0, max(0.15, min(1.0, opacity)))
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.85)
    shadow = max(1, int(round(ui_scale())))
    blf.shadow_offset(font_id, shadow, -shadow)
    blf.draw(font_id, text)
    blf.disable(font_id, blf.SHADOW)


def _draw_vp_error_labels(context: bpy.types.Context, settings) -> None:
    """Per-segment residual labels against the current camera's ideal VPs."""
    if not settings.show_vp_error_labels:
        return
    region_bounds = _region_bounds(context)
    if region_bounds is None:
        return
    bounds = scene.camera_frame_bounds(context)
    if bounds is None:
        return
    calibration = scene.calibration_from_settings(settings)
    _vanishing, _overlay_vps, ideal_endpoints = _cached_vp_overlay_solve(settings)
    opacity = settings.overlay_opacity
    for line_index, (axis, ideal_a, ideal_b) in enumerate(ideal_endpoints):
        line = settings.lines[line_index]
        point_a = scene.image_to_region(context, line.x1, line.y1, bounds=bounds)
        point_b = scene.image_to_region(context, line.x2, line.y2, bounds=bounds)
        if point_a is None or point_b is None:
            continue
        residual = core.vp_ideal_segment_residual_px(
            calibration,
            axis,
            core.LineSegment(
                float(ideal_a[0]),
                float(ideal_a[1]),
                float(ideal_b[0]),
                float(ideal_b[1]),
            ),
        )
        if residual is None:
            continue
        anchor = _error_label_anchor(point_a, point_b)
        if not _point_in_bounds(anchor, region_bounds):
            continue
        _draw_error_label(anchor, f"{residual:.1f}", opacity)
    # blf leaves blend/shader state dirty; restore so later GPU draws keep alpha.
    gpu.state.blend_set("ALPHA")


def _vp_overlay_cache_key(settings) -> tuple:
    """Fingerprint line geometry + intrinsics that affect overlay VP solves."""
    parts: list = [
        settings.vp_mode,
        float(settings.fx),
        float(settings.fy),
        float(settings.cx),
        float(settings.cy),
        float(settings.division_lambda),
        int(settings.image_width),
        int(settings.image_height),
        int(len(settings.lines)),
    ]
    for line in settings.lines:
        parts.extend(
            (
                line.axis,
                float(line.x1),
                float(line.y1),
                float(line.x2),
                float(line.y2),
            )
        )
    return tuple(parts)


def _cached_vp_overlay_solve(settings) -> tuple[dict, dict, tuple]:
    """Return vanishing points and ideal endpoints, recomputed only when needed."""
    key = _vp_overlay_cache_key(settings)
    if _vp_solve_cache["key"] == key:
        return (
            _vp_solve_cache["vanishing_points"],
            _vp_solve_cache["overlay_vanishing_points"],
            _vp_solve_cache["ideal_endpoints"],
        )
    line_bundles = scene.line_bundles_from_settings(settings)
    calibration = scene.calibration_from_settings(settings)
    ideal_line_bundles = core.undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    vanishing_points = core.collect_vanishing_points(ideal_line_bundles)
    overlay_vanishing_points = core.complete_vanishing_points(
        vanishing_points,
        calibration.intrinsics,
    )
    ideal_endpoints = []
    for line in settings.lines:
        endpoints = core.undistort_points(
            np.array([[line.x1, line.y1], [line.x2, line.y2]], dtype=np.float64),
            calibration.intrinsics.fx,
            calibration.intrinsics.fy,
            calibration.intrinsics.cx,
            calibration.intrinsics.cy,
            calibration.division_lambda,
        )
        ideal_endpoints.append((line.axis, endpoints[0].copy(), endpoints[1].copy()))
    _vp_solve_cache["key"] = key
    _vp_solve_cache["vanishing_points"] = vanishing_points
    _vp_solve_cache["overlay_vanishing_points"] = overlay_vanishing_points
    _vp_solve_cache["ideal_endpoints"] = tuple(ideal_endpoints)
    return vanishing_points, overlay_vanishing_points, _vp_solve_cache["ideal_endpoints"]


def _draw_vp_geometry(context: bpy.types.Context, fill_shader, settings) -> None:
    global _active_line_batcher
    if not settings.show_vp_overlay:
        return
    opacity = settings.overlay_opacity
    bounds = scene.camera_frame_bounds(context)
    if bounds is None:
        return
    region_bounds = _region_bounds(context)
    if region_bounds is None:
        return
    vanishing_points, overlay_vanishing_points, ideal_endpoints = _cached_vp_overlay_solve(
        settings
    )
    guide_samples = _guide_sample_count(settings)
    batcher = _LineBatcher()
    _active_line_batcher = batcher
    try:
        for line_index, (axis, ideal_a, ideal_b) in enumerate(ideal_endpoints):
            color = _with_alpha(AXIS_COLORS[axis], opacity)
            line = settings.lines[line_index]
            point_a = scene.image_to_region(
                context, line.x1, line.y1, bounds=bounds
            )
            point_b = scene.image_to_region(
                context, line.x2, line.y2, bounds=bounds
            )
            _draw_ideal_segment(
                context,
                ideal_a,
                ideal_b,
                color,
                2.4 if line_index == settings.selected_line_index else 1.8,
                frame_bounds=bounds,
            )

            vanishing = vanishing_points.get(axis)
            if vanishing is not None:
                midpoint = 0.5 * (ideal_a + ideal_b)
                target_xy = _finite_vanishing_xy(vanishing)
                if target_xy is None:
                    direction = vanishing[:2]
                else:
                    direction = target_xy - midpoint
                _draw_ideal_guide_line(
                    context,
                    midpoint,
                    direction,
                    _with_alpha(color, 0.42),
                    _GUIDE_LINE_THICKNESS,
                    target_xy=target_xy,
                    frame_bounds=bounds,
                )

            if line_index == settings.selected_line_index:
                # Endpoint handles only while Draw / Edit Lines owns the viewport.
                workspace = properties.workspace(context)
                if workspace.is_modal and workspace.work_mode == "LINE":
                    handle_color = _with_alpha(color, opacity)
                    _draw_circle(
                        fill_shader,
                        point_a,
                        7.0,
                        _with_alpha(handle_color, 0.45),
                        filled=True,
                    )
                    _draw_circle(fill_shader, point_a, 7.0, handle_color, filled=False)
                    _draw_circle(
                        fill_shader,
                        point_b,
                        7.0,
                        _with_alpha(handle_color, 0.45),
                        filled=True,
                    )
                    _draw_circle(fill_shader, point_b, 7.0, handle_color, filled=False)

        drawable_vps: dict[str, np.ndarray] = {}
        for axis, vanishing in overlay_vanishing_points.items():
            ideal_xy = _finite_vanishing_xy(vanishing)
            if ideal_xy is None or not _vanishing_within_draw_limit(settings, ideal_xy):
                continue
            drawable_vps[axis] = ideal_xy
            # Measured VPs always; implied ones only when they complete the horizon.
            if axis not in vanishing_points and axis not in ("x", "z"):
                continue
            marker = scene.ideal_to_region(
                context,
                float(ideal_xy[0]),
                float(ideal_xy[1]),
                bounds=bounds,
            )
            # Allow markers outside the plate, but still inside the 3D View region.
            if marker is not None and _point_in_bounds(marker, region_bounds):
                marker_opacity = opacity if axis in vanishing_points else opacity * 0.7
                _draw_crosshair(
                    fill_shader,
                    marker,
                    _with_alpha(AXIS_COLORS[axis], marker_opacity),
                )

        if "x" in overlay_vanishing_points and "z" in overlay_vanishing_points:
            first_xy = drawable_vps.get("x")
            second_xy = drawable_vps.get("z")
            horizon_red = _with_alpha(AXIS_COLORS["x"], opacity)
            horizon_green = _with_alpha(AXIS_COLORS["z"], opacity)
            # red · empty · green · empty …
            horizon_pattern = (horizon_red, None, horizon_green, None)
            horizon_slots = [_HORIZON_SLOT_LENGTH] * 4
            if first_xy is not None and second_xy is not None:
                _draw_ideal_segment(
                    context,
                    first_xy,
                    second_xy,
                    horizon_red,
                    _GUIDE_LINE_THICKNESS,
                    samples=guide_samples,
                    clip_bounds=region_bounds,
                    frame_bounds=bounds,
                    dash_pattern=horizon_pattern,
                    dash_slot_lengths=horizon_slots,
                )
            else:
                first = overlay_vanishing_points["x"]
                second = overlay_vanishing_points["z"]
                first_xy_raw = _finite_vanishing_xy(first)
                second_xy_raw = _finite_vanishing_xy(second)
                if first_xy_raw is not None and second_xy_raw is not None:
                    _draw_ideal_guide_line(
                        context,
                        0.5 * (first_xy_raw + second_xy_raw),
                        second_xy_raw - first_xy_raw,
                        horizon_red,
                        _GUIDE_LINE_THICKNESS,
                        frame_bounds=bounds,
                        dash_pattern=horizon_pattern,
                        dash_slot_lengths=horizon_slots,
                    )
    finally:
        batcher.flush()
        _active_line_batcher = None


def _draw_placement(context: bpy.types.Context, fill_shader, settings) -> None:
    if settings.origin_is_set:
        origin = scene.image_to_region(
            context,
            settings.origin_image[0],
            settings.origin_image[1],
        )
        _draw_crosshair(
            fill_shader,
            origin,
            _with_alpha((0.35, 1.0, 0.45, 1.0), settings.overlay_opacity),
            7.0,
        )

    # Violet PP marker when off-center, or while Manual PP Offset tool is active.
    workspace = properties.workspace(context)
    show_pp = scene.principal_point_is_off_center(settings) or (
        workspace.is_modal and workspace.work_mode == "PP"
    )
    if show_pp and settings.image_width > 0 and settings.image_height > 0:
        principal = scene.image_to_region(context, settings.cx, settings.cy)
        _draw_crosshair(
            fill_shader,
            principal,
            _with_alpha((0.72, 0.35, 1.0, 1.0), settings.overlay_opacity),
            9.0 if workspace.work_mode == "PP" else 7.0,
        )


def _draw_landmarks(context: bpy.types.Context, fill_shader, settings) -> None:
    """Draw landmark picks that belong to the active match."""
    space = properties.workspace(context)
    if not space.show_landmark_overlay:
        return
    root = properties.active_root(context)
    if root is None:
        return
    active_index = space.active_landmark_index
    opacity = settings.overlay_opacity
    for index, landmark in enumerate(space.landmarks):
        observation = scene.observation_for_match(landmark, root)
        if observation is None or not observation.is_set:
            continue
        is_active = index == active_index
        if landmark.known_object is not None:
            base = (0.35, 0.85, 1.0, 1.0) if is_active else (0.25, 0.7, 0.95, 1.0)
        else:
            base = (1.0, 0.85, 0.2, 1.0) if is_active else (0.95, 0.65, 0.15, 1.0)
        # Dim picks that are excluded from the sync solve.
        draw_opacity = opacity * (0.35 if not landmark.use_in_sync else 1.0)
        color = _with_alpha(base, draw_opacity)
        if landmark.kind == "LINE":
            point_a = scene.image_to_region(context, observation.x, observation.y)
            point_b = scene.image_to_region(context, observation.x2, observation.y2)
            if point_a is None or point_b is None:
                continue
            _draw_line(
                point_a,
                point_b,
                color,
                3.0 if is_active else 2.0,
            )
            _draw_crosshair(fill_shader, point_a, color, 7.0 if is_active else 5.0)
            _draw_crosshair(fill_shader, point_b, color, 7.0 if is_active else 5.0)
            continue
        point = scene.image_to_region(context, observation.x, observation.y)
        _draw_crosshair(fill_shader, point, color, 9.0 if is_active else 6.0)


def _draw_landmark_labels(context: bpy.types.Context, settings) -> None:
    """Name labels beside each pick in the active match (when toggled on)."""
    space = properties.workspace(context)
    if not space.show_landmark_overlay or not space.show_landmark_labels:
        return
    root = properties.active_root(context)
    if root is None:
        return
    opacity = settings.overlay_opacity
    for landmark in space.landmarks:
        observation = scene.observation_for_match(landmark, root)
        if observation is None or not observation.is_set:
            continue
        draw_opacity = opacity * (0.35 if not landmark.use_in_sync else 1.0)
        if landmark.kind == "LINE":
            point_a = scene.image_to_region(context, observation.x, observation.y)
            point_b = scene.image_to_region(context, observation.x2, observation.y2)
            if point_a is None or point_b is None:
                continue
            anchor = (point_a + point_b) * 0.5
        else:
            anchor = scene.image_to_region(context, observation.x, observation.y)
            if anchor is None:
                continue
        _draw_overlay_label(
            anchor + Vector((_s(_LANDMARK_LABEL_OFFSET_PX), _s(_LANDMARK_LABEL_OFFSET_PX))),
            landmark.name or "Landmark",
            draw_opacity,
            font_size=_LANDMARK_LABEL_FONT_SIZE,
            align="left",
        )
    # blf leaves blend/shader state dirty; restore so later GPU draws keep alpha.
    gpu.state.blend_set("ALPHA")


def _draw_rect(shader, x0: float, y0: float, x1: float, y1: float, color) -> None:
    """Axis-aligned filled rectangle in POST_PIXEL coordinates."""
    batch = batch_for_shader(
        shader,
        "TRI_FAN",
        {
            "pos": [
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            ]
        },
    )
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _region_overlap_insets(context: bpy.types.Context) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) insets when UI floats over the WINDOW.

    With Preferences → Interface → Region Overlap (default on), HEADER / TOOL_HEADER
    / UI / TOOLS sit on top of the 3D View WINDOW. POST_PIXEL draws under them unless
    we offset chrome into the uncovered plate.
    """
    area = context.area
    preferences = getattr(context, "preferences", None)
    system = getattr(preferences, "system", None) if preferences is not None else None
    if (
        area is None
        or system is None
        or not bool(getattr(system, "use_region_overlap", False))
    ):
        return 0.0, 0.0, 0.0, 0.0
    left = right = bottom = top = 0.0
    for region in area.regions:
        # Collapsed / hidden regions report tiny sizes.
        if region.width <= 1 and region.height <= 1:
            continue
        if region.type in {"HEADER", "TOOL_HEADER"} and region.alignment == "TOP":
            top += float(region.height)
        elif region.type in {"HEADER", "TOOL_HEADER", "FOOTER"} and region.alignment == "BOTTOM":
            bottom += float(region.height)
        elif region.type in {"TOOLS", "UI", "HUD"} and region.alignment == "LEFT":
            left += float(region.width)
        elif region.type in {"TOOLS", "UI", "HUD"} and region.alignment == "RIGHT":
            right += float(region.width)
    return left, right, bottom, top


def _draw_interact_mode_chrome(context: bpy.types.Context, workspace) -> None:
    """Single top banner + side/bottom frame while a Draw / Pick tool is active."""
    if not workspace.is_modal:
        return
    region = context.region
    if region is None or region.type != "WINDOW":
        return
    chrome = _MODE_CHROME.get(workspace.work_mode)
    if chrome is None:
        return
    width = float(region.width)
    height = float(region.height)
    if width < 8.0 or height < 8.0:
        return
    left, right, bottom, top = _region_overlap_insets(context)
    # Visible plate bounds inside floating Blender chrome.
    x0 = left
    x1 = width - right
    y0 = bottom
    y1 = height - top
    if x1 - x0 < 32.0 or y1 - y0 < 48.0:
        return
    accent = chrome["color"]
    fill_shader = _fill_shader()
    banner_height = _s(_MODE_BANNER_HEIGHT)
    accent_bar = _s(3.0)
    banner_top = y1
    banner_bottom = y1 - banner_height
    # One dark banner just below the overlapping header / tool header.
    _draw_rect(
        fill_shader,
        x0,
        banner_bottom,
        x1,
        banner_top,
        (0.05, 0.05, 0.05, 0.78),
    )
    _draw_rect(
        fill_shader,
        x0,
        banner_bottom,
        x1,
        banner_bottom + accent_bar,
        accent,
    )
    # Frame only left / right / bottom so it does not stack on the banner.
    thickness = _s(_MODE_FRAME_THICKNESS)
    frame = (accent[0], accent[1], accent[2], min(1.0, accent[3] * 0.95))
    _draw_rect(fill_shader, x0, y0, x1, y0 + thickness, frame)
    _draw_rect(fill_shader, x0, y0, x0 + thickness, banner_bottom, frame)
    _draw_rect(fill_shader, x1 - thickness, y0, x1, banner_bottom, frame)

    label = f"{chrome['title']}  ·  {chrome['hint']}"
    font_id = 0
    blf.size(font_id, _s(_MODE_BANNER_FONT_SIZE))
    text_width, text_height = blf.dimensions(font_id, label)
    pad = _s(12.0)
    blf.position(
        font_id,
        max(x0 + pad, x0 + (x1 - x0 - text_width) * 0.5),
        banner_bottom + (banner_height - text_height) * 0.5,
        0.0,
    )
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 0.9)
    shadow = max(1, int(round(ui_scale())))
    blf.shadow_offset(font_id, shadow, -shadow)
    blf.draw(font_id, label)
    blf.disable(font_id, blf.SHADOW)
    # blf dirties GPU blend state used by later overlay geometry.
    gpu.state.blend_set("ALPHA")


def _draw_preview(context: bpy.types.Context, settings) -> None:
    if (
        context.area is None
        or context.area.as_pointer() != _preview["area_pointer"]
        or _preview["start"] is None
        or _preview["end"] is None
        or _preview["kind"] != "LINE"
    ):
        return
    start = _preview["start"]
    end = _preview["end"]
    calibration = scene.calibration_from_settings(settings)
    ideal = core.undistort_points(
        np.asarray([start, end], dtype=np.float64),
        calibration.intrinsics.fx,
        calibration.intrinsics.fy,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
        calibration.division_lambda,
    )
    _draw_ideal_segment(
        context,
        ideal[0],
        ideal[1],
        _with_alpha(AXIS_COLORS[settings.active_axis], settings.overlay_opacity),
        2.0,
    )


def _draw_callback() -> None:
    context = bpy.context
    if (
        context.area is None
        or context.area.type != "VIEW_3D"
        or not hasattr(context.scene, "match_perspective")
    ):
        return
    workspace = context.scene.match_perspective
    gpu.state.blend_set("ALPHA")
    try:
        # Mode chrome stays up even if the user orbits out of camera view.
        _draw_interact_mode_chrome(context, workspace)
        settings = properties.active_session(context)
        if (
            settings is None
            or settings.image is None
            or not scene.is_camera_view(context)
        ):
            return
        fill_shader = _fill_shader()
        _draw_vp_geometry(context, fill_shader, settings)
        _draw_placement(context, fill_shader, settings)
        _draw_landmarks(context, fill_shader, settings)
        _draw_preview(context, settings)
        # After GPU geometry so blf cannot wipe landmark / preview alpha blending.
        _draw_landmark_labels(context, settings)
        _draw_vp_error_labels(context, settings)
    except Exception:
        # Keep the handler alive — an uncaught error can stop POST_PIXEL draws.
        import traceback

        traceback.print_exc()
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")


def register_viewport_draw_handler() -> None:
    """Register one 3D View POST_PIXEL callback."""
    ensure_viewport_draw_handler()


def ensure_viewport_draw_handler() -> None:
    """Drop a stale handle if needed and ensure a live POST_PIXEL callback."""
    global _draw_handle
    # Survive importlib.reload: previous module may have lost _draw_handle.
    namespace_key = "perspective_match_draw_handle"
    previous = bpy.app.driver_namespace.pop(namespace_key, None)
    for handle in (_draw_handle, previous):
        if handle is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ValueError, RuntimeError):
            pass
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_callback,
        (),
        "WINDOW",
        "POST_PIXEL",
    )
    bpy.app.driver_namespace[namespace_key] = _draw_handle


def unregister_viewport_draw_handler() -> None:
    """Remove the registered viewport callback."""
    global _draw_handle
    namespace_key = "perspective_match_draw_handle"
    previous = bpy.app.driver_namespace.pop(namespace_key, None)
    for handle in (_draw_handle, previous):
        if handle is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ValueError, RuntimeError):
            pass
    _draw_handle = None
    clear_preview()
