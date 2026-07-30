"""GPU overlay for VP guides, handles, origin, and landmarks."""

from __future__ import annotations

import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import core, properties, scene

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

_draw_handle = None
_preview: dict[str, object] = {
    "kind": "",
    "start": None,
    "end": None,
    "area_pointer": 0,
}


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
    shader = _line_shader()
    batch = batch_for_shader(
        shader,
        "LINES",
        {"pos": [_as_pos3(point_a), _as_pos3(point_b)]},
    )
    shader.bind()
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", thickness)
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
    lengths = slot_lengths or [_GUIDE_DASH_LENGTH] * len(pattern)
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
) -> list[list[Vector]]:
    """Map an ideal segment to clipped screen-space polylines."""
    settings = properties.active_session(context)
    if settings is None:
        return []
    first = np.asarray(point_a, dtype=np.float64)
    second = np.asarray(point_b, dtype=np.float64)
    sample_count = 2 if abs(settings.division_lambda) < 1.0e-12 else max(3, samples)
    chains: list[list[Vector]] = []
    current: list[Vector] = []
    previous = None
    for ratio in np.linspace(0.0, 1.0, sample_count):
        point = first * (1.0 - ratio) + second * ratio
        screen = scene.ideal_to_region(context, float(point[0]), float(point[1]))
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
    dash_pattern=None,
    dash_slot_lengths: list[float] | None = None,
) -> None:
    """Draw an ideal-space line, curving it on the original distorted plate."""
    settings = properties.active_session(context)
    if settings is None:
        return
    bounds = clip_bounds if clip_bounds is not None else scene.camera_frame_bounds(context)
    if bounds is None:
        return
    chains = _ideal_screen_chains(
        context,
        point_a,
        point_b,
        samples=samples,
        clip_bounds=bounds,
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


def _draw_ideal_guide_line(
    context: bpy.types.Context,
    ideal_point: np.ndarray,
    ideal_direction: np.ndarray,
    color,
    thickness: float,
    *,
    target_xy: np.ndarray | None = None,
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
        samples=96,
        clip_bounds=region_bounds,
        dash_pattern=pattern,
        dash_slot_lengths=slots,
    )


def _draw_circle(shader, center: Vector | None, radius: float, color, *, filled: bool) -> None:
    if center is None:
        return
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
    batch.draw(shader)


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
    _draw_circle(shader, center, radius, color, filled=False)
    _draw_line(center + Vector((-radius - 4.0, 0.0)), center + Vector((radius + 4.0, 0.0)), color)
    _draw_line(center + Vector((0.0, -radius - 4.0)), center + Vector((0.0, radius + 4.0)), color)


def _draw_vp_geometry(context: bpy.types.Context, fill_shader, settings) -> None:
    if not settings.show_vp_overlay:
        return
    opacity = settings.overlay_opacity
    bounds = scene.camera_frame_bounds(context)
    if bounds is None:
        return
    region_bounds = _region_bounds(context)
    if region_bounds is None:
        return
    line_bundles = scene.line_bundles_from_settings(settings)
    calibration = scene.calibration_from_settings(settings)
    ideal_line_bundles = core.undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    vanishing_points = core.collect_vanishing_points(ideal_line_bundles)
    # Derive missing orthogonal VPs so the horizon still appears with only one horizontal.
    overlay_vanishing_points = core.complete_vanishing_points(
        vanishing_points,
        calibration.intrinsics,
    )

    for line_index, line in enumerate(settings.lines):
        color = _with_alpha(AXIS_COLORS[line.axis], opacity)
        point_a = scene.image_to_region(context, line.x1, line.y1)
        point_b = scene.image_to_region(context, line.x2, line.y2)
        ideal_endpoints = core.undistort_points(
            np.array([[line.x1, line.y1], [line.x2, line.y2]], dtype=np.float64),
            calibration.intrinsics.fx,
            calibration.intrinsics.fy,
            calibration.intrinsics.cx,
            calibration.intrinsics.cy,
            calibration.division_lambda,
        )
        _draw_ideal_segment(
            context,
            ideal_endpoints[0],
            ideal_endpoints[1],
            color,
            2.4 if line_index == settings.selected_line_index else 1.8,
        )

        vanishing = vanishing_points.get(line.axis)
        if vanishing is not None:
            midpoint = 0.5 * (ideal_endpoints[0] + ideal_endpoints[1])
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
            )

        if line_index == settings.selected_line_index:
            handle_color = _with_alpha(color, opacity)
            _draw_circle(fill_shader, point_a, 7.0, _with_alpha(handle_color, 0.45), filled=True)
            _draw_circle(fill_shader, point_a, 7.0, handle_color, filled=False)
            _draw_circle(fill_shader, point_b, 7.0, _with_alpha(handle_color, 0.45), filled=True)
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
                samples=96,
                clip_bounds=region_bounds,
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
                    dash_pattern=horizon_pattern,
                    dash_slot_lengths=horizon_slots,
                )


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
        color = _with_alpha(base, opacity)
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
        AXIS_COLORS[settings.active_axis],
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
    settings = properties.active_session(context)
    if settings is None or settings.image is None or not scene.is_camera_view(context):
        return
    fill_shader = _fill_shader()
    gpu.state.blend_set("ALPHA")
    try:
        _draw_vp_geometry(context, fill_shader, settings)
        _draw_placement(context, fill_shader, settings)
        _draw_landmarks(context, fill_shader, settings)
        _draw_preview(context, settings)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")


def register_viewport_draw_handler() -> None:
    """Register one 3D View POST_PIXEL callback."""
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback,
            (),
            "WINDOW",
            "POST_PIXEL",
        )


def unregister_viewport_draw_handler() -> None:
    """Remove the registered viewport callback."""
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    clear_preview()
