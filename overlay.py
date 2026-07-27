"""GPU overlay for VP guides, handles, and origin."""

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


def _draw_ideal_segment(
    context: bpy.types.Context,
    point_a: tuple[float, float] | np.ndarray,
    point_b: tuple[float, float] | np.ndarray,
    color,
    thickness: float,
    *,
    samples: int = 20,
) -> None:
    """Draw an ideal-space line, curving it on the original distorted plate."""
    first = np.asarray(point_a, dtype=np.float64)
    second = np.asarray(point_b, dtype=np.float64)
    settings = properties.active_session(context)
    if settings is None:
        return
    bounds = scene.camera_frame_bounds(context)
    if bounds is None:
        return
    sample_count = 2 if abs(settings.division_lambda) < 1.0e-12 else max(3, samples)
    previous = None
    for ratio in np.linspace(0.0, 1.0, sample_count):
        point = first * (1.0 - ratio) + second * ratio
        screen = scene.ideal_to_region(context, float(point[0]), float(point[1]))
        if previous is not None and screen is not None:
            clipped = _clip_segment_to_bounds(previous, screen, bounds)
            if clipped is not None:
                _draw_line(clipped[0], clipped[1], color, thickness)
        previous = screen


def _draw_ideal_infinite_line(
    context: bpy.types.Context,
    ideal_point: np.ndarray,
    ideal_direction: np.ndarray,
    color,
    thickness: float,
) -> None:
    """Draw a long ideal line as a clipped-looking curved polyline."""
    direction = np.asarray(ideal_direction, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1.0e-12:
        return
    direction /= direction_length
    settings = properties.active_session(context)
    if settings is None:
        return
    image_center = np.array((settings.image_width * 0.5, settings.image_height * 0.5))
    point = np.asarray(ideal_point, dtype=np.float64)
    anchor = point + direction * float(np.dot(image_center - point, direction))
    extent = float(np.hypot(settings.image_width, settings.image_height)) * 4.0
    _draw_ideal_segment(
        context,
        anchor - direction * extent,
        anchor + direction * extent,
        color,
        thickness,
        samples=96,
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
    """Clip a finite 2D segment to a rectangular camera border."""
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
    line_bundles = scene.line_bundles_from_settings(settings)
    calibration = scene.calibration_from_settings(settings)
    ideal_line_bundles = core.undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    vanishing_points = core.collect_vanishing_points(ideal_line_bundles)

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
            if abs(float(vanishing[2])) < 1.0e-10:
                direction = vanishing[:2]
            else:
                direction = vanishing[:2] / vanishing[2] - midpoint
            _draw_ideal_infinite_line(
                context,
                midpoint,
                direction,
                _with_alpha(color, 0.42),
                0.9,
            )

        if line_index == settings.selected_line_index:
            handle_color = _with_alpha(color, settings.controls_opacity)
            _draw_circle(fill_shader, point_a, 7.0, _with_alpha(handle_color, 0.45), filled=True)
            _draw_circle(fill_shader, point_a, 7.0, handle_color, filled=False)
            _draw_circle(fill_shader, point_b, 7.0, _with_alpha(handle_color, 0.45), filled=True)
            _draw_circle(fill_shader, point_b, 7.0, handle_color, filled=False)

    for axis, vanishing in vanishing_points.items():
        if abs(float(vanishing[2])) < 1.0e-10:
            continue
        marker = scene.ideal_to_region(
            context,
            float(vanishing[0] / vanishing[2]),
            float(vanishing[1] / vanishing[2]),
        )
        if marker is not None:
            left, right, bottom, top = bounds
            if left <= marker.x <= right and bottom <= marker.y <= top:
                _draw_crosshair(fill_shader, marker, _with_alpha(AXIS_COLORS[axis], opacity))

    if "x" in vanishing_points and "z" in vanishing_points:
        first = vanishing_points["x"]
        second = vanishing_points["z"]
        if abs(float(first[2])) > 1.0e-10 and abs(float(second[2])) > 1.0e-10:
            first_ideal = first[:2] / first[2]
            second_ideal = second[:2] / second[2]
            _draw_ideal_infinite_line(
                context,
                first_ideal,
                second_ideal - first_ideal,
                _with_alpha(AXIS_COLORS["y"], opacity * 0.75),
                1.4,
            )


def _draw_placement(context: bpy.types.Context, fill_shader, settings) -> None:
    if settings.origin_is_set:
        origin = scene.image_to_region(
            context,
            settings.origin_image[0],
            settings.origin_image[1],
        )
        _draw_crosshair(fill_shader, origin, (0.35, 1.0, 0.45, settings.controls_opacity), 7.0)


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
