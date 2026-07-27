"""GPU overlay for VP guides, surfaces, handles, origin, and scale."""

from __future__ import annotations

import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import core, properties, scene

AXIS_COLORS = {
    "x": (0.91, 0.36, 0.36, 1.0),
    "y": (0.36, 0.56, 0.91, 1.0),
    "z": (0.91, 0.78, 0.36, 1.0),
}
SURFACE_COLORS = {
    "xz": (0.77, 0.94, 0.30, 1.0),
    "yz": (0.94, 0.76, 0.30, 1.0),
    "yx": (1.0, 0.42, 0.29, 1.0),
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


def _shader():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _with_alpha(color, alpha: float):
    return color[0], color[1], color[2], max(0.0, min(1.0, color[3] * alpha))


def _line_quad(
    point_a: Vector,
    point_b: Vector,
    thickness: float,
) -> list[tuple[float, float]] | None:
    delta = point_b - point_a
    if delta.length < 1.0e-6:
        return None
    perpendicular = Vector((-delta.y, delta.x)).normalized() * thickness * 0.5
    return [
        tuple(point_a - perpendicular),
        tuple(point_a + perpendicular),
        tuple(point_b + perpendicular),
        tuple(point_b - perpendicular),
    ]


def _draw_line(
    shader,
    point_a: Vector | None,
    point_b: Vector | None,
    color,
    thickness: float = 1.5,
) -> None:
    if point_a is None or point_b is None:
        return
    vertices = _line_quad(point_a, point_b, thickness)
    if vertices is None:
        return
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": vertices})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_ideal_segment(
    context: bpy.types.Context,
    shader,
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
    settings = context.scene.match_perspective
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
                _draw_line(shader, clipped[0], clipped[1], color, thickness)
        previous = screen


def _draw_ideal_infinite_line(
    context: bpy.types.Context,
    shader,
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
    settings = context.scene.match_perspective
    image_center = np.array((settings.image_width * 0.5, settings.image_height * 0.5))
    point = np.asarray(ideal_point, dtype=np.float64)
    anchor = point + direction * float(np.dot(image_center - point, direction))
    extent = float(np.hypot(settings.image_width, settings.image_height)) * 4.0
    _draw_ideal_segment(
        context,
        shader,
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


def _draw_polygon(shader, points: list[Vector], color) -> None:
    if len(points) != 4:
        return
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": [tuple(point) for point in points]},
        indices=[(0, 1, 2), (0, 2, 3)],
    )
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _clip_infinite_line(
    point: Vector,
    direction: Vector,
    bounds: tuple[float, float, float, float],
) -> tuple[Vector, Vector] | None:
    left, right, bottom, top = bounds
    intersections: list[Vector] = []
    if abs(direction.x) > 1.0e-9:
        for x_coordinate in (left, right):
            distance = (x_coordinate - point.x) / direction.x
            y_coordinate = point.y + distance * direction.y
            if bottom - 0.5 <= y_coordinate <= top + 0.5:
                intersections.append(Vector((x_coordinate, y_coordinate)))
    if abs(direction.y) > 1.0e-9:
        for y_coordinate in (bottom, top):
            distance = (y_coordinate - point.y) / direction.y
            x_coordinate = point.x + distance * direction.x
            if left - 0.5 <= x_coordinate <= right + 0.5:
                intersections.append(Vector((x_coordinate, y_coordinate)))
    unique: list[Vector] = []
    for intersection in intersections:
        if not any((intersection - existing).length < 0.5 for existing in unique):
            unique.append(intersection)
    return (unique[0], unique[1]) if len(unique) >= 2 else None


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
    _draw_line(shader, center + Vector((-radius - 4.0, 0.0)), center + Vector((radius + 4.0, 0.0)), color)
    _draw_line(shader, center + Vector((0.0, -radius - 4.0)), center + Vector((0.0, radius + 4.0)), color)


def _draw_vp_geometry(context: bpy.types.Context, shader, settings) -> None:
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
            shader,
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
                shader,
                midpoint,
                direction,
                _with_alpha(color, 0.42),
                0.9,
            )

        if line_index == settings.selected_line_index:
            handle_color = _with_alpha(color, settings.controls_opacity)
            _draw_circle(shader, point_a, 7.0, _with_alpha(handle_color, 0.45), filled=True)
            _draw_circle(shader, point_a, 7.0, handle_color, filled=False)
            _draw_circle(shader, point_b, 7.0, _with_alpha(handle_color, 0.45), filled=True)
            _draw_circle(shader, point_b, 7.0, handle_color, filled=False)

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
                _draw_crosshair(shader, marker, _with_alpha(AXIS_COLORS[axis], opacity))

    if "x" in vanishing_points and "z" in vanishing_points:
        first = vanishing_points["x"]
        second = vanishing_points["z"]
        if abs(float(first[2])) > 1.0e-10 and abs(float(second[2])) > 1.0e-10:
            first_ideal = first[:2] / first[2]
            second_ideal = second[:2] / second[2]
            _draw_ideal_infinite_line(
                context,
                shader,
                first_ideal,
                second_ideal - first_ideal,
                _with_alpha(AXIS_COLORS["y"], opacity * 0.75),
                1.4,
            )


def _draw_surfaces(context: bpy.types.Context, shader, settings) -> None:
    if not settings.show_surface_overlay:
        return
    for surface_index, surface in enumerate(settings.surfaces):
        geometry = scene.surface_ideal_geometry(settings, surface)
        if geometry is None:
            continue
        corners, grid = geometry
        region_corners = [
            scene.ideal_to_region(context, point[0], point[1])
            for point in corners
        ]
        if any(point is None for point in region_corners):
            continue
        color = _with_alpha(SURFACE_COLORS[surface.plane], settings.overlay_opacity)
        _draw_polygon(shader, region_corners, _with_alpha(color, 0.16))
        for index in range(4):
            _draw_ideal_segment(
                context,
                shader,
                corners[index],
                corners[(index + 1) % 4],
                color,
                2.4 if surface_index == settings.selected_surface_index else 1.5,
            )
        for point_a, point_b in grid:
            _draw_ideal_segment(
                context,
                shader,
                point_a,
                point_b,
                _with_alpha(color, 0.65),
                0.9,
            )
        if surface_index == settings.selected_surface_index:
            _draw_circle(shader, region_corners[0], 7.0, color, filled=True)
            _draw_circle(shader, region_corners[2], 7.0, color, filled=True)


def _draw_placement(context: bpy.types.Context, shader, settings) -> None:
    if settings.origin_is_set:
        origin = scene.image_to_region(
            context,
            settings.origin_image[0],
            settings.origin_image[1],
        )
        _draw_crosshair(shader, origin, (0.35, 1.0, 0.45, settings.controls_opacity), 7.0)
    if settings.scale_point_count >= 1:
        point_a = scene.image_to_region(
            context,
            settings.scale_point_a[0],
            settings.scale_point_a[1],
        )
        _draw_circle(shader, point_a, 6.0, (1.0, 1.0, 1.0, settings.controls_opacity), filled=True)
        if settings.scale_point_count >= 2:
            point_b = scene.image_to_region(
                context,
                settings.scale_point_b[0],
                settings.scale_point_b[1],
            )
            _draw_circle(shader, point_b, 6.0, (1.0, 1.0, 1.0, settings.controls_opacity), filled=True)
            _draw_line(shader, point_a, point_b, (1.0, 1.0, 1.0, 0.9), 1.5)


def _draw_preview(context: bpy.types.Context, shader, settings) -> None:
    if (
        context.area is None
        or context.area.as_pointer() != _preview["area_pointer"]
        or _preview["start"] is None
        or _preview["end"] is None
    ):
        return
    start = _preview["start"]
    end = _preview["end"]
    if _preview["kind"] == "LINE":
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
            shader,
            ideal[0],
            ideal[1],
            AXIS_COLORS[settings.active_axis],
            2.0,
        )
    elif _preview["kind"] == "SURFACE":
        calibration = scene.calibration_from_settings(settings)
        ideal_lines = core.undistort_line_bundles(
            scene.line_bundles_from_settings(settings),
            calibration.intrinsics,
            calibration.division_lambda,
        )
        vanishing_points = core.collect_vanishing_points(ideal_lines)
        resolved = core.surface_vanishing_points(settings.active_surface_plane, vanishing_points)
        if resolved is None:
            return
        diagonal = core.undistort_points(
            np.asarray([start, end], dtype=np.float64),
            calibration.intrinsics.fx,
            calibration.intrinsics.fy,
            calibration.intrinsics.cx,
            calibration.intrinsics.cy,
            calibration.division_lambda,
        )
        ideal_corners = core.perspective_rectangle_corners(
            tuple(diagonal[0]),
            tuple(diagonal[1]),
            resolved[0],
            resolved[1],
        )
        if ideal_corners is None:
            return
        points = [
            scene.ideal_to_region(context, float(point[0]), float(point[1]))
            for point in ideal_corners
        ]
        if any(point is None for point in points):
            return
        color = SURFACE_COLORS[settings.active_surface_plane]
        _draw_polygon(shader, points, _with_alpha(color, 0.16))
        for index in range(4):
            _draw_ideal_segment(
                context,
                shader,
                ideal_corners[index],
                ideal_corners[(index + 1) % 4],
                color,
                1.8,
            )


def _draw_callback() -> None:
    context = bpy.context
    if (
        context.area is None
        or context.area.type != "VIEW_3D"
        or not hasattr(context.scene, "match_perspective")
    ):
        return
    settings = context.scene.match_perspective
    if not settings.is_enabled or settings.image is None or not scene.is_camera_view(context):
        return
    shader = _shader()
    gpu.state.blend_set("ALPHA")
    try:
        _draw_vp_geometry(context, shader, settings)
        _draw_surfaces(context, shader, settings)
        _draw_placement(context, shader, settings)
        _draw_preview(context, shader, settings)
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
