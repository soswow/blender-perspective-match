"""Persistent scene properties for the Perspective Match extension."""

from __future__ import annotations

import bpy

AXIS_ITEMS = (
    ("x", "Red — X", "Edges parallel to Blender X"),
    ("z", "Yellow — Y", "Edges parallel to Blender Y"),
    ("y", "Blue — Z Up", "Vertical edges parallel to Blender Z"),
)

SURFACE_PLANE_ITEMS = (
    ("xz", "Floor — XY", "Red and yellow axes on Blender Z = 0"),
    ("yz", "Wall — YZ", "Blue and yellow axes on Blender X = 0"),
    ("yx", "Wall — ZX", "Blue and red axes on Blender Y = 0"),
)


def tag_viewport_redraw(context: bpy.types.Context | None = None) -> None:
    """Request redraw in every 3D View."""
    redraw_context = context or bpy.context
    window_manager = getattr(redraw_context, "window_manager", None)
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _redraw(_self, context: bpy.types.Context) -> None:
    tag_viewport_redraw(context)


class PMLineSegment(bpy.types.PropertyGroup):
    """One stored VP segment in source-image coordinates."""

    item_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    axis: bpy.props.EnumProperty(items=AXIS_ITEMS, default="x")
    x1: bpy.props.FloatProperty(default=0.0)
    y1: bpy.props.FloatProperty(default=0.0)
    x2: bpy.props.FloatProperty(default=0.0)
    y2: bpy.props.FloatProperty(default=0.0)


class PMSurface(bpy.types.PropertyGroup):
    """Perspective rectangle stored by its opposite image-space corners."""

    item_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    plane: bpy.props.EnumProperty(items=SURFACE_PLANE_ITEMS, default="xz")
    x1: bpy.props.FloatProperty(default=0.0)
    y1: bpy.props.FloatProperty(default=0.0)
    x2: bpy.props.FloatProperty(default=0.0)
    y2: bpy.props.FloatProperty(default=0.0)
    divisions: bpy.props.IntProperty(
        name="Grid",
        description="Equal world-space divisions shown in the overlay",
        default=4,
        min=1,
        max=64,
        update=_redraw,
    )
    mesh_object: bpy.props.PointerProperty(type=bpy.types.Object)


class PMSettings(bpy.types.PropertyGroup):
    """Durable calibration state saved with the Blender scene."""

    is_enabled: bpy.props.BoolProperty(
        name="Enable",
        description="Show Perspective Match tools and overlays",
        default=True,
        update=_redraw,
    )
    image: bpy.props.PointerProperty(name="Reference Image", type=bpy.types.Image)
    image_path: bpy.props.StringProperty(name="Image Path", subtype="FILE_PATH")
    project_path: bpy.props.StringProperty(name="Project Path", subtype="FILE_PATH")
    source_session_json: bpy.props.StringProperty(default="", options={"HIDDEN"})
    image_width: bpy.props.IntProperty(default=0, min=0)
    image_height: bpy.props.IntProperty(default=0, min=0)
    source_image_width: bpy.props.IntProperty(default=0, min=0)

    root_object: bpy.props.PointerProperty(type=bpy.types.Object)
    camera_object: bpy.props.PointerProperty(type=bpy.types.Object)
    match_collection: bpy.props.PointerProperty(type=bpy.types.Collection)

    vp_mode: bpy.props.EnumProperty(
        name="Perspective",
        items=(
            ("1", "1 Point", "One depth VP plus verticals; FOV stays manual"),
            ("2", "2 Point", "Two horizontal orthogonal VPs"),
            ("3", "3 Point", "Three finite orthogonal VPs"),
        ),
        default="2",
        update=_redraw,
    )
    active_axis: bpy.props.EnumProperty(
        name="Axis",
        items=AXIS_ITEMS,
        default="x",
        update=_redraw,
    )
    active_surface_plane: bpy.props.EnumProperty(
        name="Plane",
        items=SURFACE_PLANE_ITEMS,
        default="xz",
        update=_redraw,
    )
    work_mode: bpy.props.EnumProperty(
        name="Tool",
        items=(
            ("NONE", "Navigate", "Use normal viewport navigation"),
            ("LINE", "VP Lines", "Draw or edit vanishing-point lines"),
            ("SURFACE", "Surfaces", "Draw or edit perspective surfaces"),
            ("ORIGIN", "Origin", "Pick the world origin on the ground plane"),
            ("SCALE", "Scale", "Pick two ground points with a known distance"),
        ),
        default="NONE",
        update=_redraw,
    )
    is_modal: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    lines: bpy.props.CollectionProperty(type=PMLineSegment)
    selected_line_index: bpy.props.IntProperty(default=-1, options={"HIDDEN"})
    surfaces: bpy.props.CollectionProperty(type=PMSurface)
    selected_surface_index: bpy.props.IntProperty(default=-1, options={"HIDDEN"})

    lock_focal: bpy.props.BoolProperty(
        name="Manual FOV",
        description="Keep horizontal FOV fixed while refining orientation",
        default=False,
    )
    hfov_degrees: bpy.props.FloatProperty(
        name="Horizontal FOV",
        description="Horizontal field of view used by the matched camera",
        default=50.0,
        min=1.0,
        max=179.0,
        precision=2,
    )
    fx: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    fy: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    cx: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    cy: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    rotation_w2c: bpy.props.FloatVectorProperty(
        size=9,
        default=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        options={"HIDDEN"},
    )
    camera_center: bpy.props.FloatVectorProperty(
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        options={"HIDDEN"},
    )
    division_lambda: bpy.props.FloatProperty(
        name="Division λ",
        description="Fitzgibbon one-parameter radial distortion estimate",
        default=0.0,
        precision=5,
    )
    estimate_distortion: bpy.props.BoolProperty(
        name="Estimate Distortion",
        description="Estimate division λ when an axis has at least three segments",
        default=False,
    )
    lambda_saturated: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    origin_is_set: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    origin_image: bpy.props.FloatVectorProperty(size=2, default=(0.0, 0.0))
    scale_point_count: bpy.props.IntProperty(default=0, min=0, max=2, options={"HIDDEN"})
    scale_point_a: bpy.props.FloatVectorProperty(size=2, default=(0.0, 0.0))
    scale_point_b: bpy.props.FloatVectorProperty(size=2, default=(0.0, 0.0))
    measured_length: bpy.props.FloatProperty(
        name="Known Length",
        description="World-space distance between the two scale points",
        default=1.0,
        min=1.0e-6,
        unit="LENGTH",
        precision=4,
    )
    solved_scale: bpy.props.FloatProperty(default=1.0, options={"HIDDEN"})

    show_vp_overlay: bpy.props.BoolProperty(
        name="VP Guides",
        default=True,
        update=_redraw,
    )
    show_surface_overlay: bpy.props.BoolProperty(
        name="Surfaces",
        default=True,
        update=_redraw,
    )
    overlay_opacity: bpy.props.FloatProperty(
        name="Opacity",
        default=0.9,
        min=0.05,
        max=1.0,
        subtype="FACTOR",
        update=_redraw,
    )
    controls_opacity: bpy.props.FloatProperty(
        name="Handle Opacity",
        default=1.0,
        min=0.05,
        max=1.0,
        subtype="FACTOR",
        update=_redraw,
    )

    view_undistorted: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    undistorted_image: bpy.props.PointerProperty(type=bpy.types.Image)
    undistorted_path: bpy.props.StringProperty(subtype="FILE_PATH")
    undistorted_offset_x: bpy.props.FloatProperty(default=0.0)
    undistorted_offset_y: bpy.props.FloatProperty(default=0.0)
    undistorted_width: bpy.props.IntProperty(default=0, min=0)
    undistorted_height: bpy.props.IntProperty(default=0, min=0)

    fov_xy: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    fov_zy: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    fov_zx: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    residual_degrees: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    status: bpy.props.StringProperty(default="Load a reference image")
    error: bpy.props.StringProperty(default="")


CLASSES = (
    PMLineSegment,
    PMSurface,
    PMSettings,
)
