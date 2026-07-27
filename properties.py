"""Persistent properties for multi-match Perspective Match sessions."""

from __future__ import annotations

import bpy

AXIS_ITEMS = (
    ("x", "X (Red)", "Edges parallel to Blender X"),
    ("z", "Y (Green)", "Edges parallel to Blender Y"),
    ("y", "Z (Blue)", "Vertical edges parallel to Blender Z"),
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


def is_match_root(obj: bpy.types.Object | None) -> bool:
    """Return whether an object is a Perspective Match root Empty."""
    return (
        obj is not None
        and obj.name in bpy.data.objects
        and obj.type == "EMPTY"
        and bool(getattr(obj, "pm_session", None) and obj.pm_session.is_match_root)
    )


def iter_match_roots() -> list[bpy.types.Object]:
    """Return all live match roots, pruned of stale flags."""
    roots = []
    for obj in bpy.data.objects:
        if not hasattr(obj, "pm_session"):
            continue
        session = obj.pm_session
        if not session.is_match_root:
            continue
        camera = session.camera_object
        if camera is None or camera.name not in bpy.data.objects:
            # Root survived but camera was deleted — drop the match flag.
            session.is_match_root = False
            continue
        roots.append(obj)
    return sorted(roots, key=lambda item: item.name)


def workspace(context: bpy.types.Context | None = None) -> PMWorkspace:
    """Return the scene-level Perspective Match workspace."""
    blender_context = context or bpy.context
    return blender_context.scene.match_perspective


def active_root(context: bpy.types.Context | None = None) -> bpy.types.Object | None:
    """Return the active match root after pruning invalid pointers."""
    space = workspace(context)
    root = space.active_root
    if not is_match_root(root):
        if space.active_root is not None:
            space.active_root = None
            sync_active_match_enum(space, "NONE")
        return None
    return root


def active_session(context: bpy.types.Context | None = None) -> PMSession | None:
    """Return the durable session for the active match root, if any."""
    root = active_root(context)
    return None if root is None else root.pm_session


# Suppress recursive EnumProperty updates while syncing the dropdown from code.
_syncing_active_match = False


def sync_active_match_enum(space: PMWorkspace, identifier: str) -> None:
    """Set the active-match dropdown without re-entering its update callback.

    Dynamic EnumProperty values are stored as Int IDProperties. Assigning a
    string through ``space["active_match"] = ...`` raises a type error once the
    property has been initialized, so RNA assignment is required.
    """
    global _syncing_active_match
    if space.active_match == identifier:
        return
    _syncing_active_match = True
    try:
        space.active_match = identifier
    finally:
        _syncing_active_match = False


def _active_match_items(self, context):
    """Dynamic enum entries for the active-match dropdown."""
    items = [("NONE", "(Unloaded)", "No active Perspective Match session")]
    for root in iter_match_roots():
        items.append((root.name, root.name, "Activate this match camera"))
    return items


def _update_active_match(self, context) -> None:
    """Switch the active session from the dropdown and enter that camera view."""
    if _syncing_active_match:
        return
    from . import scene as scene_module

    name = self.active_match
    if name in {"", "NONE"}:
        scene_module.unload_match(context)
        return
    root = bpy.data.objects.get(name)
    if is_match_root(root):
        scene_module.set_active_match(context, root)
    else:
        self.active_root = None
        sync_active_match_enum(self, "NONE")
    tag_viewport_redraw(context)


class PMLineSegment(bpy.types.PropertyGroup):
    """One stored VP segment in source-image coordinates."""

    item_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    axis: bpy.props.EnumProperty(items=AXIS_ITEMS, default="x")
    x1: bpy.props.FloatProperty(default=0.0)
    y1: bpy.props.FloatProperty(default=0.0)
    x2: bpy.props.FloatProperty(default=0.0)
    y2: bpy.props.FloatProperty(default=0.0)


class PMSession(bpy.types.PropertyGroup):
    """Durable per-match calibration state stored on the root Empty."""

    is_match_root: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    image: bpy.props.PointerProperty(name="Reference Image", type=bpy.types.Image)
    image_path: bpy.props.StringProperty(name="Image Path", subtype="FILE_PATH")
    project_path: bpy.props.StringProperty(name="Project Path", subtype="FILE_PATH")
    source_session_json: bpy.props.StringProperty(default="", options={"HIDDEN"})
    image_width: bpy.props.IntProperty(default=0, min=0)
    image_height: bpy.props.IntProperty(default=0, min=0)
    source_image_width: bpy.props.IntProperty(default=0, min=0)

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

    lines: bpy.props.CollectionProperty(type=PMLineSegment)
    selected_line_index: bpy.props.IntProperty(default=-1, options={"HIDDEN"})

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

    show_vp_overlay: bpy.props.BoolProperty(
        name="VP Guides",
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

    # Display-only lighting; baked into a sibling *-pm-view.png, never feeds the solver.
    view_exposure: bpy.props.FloatProperty(
        name="Exposure",
        description="Display-only exposure in stops (−2…+2). Apply to bake into the view plate",
        default=0.0,
        min=-2.0,
        max=2.0,
        soft_min=-2.0,
        soft_max=2.0,
        precision=2,
        step=10,
    )
    view_contrast: bpy.props.FloatProperty(
        name="Contrast",
        description="Display-only contrast multiplier. Apply to bake into the view plate",
        default=1.0,
        min=0.5,
        max=2.0,
        soft_min=0.5,
        soft_max=2.0,
        precision=2,
        step=10,
    )
    view_lighting_applied: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    view_image: bpy.props.PointerProperty(type=bpy.types.Image)
    view_path: bpy.props.StringProperty(subtype="FILE_PATH")

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


class PMWorkspace(bpy.types.PropertyGroup):
    """Scene-level UI controller for the active match session."""

    active_root: bpy.props.PointerProperty(
        name="Active Match Root",
        type=bpy.types.Object,
        options={"HIDDEN"},
    )
    active_match: bpy.props.EnumProperty(
        name="Active Match",
        description="Perspective Match camera currently being edited",
        items=_active_match_items,
        update=_update_active_match,
    )
    work_mode: bpy.props.EnumProperty(
        name="Tool",
        items=(
            ("NONE", "Navigate", "Use normal viewport navigation"),
            ("LINE", "VP Lines", "Draw or edit vanishing-point lines"),
            ("ORIGIN", "Origin", "Pick the world origin on the ground plane"),
        ),
        default="NONE",
        options={"SKIP_SAVE"},
        update=_redraw,
    )
    is_modal: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})


# Backward-compatible alias used by older call sites during the refactor.
PMSettings = PMSession

CLASSES = (
    PMLineSegment,
    PMSession,
    PMWorkspace,
)
