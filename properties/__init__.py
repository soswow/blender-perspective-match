"""Persistent properties for multi-match Perspective Match sessions."""

from __future__ import annotations

import bpy

AXIS_ITEMS = (
    ("x", "X (Red)", "Edges parallel to Blender X"),
    ("z", "Y (Green)", "Edges parallel to Blender Y"),
    ("y", "Z (Blue)", "Vertical edges parallel to Blender Z"),
)

# Blender requires dynamic enum strings to remain referenced. Cache each count
# combination so active_axis can keep its original expanded-list UI safely.
_COUNTED_AXIS_ITEMS: dict[tuple[int, int, int], tuple] = {}


def _counted_axis_items(self, _context):
    counts = {"x": 0, "y": 0, "z": 0}
    for line in self.lines:
        counts[line.axis] += 1
    key = (counts["x"], counts["z"], counts["y"])
    if key not in _COUNTED_AXIS_ITEMS:
        _COUNTED_AXIS_ITEMS[key] = tuple(
            (
                identifier,
                f"{label} · {counts[identifier]}",
                description,
                index,
            )
            for index, (identifier, label, description) in enumerate(AXIS_ITEMS)
        )
    return _COUNTED_AXIS_ITEMS[key]


# Per-pick sync weights: High pulls harder; Low may drift in the solve.
LANDMARK_CONFIDENCE_ITEMS = (
    ("HIGH", "High", "Strong constraint — clear, precise pick"),
    ("NORMAL", "Normal", "Default constraint weight"),
    ("LOW", "Low", "Soft constraint — uncertain pick; landmark may drift"),
)

LANDMARK_KIND_ITEMS = (
    ("POINT", "Point", "Correspond a single feature point across stills"),
    ("LINE", "Line", "Correspond the same 3D edge as a 2D segment in each still"),
)

WORLD_AXIS_PARALLEL_ITEMS = (
    ("WORLD_AXIS_X", "X Axis", "Parallel to the shared-world X axis"),
    ("WORLD_AXIS_Y", "Y Axis", "Parallel to the shared-world Y axis"),
    ("WORLD_AXIS_Z", "Z Axis", "Parallel to the shared-world Z axis"),
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


def _update_vp_detect_sensitivity(self, context: bpy.types.Context) -> None:
    """Sensitivity changed — drop the cached edge plate so Debug/Detect re-run."""
    import bpy as _bpy

    cached = self.vp_detect_debug_image
    was_debug = bool(self.view_vp_detect_debug)
    self.view_vp_detect_debug = False
    self.vp_detect_debug_image = None
    self.vp_detect_debug_path = ""
    # Force Debug to re-scan even if a stale Image datablock lingered.
    self.vp_detect_sensitivity_baked = -1.0
    if cached is not None:
        try:
            if cached.users == 0:
                _bpy.data.images.remove(cached)
        except ReferenceError:
            pass
    if was_debug:
        try:
            from .. import scene

            scene.refresh_background_projection(context)
        except Exception:
            pass
    self.status = "Edge sensitivity changed — run Detect or Debug again"
    tag_viewport_redraw(context)


def _redraw(_self, context: bpy.types.Context) -> None:
    tag_viewport_redraw(context)


def _update_xy_vp_polarity(self, context: bpy.types.Context) -> None:
    """Apply a changed horizontal VP sign choice to the stored camera pose."""
    from .. import scene

    scene.apply_xy_vp_polarity(context, self)
    tag_viewport_redraw(context)


def _update_landmark_use_in_sync(_self, context: bpy.types.Context) -> None:
    """Rebuild helpers so disabled landmarks leave PM_Sync_Landmarks."""
    from .. import scene

    scene.sync_landmark_empties(context)
    tag_viewport_redraw(context)


def _update_landmark_empties(_self, context: bpy.types.Context) -> None:
    """Rebuild point Empties / line meshes when visibility or size changes."""
    from .. import scene

    scene.sync_landmark_empties(context)
    tag_viewport_redraw(context)


def _update_hide_origin_empty(self, context: bpy.types.Context) -> None:
    """Hide or show this match Origin Empty (camera and collection stay visible)."""
    from .. import scene

    root = getattr(self, "id_data", None)
    if is_match_root(root):
        scene.apply_origin_empty_hidden(root, bool(self.hide_origin_empty))
    tag_viewport_redraw(context)


def _update_landmark_kind(self, context: bpy.types.Context) -> None:
    """Clear line-only links when switching away from Line."""
    if self.kind != "LINE":
        self.parallel_to = "NONE"
    tag_viewport_redraw(context)


def _parallel_to_items(self, context):
    """Dropdown of world axes and other Lines for the parallel constraint."""
    items = [
        ("NONE", "None", "No parallel direction constraint"),
        *WORLD_AXIS_PARALLEL_ITEMS,
    ]
    if context is None:
        return items
    space = workspace(context)
    for landmark in space.landmarks:
        if landmark.kind != "LINE":
            continue
        if landmark.item_id == self.item_id or not landmark.item_id:
            continue
        items.append(
            (
                landmark.item_id,
                landmark.name or landmark.item_id[:8],
                "Same 3D direction as this line",
            )
        )
    return items


def is_match_root(obj: bpy.types.Object | None) -> bool:
    """Return whether an object is a Perspective Match root Empty."""
    return (
        obj is not None
        and obj.name in bpy.data.objects
        and obj.type == "EMPTY"
        and bool(getattr(obj, "pm_session", None) and obj.pm_session.is_match_root)
    )


def iter_match_roots() -> list[bpy.types.Object]:
    """Return all live match roots.

    Skips empties whose camera is gone without mutating RNA. Poll, draw, and
    EnumProperty items callbacks cannot write ID data; call
    ``reconcile_workspace_refs`` from operators or load handlers to drop flags.
    """
    roots = []
    for obj in bpy.data.objects:
        if not hasattr(obj, "pm_session"):
            continue
        session = obj.pm_session
        if not session.is_match_root:
            continue
        camera = session.camera_object
        if camera is None or camera.name not in bpy.data.objects:
            continue
        roots.append(obj)
    return sorted(roots, key=lambda item: item.name)


def prune_stale_match_roots() -> None:
    """Clear ``is_match_root`` on empties whose camera was deleted.

    Write-safe only (operators, load_post, timers) — not poll/draw/items.
    """
    for obj in bpy.data.objects:
        if not hasattr(obj, "pm_session"):
            continue
        session = obj.pm_session
        if not session.is_match_root:
            continue
        camera = session.camera_object
        if camera is None or camera.name not in bpy.data.objects:
            session.is_match_root = False


def match_sync_enabled(root: bpy.types.Object | None) -> bool:
    """Whether this match opts into Solve Sync / Diagnose / Refine Lenses."""
    if not is_match_root(root):
        return False
    return bool(getattr(root.pm_session, "sync_enabled", True))


def iter_sync_enabled_roots() -> list[bpy.types.Object]:
    """Match roots that participate in sync solves."""
    return [root for root in iter_match_roots() if match_sync_enabled(root)]


def workspace(context: bpy.types.Context | None = None) -> PMWorkspace:
    """Return the scene-level Perspective Match workspace."""
    blender_context = context or bpy.context
    return blender_context.scene.match_perspective


def ensure_landmark_creation_indices(space: PMWorkspace | None = None) -> None:
    """Assign creation_index to landmarks that lack one (legacy / just-added).

    Walks current collection order so existing projects keep their saved order
    as the restored “original” order when Sort A–Z is off.

    Must not be called from UI draw / UIList.filter_items (Scene ID is frozen).
    Safe from operators and load_post. During early add-on registration Blender
    may expose restricted data; migration is then deferred to load_post.
    """
    target = space
    if target is None:
        scenes = getattr(bpy.data, "scenes", None)
        if scenes is None:
            return
        for scene in scenes:
            if hasattr(scene, "match_perspective"):
                ensure_landmark_creation_indices(scene.match_perspective)
        return
    max_index = -1
    for landmark in target.landmarks:
        if landmark.creation_index >= 0:
            max_index = max(max_index, int(landmark.creation_index))
    next_index = max(max_index + 1, int(target.next_landmark_creation_index))
    for landmark in target.landmarks:
        if landmark.creation_index < 0:
            landmark.creation_index = next_index
            next_index += 1
    target.next_landmark_creation_index = next_index


def active_root(context: bpy.types.Context | None = None) -> bpy.types.Object | None:
    """Return the active match root if it is a live match (read-only)."""
    root = workspace(context).active_root
    return root if is_match_root(root) else None


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
    from .. import scene as scene_module

    items = [("NONE", "(Unloaded)", "No active Perspective Match session")]
    for root in iter_match_roots():
        # Show PM_<label> in the UI; identifier stays the Origin Empty name.
        items.append(
            (root.name, scene_module.match_prefix(root), "Activate this match camera")
        )
    return items


def _update_active_match(self, context) -> None:
    """Switch the active session from the dropdown and enter that camera view."""
    if _syncing_active_match:
        return
    from .. import scene as scene_module

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


# Suppress recursive EnumProperty updates while syncing the anchor dropdown.
_syncing_anchor_match = False


def sync_anchor_match_enum(space: PMWorkspace, identifier: str) -> None:
    """Set the sync-anchor dropdown without re-entering its update callback."""
    global _syncing_anchor_match
    if space.anchor_match == identifier:
        return
    _syncing_anchor_match = True
    try:
        space.anchor_match = identifier
    finally:
        _syncing_anchor_match = False


def _anchor_match_items(self, context):
    """Dynamic enum entries for the sync anchor dropdown."""
    from .. import scene as scene_module

    items = [("NONE", "(None)", "No sync anchor selected")]
    for root in iter_match_roots():
        items.append(
            (
                root.name,
                scene_module.match_prefix(root),
                "Use this match as the shared-world anchor",
            )
        )
    return items


def _update_anchor_match(self, context) -> None:
    """Keep the anchor pointer aligned with the dropdown selection."""
    if _syncing_anchor_match:
        return
    name = self.anchor_match
    if name in {"", "NONE"}:
        self.anchor_root = None
        return
    root = bpy.data.objects.get(name)
    if is_match_root(root):
        self.anchor_root = root
    else:
        self.anchor_root = None
        sync_anchor_match_enum(self, "NONE")
    tag_viewport_redraw(context)


def anchor_root(context: bpy.types.Context | None = None) -> bpy.types.Object | None:
    """Return the sync anchor root if it is a live match (read-only).

    PointerProperty is the source of truth. The Anchor dropdown can drift
    (dynamic enums store an index) — do not repair it here; poll/draw freeze
    Scene RNA. Call ``reconcile_workspace_refs`` from a write-safe context.
    """
    root = workspace(context).anchor_root
    return root if is_match_root(root) else None


def reconcile_workspace_refs(space: PMWorkspace) -> None:
    """Drop stale match flags and align active/anchor dropdowns with pointers.

    Must not run from poll, draw, or EnumProperty items callbacks.
    """
    prune_stale_match_roots()
    root = space.active_root
    if not is_match_root(root):
        if space.active_root is not None:
            space.active_root = None
        sync_active_match_enum(space, "NONE")
    else:
        sync_active_match_enum(space, root.name)
    root = space.anchor_root
    if not is_match_root(root):
        if space.anchor_root is not None:
            space.anchor_root = None
        sync_anchor_match_enum(space, "NONE")
    else:
        sync_anchor_match_enum(space, root.name)


class PMLineSegment(bpy.types.PropertyGroup):
    """One stored VP segment in source-image coordinates."""

    item_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    axis: bpy.props.EnumProperty(items=AXIS_ITEMS, default="x")
    x1: bpy.props.FloatProperty(default=0.0)
    y1: bpy.props.FloatProperty(default=0.0)
    x2: bpy.props.FloatProperty(default=0.0)
    y2: bpy.props.FloatProperty(default=0.0)


class PMLandmarkObservation(bpy.types.PropertyGroup):
    """One landmark pick inside a single match still."""

    match_root: bpy.props.PointerProperty(
        name="Match",
        type=bpy.types.Object,
    )
    x: bpy.props.FloatProperty(default=0.0)
    y: bpy.props.FloatProperty(default=0.0)
    # Second endpoint for LINE landmarks (ignored for POINT).
    x2: bpy.props.FloatProperty(default=0.0)
    y2: bpy.props.FloatProperty(default=0.0)
    is_set: bpy.props.BoolProperty(default=False)
    confidence: bpy.props.EnumProperty(
        name="Confidence",
        description=(
            "How strongly this pick constrains sync. Low lets the landmark Empty "
            "drift relative to this still"
        ),
        items=LANDMARK_CONFIDENCE_ITEMS,
        default="NORMAL",
        update=_redraw,
    )


class PMLandmark(bpy.types.PropertyGroup):
    """Named 3D landmark with per-match image observations."""

    item_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    # Stable insertion order for UI sort-off; assigned on add / file load migrate.
    creation_index: bpy.props.IntProperty(default=-1, options={"HIDDEN"})
    name: bpy.props.StringProperty(name="Name", default="Landmark")
    # Off = keep picks/overlay but omit from Solve Sync / Diagnose graph.
    use_in_sync: bpy.props.BoolProperty(
        name="Use in Sync",
        description=(
            "Include this landmark in Solve Sync and Diagnose. "
            "Turn off to debug which landmarks break the solve without deleting picks; "
            "also hides its Empty / line mesh from PM_Sync_Landmarks"
        ),
        default=True,
        update=_update_landmark_use_in_sync,
    )
    kind: bpy.props.EnumProperty(
        name="Kind",
        description="Point feature or the same 3D edge drawn as a line in each still",
        items=LANDMARK_KIND_ITEMS,
        default="POINT",
        update=_update_landmark_kind,
    )
    on_ground: bpy.props.BoolProperty(
        name="On Ground",
        description=(
            "Optional: landmark lies on Z=0 in the anchor world. Used only to pin "
            "absolute baseline scale after 2D↔2D relative pose is solved"
        ),
        default=False,
        update=_redraw,
    )
    known_object: bpy.props.PointerProperty(
        name="Known 3D",
        description=(
            "Optional Blender object (Empty, mesh origin, …) whose world location "
            "is a fixed landmark in shared space. For Line landmarks this is one "
            "endpoint of a known edge"
        ),
        type=bpy.types.Object,
        update=_redraw,
    )
    known_object_b: bpy.props.PointerProperty(
        name="Known 3D B",
        description=(
            "Second endpoint Empty/object for a Known 3D line landmark. "
            "With both ends set, the edge is metric in the anchor world; "
            "otherwise draw the line in ≥3 stills"
        ),
        type=bpy.types.Object,
        update=_redraw,
    )
    parallel_to: bpy.props.EnumProperty(
        name="Is Parallel To",
        description=(
            "A shared-world axis or another Line landmark that shares the same "
            "3D direction. Constrains relative orientation during sync"
        ),
        items=_parallel_to_items,
        update=_redraw,
    )
    observations: bpy.props.CollectionProperty(type=PMLandmarkObservation)
    position: bpy.props.FloatVectorProperty(
        name="Position",
        size=3,
        subtype="TRANSLATION",
        default=(0.0, 0.0, 0.0),
    )
    # Second endpoint for LINE landmarks (mesh edge viz after sync).
    position_b: bpy.props.FloatVectorProperty(
        name="Position B",
        size=3,
        subtype="TRANSLATION",
        default=(0.0, 0.0, 0.0),
    )
    has_position: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    has_line_segment: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    rmse_px: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})


class PMSession(bpy.types.PropertyGroup):
    """Durable per-match calibration state stored on the root Empty."""

    is_match_root: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    image: bpy.props.PointerProperty(name="Reference Image", type=bpy.types.Image)
    image_path: bpy.props.StringProperty(
        name="Image Path",
        description="Full path of the reference image",
        subtype="FILE_PATH",
    )
    image_width: bpy.props.IntProperty(default=0, min=0)
    image_height: bpy.props.IntProperty(default=0, min=0)
    source_image_width: bpy.props.IntProperty(default=0, min=0)

    camera_object: bpy.props.PointerProperty(type=bpy.types.Object)
    match_collection: bpy.props.PointerProperty(type=bpy.types.Collection)

    camera_control: bpy.props.EnumProperty(
        name="Camera Control",
        description=(
            "Choose whether Perspective Match or the live Blender camera "
            "owns pose and FOV"
        ),
        items=(
            (
                "MATCHED",
                "Perspective Match",
                "Apply the stored VP/origin camera whenever this match is activated",
            ),
            (
                "ADJUSTED",
                "Adjusted Camera",
                "Keep reading the camera's current transform and FOV; "
                "match activation will not replace them",
            ),
        ),
        default="MATCHED",
        update=_redraw,
    )

    vp_mode: bpy.props.EnumProperty(
        name="Perspective",
        description="Vanishing-point mode for this match",
        items=(
            ("1", "1 Point", "1-point: Y + Z lines; FOV stays manual"),
            ("2", "2 Point", "2-point: X + Y horizontals; Z uprights not used"),
            (
                "3",
                "3 Point",
                "3-point: any two axes (2+ lines each); the third is derived",
            ),
        ),
        default="3",
        update=_redraw,
    )
    active_axis: bpy.props.EnumProperty(
        name="Axis",
        items=_counted_axis_items,
        default=0,
        update=_redraw,
    )
    flip_xy_vp_polarity: bpy.props.BoolProperty(
        name="Flip X / Y Polarity",
        description=(
            "Off treats the red and green vanishing points as +X and +Y; "
            "on treats them as -X and -Y; Blender Z stays unchanged"
        ),
        default=False,
        update=_update_xy_vp_polarity,
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
    lambda_saturated: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    # OpenCV D: k1, k2, p1, p2, k3, k4, k5, k6 (imported; not estimated from VPs).
    brown_conrady: bpy.props.FloatVectorProperty(
        size=8,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        options={"HIDDEN"},
    )

    origin_is_set: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    origin_image: bpy.props.FloatVectorProperty(size=2, default=(0.0, 0.0))

    # Last camera-view zoom/pan for this match (RegionView3D framing).
    view_camera_zoom: bpy.props.FloatProperty(
        default=0.0,
        min=-30.0,
        max=600.0,
        options={"HIDDEN"},
    )
    view_camera_offset: bpy.props.FloatVectorProperty(
        size=2,
        default=(0.0, 0.0),
        options={"HIDDEN"},
    )

    show_vp_overlay: bpy.props.BoolProperty(
        name="VP Guides",
        default=True,
        update=_redraw,
    )
    show_vp_error_labels: bpy.props.BoolProperty(
        name="Show Error Label",
        description=(
            "Draw each VP segment's local direction error against the current "
            "camera's ideal vanishing point (endpoint-equivalent pixels)"
        ),
        default=False,
        update=_redraw,
    )
    snap_vp_lines_to_edges: bpy.props.BoolProperty(
        name="Snap to Edges",
        description=(
            "After releasing a VP line (or endpoint), refine it onto a nearby "
            "image edge or thin dark/bright line along the stroke"
        ),
        default=False,
    )
    vp_detect_sensitivity: bpy.props.FloatProperty(
        name="Edge Sensitivity",
        description=(
            "How eagerly auto edge detection picks faint lines. "
            "Higher finds lower-contrast edges (more noise). "
            "Lower keeps only strong edges. "
            "Changing this clears the debug edge plate"
        ),
        default=0.7,
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
        step=1,
        precision=2,
        subtype="FACTOR",
        update=_update_vp_detect_sensitivity,
    )
    # Sensitivity used to build vp_detect_debug_image (-1 = none / stale).
    vp_detect_sensitivity_baked: bpy.props.FloatProperty(
        default=-1.0,
        options={"HIDDEN"},
    )
    view_vp_detect_debug: bpy.props.BoolProperty(
        name="Debug auto detected edges",
        description=(
            "Show a black debug plate with every auto-detected edge as a white "
            "stroke. First use runs edge detection; later toggles reuse the plate"
        ),
        default=False,
        options={"HIDDEN"},
    )
    vp_detect_debug_image: bpy.props.PointerProperty(
        type=bpy.types.Image,
        options={"HIDDEN"},
    )
    vp_detect_debug_path: bpy.props.StringProperty(
        subtype="FILE_PATH",
        options={"HIDDEN"},
    )
    overlay_opacity: bpy.props.FloatProperty(
        name="Overlay Opacity",
        description="Opacity for VP guides, handles, origin marker, and landmark picks",
        default=0.9,
        min=0.05,
        max=1.0,
        subtype="FACTOR",
        update=_redraw,
    )

    # Display-only lighting; baked into post-processed/*-pm-view.png, never feeds the solver.
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
    # Last values actually baked into view_image (sliders may differ until Apply).
    view_baked_exposure: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    view_baked_contrast: bpy.props.FloatProperty(default=1.0, options={"HIDDEN"})

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
    # Length-weighted RMS of local VP-segment direction misses from the
    # current camera.
    vp_line_rms_px: bpy.props.FloatProperty(default=-1.0, options={"HIDDEN"})
    status: bpy.props.StringProperty(default="Load a reference image")
    error: bpy.props.StringProperty(default="")

    # Sync Empty transform (private → shared); identity when not registered.
    sync_enabled: bpy.props.BoolProperty(
        name="Enable Sync",
        description=(
            "Include this match in Solve Sync, Diagnose, and Refine Lenses. "
            "Turn off to leave it out of the shared-world registration. "
            "Need at least two sync-enabled matched cameras; "
            "points or lines across stills, Known 3D Empties optional"
        ),
        default=True,
        update=_redraw,
    )
    sync_is_applied: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    sync_scale: bpy.props.FloatProperty(default=1.0, options={"HIDDEN"})
    sync_rotation: bpy.props.FloatVectorProperty(
        size=4,
        default=(1.0, 0.0, 0.0, 0.0),
        options={"HIDDEN"},
    )
    sync_translation: bpy.props.FloatVectorProperty(
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        options={"HIDDEN"},
    )
    sync_rmse_px: bpy.props.FloatProperty(default=0.0, options={"HIDDEN"})
    hide_origin_empty: bpy.props.BoolProperty(
        name="Hide Origin Empty",
        description=(
            "Hide this match Origin Empty in the viewport. "
            "The match camera and collection stay visible"
        ),
        default=False,
        update=_update_hide_origin_empty,
    )


class PMWorkspace(bpy.types.PropertyGroup):
    """Scene-level UI controller for the active match session."""

    active_root: bpy.props.PointerProperty(
        name="Active Match Root",
        type=bpy.types.Object,
        options={"HIDDEN"},
    )
    active_match: bpy.props.EnumProperty(
        name="Active Match",
        description=(
            "Perspective Match camera currently being edited. "
            "Create or select a match camera to continue"
        ),
        items=_active_match_items,
        update=_update_active_match,
    )
    anchor_root: bpy.props.PointerProperty(
        name="Sync Anchor Root",
        type=bpy.types.Object,
        options={"HIDDEN"},
    )
    anchor_match: bpy.props.EnumProperty(
        name="Anchor Match",
        description="Match whose private world defines shared scale and axes before sync rotation",
        items=_anchor_match_items,
        update=_update_anchor_match,
    )
    landmarks: bpy.props.CollectionProperty(type=PMLandmark)
    active_landmark_index: bpy.props.IntProperty(default=-1, options={"HIDDEN"})
    next_landmark_creation_index: bpy.props.IntProperty(
        default=0,
        min=0,
        options={"HIDDEN"},
    )
    landmarks_sort_alphabetical: bpy.props.BoolProperty(
        name="Sort A–Z",
        description=(
            "When enabled, list landmarks alphabetically by name. "
            "When disabled, restore original add order"
        ),
        default=False,
        options={"SKIP_SAVE"},
    )
    landmarks_filter_current_match: bpy.props.BoolProperty(
        name="Filter to Current Match",
        description="Show only landmarks with a pick in the active match",
        default=False,
        options={"SKIP_SAVE"},
    )
    show_landmark_labels: bpy.props.BoolProperty(
        name="Landmark Labels",
        description="Show each landmark's name next to its pick on the reference plate",
        default=False,
        update=_redraw,
    )
    landmark_pick_confidence: bpy.props.EnumProperty(
        name="Pick Confidence",
        description="Confidence applied to the next landmark pick in the active match",
        items=LANDMARK_CONFIDENCE_ITEMS,
        default="NORMAL",
        update=_redraw,
    )
    snap_landmark_to_apriltag: bpy.props.BoolProperty(
        name="Snap to AprilTag",
        description=(
            "When picking a point landmark, snap the click onto the centre of a "
            "nearby dark AprilTag-like quadrilateral (too small or blurry to decode)"
        ),
        default=False,
    )
    show_landmark_overlay: bpy.props.BoolProperty(
        name="Landmark Guides",
        description="Show landmark picks and line segments on the reference plate",
        default=True,
        update=_redraw,
    )
    show_landmark_empties: bpy.props.BoolProperty(
        name="Landmark Empties",
        description=(
            "Show solved landmark helpers in the viewport after sync: "
            "Empties for points, single-edge meshes for lines"
        ),
        default=True,
        update=_update_landmark_empties,
    )
    landmark_empty_size: bpy.props.FloatProperty(
        name="Size",
        description="Display size of solved point landmark Empties",
        default=0.25,
        min=0.01,
        soft_max=5.0,
        step=1,
        precision=2,
        update=_update_landmark_empties,
    )
    lens_refine_span_percent: bpy.props.FloatProperty(
        name="Lens Search %",
        description=(
            "Focal search window for Refine Lenses as ± percent of each match's "
            "current fx (18 = search 82%–118% of current focal)"
        ),
        default=18.0,
        min=1.0,
        soft_max=40.0,
        max=80.0,
        step=100,
        precision=0,
    )
    share_lens: bpy.props.BoolProperty(
        name="Same Lens",
        description=(
            "When refining lenses, search one shared focal scale for every "
            "sync-enabled still (same physical camera). Turn off to search "
            "each still independently (mixed cameras or zooms)"
        ),
        default=True,
        update=_redraw,
    )
    ground_slack: bpy.props.FloatProperty(
        name="Ground Slack",
        description=(
            "How far On Ground landmarks may sit off Z=0 (scene units). "
            "0 pins them to the floor raycast. A small value (plank cup / "
            "tag thickness) lets a boarded floor flex without bending cameras"
        ),
        default=0.02,
        min=0.0,
        soft_max=0.25,
        max=2.0,
        step=1,
        precision=3,
        unit="LENGTH",
        update=_redraw,
    )
    lens_refine_progress: bpy.props.FloatProperty(
        name="Refine Progress",
        description="Progress of the running Refine Lenses job",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        options={"SKIP_SAVE"},
    )
    lock_rotation: bpy.props.BoolProperty(
        name="Lock Rotation",
        description=(
            "When solving sync, keep each match Empty's rotation on a 90° "
            "world-axis jump (identity, ±90°, 180°) and only solve translation "
            "and scale. Use when VP axes already match across stills, including "
            "an X/Y swap. With Lock Translation also on, cameras stay put and "
            "only 3D landmark positions are adjusted"
        ),
        default=False,
        update=_redraw,
    )
    lock_translation: bpy.props.BoolProperty(
        name="Lock Translation",
        description=(
            "When solving sync, keep each match Empty's translation fixed "
            "(cameras stay in place) and only solve rotation and scale. "
            "With Lock Rotation also on, cameras stay put and only 3D "
            "landmark positions are adjusted"
        ),
        default=False,
        update=_redraw,
    )
    sync_status: bpy.props.StringProperty(default="")
    work_mode: bpy.props.EnumProperty(
        name="Tool",
        items=(
            ("NONE", "Navigate", "Use normal viewport navigation"),
            ("LINE", "VP Lines", "Draw or edit vanishing-point lines"),
            ("ORIGIN", "Origin", "Pick the world origin on the ground plane"),
            ("PP", "Principal Point", "Drag the principal point on the plate"),
            ("LANDMARK", "Landmark", "Pick the active landmark in the active match"),
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
    PMLandmarkObservation,
    PMLandmark,
    PMSession,
    PMWorkspace,
)
