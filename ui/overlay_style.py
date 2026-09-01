"""Plate overlay stroke colors (no bpy)."""

# Match Blender's default axis gizmo colors (X red, Y green, Z blue).
AXIS_COLORS = {
    "x": (0.96, 0.26, 0.26, 1.0),
    "y": (0.26, 0.50, 0.96, 1.0),
    "z": (0.26, 0.80, 0.30, 1.0),
}

# Plate landmark picks: selected > Known 3D > On Ground > default.
LANDMARK_COLOR_SELECTED = (0.96, 0.22, 0.22, 1.0)
LANDMARK_COLOR_KNOWN = (0.25, 0.7, 0.95, 1.0)
LANDMARK_COLOR_GROUND = (0.92, 0.22, 0.78, 1.0)
LANDMARK_COLOR_DEFAULT = (0.95, 0.65, 0.15, 1.0)
PP_COLOR = (0.55, 0.85, 1.0, 1.0)


def landmark_pick_base_color(
    *,
    is_active: bool,
    has_known_object: bool,
    on_ground: bool,
):
    """Committed pick color. Selected wins over Known 3D / On Ground."""
    if is_active:
        return LANDMARK_COLOR_SELECTED
    if has_known_object:
        return LANDMARK_COLOR_KNOWN
    if on_ground:
        return LANDMARK_COLOR_GROUND
    return LANDMARK_COLOR_DEFAULT


def preview_line_color(kind: str, active_axis: str):
    """Rubber-band color: VP strokes follow the axis; landmark lines match selected."""
    if kind == "LANDMARK_LINE":
        return LANDMARK_COLOR_SELECTED
    return AXIS_COLORS[active_axis]
