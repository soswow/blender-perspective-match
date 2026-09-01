"""N-panel tab checks shared by overlay and operators (no bpy)."""

PERSPECTIVE_MATCH_CATEGORY = "Perspective Match"


def shows_perspective_match_tab(
    show_region_ui: bool,
    ui_width: int,
    category: str | None,
) -> bool:
    """True when the 3D View N-panel is open on the Perspective Match tab."""
    return (
        bool(show_region_ui)
        and ui_width > 1
        and category == PERSPECTIVE_MATCH_CATEGORY
    )
