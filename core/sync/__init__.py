"""Multi-match landmark sync: register private worlds into an anchor frame.

Each match keeps its VP solve in a private world. Sync finds a rigid Empty
transform ``X_shared = R X_private + t`` (scale 1) per non-anchor match, and
falls back to a similarity with free scale when a rigid pose cannot lock.

Pipeline (keep AGENTS.md in sync if this changes): register pairwise
(strongest-pair seed, then easiest-next camera) → peel cameras above
``ACCEPT_RMSE_PX`` → joint BA → peel again → resect skipped stills against
the frozen 3D (ground tags if off-plane picks disagree; frozen Is Mirror Of
lines mixed like Known 3D) → triangulate landmarks now visible in recovered
views and PnP stills that had no cloud support → pose-only BA of recovered
cameras → report.

Package layout: ``constants``, ``types``, ``projection``, ``pose``, ``ground``,
``lines``, ``mirrors``, ``ba``, ``solve``. ``from match_perspective.core import sync``
still exposes the same names as the former single module, including test helpers.
"""

from __future__ import annotations

from . import ba, constants, ground, lines, mirrors, pose, projection, solve, types

for _module in (
    constants,
    types,
    projection,
    ba,
    lines,
    mirrors,
    ground,
    pose,
    solve,
):
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value

del _module, _name, _value
