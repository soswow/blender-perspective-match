"""Multi-match landmark sync: register private worlds into an anchor frame.

Each match keeps its VP solve in a private world. Sync finds a rigid Empty
transform ``X_shared = R X_private + t`` (scale 1) per non-anchor match, and
falls back to a similarity with free scale when a rigid pose cannot lock.

Pipeline (keep AGENTS.md in sync if this changes): register pairwise
(strongest-pair seed, then easiest-next camera; Fit Only stills skipped)
with shared-ground scale from registered cameras allowed to contribute 3D
→ peel cameras above ``ACCEPT_RMSE_PX`` → joint BA (Fit Only 2D pulls pose,
not 3D) → peel again → resect skipped and Fit Only stills against the
frozen 3D (ground tags if off-plane picks disagree; frozen Is Mirror Of
lines mixed like Known 3D) → triangulate landmarks now visible in recovered
views that may move 3D and PnP stills that had no cloud support → pose-only
BA of recovered cameras → thaw 3D from recovered stills that may move 3D
→ rebuild free 3D lines from those cameras → report.

Package layout: ``constants``, ``types``, ``projection``, ``pose``, ``ground``,
``lines``, ``mirrors``, ``planes``, ``ba``, ``solve``, ``request``. ``from match_perspective.core import sync``
still exposes the same names as the former single module, including test helpers.
"""

from __future__ import annotations

from . import ba, constants, ground, lines, mirrors, planes, pose, projection, solve, types

for _module in (
    constants,
    types,
    projection,
    ba,
    lines,
    mirrors,
    planes,
    ground,
    pose,
    solve,
):
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value

del _module, _name, _value

from .request import SyncSolveRequest
