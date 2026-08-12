"""Perspective-matching geometry and multi-match solvers.

Public geometry API is re-exported from ``geometry`` so
``from match_perspective import core`` and ``core.LineSegment`` keep working.
Underscored helpers are included so unit tests can reach them.
"""

from __future__ import annotations

from . import geometry as _geometry

for _name, _value in vars(_geometry).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value, _geometry
