"""Observable overall rotation relative to supplied world directions."""

import numpy as np

from .focal_lines import line_frame
from .sync.projection import _rodrigues


ORIENTATION_INDEPENDENCE_SINE = 1e-6


def orientation_basis(point_constraints, line_constraints, points, geometry):
    """Release locally measured rotations; keep unobserved frame choices fixed."""
    rows = []
    for normal, members in point_constraints.axis_groups:
        rows.extend(np.cross(points[member] - points[members[0]], normal)
                    for member in members[1:])
    if point_constraints.mirror_enabled:
        rows.extend(line_frame(point_constraints.mirror_normal))
    for normal, members, lines in line_constraints.groups:
        if normal is None:
            continue
        origin = np.mean(points[members], axis=0)
        for line in lines:
            position, direction = geometry[line]
            rows.extend((np.cross(position - origin, normal), np.cross(direction, normal)))
    for _line, other in line_constraints.pairs:
        if isinstance(other, np.ndarray):
            rows.extend(line_frame(other))
    # Two equal-coordinate points provide only one rotational condition. Count
    # their actual independent differences, not two freedoms per axis label.
    rows = [row / np.linalg.norm(row) for row in rows if np.linalg.norm(row) > 1e-12]
    if not rows:
        return np.empty((0, 3))
    _u, singular, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.count_nonzero(singular > ORIENTATION_INDEPENDENCE_SINE * singular[0]))
    return np.eye(3) if rank == 3 else vh[:rank]


def overall_rotation(parameters, basis):
    """Map internal anchor-camera coordinates into the constrained frame."""
    return _rodrigues(parameters @ basis) if len(basis) else np.eye(3)
