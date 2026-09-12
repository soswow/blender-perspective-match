"""Point-supported line planes and unoriented parallelism for focal fitting."""

from dataclasses import dataclass

import numpy as np

from .sync.planes import fit_free_plane, grouped_plane_members, normalize_plane_groups
from .sync.constants import WORLD_AXIS_DIRECTIONS


LINE_RELATION_DIRECTION_RESIDUAL_PX = 200.0
LINE_RELATION_DIRECTION_HARD_SINE = 0.01


def validate_line_relations(point_ids, line_ids, plane_groups, parallel_pairs):
    """Validate supported references before registration; never discard a relation."""
    point_set, line_set = set(point_ids), set(line_ids)
    groups = normalize_plane_groups(plane_groups)
    if len(groups) != len(plane_groups or ()):
        raise ValueError("Plane relations contain invalid or duplicate members")
    if any(key not in point_set | line_set for key, _axis, _bucket in groups):
        raise ValueError("Plane relation contains an unsupported landmark")
    supported = []
    for (axis, bucket), members in grouped_plane_members(groups).items():
        lines = [key for key in members if key in line_set]
        if not lines:
            continue
        points = [key for key in members if key in point_set]
        minimum = 3 if axis == "FREE" else 1
        if len(points) < minimum:
            raise ValueError(
                f"Line Is in Plane {axis} #{bucket} needs "
                f"{'three non-collinear point members' if axis == 'FREE' else 'a point member'} "
                "in the same group, with two-view picks")
        supported.append((axis, points, lines))
    pairs, seen = [], set()
    for pair in parallel_pairs or ():
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                any(not isinstance(key, str) or key not in line_set | WORLD_AXIS_DIRECTIONS.keys()
                    for key in pair) or not any(key in line_set for key in pair)):
            raise ValueError("Is Parallel To needs an included line and another line or world axis")
        key = tuple(sorted(pair))
        if pair[0] == pair[1] or key in seen:
            raise ValueError("Is Parallel To contains duplicate or self links")
        seen.add(key)
        pairs.append(tuple(pair) if pair[0] in line_set else tuple(reversed(pair)))
    return supported, pairs


@dataclass
class LineFocalConstraints:
    """Geometric priors in the anchor-camera/unit-baseline coordinate frame."""

    groups: list[tuple[np.ndarray | None, list[int], list[int]]]
    pairs: list[tuple[int, int | np.ndarray]]
    plane_spring: float
    baseline_world: float
    hard_plane: bool

    @classmethod
    def from_inputs(cls, point_ids, line_ids, *, plane_groups, parallel_pairs,
                    anchor_rotation, plane_spring, baseline_world, hard_plane):
        groups, pairs = validate_line_relations(
            point_ids, line_ids, plane_groups, parallel_pairs)
        points = {key: i for i, key in enumerate(point_ids)}
        lines = {key: i for i, key in enumerate(line_ids)}
        axes = {axis: anchor_rotation @ np.eye(3)[i] for i, axis in enumerate("XYZ")}
        return cls([
            (axes.get(axis), [points[key] for key in members], [lines[key] for key in edges])
            for axis, members, edges in groups],
            [(lines[left], lines[right] if right in lines else
              anchor_rotation @ WORLD_AXIS_DIRECTIONS[right]) for left, right in pairs],
            plane_spring, baseline_world, hard_plane)

    @property
    def active(self):
        return bool(self.groups or self.pairs)

    @property
    def point_indices(self):
        return {i for _normal, members, _lines in self.groups for i in members}

    @staticmethod
    def _plane(points, normal, members):
        if normal is not None:
            return np.mean(points[members], axis=0), normal
        plane = fit_free_plane(points[members])
        if plane is None:
            raise ValueError("Line Is in Plane Free needs non-collinear point members")
        return plane

    def residual(self, points, geometry):
        """Plane position/direction and parallel direction rows, not extra picks."""
        rows = []
        for normal, members, lines in self.groups:
            origin, normal = self._plane(points, normal, members)
            for line in lines:
                position, direction = geometry[line]
                # Projection vectors remove the arbitrary sign of an SVD plane
                # normal without adding a plane-orientation parameter.
                rows.extend(self.plane_spring * normal * (normal @ (position - origin)))
                rows.extend(LINE_RELATION_DIRECTION_RESIDUAL_PX * normal * (normal @ direction))
        for left, right in self.pairs:
            other_direction = geometry[right][1] if isinstance(right, int) else right
            rows.extend(LINE_RELATION_DIRECTION_RESIDUAL_PX *
                        np.cross(geometry[left][1], other_direction))
        return np.asarray(rows, float)

    def world_gaps(self, points, geometry):
        """Maximum hard position gap and direction sine for final acceptance."""
        distance, sine = 0.0, 0.0
        for normal, members, lines in self.groups:
            origin, normal = self._plane(points, normal, members)
            for line in lines:
                position, direction = geometry[line]
                if self.hard_plane:
                    distance = max(distance, self.baseline_world * abs(float(normal @ (position - origin))))
                sine = max(sine, abs(float(normal @ direction)))
        for left, right in self.pairs:
            other_direction = geometry[right][1] if isinstance(right, int) else right
            sine = max(sine, float(np.linalg.norm(np.cross(geometry[left][1], other_direction))))
        return distance, sine
