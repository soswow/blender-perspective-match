"""Point-supported line planes and unoriented parallelism for focal fitting."""

from dataclasses import dataclass

import numpy as np

from .sync.planes import fit_free_plane, grouped_plane_members, normalize_plane_groups
from .sync.constants import WORLD_AXIS_DIRECTIONS


LINE_RELATION_DIRECTION_RESIDUAL_PX = 200.0
LINE_RELATION_DIRECTION_HARD_SINE = 0.01
LINE_HARD_PLANE_MIN_TANGENT = 1.0e-8


def validate_line_relations(point_ids, line_ids, plane_groups, parallel_pairs, *,
                            known_line_ids=()):
    """Validate supported references before registration; never discard a relation."""
    point_set, line_set = set(point_ids), set(line_ids)
    known_line_set = set(known_line_ids)
    if not known_line_set <= line_set:
        raise ValueError("Known 3D line support contains an unsupported landmark")
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
        fixed_lines = [key for key in members if key in known_line_set]
        minimum = 3 if axis == "FREE" else 1
        if len(points) + len(fixed_lines) < minimum:
            raise ValueError(
                f"Line Is in Plane {axis} #{bucket} needs "
                f"{'three non-collinear fixed members' if axis == 'FREE' else 'a fixed member'} "
                "in the same group (a fitted point or Known 3D line)")
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
    support_line_indices: list[list[int]]
    support_line_points: dict[int, np.ndarray]
    pairs: list[tuple[int, int | np.ndarray]]
    plane_spring: float
    baseline_world: float
    hard_plane: bool
    derived_endpoints: dict[int, tuple[int, int]]

    @classmethod
    def from_inputs(cls, point_ids, line_ids, *, plane_groups, parallel_pairs,
                    anchor_rotation, plane_spring, baseline_world, hard_plane,
                    known_line_ids=(), known_line_positions=None):
        return cls.from_inputs_with_derived(
            point_ids, line_ids, plane_groups=plane_groups,
            parallel_pairs=parallel_pairs, anchor_rotation=anchor_rotation,
            plane_spring=plane_spring, baseline_world=baseline_world,
            hard_plane=hard_plane, known_line_ids=known_line_ids,
            known_line_positions=known_line_positions, derived_lines=())

    @classmethod
    def from_inputs_with_derived(cls, point_ids, line_ids, *, plane_groups,
                    parallel_pairs, anchor_rotation, plane_spring,
                    baseline_world, hard_plane, known_line_ids=(),
                    known_line_positions=None, derived_lines=()):
        known = set(known_line_ids)
        groups, pairs = validate_line_relations(
            point_ids, line_ids, plane_groups, parallel_pairs,
            known_line_ids=known)
        points = {key: i for i, key in enumerate(point_ids)}
        lines = {key: i for i, key in enumerate(line_ids)}
        axes = {axis: anchor_rotation @ np.eye(3)[i] for i, axis in enumerate("XYZ")}
        derived = {line_id: (first, second)
                   for line_id, first, second in derived_lines}
        effective_edges = [
            [key for key in edges
             if (not hard_plane or key not in derived
                 or not set(derived[key]) <= set(members))]
            for _axis, members, edges in groups
        ]
        return cls([
            (axes.get(axis), [points[key] for key in members],
             [lines[key] for key in filtered])
            for (axis, members, _edges), filtered in zip(groups, effective_edges)],
            [[lines[key] for key in filtered if key in known]
             for filtered in effective_edges],
            {lines[key]: np.asarray(value, float) for key, value in
             (known_line_positions or {}).items() if key in lines},
            [(lines[left], lines[right] if right in lines else
              anchor_rotation @ WORLD_AXIS_DIRECTIONS[right]) for left, right in pairs],
            plane_spring, baseline_world, hard_plane,
            {lines[line_id]: (points[first], points[second])
             for line_id, first, second in derived_lines})

    @property
    def active(self):
        return bool(self.groups or self.pairs)

    @property
    def point_indices(self):
        return ({i for _normal, members, _lines in self.groups for i in members} |
                {i for pair in self.derived_endpoints.values() for i in pair})

    def _plane(self, points, geometry, normal, members, support_lines):
        locations = [points[member] for member in members]
        locations.extend(self.support_line_points.get(line, geometry[line][0])
                         for line in support_lines)
        locations = np.asarray(locations, dtype=float)
        if normal is not None:
            return np.mean(locations, axis=0), normal
        plane = fit_free_plane(locations)
        if plane is None:
            raise ValueError("Line Is in Plane Free needs non-collinear fixed members")
        return plane

    def hard_direction_normals(self, points, geometry):
        """Return defining normals for fitted lines in hard plane groups."""
        if not self.hard_plane:
            return {}
        normals = {}
        for (normal, members, lines), support_lines in zip(self.groups, self.support_line_indices):
            _origin, normal = self._plane(points, geometry, normal, members, support_lines)
            for line in lines:
                if line not in support_lines and line not in self.derived_endpoints:
                    normals[line] = normal
        return normals

    def constrain_hard_directions(self, points, geometry):
        """Use an in-plane infinite line throughout fitting and result emission."""
        normals = self.hard_direction_normals(points, geometry)
        if not normals:
            return geometry
        constrained = list(geometry)
        for line, normal in normals.items():
            position, direction = geometry[line]
            tangent = direction - normal * float(normal @ direction)
            length = float(np.linalg.norm(tangent))
            if length < LINE_HARD_PLANE_MIN_TANGENT:
                raise ValueError("Line direction is normal to its hard plane")
            constrained[line] = (position, tangent / length)
        return constrained

    def residual(self, points, geometry):
        """Plane position/direction and parallel direction rows, not extra picks."""
        rows = []
        for (normal, members, lines), support_lines in zip(self.groups, self.support_line_indices):
            origin, normal = self._plane(points, geometry, normal, members, support_lines)
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
        for (normal, members, lines), support_lines in zip(self.groups, self.support_line_indices):
            origin, normal = self._plane(points, geometry, normal, members, support_lines)
            for line in lines:
                position, direction = geometry[line]
                if self.hard_plane:
                    distance = max(distance, self.baseline_world * abs(float(normal @ (position - origin))))
                sine = max(sine, abs(float(normal @ direction)))
        for left, right in self.pairs:
            other_direction = geometry[right][1] if isinstance(right, int) else right
            sine = max(sine, float(np.linalg.norm(np.cross(geometry[left][1], other_direction))))
        return distance, sine
