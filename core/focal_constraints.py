"""Point-only plane and supplied-mirror residuals for independent FOV fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sync.constants import (
    MIRROR_PAIR_HARD_GAP, MIRROR_PAIR_RESIDUAL_PX, MIRROR_PLANE_RESIDUAL_PX,
    PLANE_AXIS_ALIGNED_MIN, PLANE_FREE_MIN, PLANE_HARD_SLACK, PLANE_RESIDUAL_PX,
)
from .sync.planes import fit_free_plane, grouped_plane_members, normalize_plane_groups
from .sync.mirrors import _dedupe_mirror_pairs


# Allow float32 Blender RNA locations/normals around a truly through-anchor
# plane without adding an almost unobservable metric-scale parameter.
MIRROR_ANCHOR_PLANE_RELATIVE_TOLERANCE = 1e-5


@dataclass
class PointFocalConstraints:
    """Compiled constraints in the fixed anchor-camera/unit-seed-baseline chart."""

    axis_groups: list[tuple[np.ndarray, list[int]]]
    free_groups: list[list[int]]
    mirror_pairs: list[tuple[int, int]]
    mirror_normal: np.ndarray | None
    mirror_distance: float
    baseline_world: float
    plane_spring: float
    mirror_pair_spring: float
    mirror_offset_spring: float
    free_mirror_offset: bool
    free_baseline: bool
    hard_plane: bool

    @property
    def active(self) -> bool:
        return bool(self.axis_groups or self.free_groups or self.mirror_pairs)

    @classmethod
    def from_inputs(
        cls, point_ids: list[str], *, anchor_rotation: np.ndarray,
        anchor_center: np.ndarray, baseline_world: float,
        plane_groups: list[tuple[str, str, int]] | None,
        plane_slack: float, mirror_pairs: list[tuple[str, str]] | None,
        mirror_plane: tuple[np.ndarray, np.ndarray] | None,
        mirror_slack: float,
    ) -> "PointFocalConstraints":
        """Reject unsupported references rather than dropping relation members."""
        index = {key: value for value, key in enumerate(point_ids)}
        if not np.isfinite((plane_slack, mirror_slack)).all() or min(plane_slack, mirror_slack) < 0:
            raise ValueError("Plane and Mirror Slack must be finite and nonnegative")
        groups = normalize_plane_groups(plane_groups)
        if len(groups) != len(plane_groups or ()):
            raise ValueError("Point plane groups contain invalid or duplicate members")
        if any(item[0] not in index for item in groups):
            raise ValueError("Point plane group contains a landmark without two-view picks")
        axis_groups: list[tuple[np.ndarray, list[int]]] = []
        free_groups: list[list[int]] = []
        world_axes = {"X": np.array((1.0, 0.0, 0.0)),
                      "Y": np.array((0.0, 1.0, 0.0)),
                      "Z": np.array((0.0, 0.0, 1.0))}
        for (axis, _bucket), members in grouped_plane_members(groups).items():
            minimum = PLANE_FREE_MIN if axis == "FREE" else PLANE_AXIS_ALIGNED_MIN
            if len(members) < minimum:
                continue  # Matches Sync's inactive-bucket meaning.
            member_index = [index[key] for key in members]
            if axis == "FREE":
                free_groups.append(member_index)
            else:
                axis_groups.append((anchor_rotation @ world_axes[axis], member_index))
        pairs = _dedupe_mirror_pairs(mirror_pairs)
        if len(pairs) != len(mirror_pairs or ()):
            raise ValueError("Point mirror pairs contain invalid or duplicate links")
        if pairs and mirror_plane is None:
            raise ValueError("Point mirror pairs need a supplied Mirror Empty")
        if any(left not in index or right not in index for left, right in pairs):
            raise ValueError("Point mirror pair contains a landmark without two-view picks")
        mirror_normal = None
        mirror_distance = 0.0
        mirror_distance_world = 0.0
        mirror_scale_tolerance = 0.0
        if mirror_plane is not None:
            origin = np.asarray(mirror_plane[0], float).reshape(3)
            normal_world = np.array(mirror_plane[1], dtype=float, copy=True).reshape(3)
            length = float(np.linalg.norm(normal_world))
            if not np.isfinite(origin).all() or not np.isfinite(normal_world).all() or length < 1e-12:
                raise ValueError("Supplied mirror plane is invalid")
            normal_world /= length
            mirror_normal = anchor_rotation @ normal_world
            mirror_distance_world = float(normal_world @ (origin - anchor_center))
            mirror_distance = mirror_distance_world / baseline_world
            mirror_scale_tolerance = MIRROR_ANCHOR_PLANE_RELATIVE_TOLERANCE * max(
                baseline_world, float(np.linalg.norm(origin - anchor_center)), 1.0)
        mirror_enabled = bool(pairs)
        free_baseline = bool(mirror_enabled and
                             abs(mirror_distance_world) > mirror_scale_tolerance)
        hard_plane_slack = plane_slack if plane_slack > 1e-12 else PLANE_HARD_SLACK
        return cls(
            axis_groups=axis_groups, free_groups=free_groups,
            mirror_pairs=[(index[left], index[right]) for left, right in pairs],
            mirror_normal=mirror_normal, mirror_distance=mirror_distance,
            baseline_world=baseline_world,
            plane_spring=PLANE_RESIDUAL_PX * baseline_world / hard_plane_slack,
            mirror_pair_spring=MIRROR_PAIR_RESIDUAL_PX * baseline_world / MIRROR_PAIR_HARD_GAP,
            mirror_offset_spring=(MIRROR_PLANE_RESIDUAL_PX * baseline_world / mirror_slack
                                  if mirror_enabled and mirror_slack > 1e-12 else 0.0),
            free_mirror_offset=bool(mirror_enabled and mirror_slack > 1e-12),
            free_baseline=free_baseline,
            hard_plane=plane_slack <= 1e-12,
        )

    def world_gaps(self, points: np.ndarray, *, mirror_offset: float = 0.0) -> tuple[float, float]:
        """Maximum active hard-plane distance and point-reflection gap in world units."""
        plane_max = 0.0
        if self.hard_plane:
            for normal, members in self.axis_groups:
                coordinates = points[members] @ normal
                plane_max = max(plane_max, self.baseline_world * float(np.ptp(coordinates)))
            for members in self.free_groups:
                local = points[members]
                fitted = fit_free_plane(local)
                if fitted is None:
                    return float("inf"), float("inf")
                centre, normal = fitted
                plane_max = max(plane_max, self.baseline_world *
                                float(np.max(abs((local - centre) @ normal))))
        mirror_max = 0.0
        if self.mirror_pairs:
            normal = self.mirror_normal
            assert normal is not None
            householder = np.eye(3) - 2.0 * np.outer(normal, normal)
            shift = 2.0 * (self.mirror_distance + mirror_offset) * normal
            for left, right in self.mirror_pairs:
                gap = points[right] - (householder @ points[left] + shift)
                mirror_max = max(mirror_max, self.baseline_world * float(np.linalg.norm(gap)))
        return plane_max, mirror_max

    def residual_and_jacobian(
        self, points: np.ndarray, *, point_offset: int, parameter_count: int,
        mirror_offset: float = 0.0, mirror_offset_column: int | None = None,
        jacobian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return world-unit geometric springs, never image measurements."""
        residuals: list[float] = []
        rows: list[np.ndarray] = []
        for normal, members in self.axis_groups:
            reference = members[0]
            for member in members[1:]:
                residuals.append(self.plane_spring * float(normal @ (points[member] - points[reference])))
                if jacobian:
                    row = np.zeros(parameter_count)
                    row[point_offset + 3 * member:point_offset + 3 * member + 3] = self.plane_spring * normal
                    row[point_offset + 3 * reference:point_offset + 3 * reference + 3] = -self.plane_spring * normal
                    rows.append(row)
        for members in self.free_groups:
            local = points[members]
            fitted = fit_free_plane(local)
            if fitted is None:
                raise ValueError("Free plane members are collinear")
            centre, normal = fitted
            values = self.plane_spring * ((local - centre) @ normal)
            residuals.extend(values.tolist())
            if jacobian:
                block = np.zeros((len(members), parameter_count))
                # The fitted normal and centre depend on every member. The
                # group is small; local finite differences include that motion.
                for local_index, member in enumerate(members):
                    for axis in range(3):
                        step = 1e-6 * max(1.0, abs(local[local_index, axis]))
                        shifted = local.copy()
                        shifted[local_index, axis] += step
                        trial = fit_free_plane(shifted)
                        if trial is None:
                            raise ValueError("Free plane derivative is degenerate")
                        trial_centre, trial_normal = trial
                        if trial_normal @ normal < 0:
                            trial_normal = -trial_normal
                        changed = self.plane_spring * ((shifted - trial_centre) @ trial_normal)
                        block[:, point_offset + 3 * member + axis] = (changed - values) / step
                rows.extend(block)
        if self.mirror_pairs:
            normal = self.mirror_normal
            assert normal is not None
            householder = np.eye(3) - 2.0 * np.outer(normal, normal)
            reflected_offset = 2.0 * (self.mirror_distance + mirror_offset) * normal
            for left, right in self.mirror_pairs:
                gap = points[right] - (householder @ points[left] + reflected_offset)
                residuals.extend((self.mirror_pair_spring * gap).tolist())
                if jacobian:
                    block = np.zeros((3, parameter_count))
                    block[:, point_offset + 3 * left:point_offset + 3 * left + 3] = -self.mirror_pair_spring * householder
                    block[:, point_offset + 3 * right:point_offset + 3 * right + 3] = self.mirror_pair_spring * np.eye(3)
                    if mirror_offset_column is not None:
                        block[:, mirror_offset_column] = -2.0 * self.mirror_pair_spring * normal
                    rows.extend(block)
            if self.free_mirror_offset:
                residuals.append(self.mirror_offset_spring * mirror_offset)
                if jacobian:
                    row = np.zeros(parameter_count)
                    assert mirror_offset_column is not None
                    row[mirror_offset_column] = self.mirror_offset_spring
                    rows.append(row)
        residual = np.asarray(residuals, float)
        return residual, (np.asarray(rows, float).reshape(-1, parameter_count)
                          if jacobian else None)
