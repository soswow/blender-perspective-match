"""Known 3D and On Ground point priors for joint focal fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .sync.constants import (
    GROUND_Z_RESIDUAL_PX, KNOWN_3D_RESIDUAL_PX,
)


ROTATION_ROUNDOFF_ATOL = 4.0 * np.finfo(np.float32).eps


@dataclass
class PointReferenceConstraints:
    """World-reference point constraints in an anchor/unit-baseline chart."""

    point_count: int
    anchor_rotation: np.ndarray
    anchor_center: np.ndarray
    baseline_world: float
    known_world: dict[int, np.ndarray]
    ground_indices: tuple[int, ...]
    known_3d_slack: float
    ground_slack: float
    hard_xyz: dict[int, np.ndarray]
    hard_z: dict[int, tuple[np.ndarray, float]]

    @property
    def active(self) -> bool:
        return bool(self.known_world or self.ground_indices)

    @classmethod
    def from_inputs(
        cls, point_ids: list[str], *, anchor_rotation: np.ndarray,
        anchor_center: np.ndarray, baseline_world: float,
        known_world: Mapping[str, np.ndarray] | None,
        known_3d_slack: float, ground_landmark_ids: list[str] | None,
        ground_slack: float,
    ) -> "PointReferenceConstraints":
        """Compile complete references; reject malformed or missing point IDs."""
        index = {key: value for value, key in enumerate(point_ids)}
        if (len(index) != len(point_ids) or
                any(not isinstance(key, str) or not key for key in point_ids)):
            raise ValueError("Point IDs must be distinct nonempty strings")
        rotation = np.asarray(anchor_rotation, dtype=float).reshape(3, 3).copy()
        center = np.asarray(anchor_center, dtype=float).reshape(3).copy()
        baseline = float(baseline_world)
        known_slack = float(known_3d_slack)
        ground_slack = float(ground_slack)
        if (not np.isfinite(rotation).all() or not np.isfinite(center).all() or
                not np.isfinite((baseline, known_slack, ground_slack)).all() or
                baseline <= 0 or min(known_slack, ground_slack) < 0):
            raise ValueError("Point reference frame and slacks must be finite and valid")
        # Blender persists each matrix entry as float32; products of those
        # entries need a float32-sized roundoff allowance after reopening.
        if (not np.allclose(rotation @ rotation.T, np.eye(3),
                            rtol=0.0, atol=ROTATION_ROUNDOFF_ATOL) or
                not np.isclose(np.linalg.det(rotation), 1.0,
                               rtol=0.0, atol=ROTATION_ROUNDOFF_ATOL)):
            raise ValueError("Anchor rotation must be a proper orthonormal matrix")
        known: dict[int, np.ndarray] = {}
        for key, location in (known_world or {}).items():
            if key not in index:
                raise ValueError("Known 3D reference lacks a fitted point")
            world = np.asarray(location, dtype=float).reshape(3).copy()
            if not np.isfinite(world).all():
                raise ValueError("Known 3D location must be finite")
            known[index[key]] = world
        ground_ids = tuple(ground_landmark_ids or ())
        if len(set(ground_ids)) != len(ground_ids):
            raise ValueError("On Ground point IDs must be distinct")
        if any(key not in index for key in ground_ids):
            raise ValueError("On Ground reference lacks a fitted point")
        ground = tuple(index[key] for key in ground_ids)
        if known_slack <= 1e-12 and ground_slack <= 1e-12:
            for point in ground:
                if point in known and abs(float(known[point][2])) > 1e-7:
                    raise ValueError("Hard Known 3D and On Ground references disagree")
        hard_xyz = ({point: rotation @ (world - center) / baseline
                     for point, world in known.items()}
                    if known_slack <= 1e-12 else {})
        # A world-Z plane is generally oblique in the anchor chart.
        normal = rotation[:, 2].copy()
        offset = -float(center[2]) / baseline
        hard_z = ({point: (normal.copy(), offset) for point in ground
                   if point not in hard_xyz}
                  if ground_slack <= 1e-12 else {})
        return cls(
            point_count=len(point_ids), anchor_rotation=rotation,
            anchor_center=center, baseline_world=baseline,
            known_world=known, ground_indices=ground,
            known_3d_slack=known_slack, ground_slack=ground_slack,
            hard_xyz=hard_xyz, hard_z=hard_z,
        )

    def project_hard(self, points: np.ndarray) -> np.ndarray:
        """Return chart points with exact Known 3D and ground pins applied."""
        projected = np.array(points, dtype=float, copy=True).reshape(self.point_count, 3)
        for point, target in self.hard_xyz.items():
            projected[point] = target
        for point, (normal, offset) in self.hard_z.items():
            projected[point] += (offset - float(normal @ projected[point])) * normal
        return projected

    def world_gaps(self, points: np.ndarray) -> tuple[float, float]:
        """Maximum Known 3D distance and ground height, both in world units."""
        chart = np.asarray(points, dtype=float).reshape(self.point_count, 3)
        world = (chart * self.baseline_world) @ self.anchor_rotation + self.anchor_center
        known_gap = max((float(np.linalg.norm(world[point] - target))
                         for point, target in self.known_world.items()), default=0.0)
        ground_gap = max((abs(float(world[point, 2])) for point in self.ground_indices),
                         default=0.0)
        return known_gap, ground_gap

    def residual_and_jacobian(
        self, points: np.ndarray, *, point_offset: int, parameter_count: int,
        jacobian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return soft world-unit springs; hard coordinates have no rows."""
        chart = np.asarray(points, dtype=float).reshape(self.point_count, 3)
        world = (chart * self.baseline_world) @ self.anchor_rotation + self.anchor_center
        residuals: list[float] = []
        rows: list[np.ndarray] = []
        for point in self.ground_indices:
            if point in self.hard_xyz or point in self.hard_z:
                continue
            slack = self.ground_slack
            if point in self.known_world and self.known_3d_slack > 1e-12:
                slack = min(slack, self.known_3d_slack)
            spring = GROUND_Z_RESIDUAL_PX / slack
            residuals.append(spring * float(world[point, 2]))
            if jacobian:
                row = np.zeros(parameter_count)
                start = point_offset + 3 * point
                row[start:start + 3] = spring * self.baseline_world * self.anchor_rotation[:, 2]
                rows.append(row)
        if self.known_3d_slack > 1e-12:
            spring = KNOWN_3D_RESIDUAL_PX / self.known_3d_slack
            ground_set = set(self.ground_indices)
            for point, target in self.known_world.items():
                for axis in ((0, 1) if point in ground_set else (0, 1, 2)):
                    residuals.append(spring * float(world[point, axis] - target[axis]))
                    if jacobian:
                        row = np.zeros(parameter_count)
                        start = point_offset + 3 * point
                        row[start:start + 3] = (
                            spring * self.baseline_world * self.anchor_rotation[:, axis])
                        rows.append(row)
        return (np.asarray(residuals, dtype=float),
                np.asarray(rows, dtype=float).reshape(-1, parameter_count)
                if jacobian else None)
