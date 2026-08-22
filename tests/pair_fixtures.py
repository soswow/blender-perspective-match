"""Build sync scenes from a true camera; solver sees stored K/pose.

Picks are always projected with the true calibration. Stored intrinsics come
from ``remap_intrinsics_to_size`` (what Import YAML / copy-K would write).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from match_perspective import core
from match_perspective.core import ros_camera_info, sync
from sync_fixtures import _look_at_rotation, _project

# Portrait calibration plate (same shape as a typical imported YAML).
YAML_WIDTH = 3000
YAML_HEIGHT = 4000
YAML_FX = 2866.0
YAML_FY = 2868.0
YAML_CX = 1497.0
YAML_CY = 2021.0

GROUND_POINTS = (
    np.array((-1.5, -1.2, 0.0)),
    np.array((1.6, -1.0, 0.0)),
    np.array((1.8, 1.4, 0.0)),
    np.array((-1.3, 1.6, 0.0)),
    np.array((0.1, -0.2, 0.0)),
    np.array((0.8, 0.6, 0.0)),
    np.array((-0.4, 0.3, 0.0)),
)
RAISED_POINTS = (
    np.array((-0.6, -0.5, 0.68)),
    np.array((0.5, -0.4, 0.68)),
    np.array((0.4, 0.5, 0.68)),
    np.array((-0.5, 0.4, 0.68)),
    np.array((0.0, 0.0, 0.68)),
    np.array((0.7, 0.4, 0.68)),
)


def k_for_plate(width: int, height: int) -> core.CameraIntrinsics:
    """Intrinsics the UI would store after remapping the portrait YAML."""
    fx, fy, cx, cy, _kind = ros_camera_info.remap_intrinsics_to_size(
        YAML_FX,
        YAML_FY,
        YAML_CX,
        YAML_CY,
        YAML_WIDTH,
        YAML_HEIGHT,
        width,
        height,
    )
    return core.CameraIntrinsics(fx, fy, cx, cy, width, height)


def look_at_calibration(
    center: np.ndarray,
    target: np.ndarray,
    intrinsics: core.CameraIntrinsics,
) -> core.Calibration:
    return core.Calibration(
        intrinsics,
        _look_at_rotation(center, target),
        np.asarray(center, dtype=np.float64),
    )


@dataclass
class ViewSpec:
    """One still: plate size, true pose, optional leftover pose for the solver."""

    match_id: str
    width: int
    height: int
    center: np.ndarray
    target: np.ndarray
    leftover_center: np.ndarray | None = None
    leftover_target: np.ndarray | None = None


def build_views(
    views: list[ViewSpec],
    *,
    include_raised: bool = True,
    nadir_ground: bool = True,
    nadir_raised: bool = True,
    nadir_id: str = "nadir",
) -> tuple[list[sync.SyncMatchInput], list[sync.SyncObservation], dict[str, core.Calibration]]:
    """Return solver matches (stored pose/K), observations (true projections), true cals."""
    true_by_id: dict[str, core.Calibration] = {}
    matches: list[sync.SyncMatchInput] = []
    for view in views:
        intrinsics = k_for_plate(view.width, view.height)
        true_cal = look_at_calibration(view.center, view.target, intrinsics)
        true_by_id[view.match_id] = true_cal
        if view.leftover_center is not None:
            leftover_target = (
                view.leftover_target
                if view.leftover_target is not None
                else np.array((0.5, 0.5, 0.0))
            )
            stored = look_at_calibration(
                view.leftover_center, leftover_target, intrinsics
            )
        else:
            stored = true_cal
        matches.append(sync.SyncMatchInput(view.match_id, stored))

    observations: list[sync.SyncObservation] = []

    def add_point(
        landmark_id: str,
        point: np.ndarray,
        on_ground: bool,
        match_id: str,
    ) -> None:
        observations.append(
            sync.SyncObservation(
                match_id,
                landmark_id,
                *_project(point, true_by_id[match_id]),
                on_ground,
            )
        )

    for index, point in enumerate(GROUND_POINTS):
        for view in views:
            if view.match_id == nadir_id and not nadir_ground:
                continue
            add_point(f"g{index}", point, True, view.match_id)
    if include_raised:
        for index, point in enumerate(RAISED_POINTS):
            for view in views:
                if view.match_id == nadir_id and not nadir_raised:
                    continue
                add_point(f"e{index}", point, False, view.match_id)
    return matches, observations, true_by_id


def optical_axis_z(calibration: core.Calibration, similarity: sync.SimilarityTransform) -> float:
    """Shared-world |Z| of the camera +Z axis after the Empty transform."""
    axis_private = calibration.rotation_w2c.T @ np.array((0.0, 0.0, 1.0))
    axis_shared = similarity.rotation @ axis_private
    axis_shared = axis_shared / np.linalg.norm(axis_shared)
    return abs(float(axis_shared[2]))
