"""Shared cameras and observations for landmark-graph sync tests."""

from __future__ import annotations

import numpy as np

from match_perspective import core
from match_perspective.core import sync

def _look_at_rotation(camera_center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build an OpenCV world-to-camera rotation looking toward target (Z forward)."""
    forward = target - camera_center
    forward = forward / np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.9:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def _rodrigues_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _project(point, calibration: core.Calibration) -> tuple[float, float]:
    projected = sync.project_private_point(point, calibration)
    assert projected is not None
    return float(projected[0]), float(projected[1])


def _synthetic_scene(*, with_ground: bool, yaw: float = 0.35) -> tuple:
    """Build two calibrated matches + landmark observations for sync tests."""
    intrinsics = core.CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=400.0,
        cy=300.0,
        image_width=800,
        image_height=600,
    )
    true_landmarks = {
        "p0": np.array((0.0, 0.0, 0.0), dtype=np.float64),
        "p1": np.array((2.0, 0.0, 0.0), dtype=np.float64),
        "p2": np.array((0.0, 2.5, 0.0), dtype=np.float64),
        "p3": np.array((1.5, 1.0, 0.0), dtype=np.float64),
        "p4": np.array((1.0, 0.5, 2.0), dtype=np.float64),
        "p5": np.array((-0.5, 1.2, 1.5), dtype=np.float64),
        "p6": np.array((0.8, -0.4, 0.9), dtype=np.float64),
    }
    ground_ids = {"p0", "p1", "p2", "p3"} if with_ground else set()

    true_sim = sync.SimilarityTransform(
        scale=1.0,
        rotation=_rodrigues_z(yaw),
        translation=np.array((3.0, -2.0, 0.5), dtype=np.float64),
    )

    anchor_center = np.array((-3.0, -4.0, 2.0), dtype=np.float64)
    anchor_rotation = _look_at_rotation(anchor_center, np.array((0.5, 0.5, 0.0)))
    anchor_calibration = core.Calibration(
        intrinsics=intrinsics,
        rotation_w2c=anchor_rotation,
        camera_center=anchor_center,
    )

    shared_center = np.array((4.0, -3.0, 2.5), dtype=np.float64)
    shared_rotation = _look_at_rotation(shared_center, np.array((0.5, 0.5, 0.5)))
    rotation_private = shared_rotation @ true_sim.rotation
    center_private = true_sim.rotation.T @ (shared_center - true_sim.translation)
    other_calibration = core.Calibration(
        intrinsics=intrinsics,
        rotation_w2c=rotation_private,
        camera_center=center_private,
    )
    private_landmarks = {
        key: true_sim.inverse_point(point) for key, point in true_landmarks.items()
    }

    matches = [
        sync.SyncMatchInput("anchor", anchor_calibration),
        sync.SyncMatchInput("other", other_calibration),
    ]
    observations = []
    for landmark_id, shared_point in true_landmarks.items():
        observations.append(
            sync.SyncObservation(
                "anchor",
                landmark_id,
                *_project(shared_point, anchor_calibration),
                on_ground=landmark_id in ground_ids,
            )
        )
        observations.append(
            sync.SyncObservation(
                "other",
                landmark_id,
                *_project(private_landmarks[landmark_id], other_calibration),
                on_ground=landmark_id in ground_ids,
            )
        )
    return matches, observations, true_sim, center_private, shared_center

