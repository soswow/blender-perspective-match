"""Dataclasses and similarity helpers for landmark-graph sync."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import geometry as core


class SyncCancelled(RuntimeError):
    """Raised when a cancellable sync/Diagnose solve is stopped."""


@dataclass
class SimilarityTransform:
    """Maps a match private world into shared (anchor) world: ``s R x + t``."""

    scale: float = 1.0
    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64),
    )
    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )

    def matrix(self) -> np.ndarray:
        """Return a 4×4 homogeneous matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Map a private-world point into shared world."""
        return self.scale * (self.rotation @ point) + self.translation

    def inverse_point(self, point: np.ndarray) -> np.ndarray:
        """Map a shared-world point into private world."""
        scale = max(float(self.scale), 1.0e-12)
        return self.rotation.T @ ((point - self.translation) / scale)


def _compose_similarities(
    outer: SimilarityTransform,
    inner: SimilarityTransform,
) -> SimilarityTransform:
    """Return the transform that applies ``inner`` and then ``outer``."""
    return SimilarityTransform(
        scale=float(outer.scale) * float(inner.scale),
        rotation=outer.rotation @ inner.rotation,
        translation=(
            float(outer.scale) * (outer.rotation @ inner.translation)
            + outer.translation
        ),
    )


def _inverse_similarity(similarity: SimilarityTransform) -> SimilarityTransform:
    """Return the inverse private/shared similarity."""
    scale = max(float(similarity.scale), 1.0e-12)
    rotation = similarity.rotation.T
    return SimilarityTransform(
        scale=1.0 / scale,
        rotation=rotation,
        translation=-(rotation @ similarity.translation) / scale,
    )


# Relative least-squares weights for pick confidence (UI: High / Normal / Low).
CONFIDENCE_WEIGHTS = {
    "HIGH": 4.0,
    "NORMAL": 1.0,
    "LOW": 0.25,
}


@dataclass
class SyncObservation:
    """One landmark click in one match, in source-image pixels."""

    match_id: str
    landmark_id: str
    u: float
    v: float
    on_ground: bool = False
    landmark_name: str = ""
    # Relative least-squares weight (High=4, Normal=1, Low=0.25).
    weight: float = 1.0


@dataclass
class SyncLineObservation:
    """One 2D segment observation of a shared 3D edge, in source-image pixels."""

    match_id: str
    landmark_id: str
    u1: float
    v1: float
    u2: float
    v2: float
    landmark_name: str = ""
    weight: float = 1.0


def confidence_weight(confidence: str) -> float:
    """Map a UI confidence enum to a sync residual weight."""
    return float(CONFIDENCE_WEIGHTS.get(confidence, 1.0))


def _observation_scale(observation: SyncObservation) -> float:
    """sqrt(weight) so cost uses weight * r^2."""
    return float(np.sqrt(max(float(observation.weight), 1.0e-12)))


def _pair_scale(
    anchor_obs: SyncObservation,
    other_obs: SyncObservation,
) -> float:
    """Correspondence scale from geometric-mean weight of both picks."""
    return float(
        np.sqrt(
            max(float(anchor_obs.weight) * float(other_obs.weight), 1.0e-12)
        )
    )


@dataclass
class SyncMatchInput:
    """Frozen private-frame calibration for one match."""

    match_id: str
    calibration: core.Calibration


@dataclass
class GroundPlaneInitialization:
    """Calibrated planar initialization rooted in the anchor camera frame."""

    # Unit normal pointing from the anchor camera toward the observed plane.
    plane_normal_camera: np.ndarray
    # Anchor-camera coordinates → each supporting camera's coordinates.
    relative_rotations: dict[str, np.ndarray]
    # Camera-to-plane distance relative to the anchor distance.
    plane_distance_ratios: dict[str, float]
    supporting_match_ids: list[str]
    mean_normal_deviation_degrees: float = 0.0


@dataclass
class _GroundHomographyCandidate:
    normal_camera: np.ndarray
    relative_rotation: np.ndarray
    plane_distance_ratio: float
    positive_depth_count: int
    point_count: int
    strength: float


@dataclass
class SyncSolveResult:
    """Result of a landmark-graph sync solve."""

    similarities: dict[str, SimilarityTransform]
    landmarks: dict[str, np.ndarray]
    mean_reprojection_px: float
    per_match_rmse_px: dict[str, float]
    per_landmark_rmse_px: dict[str, float]
    message: str
    success: bool = True
    # Finite 3D segments for LINE landmarks (debug mesh viz).
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )
    # Landmarks soft-downweighted before the joint BA pass.
    downweighted_landmark_ids: list[str] = field(default_factory=list)
    bundle_adjusted: bool = False
    # Leave-one-out Diagnose: (name, with_rmse, without_rmse) for worst picks.
    leave_one_out: list[tuple[str, float, float]] = field(default_factory=list)
    # Skipped still whose other picks fit: (match_id, landmark_name, error_px).
    inconsistent_picks: list[tuple[str, str, float]] = field(default_factory=list)
