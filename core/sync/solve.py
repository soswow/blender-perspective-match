"""Top-level landmark-graph solve: register, peel, BA, resect skipped."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Callable

import numpy as np

from ..derived_lines import derived_line_geometry, validate_derived_lines

from .ba import (
    _auto_downweight_outlier_observations,
    _balance_observation_weights,
    _bundle_adjust_registration,
    _collect_ba_line_constraints,
    _pack_params,
    _per_match_rmse_snapshot,
    _point_landmark_rmse_snapshot,
    _residual_vector,
    _triangulate_landmarks,
    observations_for_location,
)
from .constants import (
    ACCEPT_RMSE_PX,
    BA_ACCEPT_RMSE_FLOOR_PX,
    BA_ACCEPT_RMSE_SLACK_PX,
    BA_FREE_LANDMARK_LIMIT,
    GROUND_PLANE_Z_FRACTION,
    GROUND_SLACK_DEFAULT,
    KNOWN_3D_SLACK_DEFAULT,
    LINE_PLANE_MIN_SINE,
    MIRROR_PAIR_SLACK_DEFAULT,
    MIRROR_SLACK_DEFAULT,
    PLANE_SLACK_DEFAULT,
    RECOVERED_HUBER_DELTA_PX,
    RESECT_MISMATCH_CANDIDATE_LIMIT,
)
from .lines import (
    _enforce_parallel_line_segments,
    _finite_segment_from_line_observations,
    _line_anchor_match_ids,
    _line_observation_reprojection_errors,
    _parallel_direction_error,
    _reconstruct_line_from_observations,
    line_support_angles,
)
from .mirrors import (
    _dedupe_mirror_pairs,
    apply_mirror_seed,
    enforce_mirror_line_segments,
    frozen_mirror_line_segments,
    mirror_plane_offset,
    reflect_point,
    seed_mirror_landmarks,
    seed_mirror_line_segments,
)
from .planes import (
    active_plane_group_count,
    apply_plane_seed,
    seed_plane_points,
    supported_line_planes,
    enforce_plane_line_segments,
    normalize_plane_groups,
    plane_slack_excesses,
)
from .pose import (
    _consistent_metric_landmarks,
    _format_worst_landmarks,
    _landmark_names,
    _metric_landmarks,
    _metric_pnp_correspondences,
    _posed_ground_landmarks,
    _register_from_relative_pose,
    _relative_pose_from_correspondences,
    _reprojection_errors_for_similarity,
    _square_pixel_intrinsics_if_stretched,
)
from .projection import project_private_point
from .types import (
    SimilarityTransform,
    SyncCancelled,
    SyncLineObservation,
    SyncMatchInput,
    SyncObservation,
    SyncSolutionSeed,
    SyncSolveResult,
)


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise SyncCancelled("Sync cancelled")


def _report_progress(
    progress_callback: Callable[[str], None] | None,
    label: str,
) -> None:
    if progress_callback is not None:
        progress_callback(label)


def _validated_solution_seed(
    seed: SyncSolutionSeed | None,
    match_map: dict[str, SyncMatchInput],
    anchor_id: str,
    point_observations: dict[str, list[SyncObservation]],
    line_ids: set[str],
) -> tuple[dict[str, SimilarityTransform], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Keep finite, compatible portions of a persisted solution as a warm start."""
    if seed is None or anchor_id not in seed.similarities:
        return {}, {}, {}
    anchor = seed.similarities[anchor_id]
    if (
        abs(float(anchor.scale) - 1.0) > 1.0e-5
        or np.linalg.norm(np.asarray(anchor.rotation) - np.eye(3)) > 1.0e-4
        or np.linalg.norm(np.asarray(anchor.translation)) > 1.0e-4
    ):
        return {}, {}, {}
    similarities = {}
    for match_id, item in seed.similarities.items():
        if match_id not in match_map or match_id not in seed.calibrations:
            continue
        rotation = np.asarray(item.rotation, dtype=np.float64)
        translation = np.asarray(item.translation, dtype=np.float64)
        scale = float(item.scale)
        if (
            rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.isfinite(scale) or not 1.0e-9 < scale < 1.0e9
            or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1.0e-3
            or np.linalg.det(rotation) < 0.999
        ):
            continue
        similarities[match_id] = SimilarityTransform(scale, rotation.copy(), translation.copy())
    if anchor_id not in similarities:
        return {}, {}, {}
    landmarks = {}
    for landmark_id, value in seed.landmarks.items():
        point = np.asarray(value, dtype=np.float64)
        if landmark_id in point_observations and point.shape == (3,) and np.isfinite(point).all():
            landmarks[landmark_id] = point.copy()
    segments = {}
    for landmark_id, value in seed.line_segments.items():
        segment = np.asarray(value, dtype=np.float64)
        if (
            landmark_id in line_ids and segment.shape == (2, 3)
            and np.isfinite(segment).all()
            and np.linalg.norm(segment[1] - segment[0]) > 1.0e-9
        ):
            segments[landmark_id] = (segment[0].copy(), segment[1].copy())
    # A manually changed calibration or grossly incompatible pick graph must
    # register afresh. A few edited observations may still use the good pose.
    errors_by_match: dict[str, list[float]] = {}
    for landmark_id, items in point_observations.items():
        point = landmarks.get(landmark_id)
        if point is None:
            continue
        for observation in items:
            similarity = similarities.get(observation.match_id)
            if similarity is None:
                continue
            projected = project_private_point(
                similarity.inverse_point(point),
                match_map[observation.match_id].calibration,
            )
            error = float("inf") if projected is None else float(np.linalg.norm(
                np.asarray(projected) - np.array((observation.u, observation.v))
            ))
            errors_by_match.setdefault(observation.match_id, []).append(error)
    for match_id, errors in errors_by_match.items():
        if match_id == anchor_id or len(errors) < 3:
            continue
        finite = np.asarray(errors, dtype=np.float64)
        if (
            np.mean(np.isfinite(finite)) < 0.8
            or float(np.median(finite)) > 2.0 * ACCEPT_RMSE_PX
        ):
            similarities.pop(match_id, None)
    return similarities, landmarks, segments


def solution_result_from_seed(seed: SyncSolutionSeed) -> SyncSolveResult | None:
    """Rebuild exact applied reporting alongside a persisted numerical solution."""
    diagnostics = seed.diagnostics
    if diagnostics is None:
        return None
    return SyncSolveResult(
        similarities=deepcopy(seed.similarities),
        landmarks=deepcopy(seed.landmarks),
        line_segments=deepcopy(seed.line_segments),
        calibrations=deepcopy(seed.calibrations),
        mean_reprojection_px=float(diagnostics.mean_reprojection_px),
        per_match_rmse_px=dict(diagnostics.per_match_rmse_px),
        per_landmark_rmse_px=dict(diagnostics.per_landmark_rmse_px),
        point_rmse_px=diagnostics.point_rmse_px,
        line_rmse_px=diagnostics.line_rmse_px,
        per_match_point_rmse_px=dict(diagnostics.per_match_point_rmse_px),
        per_match_line_rmse_px=dict(diagnostics.per_match_line_rmse_px),
        message=str(diagnostics.message),
        line_support_angles_deg=dict(diagnostics.line_support_angles_deg),
        weak_line_ids=list(diagnostics.weak_line_ids),
        plane_seeded_landmark_ids=list(diagnostics.plane_seeded_landmark_ids),
        downweighted_landmark_ids=list(diagnostics.downweighted_landmark_ids),
        bundle_adjusted=bool(diagnostics.bundle_adjusted),
        inconsistent_picks=list(diagnostics.inconsistent_picks),
        joint_initial_objective=diagnostics.joint_initial_objective,
        joint_final_objective=diagnostics.joint_final_objective,
        joint_constraint_gaps=dict(diagnostics.joint_constraint_gaps),
        joint_mirror_offset_m=float(diagnostics.joint_mirror_offset_m),
        joint_support_coverage=deepcopy(diagnostics.joint_support_coverage),
        joint_refusal_reason=str(diagnostics.joint_refusal_reason),
        joint_point_weights=deepcopy(diagnostics.joint_point_weights),
    )


def _connected_match_ids(
    anchor_id: str,
    observations: list[SyncObservation],
    *,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> set[str]:
    """Matches reachable from the anchor through shared landmarks.

    Known-world landmarks (Blender Empties) also bridge: a pick in any match
    links that match to the anchor even without an anchor 2D observation.
    """
    landmark_to_matches: dict[str, set[str]] = {}
    for observation in observations:
        landmark_to_matches.setdefault(observation.landmark_id, set()).add(
            observation.match_id,
        )
    for observation in line_observations or ():
        landmark_to_matches.setdefault(observation.landmark_id, set()).add(
            observation.match_id,
        )
    if known_world:
        for landmark_id in known_world:
            landmark_to_matches.setdefault(landmark_id, set()).add(anchor_id)
    if known_lines:
        for landmark_id in known_lines:
            landmark_to_matches.setdefault(landmark_id, set()).add(anchor_id)
    adjacency: dict[str, set[str]] = {}
    for match_ids in landmark_to_matches.values():
        for match_id in match_ids:
            adjacency.setdefault(match_id, set()).update(match_ids - {match_id})
    reached = {anchor_id}
    queue = [anchor_id]
    while queue:
        current = queue.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _observations_for_landmark_ids(
    observations_by_landmark: dict[str, list[SyncObservation]],
    keep_ids: set[str],
) -> dict[str, list[SyncObservation]]:
    """Copy the observation lists for a landmark subset."""
    return {
        landmark_id: list(items)
        for landmark_id, items in observations_by_landmark.items()
        if landmark_id in keep_ids
    }


def _ground_like_landmark_ids(
    cloud: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
    *,
    ground_slack: float = 0.0,
) -> set[str]:
    """Landmarks tagged On Ground, or whose triangulated Z is near the plane."""
    keep: set[str] = set()
    slack = max(float(ground_slack), 0.0)
    for landmark_id, point in cloud.items():
        items = observations_by_landmark.get(landmark_id, [])
        if any(item.on_ground for item in items):
            keep.add(landmark_id)
            continue
        scale = max(float(np.linalg.norm(point)), 1.0e-3)
        limit = slack if slack > 1.0e-12 else GROUND_PLANE_Z_FRACTION * scale
        if abs(float(point[2])) <= limit:
            keep.add(landmark_id)
    return keep


def _similarity_rmse_against_cloud(
    similarity: SimilarityTransform,
    match_id: str,
    cloud: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    anchor_id: str,
) -> float | None:
    """RMSE of a candidate Empty pose vs shared-world 3D ↔ this still's 2D."""
    points_shared, points_image, _ids, weights = _metric_pnp_correspondences(
        match_id, cloud, observations_by_landmark
    )
    if len(points_shared) < 4:
        return None
    errors = _reprojection_errors_for_similarity(
        similarity,
        [],
        matches[anchor_id].calibration,
        matches[match_id].calibration,
        points_shared,
        points_image,
        point_weights=weights,
    )
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def _try_register_against_cloud(
    match_id: str,
    cloud: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
    matches: dict[str, SyncMatchInput],
    anchor_id: str,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] | None,
    parallel_pairs: list[tuple[str, str]] | None,
    *,
    lock_rotation: bool,
    lock_translation: bool,
    rmse_limit: float = ACCEPT_RMSE_PX,
    use_pose_cache: bool = False,
    initial_similarity: SimilarityTransform | None = None,
    initial_only: bool = False,
    best_candidate_out: list[SimilarityTransform] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SimilarityTransform | None:
    """PnP a skipped still against a frozen 3D cloud (no free 2D↔2D pairs)."""
    _check_cancelled(cancel_check)
    if len(cloud) < 4:
        return None
    subset = _observations_for_landmark_ids(observations_by_landmark, set(cloud))
    solved, _detail = _relative_pose_from_correspondences(
        anchor_id,
        match_id,
        subset,
        matches,
        cloud,
        known_lines=known_lines,
        line_observations_by_landmark=line_observations_by_landmark,
        parallel_pairs=parallel_pairs,
        initial_similarity=initial_similarity,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        use_pose_cache=use_pose_cache,
        cancel_check=cancel_check,
        initial_only=initial_only,
        best_candidate_out=best_candidate_out,
    )
    if solved is None:
        return None
    rmse = _similarity_rmse_against_cloud(
        solved, match_id, cloud, subset, matches, anchor_id
    )
    if rmse is None or rmse > rmse_limit:
        return None
    return solved


def _pick_reprojection_px(
    similarity: SimilarityTransform,
    observation: SyncObservation,
    calibration,
    point: np.ndarray,
) -> float:
    """Pixel error of one 2D pick vs a shared-world 3D point."""
    projected = project_private_point(similarity.inverse_point(point), calibration)
    if projected is None:
        return 1.0e3
    return float(np.hypot(projected[0] - observation.u, projected[1] - observation.v))


def _pick_errors_for_match(
    match_id: str,
    similarity: SimilarityTransform,
    cloud: dict[str, np.ndarray],
    observations: list[SyncObservation],
    matches: dict[str, SyncMatchInput],
) -> list[tuple[str, str, float]]:
    """Per-landmark reprojection of ``match_id`` against ``cloud``."""
    calibration = matches[match_id].calibration
    rows: list[tuple[str, str, float]] = []
    for observation in observations:
        if observation.match_id != match_id:
            continue
        point = cloud.get(observation.landmark_id)
        if point is None:
            continue
        error = _pick_reprojection_px(similarity, observation, calibration, point)
        name = observation.landmark_name or observation.landmark_id
        rows.append((observation.landmark_id, name, error))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def _dominant_mismatch_picks(
    errors: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    """Picks that blow a still's RMSE while the others still fit."""
    if len(errors) < 4:
        return []
    median = float(np.median([item[2] for item in errors]))
    floor = max(3.0 * max(median, 1.0), ACCEPT_RMSE_PX)
    flagged = [item for item in errors if item[2] > floor]
    return flagged[:3]


def _record_mismatch_picks(state: _SolveState, match_id: str) -> None:
    """Remember a peel-time pick that dominates this still's RMSE."""
    similarity = state.similarities.get(match_id)
    if similarity is None:
        return
    errors = _pick_errors_for_match(
        match_id,
        similarity,
        state.landmarks,
        state.usable_observations,
        state.match_map,
    )
    flagged = _dominant_mismatch_picks(errors)
    if flagged:
        state.inconsistent_picks[match_id] = flagged


def _mismatch_reason(picks: list[tuple[str, str, float]]) -> str:
    """Short status clause for a skipped still with a wrong correspondence."""
    bits = [f"{name} {error:.0f}px" for _landmark_id, name, error in picks]
    noun = "pick" if len(picks) == 1 else "picks"
    return f"{', '.join(bits)} in that still — likely a mismatched {noun}"


def _resect_mismatch_picks(
    match_id: str,
    cloud: dict[str, np.ndarray],
    retry_kwargs: dict,
    initial_similarity: SimilarityTransform | None,
) -> list[tuple[str, str, float]]:
    """Warm-refit without each worst pick; report one-pick pose recoveries."""
    if initial_similarity is None:
        return []
    _check_cancelled(retry_kwargs.get("cancel_check"))
    observations = retry_kwargs["observations_by_landmark"]
    matches = retry_kwargs["matches"]
    seen = [
        observation
        for items in observations.values()
        for observation in items
        if observation.match_id == match_id and observation.landmark_id in cloud
    ]
    if len(seen) < 5:
        return []
    calibration = matches[match_id].calibration
    seen.sort(
        key=lambda observation: _pick_reprojection_px(
            initial_similarity,
            observation,
            calibration,
            cloud[observation.landmark_id],
        ),
        reverse=True,
    )
    found: list[tuple[str, str, float]] = []
    for observation in seen[:RESECT_MISMATCH_CANDIDATE_LIMIT]:
        _check_cancelled(retry_kwargs.get("cancel_check"))
        reduced = {
            landmark_id: point
            for landmark_id, point in cloud.items()
            if landmark_id != observation.landmark_id
        }
        solved = _try_register_against_cloud(
            match_id,
            reduced,
            initial_similarity=initial_similarity,
            initial_only=True,
            **retry_kwargs,
        )
        if solved is None:
            continue
        point = cloud[observation.landmark_id]
        error = _pick_reprojection_px(
            solved, observation, matches[match_id].calibration, point
        )
        if error <= ACCEPT_RMSE_PX:
            continue
        name = observation.landmark_name or observation.landmark_id
        found.append((observation.landmark_id, name, error))
    found.sort(key=lambda item: item[2], reverse=True)
    return found[:3]


@dataclass
class _SolveState:
    """Mutable graph for the named stages in ``solve_landmark_sync``."""

    match_map: dict[str, SyncMatchInput]
    anchor_id: str
    known_world: dict[str, np.ndarray]
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]]
    parallel_pairs: list[tuple[str, str]] | None
    mirror_pairs: list[tuple[str, str]] | None
    mirror_plane: tuple[np.ndarray, np.ndarray] | None
    mirror_slack: float
    mirror_landmark_id: str | None
    plane_groups: list[tuple[str, str, int]]
    plane_slack: float
    lock_rotation: bool
    lock_translation: bool
    use_pose_cache: bool
    cancel_check: Callable[[], bool] | None
    ground_slack: float
    known_3d_slack: float
    identity_result: dict[str, SimilarityTransform]
    valid_observations: list[SyncObservation]
    observations_by_landmark_all: dict[str, list[SyncObservation]]
    line_observations_by_landmark_all: dict[str, list[SyncLineObservation]]
    observations_by_landmark: dict[str, list[SyncObservation]]
    line_observations_by_landmark: dict[str, list[SyncLineObservation]]
    usable_observations: list[SyncObservation]
    landmark_ids: list[str]
    free_match_ids: list[str]
    fixed_match_ids: set[str]
    similarities: dict[str, SimilarityTransform]
    skipped_unregistered: list[str]
    failure_detail: str
    connected: set[str]
    derived_lines: list[tuple[str, str, str]] = field(default_factory=list)
    location_match_ids: set[str] | None = None
    readonly_match_ids: set[str] = field(default_factory=set)
    mirror_offset: float = 0.0
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    consistent_metric: dict[str, np.ndarray] = field(default_factory=dict)
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    initial_line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    seed_unchanged: bool = False
    recovered: list[str] = field(default_factory=list)
    peeled_similarities: dict[str, SimilarityTransform] = field(default_factory=dict)
    skip_notes: dict[str, str] = field(default_factory=dict)
    downweighted_ids: list[str] = field(default_factory=list)
    did_bundle_adjust: bool = False
    kept_joint_geometry: bool = False
    plane_seeded_ids: set[str] = field(default_factory=set)
    pre_ba_match_rmse: dict[str, float] = field(default_factory=dict)
    inconsistent_picks: dict[str, list[tuple[str, str, float]]] = field(
        default_factory=dict
    )

    def drop_matches(self, match_ids: list[str]) -> None:
        """Remove cameras from the joint graph and remember them as skipped."""
        match_ids = [
            match_id for match_id in match_ids if match_id not in self.fixed_match_ids
        ]
        if not match_ids:
            return
        if self.mirror_landmark_id:
            self.mirror_offset = 0.0
        skip = set(match_ids)
        self.skipped_unregistered.extend(
            match_id
            for match_id in match_ids
            if match_id not in self.skipped_unregistered
        )
        self.free_match_ids = [
            item for item in self.free_match_ids if item not in skip
        ]
        for match_id in skip:
            similarity = self.similarities.pop(match_id, None)
            if similarity is not None:
                self.peeled_similarities[match_id] = similarity
        self.usable_observations = [
            observation
            for observation in self.usable_observations
            if observation.match_id not in skip
        ]
        self.observations_by_landmark = {}
        for observation in self.usable_observations:
            self.observations_by_landmark.setdefault(
                observation.landmark_id, []
            ).append(observation)
        self.landmark_ids = sorted(self.observations_by_landmark.keys())
        self.line_observations_by_landmark = {
            landmark_id: [
                item for item in items if item.match_id not in skip
            ]
            for landmark_id, items in self.line_observations_by_landmark.items()
        }

    def ba_constraint_kwargs(self) -> dict:
        """The same geometric priors for initial and recovered-camera joint BA."""
        return dict(
            ground_landmark_ids=sorted({o.landmark_id for o in self.usable_observations if o.on_ground}),
            ground_slack=self.ground_slack,
            known_world_priors=self.known_world,
            known_3d_slack=self.known_3d_slack,
            mirror_pairs=self.mirror_pairs,
            mirror_plane=self.mirror_plane,
            mirror_slack=self.mirror_slack,
            mirror_landmark_id=self.mirror_landmark_id,
            plane_groups=self.plane_groups,
            plane_slack=self.plane_slack,
        )

    def effective_mirror_plane(
        self, landmarks: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Use the current reference point with the supplied plane normal."""
        if self.mirror_plane is None:
            return None
        if self.mirror_landmark_id is None:
            return self.mirror_plane
        points = self.landmarks if landmarks is None else landmarks
        point = points.get(self.mirror_landmark_id)
        if point is None:
            raise ValueError("Mirror reference point lost its two-view reconstruction; add supporting picks")
        normal = self.mirror_plane[1]
        unit = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
        return point + self.mirror_offset * unit, normal

    def rebuild_landmarks(
        self, retained_landmarks: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Triangulate points/lines; pin Known 3D and consistent On Ground."""
        self.plane_seeded_ids.clear()
        rebuilt = _triangulate_landmarks(
            sorted(self.observations_by_landmark),
            self.observations_by_landmark,
            self.similarities,
            self.match_map,
            location_match_ids=self.location_match_ids,
        )
        metric_points = _metric_landmarks(
            self.observations_by_landmark,
            self.anchor_id,
            self.match_map[self.anchor_id].calibration,
            self.known_world,
        )
        for landmark_id, point in _posed_ground_landmarks(
            self.observations_by_landmark, self.similarities, self.match_map,
            location_match_ids=self.location_match_ids,
        ).items():
            metric_points.setdefault(landmark_id, point)
        consistent = _consistent_metric_landmarks(
            metric_points,
            rebuilt,
            set(self.known_world),
            ground_slack=self.ground_slack,
        )
        rebuilt.update(consistent)
        for landmark_id, point in self.known_world.items():
            rebuilt[landmark_id] = point
        # Slack leaves On Ground unpinned; still seed them when no second
        # location view could triangulate.
        for landmark_id, point in metric_points.items():
            rebuilt.setdefault(landmark_id, point)
        rebuilt.update({
            landmark_id: point.copy()
            for landmark_id, point in (retained_landmarks or {}).items()
            if (self.seed_unchanged or landmark_id in rebuilt)
            and landmark_id in self.observations_by_landmark
            and landmark_id not in consistent
            and landmark_id not in self.known_world
        })
        self.landmarks = rebuilt
        self.consistent_metric = consistent
        _rebuild_free_line_segments(self)
        self.initial_line_segments = {}
        _attach_mirror_landmarks(self)
        ground_ids = {
            observation.landmark_id
            for observation in self.valid_observations
            if observation.on_ground
        }
        apply_plane_seed(
            self.landmarks,
            self.line_segments,
            self.plane_groups,
            known_ids=set(self.known_world),
            known_line_ids=set(self.known_lines),
            ground_ids=ground_ids,
        )
        _snap_mirror_landmarks(self)
        self.seed_plane_landmarks()

    def seed_plane_landmarks(self) -> None:
        """Use already reconstructed planes for permitted single-view points."""
        seeded = seed_plane_points(
            self.landmarks, self.line_segments, self.plane_groups,
            self.observations_by_landmark_all, self.similarities, self.match_map,
            plane_slack=self.plane_slack, location_match_ids=self.location_match_ids,
        )
        self.landmarks.update(seeded)
        self.plane_seeded_ids.update(seeded)


def _snap_mirror_landmarks(state: _SolveState) -> None:
    """Project reconstructed two-sided pairs onto the current mirror plane."""
    if state.mirror_plane is None or not state.mirror_pairs:
        return
    plane_point, plane_normal = state.effective_mirror_plane()
    apply_mirror_seed(
        state.landmarks,
        state.mirror_pairs,
        plane_point,
        plane_normal,
        known_ids=set(state.known_world),
        skip_ids=set(state.line_segments) | set(state.known_lines),
    )
    if not state.line_segments:
        return
    location_ids = getattr(state, "location_match_ids", None)
    line_support = {
        key: observations_for_location(items, location_ids)
        for key, items in state.line_observations_by_landmark.items()
    }
    enforce_mirror_line_segments(
        state.line_segments,
        state.landmarks,
        state.mirror_pairs,
        plane_point,
        plane_normal,
        line_support,
        state.similarities,
        state.match_map,
        state.known_lines,
        state.fixed_match_ids,
        parallel_pairs=state.parallel_pairs,
        line_planes=supported_line_planes(
            state.landmarks, state.line_segments, state.plane_groups, state.known_lines,
            ground_landmark_ids=sorted({item.landmark_id for item in state.valid_observations if item.on_ground}),
            excluded_support_ids=state.plane_seeded_ids,
        ) if state.plane_slack <= 1e-12 else None,
    )


def _attach_mirror_landmarks(state: _SolveState) -> None:
    """Seed mirror geometry from location-enabled views; retain all posed picks."""
    if state.mirror_plane is None or not state.mirror_pairs:
        return
    plane_point, plane_normal = state.effective_mirror_plane()
    location_ids = getattr(state, "location_match_ids", None)
    point_support = {
        key: observations_for_location(items, location_ids)
        for key, items in state.observations_by_landmark_all.items()
    }
    line_support = {
        key: observations_for_location(items, location_ids)
        for key, items in state.line_observations_by_landmark.items()
    }
    seed_mirror_landmarks(
        state.landmarks,
        point_support,
        state.similarities,
        state.match_map,
        state.mirror_pairs,
        plane_point,
        plane_normal,
    )
    # An unpicked partner may have been seeded before BA fitted a soft plane
    # offset. Keep that dependent point on the accepted effective plane.
    if state.mirror_landmark_id:
        for landmark_a, landmark_b in _dedupe_mirror_pairs(state.mirror_pairs):
            for source_id, partner_id in ((landmark_a, landmark_b), (landmark_b, landmark_a)):
                if (
                    source_id in state.landmarks
                    and partner_id in state.landmarks
                    and point_support.get(source_id)
                    and not state.observations_by_landmark_all.get(partner_id)
                    and partner_id not in state.known_world
                    and partner_id not in state.line_segments
                ):
                    state.landmarks[partner_id] = reflect_point(
                        state.landmarks[source_id], plane_point, plane_normal,
                    )
        for landmark_a, landmark_b in _dedupe_mirror_pairs(state.mirror_pairs):
            for source_id, partner_id in ((landmark_a, landmark_b), (landmark_b, landmark_a)):
                if (
                    source_id in state.line_segments
                    and line_support.get(source_id)
                    and not state.line_observations_by_landmark.get(partner_id)
                    and not state.observations_by_landmark_all.get(partner_id)
                    and partner_id not in state.known_lines
                ):
                    state.line_segments.pop(partner_id, None)
                    state.landmarks.pop(partner_id, None)
    seed_mirror_line_segments(
        state.line_segments,
        state.landmarks,
        line_support,
        state.similarities,
        state.match_map,
        state.mirror_pairs,
        plane_point,
        plane_normal,
        state.known_lines,
        state.fixed_match_ids,
    )
    ground_ids = sorted({item.landmark_id for item in state.valid_observations if item.on_ground})
    enforce_plane_line_segments(
        state.line_segments,state.landmarks,state.plane_groups,line_support,
        state.similarities,state.match_map,state.known_lines,
        plane_slack=state.plane_slack,ground_landmark_ids=ground_ids,
        excluded_support_ids=state.plane_seeded_ids,
    )
    enforce_mirror_line_segments(
        state.line_segments,
        state.landmarks,
        state.mirror_pairs,
        plane_point,
        plane_normal,
        line_support,
        state.similarities,
        state.match_map,
        state.known_lines,
        state.fixed_match_ids,
        line_planes=supported_line_planes(
            state.landmarks,state.line_segments,state.plane_groups,state.known_lines,
            ground_landmark_ids=ground_ids,
            excluded_support_ids=state.plane_seeded_ids,
        ) if state.plane_slack <= 1e-12 else None,
        parallel_pairs=state.parallel_pairs,
    )
    existing = {
        (observation.match_id, observation.landmark_id)
        for observation in state.usable_observations
    }
    for landmark_a, landmark_b in _dedupe_mirror_pairs(state.mirror_pairs):
        for landmark_id in (landmark_a, landmark_b):
            if landmark_id not in state.landmarks:
                continue
            for observation in state.observations_by_landmark_all.get(
                landmark_id, []
            ):
                if observation.match_id not in state.similarities:
                    continue
                key = (observation.match_id, observation.landmark_id)
                if key in existing:
                    continue
                state.usable_observations.append(observation)
                existing.add(key)
    state.observations_by_landmark = {}
    for observation in state.usable_observations:
        state.observations_by_landmark.setdefault(
            observation.landmark_id, []
        ).append(observation)
    state.landmark_ids = sorted(state.observations_by_landmark.keys())


def _peel_cameras_above_rmse(state: _SolveState) -> SyncSolveResult | None:
    """Drop the worst free camera while its RMSE is above ACCEPT_RMSE_PX."""
    while state.free_match_ids:
        state.pre_ba_match_rmse = _per_match_rmse_snapshot(
            state.free_match_ids,
            state.landmark_ids,
            state.similarities,
            state.landmarks,
            state.anchor_id,
            state.match_map,
            state.usable_observations,
        )
        ranked = sorted(
            (
                (state.pre_ba_match_rmse.get(match_id, 0.0), match_id)
                for match_id in state.free_match_ids
                if match_id not in state.fixed_match_ids
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] <= ACCEPT_RMSE_PX:
            return None
        _record_mismatch_picks(state, ranked[0][1])
        state.drop_matches([ranked[0][1]])
        if not state.free_match_ids:
            return SyncSolveResult(
                similarities=state.identity_result,
                landmarks={},
                mean_reprojection_px=ranked[0][0],
                per_match_rmse_px=state.pre_ba_match_rmse,
                per_landmark_rmse_px={},
                message=(
                    state.failure_detail
                    or (
                        f"Could not keep a camera under {ACCEPT_RMSE_PX:.0f} px "
                        "after registration"
                    )
                ),
                success=False,
            )
        state.rebuild_landmarks()
    return None


def _cloud_observation_count(
    match_id: str,
    cloud: dict[str, np.ndarray],
    observations_by_landmark: dict[str, list[SyncObservation]],
) -> int:
    """How many cloud landmarks this still actually picked."""
    count = 0
    for landmark_id in cloud:
        items = observations_by_landmark.get(landmark_id, [])
        if any(item.match_id == match_id for item in items):
            count += 1
    return count


def _posed_observations(
    state: _SolveState,
) -> dict[str, list[SyncObservation]]:
    """Observations whose cameras currently have a pose."""
    posed = set(state.similarities)
    grouped: dict[str, list[SyncObservation]] = {}
    for observation in state.valid_observations:
        if observation.match_id not in posed:
            continue
        grouped.setdefault(observation.landmark_id, []).append(observation)
    return grouped


def _expand_landmarks_after_resect(state: _SolveState) -> None:
    """Triangulate landmarks now visible in recovered views; pose hanging stills.

    A still that only shares tags with peeled cameras can stay in the graph with
    a leftover pose and zero residuals after those cameras drop. Exclusive
    landmarks then keep a stale Empty / list RMSE. Fill 3D from the recovered
    cameras, then PnP stills that never had four cloud hits.
    """
    if len(state.similarities) < 2:
        return
    original_cloud = dict(state.landmarks)
    observations_posed = _posed_observations(state)
    extra = _triangulate_landmarks(
        sorted(observations_posed.keys()),
        observations_posed,
        state.similarities,
        state.match_map,
        location_match_ids=state.location_match_ids,
    )
    added_ids: set[str] = set()
    for landmark_id, point in extra.items():
        if landmark_id in state.landmarks:
            continue
        state.landmarks[landmark_id] = point
        added_ids.add(landmark_id)
    if not added_ids:
        return
    retry_kwargs = {
        "observations_by_landmark": state.observations_by_landmark_all,
        "matches": state.match_map,
        "anchor_id": state.anchor_id,
        "known_lines": state.known_lines,
        "line_observations_by_landmark": state.line_observations_by_landmark_all,
        "parallel_pairs": state.parallel_pairs,
        "lock_rotation": state.lock_rotation,
        "lock_translation": state.lock_translation,
        "use_pose_cache": state.use_pose_cache,
        "cancel_check": state.cancel_check,
    }
    weak_ids = [
        match_id
        for match_id in list(state.similarities)
        if match_id != state.anchor_id
        and match_id not in state.fixed_match_ids
        and _cloud_observation_count(
            match_id,
            original_cloud,
            state.observations_by_landmark_all,
        )
        < 4
    ]
    for match_id in weak_ids:
        solved = _try_register_against_cloud(
            match_id, state.landmarks, **retry_kwargs
        )
        if solved is None:
            state.similarities.pop(match_id, None)
            if match_id not in state.skipped_unregistered:
                state.skipped_unregistered.append(match_id)
            state.skip_notes[match_id] = (
                "could not lock from landmarks only visible on recovered stills"
            )
            state.free_match_ids = [
                item for item in state.free_match_ids if item != match_id
            ]
            continue
        state.similarities[match_id] = solved
        if match_id not in state.recovered:
            state.recovered.append(match_id)
        state.skipped_unregistered = [
            item for item in state.skipped_unregistered if item != match_id
        ]
        if match_id not in state.free_match_ids:
            state.free_match_ids.append(match_id)
    observations_posed = _posed_observations(state)
    refined = _triangulate_landmarks(
        sorted(observations_posed.keys()),
        observations_posed,
        state.similarities,
        state.match_map,
        location_match_ids=state.location_match_ids,
    )
    for landmark_id, point in refined.items():
        if landmark_id not in state.landmarks or landmark_id in added_ids:
            state.landmarks[landmark_id] = point
    _attach_mirror_landmarks(state)


def _rebuild_usable_observations(state: _SolveState) -> None:
    """Keep observations whose camera and landmark both survived the solve."""
    skip = set(state.skipped_unregistered)
    state.usable_observations = [
        observation
        for observation in state.valid_observations
        if observation.match_id in state.similarities
        and observation.match_id not in skip
        and observation.landmark_id in state.landmarks
    ]
    state.observations_by_landmark = {}
    for observation in state.usable_observations:
        state.observations_by_landmark.setdefault(
            observation.landmark_id, []
        ).append(observation)
    state.landmark_ids = sorted(state.landmarks.keys())
    state.line_observations_by_landmark = {
        landmark_id: [
            item
            for item in items
            if item.match_id in state.similarities and item.match_id not in skip
        ]
        for landmark_id, items in state.line_observations_by_landmark_all.items()
    }


def _rebuild_free_line_segments(state: _SolveState) -> None:
    """Re-intersect free 3D lines from cameras allowed to move 3D."""
    line_support = {
        key: observations_for_location(items, getattr(state, "location_match_ids", None))
        for key, items in state.line_observations_by_landmark.items()
    }
    segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for landmark_id, (point_a, point_b) in state.known_lines.items():
        segments[landmark_id] = (point_a.copy(), point_b.copy())
        state.landmarks[landmark_id] = 0.5 * (point_a + point_b)
    for landmark_id, items in line_support.items():
        if landmark_id in segments:
            continue
        posed_items = [item for item in items if item.match_id in state.similarities]
        seeded = state.initial_line_segments.get(landmark_id)
        if seeded is not None and posed_items and (
            state.seed_unchanged or len({item.match_id for item in posed_items}) >= 2
        ):
            segments[landmark_id] = (seeded[0].copy(), seeded[1].copy())
            state.landmarks[landmark_id] = 0.5 * (seeded[0] + seeded[1])
            continue
        anchor_ids = _line_anchor_match_ids(posed_items, state.fixed_match_ids)
        reconstructed = _reconstruct_line_from_observations(
            posed_items,
            state.similarities,
            state.match_map,
            prefer_match_ids=anchor_ids,
        )
        if reconstructed is None:
            continue
        point, direction = reconstructed
        segment = _finite_segment_from_line_observations(
            point,
            direction,
            posed_items,
            state.similarities,
            state.match_map,
            extent_match_ids=anchor_ids,
        )
        state.landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
        segments[landmark_id] = segment
    _enforce_parallel_line_segments(
        segments,
        state.landmarks,
        state.parallel_pairs,
        line_support,
        state.similarities,
        state.match_map,
        state.known_lines,
    )
    plane_groups = list(getattr(state, "plane_groups", ()) or ())
    if plane_groups:
        ground_ids = sorted(
            {
                observation.landmark_id
                for observation in getattr(state, "valid_observations", ())
                if observation.on_ground
            }
        )
        enforce_plane_line_segments(
            segments,
            state.landmarks,
            plane_groups,
            line_support,
            state.similarities,
            state.match_map,
            state.known_lines,
            plane_slack=float(getattr(state, "plane_slack", 0.0) or 0.0),
            parallel_pairs=state.parallel_pairs,
            ground_landmark_ids=ground_ids,
            excluded_support_ids=getattr(state,"plane_seeded_ids",set()),
        )
    mirror_plane = (
        state.effective_mirror_plane()
        if isinstance(state, _SolveState)
        else getattr(state, "mirror_plane", None)
    )
    mirror_pairs = getattr(state, "mirror_pairs", None)
    if mirror_plane is not None and mirror_pairs:
        enforce_mirror_line_segments(
            segments,
            state.landmarks,
            mirror_pairs,
            mirror_plane[0],
            mirror_plane[1],
            line_support,
            state.similarities,
            state.match_map,
            state.known_lines,
            getattr(state, "fixed_match_ids", None),
            parallel_pairs=state.parallel_pairs,
            line_planes=supported_line_planes(
                state.landmarks, segments, plane_groups, state.known_lines,
                ground_landmark_ids=ground_ids if plane_groups else [],
                excluded_support_ids=getattr(state, "plane_seeded_ids", set()),
            ) if float(getattr(state, "plane_slack", 0.0) or 0.0) <= 1e-12 else None,
        )
    try:
        derived_segments, _geometry = derived_line_geometry(
            state.landmarks, getattr(state, "derived_lines", ()))
        segments.update(derived_segments)
        for line_id, segment in derived_segments.items():
            state.landmarks[line_id] = 0.5 * (segment[0] + segment[1])
    except ValueError:
        # Endpoint recovery can happen after this intermediate rebuild.
        pass
    state.line_segments = segments


def _resect_known_lines(state: _SolveState) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """User Known 3D lines plus frozen mirrored partners (not merged into state)."""
    merged = {
        landmark_id: (segment[0].copy(), segment[1].copy())
        for landmark_id, segment in state.known_lines.items()
    }
    merged.update(
        frozen_mirror_line_segments(
            state.line_segments,
            state.mirror_pairs,
            state.known_lines,
        )
    )
    return merged


def _recovered_combo_rmse(
    match_ids: list[str],
    similarities: dict[str, SimilarityTransform],
    landmarks: dict[str, np.ndarray],
    matches: dict[str, SyncMatchInput],
    observations: list[SyncObservation],
    line_constraints: list[tuple[str, np.ndarray, np.ndarray, SyncLineObservation]],
) -> dict[str, float]:
    """Unweighted point + line RMSE per recovered still (failed projections skipped)."""
    errors_by_match: dict[str, list[float]] = {match_id: [] for match_id in match_ids}
    for observation in observations:
        bucket = errors_by_match.get(observation.match_id)
        if bucket is None:
            continue
        point = landmarks.get(observation.landmark_id)
        if point is None:
            continue
        error = _pick_reprojection_px(
            similarities[observation.match_id],
            observation,
            matches[observation.match_id].calibration,
            point,
        )
        if error < 500.0:
            bucket.append(error)
    for _landmark_id, point, direction, line_obs in line_constraints:
        bucket = errors_by_match.get(line_obs.match_id)
        if bucket is None or line_obs.match_id not in similarities:
            continue
        stroke = replace(line_obs, weight=1.0)
        bucket.extend(
            _line_observation_reprojection_errors(
                point,
                direction,
                stroke,
                matches[line_obs.match_id].calibration,
                similarities[line_obs.match_id],
            )
        )
    return {
        match_id: float(np.sqrt(np.mean(np.square(values))))
        for match_id, values in errors_by_match.items()
        if values
    }


def _mean_rmse(values: dict[str, float]) -> float:
    if not values:
        return 0.0
    squares = [value * value for value in values.values()]
    return float(np.sqrt(np.mean(squares)))


def _polish_recovered_poses(state: _SolveState) -> None:
    """Pose-only BA of resected stills vs frozen 3D (lines keep orientation)."""
    recovered = [
        match_id
        for match_id in state.recovered
        if match_id in state.similarities and match_id not in state.fixed_match_ids
    ]
    if not recovered:
        return
    recovered_set = set(recovered)
    polish_observations = []
    for observation in state.usable_observations:
        if observation.match_id not in recovered_set:
            continue
        point = state.landmarks.get(observation.landmark_id)
        if point is None:
            continue
        error = _pick_reprojection_px(
            state.similarities[observation.match_id],
            observation,
            state.match_map[observation.match_id].calibration,
            point,
        )
        if error >= 500.0:
            continue
        polish_observations.append(observation)
    polish_observations = _balance_observation_weights(
        polish_observations, state.match_map
    )
    line_constraints = _collect_ba_line_constraints(
        state.line_segments,
        state.known_lines,
        state.line_observations_by_landmark,
        recovered_set,
    )
    if line_constraints and polish_observations:
        line_boost = max(
            1.0,
            float(len(polish_observations))
            / (2.0 * float(len(line_constraints))),
        )
        line_constraints = [
            (
                landmark_id,
                point,
                direction,
                replace(observation, weight=float(observation.weight) * line_boost),
            )
            for landmark_id, point, direction, observation in line_constraints
        ]
    if not polish_observations and not line_constraints:
        return
    pre_similarities = {
        match_id: SimilarityTransform(
            scale=item.scale,
            rotation=np.array(item.rotation, copy=True),
            translation=np.array(item.translation, copy=True),
        )
        for match_id, item in state.similarities.items()
    }
    pre_combo = _recovered_combo_rmse(
        recovered,
        state.similarities,
        state.landmarks,
        state.match_map,
        polish_observations,
        line_constraints,
    )

    fixed_landmarks = {
        landmark_id: point.copy()
        for landmark_id, point in state.landmarks.items()
    }
    similarities, landmarks, line_segments, ran = _bundle_adjust_registration(
        recovered,
        [],
        fixed_landmarks,
        state.similarities,
        state.landmarks,
        state.anchor_id,
        state.match_map,
        polish_observations,
        line_constraints,
        known_line_ids=set(state.line_segments) | set(state.known_lines),
        line_segments=state.line_segments,
        lock_rotation=state.lock_rotation,
        lock_translation=state.lock_translation,
        max_iterations=12,
        max_free_lines=0,
        huber_delta=RECOVERED_HUBER_DELTA_PX,
        location_match_ids=getattr(state, "location_match_ids", None),
    )
    if not ran:
        return
    post_combo = _recovered_combo_rmse(
        recovered,
        similarities,
        landmarks,
        state.match_map,
        polish_observations,
        line_constraints,
    )
    post_points = _per_match_rmse_snapshot(
        recovered,
        state.landmark_ids,
        similarities,
        landmarks,
        state.anchor_id,
        state.match_map,
        polish_observations,
    )
    if _mean_rmse(post_points) > ACCEPT_RMSE_PX:
        state.similarities = pre_similarities
        return
    if _mean_rmse(post_combo) > _mean_rmse(pre_combo) + 2.0:
        state.similarities = pre_similarities
        return
    state.similarities = similarities
    state.landmarks = landmarks
    state.line_segments = line_segments


def _thaw_recovered_location(state: _SolveState) -> None:
    """Commit a constrained 3D update only if previously solved views still fit."""
    location_ids = state.location_match_ids
    if location_ids is None:
        return
    if not any(
        match_id in location_ids and match_id in state.similarities
        for match_id in state.recovered
    ):
        return
    # Execution control belongs to the caller (often a threading.Event method),
    # not to the numerical candidate. Copy evidence while sharing that callback.
    trial = deepcopy(state, {id(state.cancel_check): state.cancel_check})
    _refine_recovered_location(trial)
    recovered = set(state.recovered)
    protected = [o for o in state.usable_observations
                 if o.match_id not in recovered and o.landmark_id in state.landmarks]
    required_ids = {o.landmark_id for o in protected}
    if not required_ids <= set(trial.landmarks):
        state.kept_joint_geometry = True
        return
    scores = []
    for candidate in (state, trial):
        scores.append(_per_match_rmse_snapshot(
            state.free_match_ids, sorted(required_ids), candidate.similarities,
            candidate.landmarks, state.anchor_id, state.match_map, protected,
        ))
    before, after = scores
    if any(not np.isfinite(after.get(key, np.inf)) or after[key] > min(
        ACCEPT_RMSE_PX, max(BA_ACCEPT_RMSE_FLOOR_PX, value + BA_ACCEPT_RMSE_SLACK_PX)
    ) for key, value in before.items()):
        state.kept_joint_geometry = True
        return
    state.__dict__.update(trial.__dict__)


def _refine_recovered_location(state: _SolveState) -> None:
    """Build the proposed recovered-camera update with the joint solve's priors."""
    previous_known = {
        key: point.copy() for key, point in state.landmarks.items()
        if key in state.known_world and state.known_3d_slack > 1.0e-12
    }
    if state.mirror_landmark_id:
        state.rebuild_landmarks(retained_landmarks=previous_known)
    else:
        state.rebuild_landmarks()
        state.landmarks.update(previous_known)
    _rebuild_usable_observations(state)
    free_match_ids = [
        match_id
        for match_id in state.free_match_ids
        if match_id in state.similarities
    ]
    fixed_ids = set(state.known_world) if state.known_3d_slack <= 1.0e-12 else set()
    if state.ground_slack <= 1.0e-12:
        fixed_ids.update(set(state.consistent_metric)-set(state.known_world))
    free_landmark_ids = [
        landmark_id
        for landmark_id in state.landmark_ids
        if landmark_id in state.landmarks and landmark_id not in fixed_ids
    ]
    if len(free_landmark_ids) > BA_FREE_LANDMARK_LIMIT:
        free_landmark_ids = []
    line_constraints = _collect_ba_line_constraints(
        state.line_segments,
        state.known_lines,
        state.line_observations_by_landmark,
        set(state.similarities),
    )
    if not state.usable_observations and not line_constraints:
        return
    fixed_landmarks = {
        landmark_id: point.copy()
        for landmark_id, point in state.landmarks.items()
        if landmark_id not in set(free_landmark_ids)
    }
    plane_offset_out: list[float] = []
    similarities, landmarks, line_segments, ran = _bundle_adjust_registration(
        free_match_ids,
        free_landmark_ids,
        fixed_landmarks,
        state.similarities,
        state.landmarks,
        state.anchor_id,
        state.match_map,
        state.usable_observations,
        line_constraints,
        known_line_ids=set(state.known_lines),
        line_segments=state.line_segments,
        lock_rotation=state.lock_rotation,
        lock_translation=state.lock_translation,
        fixed_similarities={
            match_id: state.similarities[match_id]
            for match_id in state.fixed_match_ids
            if match_id in state.similarities
        },
        max_iterations=8,
        location_match_ids=state.location_match_ids,
        **state.ba_constraint_kwargs(),
        free_plane_offset=(state.mirror_slack > 1.0e-12 and bool(state.mirror_pairs) and state.mirror_plane is not None),
        plane_offset_start=state.mirror_offset if state.mirror_landmark_id else 0.0,
        plane_offset_out=plane_offset_out,
    )
    if not ran:
        return
    state.similarities = similarities
    state.landmarks = landmarks
    state.line_segments = line_segments
    if state.mirror_landmark_id:
        state.mirror_offset = plane_offset_out[0]
        _attach_mirror_landmarks(state)


def _resect_skipped_matches(state: _SolveState) -> None:
    """PnP skipped stills against the frozen cloud; ground-only if that fails."""
    if not state.landmarks:
        return
    _attach_mirror_landmarks(state)
    cloud = dict(state.known_world)
    cloud.update(state.landmarks)
    ground_ids = _ground_like_landmark_ids(
        cloud,
        state.observations_by_landmark_all,
        ground_slack=state.ground_slack,
    )
    ground_cloud = {
        landmark_id: cloud[landmark_id] for landmark_id in ground_ids
    }
    retry_kwargs = {
        "observations_by_landmark": state.observations_by_landmark_all,
        "matches": state.match_map,
        "anchor_id": state.anchor_id,
        "known_lines": _resect_known_lines(state),
        "line_observations_by_landmark": state.line_observations_by_landmark_all,
        "parallel_pairs": state.parallel_pairs,
        "lock_rotation": state.lock_rotation,
        "lock_translation": state.lock_translation,
        "use_pose_cache": state.use_pose_cache,
        "cancel_check": state.cancel_check,
    }
    recovered: list[str] = []
    for match_id in list(state.skipped_unregistered):
        _check_cancelled(state.cancel_check)
        failed_candidates: list[SimilarityTransform] = []
        solved = _try_register_against_cloud(
            match_id,
            cloud,
            best_candidate_out=failed_candidates,
            initial_similarity=state.peeled_similarities.get(match_id),
            **retry_kwargs,
        )
        ground_candidates: list[SimilarityTransform] = []
        if solved is None:
            solved = _try_register_against_cloud(
                match_id,
                ground_cloud,
                best_candidate_out=ground_candidates,
                **retry_kwargs,
            )
        if solved is None:
            if match_id not in state.inconsistent_picks:
                mismatches = _resect_mismatch_picks(
                    match_id,
                    cloud,
                    retry_kwargs,
                    failed_candidates[-1] if failed_candidates else None,
                )
                if not mismatches:
                    mismatches = _resect_mismatch_picks(
                        match_id,
                        ground_cloud,
                        retry_kwargs,
                        ground_candidates[-1] if ground_candidates else None,
                    )
                if mismatches:
                    state.inconsistent_picks[match_id] = mismatches
            continue
        state.similarities[match_id] = solved
        recovered.append(match_id)
    recovered_set = set(recovered)
    for match_id in recovered:
        state.inconsistent_picks.pop(match_id, None)
    state.skipped_unregistered = [
        match_id
        for match_id in state.skipped_unregistered
        if match_id not in recovered_set
    ]
    state.free_match_ids.extend(
        match_id for match_id in recovered if match_id not in state.free_match_ids
    )
    state.recovered = recovered
    _expand_landmarks_after_resect(state)
    state.seed_plane_landmarks()
    _rebuild_usable_observations(state)
    _attach_mirror_landmarks(state)
    _rebuild_free_line_segments(state)
    _polish_recovered_poses(state)
    _thaw_recovered_location(state)
    _rebuild_free_line_segments(state)
    _attach_mirror_landmarks(state)
    _rebuild_usable_observations(state)


def solve_landmark_sync(
    matches: list[SyncMatchInput],
    observations: list[SyncObservation],
    *,
    anchor_id: str,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    derived_lines: list[tuple[str, str, str]] | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
    initial_similarities: dict[str, SimilarityTransform] | None = None,
    initial_solution: SyncSolutionSeed | None = None,
    fixed_similarities: dict[str, SimilarityTransform] | None = None,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    use_pose_cache: bool = False,
    ground_slack: float | None = None,
    known_3d_slack: float | None = None,
    mirror_pairs: list[tuple[str, str]] | None = None,
    mirror_plane: tuple[np.ndarray, np.ndarray] | None = None,
    mirror_slack: float | None = None,
    mirror_pair_slack: float | None = None,
    mirror_landmark_id: str | None = None,
    plane_groups: list[tuple[str, str, int]] | None = None,
    plane_slack: float | None = None,
    location_match_ids: set[str] | None = None,
    readonly_match_ids: set[str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SyncSolveResult:
    """Register non-anchor matches from 2D correspondences and/or known 3D.

    Stages: pairwise register → peel cameras above ``ACCEPT_RMSE_PX`` → joint
    BA → peel again → resect skipped stills against frozen 3D (ground tags
    if off-plane picks disagree; frozen Is Mirror Of lines mixed like Known 3D
    lines) → triangulate landmarks now visible in recovered views and PnP
    stills that had no cloud support → pose-only BA of recovered cameras →
    rebuild free 3D lines from every posed camera → report.
    Recovered cameras must not fail the joint RMSE. fy=fx when pixels were
    aspect-stretched. ``ground_slack`` is how far On Ground landmarks may leave
    Z=0 in joint BA (0 pins them when triangulation agrees with the raycast).
    ``known_3d_slack`` is how far Known 3D points may leave their Empty
    (0 pins them; pairwise registration still uses the Empty). On Ground
    Known 3D uses the tighter of the two slacks for Z. ``mirror_pairs`` seed
    joint BA and, after that, mixed resection of a one-view partner line.
    ``fixed_similarities`` keeps those non-anchor
    match transforms unchanged while their observations still constrain the
    solve. ``location_match_ids`` is which cameras' 2D may move 3D (points
    and lines); ``None`` keeps every camera. The Anchor always contributes
    when a set is given. ``readonly_match_ids`` skip pairwise and are
    resected against the frozen cloud. ``mirror_plane`` is ``(point, normal)``;
    ``mirror_slack`` is how far that plane may slide along the normal (the
    Empty is not moved). ``mirror_landmark_id`` uses the current solved point
    as the plane origin. ``plane_groups`` is ``(landmark_id, axis, bucket)``
    with axis X/Y/Z/FREE and bucket 1..10; ``plane_slack`` is how far those
    members may leave the shared plane.
    """
    seed_unchanged = False
    if initial_solution is not None:
        from .request import SyncSolveRequest

        input_values = locals().copy()
        current_request = SyncSolveRequest(**{
            item.name: input_values[item.name] for item in fields(SyncSolveRequest)
        })
        seed_unchanged = initial_solution.evidence_sha256 == current_request.evidence_sha256()
    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Preparing sync graph")
    if ground_slack is None:
        ground_slack = GROUND_SLACK_DEFAULT
    ground_slack = max(float(ground_slack), 0.0)
    if known_3d_slack is None:
        known_3d_slack = KNOWN_3D_SLACK_DEFAULT
    known_3d_slack = max(float(known_3d_slack), 0.0)
    if mirror_slack is None:
        mirror_slack = MIRROR_SLACK_DEFAULT
    mirror_slack = max(float(mirror_slack), 0.0)
    if mirror_pair_slack is None:
        mirror_pair_slack = MIRROR_PAIR_SLACK_DEFAULT
    mirror_pair_slack = max(float(mirror_pair_slack), 0.0)
    if plane_slack is None:
        plane_slack = PLANE_SLACK_DEFAULT
    plane_slack = max(float(plane_slack), 0.0)
    plane_groups = normalize_plane_groups(plane_groups)
    mirror_pairs = _dedupe_mirror_pairs(mirror_pairs)
    if mirror_plane is not None:
        mirror_plane = (
            np.asarray(mirror_plane[0], dtype=np.float64).reshape(3).copy(),
            np.asarray(mirror_plane[1], dtype=np.float64).reshape(3).copy(),
        )
        if float(np.linalg.norm(mirror_plane[1])) < 1.0e-12:
            mirror_plane = None
    if mirror_landmark_id is not None and (
        not isinstance(mirror_landmark_id, str) or not mirror_landmark_id
    ):
        raise ValueError("Mirror reference must identify a point landmark")
    ignored_mirror_pairs = 0
    if mirror_pairs and mirror_plane is None:
        ignored_mirror_pairs = len(mirror_pairs)
        mirror_pairs = []
    known_world = {
        landmark_id: np.asarray(point, dtype=np.float64).reshape(3)
        for landmark_id, point in (known_world or {}).items()
    }
    known_lines = {
        landmark_id: (
            np.asarray(pair[0], dtype=np.float64).reshape(3),
            np.asarray(pair[1], dtype=np.float64).reshape(3),
        )
        for landmark_id, pair in (known_lines or {}).items()
    }
    match_map = {item.match_id: item for item in matches}
    for item in matches:
        _square_pixel_intrinsics_if_stretched(item.calibration)
    identity_result = {item.match_id: SimilarityTransform() for item in matches}
    def _mirror_failure(reason: str) -> SyncSolveResult:
        return SyncSolveResult(
            similarities=identity_result, landmarks={}, mean_reprojection_px=0.0,
            per_match_rmse_px={}, per_landmark_rmse_px={},
            message=f"Mirror reference unavailable: {reason}", success=False,
        )

    if mirror_landmark_id is not None and mirror_plane is None:
        return _mirror_failure("choose a mirror plane normal")
    if anchor_id not in match_map:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="Anchor match is missing",
            success=False,
        )
    fixed_similarities = {
        match_id: SimilarityTransform(
            scale=float(similarity.scale),
            rotation=np.asarray(similarity.rotation, dtype=np.float64)
            .reshape(3, 3)
            .copy(),
            translation=np.asarray(similarity.translation, dtype=np.float64)
            .reshape(3)
            .copy(),
        )
        for match_id, similarity in (fixed_similarities or {}).items()
        if match_id in match_map and match_id != anchor_id
    }

    valid_observations = [
        observation
        for observation in observations
        if observation.match_id in match_map
    ]
    valid_line_observations = [
        observation
        for observation in (line_observations or [])
        if observation.match_id in match_map
    ]
    observations_by_landmark: dict[str, list[SyncObservation]] = {}
    for observation in valid_observations:
        observations_by_landmark.setdefault(observation.landmark_id, []).append(
            observation,
        )
    line_observations_by_landmark: dict[str, list[SyncLineObservation]] = {}
    for observation in valid_line_observations:
        line_observations_by_landmark.setdefault(
            observation.landmark_id, []
        ).append(observation)
    observations_by_landmark_all = {
        landmark_id: list(items)
        for landmark_id, items in observations_by_landmark.items()
    }
    line_observations_by_landmark_all = {
        landmark_id: list(items)
        for landmark_id, items in line_observations_by_landmark.items()
    }
    derived_lines = validate_derived_lines(
        derived_lines,
        set(observations_by_landmark) | set(known_world),
        other_line_ids=set(line_observations_by_landmark) | set(known_lines),
    )
    if mirror_landmark_id is not None:
        if mirror_landmark_id in known_lines or mirror_landmark_id in line_observations_by_landmark:
            return _mirror_failure("select a point landmark, not a line")
        if mirror_landmark_id not in observations_by_landmark and mirror_landmark_id not in known_world:
            return _mirror_failure("the selected point is missing")
        if any(mirror_landmark_id in pair for pair in mirror_pairs):
            return _mirror_failure("the plane point cannot be a member of an active mirror pair")
        location_ids = set(match_map) if location_match_ids is None else set(location_match_ids) | {anchor_id}
        supporting_views = {
            item.match_id for item in observations_by_landmark.get(mirror_landmark_id, ())
            if item.match_id in location_ids and item.match_id not in (readonly_match_ids or ())
        }
        if mirror_landmark_id not in known_world and len(supporting_views) < 2:
            return _mirror_failure("add picks in two location-enabled cameras")

    multi_ids = {
        landmark_id
        for landmark_id, items in observations_by_landmark.items()
        if len({item.match_id for item in items}) >= 2
    }
    known_observed_ids = {
        landmark_id
        for landmark_id in known_world
        if landmark_id in observations_by_landmark
    }
    ground_observed_ids = {
        landmark_id
        for landmark_id, items in observations_by_landmark.items()
        if any(item.on_ground for item in items)
    }
    known_line_metric_ids = {
        landmark_id
        for landmark_id in known_lines
        if any(
            item.match_id != anchor_id
            for item in line_observations_by_landmark.get(landmark_id, [])
        )
    }
    # Free lines seen in ≥3 stills can constrain after two matches register.
    free_line_multi_ids = {
        landmark_id
        for landmark_id, items in line_observations_by_landmark.items()
        if landmark_id not in known_lines
        and len({item.match_id for item in items}) >= 3
    }
    metric_ids = known_observed_ids | ground_observed_ids | known_line_metric_ids
    usable_ids = multi_ids | known_observed_ids | ground_observed_ids
    if (
        len(multi_ids) < 5
        and len(metric_ids) < 3
        and len(known_line_metric_ids) < 3
        and not free_line_multi_ids
    ):
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message=(
                "Need ≥5 point landmarks in two+ matches, ≥3 Known 3D points / "
                "On Ground / Known 3D lines, or line landmarks shared across ≥3 stills"
            ),
            success=False,
        )
    if not usable_ids and not known_line_metric_ids and not free_line_multi_ids:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="No usable landmarks for sync",
            success=False,
        )

    usable_observations = [
        observation
        for observation in valid_observations
        if observation.landmark_id in usable_ids
    ]
    connected = _connected_match_ids(
        anchor_id,
        usable_observations,
        known_world=known_world,
        line_observations=valid_line_observations,
        known_lines=known_lines,
    )
    free_match_ids = sorted(
        match_id for match_id in connected if match_id != anchor_id
    )
    if not free_match_ids:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="No non-anchor matches are connected through landmarks",
            success=False,
        )

    usable_observations = [
        observation
        for observation in usable_observations
        if observation.match_id in connected
    ]
    observations_by_landmark = {}
    for observation in usable_observations:
        observations_by_landmark.setdefault(observation.landmark_id, []).append(
            observation,
        )
    line_observations_by_landmark = {
        landmark_id: [
            item for item in items if item.match_id in connected
        ]
        for landmark_id, items in line_observations_by_landmark.items()
    }
    landmark_ids = sorted(observations_by_landmark.keys())

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Registering cameras")
    seed_similarities, seed_landmarks, seed_segments = _validated_solution_seed(
        initial_solution, match_map, anchor_id,
        observations_by_landmark, set(line_observations_by_landmark),
    )
    registration_seeds = dict(seed_similarities)
    registration_seeds.update(initial_similarities or {})
    registration_seeds.update(fixed_similarities)
    if location_match_ids is None:
        resolved_location = None
    else:
        resolved_location = {
            match_id
            for match_id in location_match_ids
            if match_id in match_map
        }
        resolved_location.add(anchor_id)
    readonly_ids = {
        match_id
        for match_id in (readonly_match_ids or ())
        if match_id in match_map and match_id != anchor_id
    }
    readonly_unlocked = {
        match_id
        for match_id in readonly_ids
        if match_id not in fixed_similarities
    }
    pairwise_ids = [
        match_id for match_id in free_match_ids if match_id not in readonly_unlocked
    ]
    similarities, failure_detail = _register_from_relative_pose(
        anchor_id,
        pairwise_ids,
        observations_by_landmark,
        match_map,
        known_world,
        known_lines=known_lines,
        line_observations_by_landmark=line_observations_by_landmark,
        parallel_pairs=parallel_pairs,
        initial_similarities=registration_seeds,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        use_pose_cache=use_pose_cache,
        cancel_check=cancel_check,
        location_match_ids=resolved_location,
        free_point_graph_only=not (
            known_world or known_lines or line_observations_by_landmark
            or parallel_pairs or mirror_pairs or plane_groups
            or fixed_similarities or readonly_ids or initial_similarities or seed_similarities
            or lock_rotation or lock_translation
            or (resolved_location is not None and resolved_location != set(match_map))
            or any(observation.on_ground for observation in usable_observations)
        ),
        progress_callback=progress_callback,
    )
    if similarities is None:
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message=failure_detail
            or (
                "Could not register every match — add ≥5 well-spread ordinary "
                "point landmarks shared with any registered match, or ≥3 "
                "non-collinear 2D↔3D point picks (Known 3D / reconstructed On "
                "Ground), or ≥3 Known 3D line picks"
            ),
            success=False,
        )

    skipped_unregistered = [
        match_id for match_id in free_match_ids if match_id not in similarities
    ]
    if skipped_unregistered:
        skip = set(skipped_unregistered)
        free_match_ids = [
            match_id for match_id in free_match_ids if match_id not in skip
        ]
        usable_observations = [
            observation
            for observation in usable_observations
            if observation.match_id not in skip
        ]
        observations_by_landmark = {}
        for observation in usable_observations:
            observations_by_landmark.setdefault(observation.landmark_id, []).append(
                observation,
            )
        landmark_ids = sorted(observations_by_landmark.keys())
        line_observations_by_landmark = {
            landmark_id: [
                item for item in items if item.match_id not in skip
            ]
            for landmark_id, items in line_observations_by_landmark.items()
        }
    similarities[anchor_id] = SimilarityTransform()
    state = _SolveState(
        match_map=match_map,
        anchor_id=anchor_id,
        known_world=known_world,
        known_lines=known_lines,
        derived_lines=derived_lines,
        parallel_pairs=parallel_pairs,
        mirror_pairs=mirror_pairs,
        mirror_plane=mirror_plane,
        mirror_slack=mirror_slack,
        mirror_landmark_id=mirror_landmark_id,
        plane_groups=plane_groups,
        plane_slack=plane_slack,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        use_pose_cache=use_pose_cache,
        cancel_check=cancel_check,
        ground_slack=ground_slack,
        known_3d_slack=known_3d_slack,
        identity_result=identity_result,
        valid_observations=valid_observations,
        observations_by_landmark_all=observations_by_landmark_all,
        line_observations_by_landmark_all=line_observations_by_landmark_all,
        observations_by_landmark=observations_by_landmark,
        line_observations_by_landmark=line_observations_by_landmark,
        usable_observations=usable_observations,
        landmark_ids=landmark_ids,
        free_match_ids=free_match_ids,
        fixed_match_ids=set(fixed_similarities) & set(free_match_ids),
        similarities=similarities,
        skipped_unregistered=skipped_unregistered,
        failure_detail=failure_detail or "",
        connected=connected,
        location_match_ids=resolved_location,
        readonly_match_ids=readonly_ids,
        initial_line_segments=seed_segments,
        seed_unchanged=seed_unchanged,
        peeled_similarities={
            match_id: registration_seeds[match_id]
            for match_id in readonly_unlocked if match_id in registration_seeds
        },
    )
    # 1. register (done)  2. peel weak cameras  3. BA  4. peel  5. resect
    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Triangulating landmarks")
    try:
        state.rebuild_landmarks(retained_landmarks=seed_landmarks)
        peeled = _peel_cameras_above_rmse(state)
    except ValueError as error:
        if mirror_landmark_id is None or not str(error).startswith("Mirror reference point lost"):
            raise
        return _mirror_failure("supporting cameras could not be retained; add reliable picks")
    if peeled is not None:
        return peeled
    free_match_ids = state.free_match_ids
    usable_observations = state.usable_observations
    observations_by_landmark = state.observations_by_landmark
    landmark_ids = state.landmark_ids
    line_observations_by_landmark = state.line_observations_by_landmark
    similarities = state.similarities
    skipped_unregistered = state.skipped_unregistered
    landmarks = state.landmarks
    consistent_metric = state.consistent_metric
    line_segments = state.line_segments
    pre_ba_match_rmse = state.pre_ba_match_rmse

    # Soft-downweight severe point outliers, then jointly refine poses + 3D.
    seed_rmse = _point_landmark_rmse_snapshot(
        free_match_ids,
        landmark_ids,
        similarities,
        landmarks,
        anchor_id,
        match_map,
        usable_observations,
    )
    ba_observations, downweighted_ids = _auto_downweight_outlier_observations(
        usable_observations,
        seed_rmse,
        protected_ids={
            landmark_id
            for pair in (mirror_pairs or [])
            for landmark_id in pair
        }
        | {
            observation.landmark_id
            for observation in usable_observations
            if observation.protect_outlier
        },
    )
    ba_observations = _balance_observation_weights(ba_observations, match_map)
    ground_landmark_ids = state.ba_constraint_kwargs()["ground_landmark_ids"]
    # Slack > 0: keep On Ground free so BA can spring Z toward the plane.
    if ground_slack > 1.0e-12:
        fixed_landmark_ids = set(known_world)
    else:
        fixed_landmark_ids = set(known_world) | set(consistent_metric)
    # Slack > 0: thaw Known 3D so BA can spring XYZ toward the Empty
    # while 2D picks pull the point along each still's ray.
    if known_3d_slack > 1.0e-12:
        fixed_landmark_ids -= set(known_world)
    free_landmark_ids = [
        landmark_id
        for landmark_id in landmark_ids
        if landmark_id in landmarks and landmark_id not in fixed_landmark_ids
    ]
    froze_structure = len(free_landmark_ids) > BA_FREE_LANDMARK_LIMIT
    ba_free_landmark_ids = [] if froze_structure else list(free_landmark_ids)
    ba_iterations = 12 if froze_structure else 20
    line_constraints = _collect_ba_line_constraints(
        line_segments,
        known_lines,
        line_observations_by_landmark,
        set(similarities),
    )
    pre_ba_similarities = {
        match_id: SimilarityTransform(
            scale=item.scale,
            rotation=np.array(item.rotation, copy=True),
            translation=np.array(item.translation, copy=True),
        )
        for match_id, item in similarities.items()
    }
    pre_ba_landmarks = {
        landmark_id: point.copy() for landmark_id, point in landmarks.items()
    }
    pre_ba_segments = {
        landmark_id: (point_a.copy(), point_b.copy())
        for landmark_id, (point_a, point_b) in line_segments.items()
    }

    def _mean_rmse(values: dict[str, float]) -> float:
        if not values:
            return 0.0
        squares = [value * value for value in values.values()]
        return float(np.sqrt(np.mean(squares)))

    def _copy_similarities():
        return {
            match_id: SimilarityTransform(
                scale=item.scale,
                rotation=np.array(item.rotation, copy=True),
                translation=np.array(item.translation, copy=True),
            )
            for match_id, item in similarities.items()
        }

    def _copy_landmarks():
        return {
            landmark_id: point.copy() for landmark_id, point in landmarks.items()
        }

    def _copy_segments():
        return {
            landmark_id: (point_a.copy(), point_b.copy())
            for landmark_id, (point_a, point_b) in line_segments.items()
        }

    def _fixed_landmarks_for(free_ids: list[str]) -> dict[str, np.ndarray]:
        if not free_ids:
            return {
                landmark_id: landmarks[landmark_id].copy()
                for landmark_id in landmark_ids
                if landmark_id in landmarks
            }
        free_set = set(free_ids)
        return {
            landmark_id: point.copy()
            for landmark_id, point in landmarks.items()
            if landmark_id in fixed_landmark_ids or landmark_id not in free_set
        }

    def _refresh_free_lines() -> None:
        line_support = {
            key: observations_for_location(items, state.location_match_ids)
            for key, items in line_observations_by_landmark.items()
        }
        for landmark_id, items in line_support.items():
            if landmark_id in known_lines:
                continue
            if not items:
                continue
            anchor_ids = _line_anchor_match_ids(items, state.fixed_match_ids)
            if landmark_id in line_segments:
                point_a, point_b = line_segments[landmark_id]
                direction = point_b - point_a
                span = float(np.linalg.norm(direction))
                if span > 1.0e-9:
                    direction = direction / span
                    if anchor_ids is not None:
                        reconstructed = _reconstruct_line_from_observations(
                            items,
                            similarities,
                            match_map,
                            prefer_match_ids=anchor_ids,
                        )
                        if reconstructed is not None:
                            point, direction = reconstructed
                        else:
                            point = 0.5 * (point_a + point_b)
                    else:
                        point = 0.5 * (point_a + point_b)
                    segment = _finite_segment_from_line_observations(
                        point,
                        direction,
                        items,
                        similarities,
                        match_map,
                        extent_match_ids=anchor_ids,
                    )
                    landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
                    line_segments[landmark_id] = segment
                    continue
            reconstructed = _reconstruct_line_from_observations(
                items,
                similarities,
                match_map,
                prefer_match_ids=anchor_ids,
            )
            if reconstructed is None:
                continue
            point, direction = reconstructed
            segment = _finite_segment_from_line_observations(
                point,
                direction,
                items,
                similarities,
                match_map,
                extent_match_ids=anchor_ids,
            )
            landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
            line_segments[landmark_id] = segment
        _enforce_parallel_line_segments(
            line_segments,
            landmarks,
            parallel_pairs,
            line_support,
            similarities,
            match_map,
            known_lines,
        )
        enforce_plane_line_segments(
            line_segments,
            landmarks,
            plane_groups,
            line_support,
            similarities,
            match_map,
            known_lines,
            plane_slack=plane_slack,
            ground_landmark_ids=ground_landmark_ids,
            excluded_support_ids=state.plane_seeded_ids,
        )
        current_mirror_plane = state.effective_mirror_plane(landmarks)
        if current_mirror_plane is not None and mirror_pairs:
            seed_mirror_line_segments(
                line_segments,
                landmarks,
                line_support,
                similarities,
                match_map,
                mirror_pairs,
                current_mirror_plane[0],
                current_mirror_plane[1],
                known_lines,
                state.fixed_match_ids,
            )
            enforce_mirror_line_segments(
                line_segments,
                landmarks,
                mirror_pairs,
                current_mirror_plane[0],
                current_mirror_plane[1],
                line_support,
                similarities,
                match_map,
                known_lines,
                state.fixed_match_ids,
                parallel_pairs=parallel_pairs,
                line_planes=supported_line_planes(
                    landmarks, line_segments, plane_groups, known_lines,
                    ground_landmark_ids=ground_landmark_ids,
                    excluded_support_ids=state.plane_seeded_ids,
                ) if plane_slack <= 1e-12 else None,
            )

    def _run_ba(
        free_ids: list[str],
        iterations: int,
        *,
        free_plane_offset: bool = False,
    ) -> bool:
        nonlocal similarities, landmarks, line_segments
        _check_cancelled(cancel_check)
        plane_offset_out: list[float] = []
        similarities, landmarks, line_segments, ran = (
            _bundle_adjust_registration(
                free_match_ids,
                free_ids,
                _fixed_landmarks_for(free_ids),
                similarities,
                landmarks,
                anchor_id,
                match_map,
                ba_observations,
                line_constraints,
                known_line_ids=set(known_lines),
                line_segments=line_segments,
                lock_rotation=lock_rotation,
                lock_translation=lock_translation,
                fixed_similarities={
                    match_id: similarities[match_id]
                    for match_id in state.fixed_match_ids
                    if match_id in similarities
                },
                max_iterations=iterations,
                **state.ba_constraint_kwargs(),
                free_plane_offset=free_plane_offset,
                plane_offset_start=state.mirror_offset if mirror_landmark_id else 0.0,
                plane_offset_out=plane_offset_out,
                location_match_ids=state.location_match_ids,
            )
        )
        if ran:
            if mirror_landmark_id:
                state.mirror_offset = plane_offset_out[0]
            _refresh_free_lines()
        return ran

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Bundle adjustment")
    use_plane_offset = (
        mirror_slack > 1.0e-12
        and bool(mirror_pairs)
        and mirror_plane is not None
    )
    # Mirror pairs are extra joint-BA pull, not a pairwise seed. Slack > 0
    # then thaws δ with non-mirror 3D frozen so a biased Empty cannot drag the
    # cloud.
    did_bundle_adjust = _run_ba(
        ba_free_landmark_ids,
        ba_iterations,
        free_plane_offset=use_plane_offset,
    )

    post_ba_match_rmse = _per_match_rmse_snapshot(
        free_match_ids,
        landmark_ids,
        similarities,
        landmarks,
        anchor_id,
        match_map,
        usable_observations,
    )

    if did_bundle_adjust and _mean_rmse(post_ba_match_rmse) > max(
        BA_ACCEPT_RMSE_FLOOR_PX, _mean_rmse(pre_ba_match_rmse) + BA_ACCEPT_RMSE_SLACK_PX
    ):
        similarities = pre_ba_similarities
        landmarks = pre_ba_landmarks
        line_segments = pre_ba_segments
        state.mirror_offset = 0.0
        did_bundle_adjust = False
        post_ba_match_rmse = pre_ba_match_rmse
    elif did_bundle_adjust and froze_structure:
        rmse_a = _mean_rmse(post_ba_match_rmse)
        pass_a_similarities = _copy_similarities()
        pass_a_landmarks = _copy_landmarks()
        pass_a_segments = _copy_segments()
        pass_a_offset = state.mirror_offset
        if _run_ba(
            list(free_landmark_ids),
            12,
            free_plane_offset=use_plane_offset,
        ):
            rmse_b = _mean_rmse(
                _per_match_rmse_snapshot(
                    free_match_ids,
                    landmark_ids,
                    similarities,
                    landmarks,
                    anchor_id,
                    match_map,
                    usable_observations,
                )
            )
            if rmse_b <= rmse_a + 1.0e-6:
                pass_b_similarities = _copy_similarities()
                pass_b_landmarks = _copy_landmarks()
                pass_b_segments = _copy_segments()
                pass_b_offset = state.mirror_offset
                if _run_ba([], 8, free_plane_offset=use_plane_offset):
                    rmse_c = _mean_rmse(
                        _per_match_rmse_snapshot(
                            free_match_ids,
                            landmark_ids,
                            similarities,
                            landmarks,
                            anchor_id,
                            match_map,
                            usable_observations,
                        )
                    )
                    if rmse_c > rmse_b + 1.0e-6:
                        similarities = pass_b_similarities
                        landmarks = pass_b_landmarks
                        line_segments = pass_b_segments
                        state.mirror_offset = pass_b_offset
                post_ba_match_rmse = _per_match_rmse_snapshot(
                    free_match_ids,
                    landmark_ids,
                    similarities,
                    landmarks,
                    anchor_id,
                    match_map,
                    usable_observations,
                )
            else:
                similarities = pass_a_similarities
                landmarks = pass_a_landmarks
                line_segments = pass_a_segments
                state.mirror_offset = pass_a_offset
                post_ba_match_rmse = {
                    match_id: rmse_a for match_id in post_ba_match_rmse
                }
                post_ba_match_rmse = _per_match_rmse_snapshot(
                    free_match_ids,
                    landmark_ids,
                    similarities,
                    landmarks,
                    anchor_id,
                    match_map,
                    usable_observations,
                )

    mirror_free_ids = sorted(
        {
            landmark_id
            for pair in (mirror_pairs or [])
            for landmark_id in pair
            if landmark_id in landmarks
        }
    )
    if use_plane_offset and mirror_free_ids:
        thaw_rmse = _mean_rmse(post_ba_match_rmse)
        thaw_similarities = _copy_similarities()
        thaw_landmarks = _copy_landmarks()
        thaw_segments = _copy_segments()
        thaw_offset = state.mirror_offset
        if _run_ba(
            mirror_free_ids,
            12,
            free_plane_offset=True,
        ):
            thawed_rmse = _mean_rmse(
                _per_match_rmse_snapshot(
                    free_match_ids,
                    landmark_ids,
                    similarities,
                    landmarks,
                    anchor_id,
                    match_map,
                    usable_observations,
                )
            )
            if thawed_rmse > thaw_rmse + 2.0:
                similarities = thaw_similarities
                landmarks = thaw_landmarks
                line_segments = thaw_segments
                state.mirror_offset = thaw_offset
            else:
                post_ba_match_rmse = _per_match_rmse_snapshot(
                    free_match_ids,
                    landmark_ids,
                    similarities,
                    landmarks,
                    anchor_id,
                    match_map,
                    usable_observations,
                )

    state.similarities = similarities
    state.landmarks = landmarks
    state.line_segments = line_segments
    state.free_match_ids = free_match_ids
    state.usable_observations = usable_observations
    state.observations_by_landmark = observations_by_landmark
    state.line_observations_by_landmark = line_observations_by_landmark
    state.landmark_ids = landmark_ids
    state.did_bundle_adjust = bool(did_bundle_adjust)
    if mirror_landmark_id and did_bundle_adjust:
        _attach_mirror_landmarks(state)

    weak_after = [
        match_id
        for match_id, rmse in sorted(
            (
                (match_id, post_ba_match_rmse.get(match_id, 0.0))
                for match_id in free_match_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if rmse > ACCEPT_RMSE_PX and match_id not in state.fixed_match_ids
    ][:1]
    if weak_after:
        _record_mismatch_picks(state, weak_after[0])
        state.drop_matches(weak_after)
        if not state.free_match_ids:
            return SyncSolveResult(
                similarities=identity_result,
                landmarks={},
                mean_reprojection_px=_mean_rmse(post_ba_match_rmse),
                per_match_rmse_px=post_ba_match_rmse,
                per_landmark_rmse_px={},
                message=(
                    f"Sync rejected — every non-anchor camera stayed above "
                    f"{ACCEPT_RMSE_PX:.0f} px. "
                    "Uncheck On Ground on off-plane landmarks, or re-pick "
                    "the worst landmarks"
                ),
                success=False,
            )
        try:
            state.rebuild_landmarks()
        except ValueError as error:
            if mirror_landmark_id is None or not str(error).startswith("Mirror reference point lost"):
                raise
            return _mirror_failure("supporting cameras could not be retained; add reliable picks")

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Retrying skipped cameras")
    try:
        _resect_skipped_matches(state)
    except ValueError as error:
        if mirror_landmark_id is None or not str(error).startswith("Mirror reference point lost"):
            raise
        return _mirror_failure("supporting cameras could not be retained; add reliable picks")
    similarities = state.similarities
    landmarks = state.landmarks
    line_segments = state.line_segments
    free_match_ids = state.free_match_ids
    usable_observations = state.usable_observations
    observations_by_landmark = state.observations_by_landmark
    line_observations_by_landmark = state.line_observations_by_landmark
    landmark_ids = state.landmark_ids
    skipped_unregistered = state.skipped_unregistered
    recovered = state.recovered
    consistent_metric = state.consistent_metric

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Final diagnostics")
    residual_landmark_ids = [
        landmark_id for landmark_id in landmark_ids if landmark_id in landmarks
    ]
    residual_observations = [
        observation
        for observation in usable_observations
        if observation.landmark_id in landmarks
    ]
    residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {landmark_id: landmarks[landmark_id] for landmark_id in residual_landmark_ids},
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        match_map,
        residual_observations,
        weighted=False,
    )
    weighted_residuals = _residual_vector(
        _pack_params(
            free_match_ids,
            residual_landmark_ids,
            {match_id: similarities[match_id] for match_id in free_match_ids},
            {landmark_id: landmarks[landmark_id] for landmark_id in residual_landmark_ids},
        ),
        free_match_ids,
        residual_landmark_ids,
        anchor_id,
        match_map,
        residual_observations,
        weighted=True,
    )
    per_match_sse: dict[str, list[float]] = {
        match_id: [] for match_id in similarities
    }
    per_landmark_sse: dict[str, list[float]] = {
        landmark_id: [] for landmark_id in residual_landmark_ids
    }
    recovered_set = set(recovered)
    residual_index = 0
    joint_point_sse: list[float] = []
    joint_weighted_sse: list[float] = []
    for observation in residual_observations:
        error_u = float(residuals[residual_index])
        error_v = float(residuals[residual_index + 1])
        weighted_u = (
            float(weighted_residuals[residual_index])
            if weighted_residuals.size
            else error_u
        )
        weighted_v = (
            float(weighted_residuals[residual_index + 1])
            if weighted_residuals.size
            else error_v
        )
        residual_index += 2
        squared = error_u * error_u + error_v * error_v
        per_match_sse[observation.match_id].append(squared)
        per_landmark_sse[observation.landmark_id].append(squared)
        # A still recovered after peel can miss off-plane features; keep that
        # in per-match Diagnose, not in the joint accept/reject RMSE.
        if observation.match_id not in recovered_set:
            joint_point_sse.append(squared)
            joint_weighted_sse.append(weighted_u * weighted_u)
            joint_weighted_sse.append(weighted_v * weighted_v)

    # Pose quality = point residuals only. Line px (esp. after Parallel lock)
    # is diagnostic and must not reject a good camera solve.
    point_sse = joint_point_sse
    per_match_point_sse = {
        match_id: list(values) for match_id, values in per_match_sse.items()
    }
    weighted_point_sse = joint_weighted_sse

    parallel_landmark_ids: set[str] = set()
    for landmark_a, landmark_b in parallel_pairs or ():
        parallel_landmark_ids.add(landmark_a)
        parallel_landmark_ids.add(landmark_b)

    line_error_values: list[float] = []
    weighted_line_error_values: list[float] = []
    per_line_sse: dict[str, list[float]] = {}
    per_match_line_sse: dict[str, list[float]] = {}
    for landmark_id, items in line_observations_by_landmark.items():
        if landmark_id in line_segments:
            point_a, point_b = line_segments[landmark_id]
            direction = point_b - point_a
            span = float(np.linalg.norm(direction))
            if span < 1.0e-9:
                continue
            direction = direction / span
            point = 0.5 * (point_a + point_b)
        elif landmark_id in known_lines:
            point_a, point_b = known_lines[landmark_id]
            direction = point_b - point_a
            span = float(np.linalg.norm(direction))
            if span < 1.0e-9:
                continue
            direction = direction / span
            point = point_a
        else:
            reconstructed = _reconstruct_line_from_observations(
                observations_for_location(items, state.location_match_ids),
                similarities,
                match_map,
            )
            if reconstructed is None:
                continue
            point, direction = reconstructed
        for observation in items:
            similarity = similarities.get(observation.match_id)
            match = match_map.get(observation.match_id)
            if similarity is None or match is None:
                continue
            raw = _line_observation_reprojection_errors(
                point,
                direction,
                SyncLineObservation(
                    match_id=observation.match_id,
                    landmark_id=observation.landmark_id,
                    u1=observation.u1,
                    v1=observation.v1,
                    u2=observation.u2,
                    v2=observation.v2,
                    weight=1.0,
                ),
                match.calibration,
                similarity,
            )
            weighted = _line_observation_reprojection_errors(
                point,
                direction,
                observation,
                match.calibration,
                similarity,
            )
            for value in raw:
                squared = value * value
                line_error_values.append(squared)
                per_line_sse.setdefault(landmark_id, []).append(squared)
                # Keep lines visible in per-landmark Diagnose, not in pose reject.
                per_landmark_sse.setdefault(observation.landmark_id, []).append(squared)
                per_match_sse.setdefault(observation.match_id, []).append(squared)
                per_match_line_sse.setdefault(observation.match_id, []).append(squared)
            weighted_line_error_values.extend(value * value for value in weighted)

    # Parallel pairs: report residual angle after enforcement (should be ~0°).
    parallel_angles_deg: list[float] = []
    for landmark_a, landmark_b in parallel_pairs or ():
        direction_a = None
        direction_b = None
        if landmark_a in line_segments:
            point_a, point_b = line_segments[landmark_a]
            direction_a = point_b - point_a
        if landmark_b in line_segments:
            point_a, point_b = line_segments[landmark_b]
            direction_b = point_b - point_a
        if direction_a is None or direction_b is None:
            continue
        parallel_error = _parallel_direction_error(direction_a, direction_b)
        parallel_angles_deg.append(float(np.degrees(np.arcsin(min(parallel_error, 1.0)))))

    def _rmse(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.sqrt(np.mean(values)))

    # Keep the camera-level number consistent with pose acceptance and the
    # headline RMSE. A badly drawn free line remains visible on that landmark,
    # but no longer makes an otherwise good camera report hundreds of pixels.
    per_match_source = per_match_point_sse if point_sse else per_match_sse
    per_match_rmse = {
        match_id: _rmse(values) for match_id, values in per_match_source.items()
    }
    per_landmark_rmse = {
        landmark_id: _rmse(values) for landmark_id, values in per_landmark_sse.items()
    }
    per_line_rmse = {
        landmark_id: _rmse(values) for landmark_id, values in per_line_sse.items()
    }
    mean_rmse = _rmse(point_sse) if point_sse else _rmse(line_error_values)
    mean_weighted_rmse = (
        _rmse(weighted_point_sse) if weighted_point_sse else mean_rmse
    )
    names = _landmark_names(observations_by_landmark)
    for landmark_id, items in line_observations_by_landmark.items():
        for item in items:
            if item.landmark_name:
                names[landmark_id] = item.landmark_name
                break
        names.setdefault(landmark_id, landmark_id[:8])
    if mean_rmse > ACCEPT_RMSE_PX and point_sse:
        # Worst among points — line Parallel miss is not a pose failure.
        point_only_rmse = {
            landmark_id: rmse
            for landmark_id, rmse in per_landmark_rmse.items()
            if landmark_id in residual_landmark_ids
        }
        worst = _format_worst_landmarks(point_only_rmse, names)
        hint = (
            "Re-pick the worst landmarks on both stills. "
            "If several ordinary landmarks are all high, FOV/VP may be off."
        )
        message = (
            f"Sync rejected (reproj {mean_rmse:.0f} px, "
            f"weighted {mean_weighted_rmse:.0f} px)."
        )
        if worst:
            message += f" {worst}."
        message += f" {hint}"
        return SyncSolveResult(
            similarities=identity_result,
            landmarks={},
            mean_reprojection_px=mean_rmse,
            per_match_rmse_px=per_match_rmse,
            per_landmark_rmse_px=per_landmark_rmse,
            message=message,
            success=False,
        )
    disconnected = sorted(set(match_map) - connected)
    known_count = sum(1 for landmark_id in known_world if landmark_id in landmarks)
    known_line_count = len(known_lines)
    ground_count = sum(
        1
        for landmark_id, items in observations_by_landmark.items()
        if any(item.on_ground for item in items) and landmark_id not in known_world
    )
    free_line_count = sum(
        1
        for landmark_id in line_observations_by_landmark
        if landmark_id not in known_lines
    )
    message = (
        f"Synced {len(free_match_ids)} match(es) · {len(landmarks)} landmarks · "
        f"RMSE {mean_rmse:.2f} px"
    )
    if state.fixed_match_ids:
        message += f" · {len(state.fixed_match_ids)} pose locked"
    if state.readonly_match_ids:
        message += f" · {len(state.readonly_match_ids)} fit only"
    constraint_bits = []
    if known_count:
        constraint_bits.append(f"{known_count} known 3D")
    if known_line_count:
        constraint_bits.append(f"{known_line_count} known lines")
    if ground_count:
        constraint_bits.append(f"{ground_count} ground")
    if free_line_count:
        constraint_bits.append(f"{free_line_count} free lines")
    if parallel_pairs:
        constraint_bits.append(f"{len(parallel_pairs)} parallel")
    if mirror_pairs:
        constraint_bits.append(f"{len(mirror_pairs)} mirror")
    plane_count = active_plane_group_count(
        plane_groups, landmarks, line_segments
    )
    if plane_count:
        constraint_bits.append(f"{plane_count} plane")
    if constraint_bits:
        message += " · constraints: " + " + ".join(constraint_bits)
    if mirror_pairs and mirror_landmark_id:
        message += " · mirror plane follows point"
    if did_bundle_adjust:
        message += " · joint BA"
        if froze_structure:
            message += " (thaw 3D)"
    if state.kept_joint_geometry:
        message += " · kept existing geometry after camera recovery"
    plane_seeded_ids = sorted(
        key for key in state.plane_seeded_ids if key in landmarks and len([
            observation for observation in observations_for_location(
                state.observations_by_landmark_all.get(key, []), state.location_match_ids
            ) if observation.match_id in similarities
        ]) == 1
    )
    if plane_seeded_ids:
        noun = "point" if len(plane_seeded_ids) == 1 else "points"
        message += f" · {len(plane_seeded_ids)} {noun} from plane + one view"
    if ground_slack > 1.0e-12:
        drifted = []
        for landmark_id, items in observations_by_landmark.items():
            if landmark_id not in landmarks:
                continue
            if not any(item.on_ground for item in items):
                continue
            z_slack = ground_slack
            if landmark_id in known_world and known_3d_slack > 1.0e-12:
                z_slack = min(ground_slack, known_3d_slack)
            height = abs(float(landmarks[landmark_id][2]))
            if height > z_slack:
                drifted.append(
                    (names.get(landmark_id, landmark_id[:8]), height)
                )
        drifted.sort(key=lambda item: -item[1])
        if drifted:
            bits = [f"{name} Z={height:.3f}" for name, height in drifted[:4]]
            message += (
                f" · ground slack {ground_slack:g} exceeded: " + ", ".join(bits)
            )
    if known_3d_slack > 1.0e-12:
        drifted = []
        for landmark_id, origin in known_world.items():
            point = landmarks.get(landmark_id)
            if point is None:
                continue
            offset = float(np.linalg.norm(point - origin))
            if offset > known_3d_slack:
                drifted.append(
                    (names.get(landmark_id, landmark_id[:8]), offset)
                )
        drifted.sort(key=lambda item: -item[1])
        if drifted:
            bits = [f"{name} Δ={offset:.3f}" for name, offset in drifted[:4]]
            message += (
                f" · known 3D slack {known_3d_slack:g} exceeded: "
                + ", ".join(bits)
            )
    if ignored_mirror_pairs:
        message += (
            f" · {ignored_mirror_pairs} mirror pair(s) ignored — no Mirror Empty"
        )
    if (
        mirror_slack > 1.0e-12
        and mirror_plane is not None
        and mirror_pairs
    ):
        if mirror_landmark_id:
            offset = state.mirror_offset
        else:
            offset = mirror_plane_offset(
                landmarks, mirror_pairs, mirror_plane[0], mirror_plane[1],
            )
        if offset is not None and abs(offset) > mirror_slack:
            message += (
                f" · mirror slack {mirror_slack:g} exceeded: "
                f"plane Δ={offset:.3f}"
            )
    plane_drifted = plane_slack_excesses(
        landmarks,
        plane_groups,
        plane_slack,
        line_segments=line_segments,
        ground_landmark_ids=ground_landmark_ids,
        names=names,
    )
    if plane_drifted:
        bits = [
            f"{name} d={distance:.3f}" for name, distance in plane_drifted[:4]
        ]
        message += (
            f" · plane slack {plane_slack:g} exceeded: " + ", ".join(bits)
        )
    if recovered:
        recovered_list = ", ".join(f"'{name}'" for name in sorted(recovered))
        message += f" · recovered {recovered_list} after joint lock"
    if downweighted_ids:
        message += f" · downweighted {len(downweighted_ids)} outlier(s)"
    if parallel_angles_deg:
        mean_angle = float(np.mean(parallel_angles_deg))
        message += f" · parallel Δ {mean_angle:.1f}°"
        if mean_angle > 1.0:
            message += " (could not lock family — redraw / Known 3D)"
        else:
            message += " (direction locked)"
    if disconnected:
        message += f" · skipped {len(disconnected)} disconnected"
    if skipped_unregistered:
        skip_bits = []
        for match_id in sorted(skipped_unregistered):
            picks = state.inconsistent_picks.get(match_id)
            if picks:
                skip_bits.append(f"'{match_id}' ({_mismatch_reason(picks)})")
            elif match_id in state.skip_notes:
                skip_bits.append(f"'{match_id}' ({state.skip_notes[match_id]})")
            else:
                skip_bits.append(f"'{match_id}'")
        message += " · skipped " + ", ".join(skip_bits)
        unnamed = [
            match_id
            for match_id in skipped_unregistered
            if match_id not in state.inconsistent_picks
            and match_id not in state.skip_notes
        ]
        if unnamed and failure_detail and any(
            match_id in failure_detail for match_id in unnamed
        ):
            message += f" ({failure_detail})"
    # Soft warn: accepted but likely inaccurate picks or intrinsics.
    if mean_rmse > 8.0:
        point_only_rmse = {
            landmark_id: rmse
            for landmark_id, rmse in per_landmark_rmse.items()
            if landmark_id in residual_landmark_ids
        }
        worst = _format_worst_landmarks(point_only_rmse, names)
        message += " · WARN high error"
        if worst:
            message += f" ({worst})"
        message += " — check picks / FOV"
    # Line miss after Parallel is expected when drawings disagree with the lock.
    high_parallel_lines = [
        (names.get(landmark_id, landmark_id), rmse)
        for landmark_id, rmse in per_line_rmse.items()
        if landmark_id in parallel_landmark_ids and rmse > 20.0
    ]
    high_parallel_lines.sort(key=lambda item: item[1], reverse=True)
    if high_parallel_lines:
        bits = ", ".join(
            f"{name} {rmse:.0f}px" for name, rmse in high_parallel_lines[:3]
        )
        message += (
            f" · parallel line miss ({bits}) — 2D drawings vs locked 3D direction"
        )

    support_angles = line_support_angles(
        line_segments,
        {key: observations_for_location(items, state.location_match_ids)
         for key, items in line_observations_by_landmark.items()},
        similarities, match_map,
        known_lines=known_lines, mirror_pairs=mirror_pairs,
        mirror_normal=mirror_plane[1] if mirror_plane is not None else None,
        support_planes=supported_line_planes(
            landmarks,line_segments,plane_groups,known_lines,
            ground_landmark_ids=ground_landmark_ids,
            excluded_support_ids=state.plane_seeded_ids,
        ) if plane_slack <= 1e-12 else None,
    )
    weak_line_ids = sorted(key for key, angle in support_angles.items()
                           if float(np.sin(np.radians(angle))) < LINE_PLANE_MIN_SINE)
    if weak_line_ids:
        weak_names = ", ".join(names.get(key,key) for key in weak_line_ids[:3])
        message += (f" · weak 3D line support ({weak_names}) — "
                    "small stroke edits may move these lines; add a distinct view or longer strokes")

    try:
        derived_segments, _geometry = derived_line_geometry(
            landmarks, state.derived_lines)
    except ValueError as exc:
        return SyncSolveResult(
            similarities=similarities, landmarks=landmarks,
            mean_reprojection_px=mean_rmse,
            per_match_rmse_px=per_match_rmse,
            per_landmark_rmse_px=per_landmark_rmse,
            message=str(exc), success=False,
            line_segments=line_segments,
        )
    line_segments.update(derived_segments)
    for line_id, segment in derived_segments.items():
        landmarks[line_id] = 0.5 * (segment[0] + segment[1])

    return SyncSolveResult(
        similarities=similarities,
        landmarks=landmarks,
        mean_reprojection_px=mean_rmse,
        per_match_rmse_px=per_match_rmse,
        point_rmse_px=_rmse(point_sse) if point_sse else None,
        line_rmse_px=_rmse(line_error_values) if line_error_values else None,
        per_match_point_rmse_px={key: _rmse(values) for key, values in per_match_point_sse.items() if values},
        per_match_line_rmse_px={key: _rmse(values) for key, values in per_match_line_sse.items() if values},
        per_landmark_rmse_px=per_landmark_rmse,
        message=message,
        success=True,
        line_segments=line_segments,
        line_support_angles_deg=support_angles,
        weak_line_ids=weak_line_ids,
        plane_seeded_landmark_ids=plane_seeded_ids,
        downweighted_landmark_ids=downweighted_ids,
        bundle_adjusted=bool(did_bundle_adjust),
        joint_mirror_offset_m=float(state.mirror_offset),
        inconsistent_picks=[
            (match_id, name, error)
            for match_id, picks in state.inconsistent_picks.items()
            if match_id in skipped_unregistered
            for _landmark_id, name, error in picks
        ],
    )
