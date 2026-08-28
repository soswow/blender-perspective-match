"""Top-level landmark-graph solve: register, peel, BA, resect skipped."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

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
)
from .constants import (
    ACCEPT_RMSE_PX,
    BA_FREE_LANDMARK_LIMIT,
    GROUND_PLANE_Z_FRACTION,
    GROUND_SLACK_DEFAULT,
    RESECT_MISMATCH_CANDIDATE_LIMIT,
)
from .lines import (
    _enforce_parallel_line_segments,
    _finite_segment_from_line_observations,
    _line_observation_reprojection_errors,
    _parallel_direction_error,
    _reconstruct_line_from_observations,
)
from .pose import (
    _consistent_metric_landmarks,
    _format_worst_landmarks,
    _landmark_names,
    _metric_landmarks,
    _metric_pnp_correspondences,
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
    lock_rotation: bool
    lock_translation: bool
    use_pose_cache: bool
    cancel_check: Callable[[], bool] | None
    ground_slack: float
    identity_result: dict[str, SimilarityTransform]
    valid_observations: list[SyncObservation]
    observations_by_landmark_all: dict[str, list[SyncObservation]]
    line_observations_by_landmark_all: dict[str, list[SyncLineObservation]]
    observations_by_landmark: dict[str, list[SyncObservation]]
    line_observations_by_landmark: dict[str, list[SyncLineObservation]]
    usable_observations: list[SyncObservation]
    landmark_ids: list[str]
    free_match_ids: list[str]
    similarities: dict[str, SimilarityTransform]
    skipped_unregistered: list[str]
    failure_detail: str
    connected: set[str]
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    consistent_metric: dict[str, np.ndarray] = field(default_factory=dict)
    line_segments: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    recovered: list[str] = field(default_factory=list)
    skip_notes: dict[str, str] = field(default_factory=dict)
    downweighted_ids: list[str] = field(default_factory=list)
    did_bundle_adjust: bool = False
    pre_ba_match_rmse: dict[str, float] = field(default_factory=dict)
    inconsistent_picks: dict[str, list[tuple[str, str, float]]] = field(
        default_factory=dict
    )

    def drop_matches(self, match_ids: list[str]) -> None:
        """Remove cameras from the joint graph and remember them as skipped."""
        if not match_ids:
            return
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
            self.similarities.pop(match_id, None)
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

    def rebuild_landmarks(self) -> None:
        """Triangulate points/lines; pin Known 3D and consistent On Ground."""
        rebuilt = _triangulate_landmarks(
            self.landmark_ids,
            self.observations_by_landmark,
            self.similarities,
            self.match_map,
        )
        metric_points = _metric_landmarks(
            self.observations_by_landmark,
            self.anchor_id,
            self.match_map[self.anchor_id].calibration,
            self.known_world,
        )
        consistent = _consistent_metric_landmarks(
            metric_points,
            rebuilt,
            set(self.known_world),
            ground_slack=self.ground_slack,
        )
        rebuilt.update(consistent)
        for landmark_id, point in self.known_world.items():
            rebuilt[landmark_id] = point
        segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for landmark_id, (point_a, point_b) in self.known_lines.items():
            rebuilt[landmark_id] = 0.5 * (point_a + point_b)
            segments[landmark_id] = (point_a.copy(), point_b.copy())
        for landmark_id, items in self.line_observations_by_landmark.items():
            if landmark_id in segments:
                continue
            reconstructed = _reconstruct_line_from_observations(
                items, self.similarities, self.match_map
            )
            if reconstructed is None:
                continue
            point, direction = reconstructed
            segment = _finite_segment_from_line_observations(
                point, direction, items, self.similarities, self.match_map
            )
            rebuilt[landmark_id] = 0.5 * (segment[0] + segment[1])
            segments[landmark_id] = segment
        _enforce_parallel_line_segments(
            segments,
            rebuilt,
            self.parallel_pairs,
            self.line_observations_by_landmark,
            self.similarities,
            self.match_map,
            self.known_lines,
        )
        self.landmarks = rebuilt
        self.consistent_metric = consistent
        self.line_segments = segments


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
    )
    for landmark_id, point in refined.items():
        if landmark_id not in state.landmarks or landmark_id in added_ids:
            state.landmarks[landmark_id] = point


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


def _resect_skipped_matches(state: _SolveState) -> None:
    """PnP skipped stills against the frozen cloud; ground-only if that fails."""
    if not state.landmarks:
        return
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
        "known_lines": state.known_lines,
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
    _rebuild_usable_observations(state)


def solve_landmark_sync(
    matches: list[SyncMatchInput],
    observations: list[SyncObservation],
    *,
    anchor_id: str,
    known_world: dict[str, np.ndarray] | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
    initial_similarities: dict[str, SimilarityTransform] | None = None,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    use_pose_cache: bool = False,
    ground_slack: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SyncSolveResult:
    """Register non-anchor matches from 2D correspondences and/or known 3D.

    Stages: pairwise register → peel cameras above ``ACCEPT_RMSE_PX`` → joint
    BA → peel again → resect skipped stills against frozen 3D (ground tags
    if off-plane picks disagree) → triangulate landmarks now visible in
    recovered views and PnP stills that had no cloud support → report.
    Recovered cameras must not fail the joint RMSE. fy=fx when pixels were
    aspect-stretched. ``ground_slack`` is how far On Ground landmarks may leave
    Z=0 in joint BA (0 pins them when triangulation agrees with the raycast).
    """
    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Preparing sync graph")
    if ground_slack is None:
        ground_slack = GROUND_SLACK_DEFAULT
    ground_slack = max(float(ground_slack), 0.0)
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
    similarities, failure_detail = _register_from_relative_pose(
        anchor_id,
        free_match_ids,
        observations_by_landmark,
        match_map,
        known_world,
        known_lines=known_lines,
        line_observations_by_landmark=line_observations_by_landmark,
        parallel_pairs=parallel_pairs,
        initial_similarities=initial_similarities,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        use_pose_cache=use_pose_cache,
        cancel_check=cancel_check,
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
                "Could not register every match — need ≥5 well-spread 2D "
                "landmarks or ≥3 Known 3D / On Ground / Known 3D line picks"
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
        parallel_pairs=parallel_pairs,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        use_pose_cache=use_pose_cache,
        cancel_check=cancel_check,
        ground_slack=ground_slack,
        identity_result=identity_result,
        valid_observations=valid_observations,
        observations_by_landmark_all=observations_by_landmark_all,
        line_observations_by_landmark_all=line_observations_by_landmark_all,
        observations_by_landmark=observations_by_landmark,
        line_observations_by_landmark=line_observations_by_landmark,
        usable_observations=usable_observations,
        landmark_ids=landmark_ids,
        free_match_ids=free_match_ids,
        similarities=similarities,
        skipped_unregistered=skipped_unregistered,
        failure_detail=failure_detail or "",
        connected=connected,
    )
    # 1. register (done)  2. peel weak cameras  3. BA  4. peel  5. resect
    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Triangulating landmarks")
    state.rebuild_landmarks()
    peeled = _peel_cameras_above_rmse(state)
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
    )
    ba_observations = _balance_observation_weights(ba_observations, match_map)
    ground_landmark_ids = sorted(
        {
            observation.landmark_id
            for observation in ba_observations
            if observation.on_ground and observation.landmark_id not in known_world
        }
    )
    # Slack > 0: keep On Ground free so BA can spring Z toward the plane.
    if ground_slack > 1.0e-12:
        fixed_landmark_ids = set(known_world)
    else:
        fixed_landmark_ids = set(known_world) | set(consistent_metric)
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
        for landmark_id, items in line_observations_by_landmark.items():
            if landmark_id in known_lines:
                continue
            if landmark_id in line_segments:
                point_a, point_b = line_segments[landmark_id]
                direction = point_b - point_a
                span = float(np.linalg.norm(direction))
                if span > 1.0e-9:
                    direction = direction / span
                    segment = _finite_segment_from_line_observations(
                        0.5 * (point_a + point_b),
                        direction,
                        items,
                        similarities,
                        match_map,
                    )
                    landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
                    line_segments[landmark_id] = segment
                    continue
            reconstructed = _reconstruct_line_from_observations(
                items, similarities, match_map
            )
            if reconstructed is None:
                continue
            point, direction = reconstructed
            segment = _finite_segment_from_line_observations(
                point, direction, items, similarities, match_map
            )
            landmarks[landmark_id] = 0.5 * (segment[0] + segment[1])
            line_segments[landmark_id] = segment
        _enforce_parallel_line_segments(
            line_segments,
            landmarks,
            parallel_pairs,
            line_observations_by_landmark,
            similarities,
            match_map,
            known_lines,
        )

    def _run_ba(free_ids: list[str], iterations: int) -> bool:
        nonlocal similarities, landmarks, line_segments
        _check_cancelled(cancel_check)
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
                max_iterations=iterations,
                ground_landmark_ids=ground_landmark_ids,
                ground_slack=ground_slack,
            )
        )
        if ran:
            _refresh_free_lines()
        return ran

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Bundle adjustment")
    did_bundle_adjust = _run_ba(ba_free_landmark_ids, ba_iterations)

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
        8.0, _mean_rmse(pre_ba_match_rmse) + 2.0
    ):
        similarities = pre_ba_similarities
        landmarks = pre_ba_landmarks
        line_segments = pre_ba_segments
        did_bundle_adjust = False
        post_ba_match_rmse = pre_ba_match_rmse
    elif did_bundle_adjust and froze_structure:
        rmse_a = _mean_rmse(post_ba_match_rmse)
        pass_a_similarities = _copy_similarities()
        pass_a_landmarks = _copy_landmarks()
        pass_a_segments = _copy_segments()
        if _run_ba(list(free_landmark_ids), 12):
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
                if _run_ba([], 8):
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


    state.similarities = similarities
    state.landmarks = landmarks
    state.line_segments = line_segments
    state.free_match_ids = free_match_ids
    state.usable_observations = usable_observations
    state.observations_by_landmark = observations_by_landmark
    state.line_observations_by_landmark = line_observations_by_landmark
    state.landmark_ids = landmark_ids
    state.did_bundle_adjust = bool(did_bundle_adjust)

    weak_after = [
        match_id
        for match_id, rmse in sorted(
            ((match_id, post_ba_match_rmse.get(match_id, 0.0)) for match_id in free_match_ids),
            key=lambda item: item[1],
            reverse=True,
        )
        if rmse > ACCEPT_RMSE_PX
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
        state.rebuild_landmarks()

    _check_cancelled(cancel_check)
    _report_progress(progress_callback, "Retrying skipped cameras")
    _resect_skipped_matches(state)
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
                items, similarities, match_map
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
    if mean_weighted_rmse > ACCEPT_RMSE_PX and point_sse:
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
    scale_bits = []
    if known_count:
        scale_bits.append(f"{known_count} known 3D")
    if known_line_count:
        scale_bits.append(f"{known_line_count} known lines")
    if ground_count:
        scale_bits.append(f"{ground_count} ground")
    if free_line_count:
        scale_bits.append(f"{free_line_count} free lines")
    if parallel_pairs:
        scale_bits.append(f"{len(parallel_pairs)} parallel")
    if scale_bits:
        message += " · scale from " + " + ".join(scale_bits)
    else:
        message += " · scale from depth heuristic"
    if did_bundle_adjust:
        message += " · joint BA"
        if froze_structure:
            message += " (thaw 3D)"
    if ground_slack > 1.0e-12:
        drifted = []
        for landmark_id, items in observations_by_landmark.items():
            if landmark_id not in landmarks:
                continue
            if not any(item.on_ground for item in items):
                continue
            if landmark_id in known_world:
                continue
            height = abs(float(landmarks[landmark_id][2]))
            if height > ground_slack:
                drifted.append(
                    (names.get(landmark_id, landmark_id[:8]), height)
                )
        drifted.sort(key=lambda item: -item[1])
        if drifted:
            bits = [f"{name} Z={height:.3f}" for name, height in drifted[:4]]
            message += (
                f" · ground slack {ground_slack:g} exceeded: " + ", ".join(bits)
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

    return SyncSolveResult(
        similarities=similarities,
        landmarks=landmarks,
        mean_reprojection_px=mean_rmse,
        per_match_rmse_px=per_match_rmse,
        per_landmark_rmse_px=per_landmark_rmse,
        message=message,
        success=True,
        line_segments=line_segments,
        downweighted_landmark_ids=downweighted_ids,
        bundle_adjusted=bool(did_bundle_adjust),
        inconsistent_picks=[
            (match_id, name, error)
            for match_id, picks in state.inconsistent_picks.items()
            if match_id in skipped_unregistered
            for _landmark_id, name, error in picks
        ],
    )
