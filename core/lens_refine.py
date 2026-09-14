"""Refine focal length from landmark sync with VP priors or free 2D points.

Sync keeps intrinsics frozen. This outer loop varies ``fx`` (= ``fy``), rebuilds
orientation from VP lines at each candidate (locked-focal refine), and scores
``supported_point_rmse + vp_weight * Σ line_rms²``. Coordinate descent — one match at a
time — is followed by a coupled polish that jointly moves landmark-sharing
pairs (and a global relative-scale probe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as core
from . import sync as sync_module
from .focal_line_constraints import validate_line_relations
from .focal_bundle import (
    DEFAULT_POINT_FOCAL_SPAN, MAX_CAMERAS, MAX_POINTS, MAX_LINES, MAX_LINE_STROKES,
    fit_independent_focals, FocalFitCandidate,
)
from .joint_fit_score import JointFitScorer, joint_line_support_diagnostics
from .focal_startup import provisional_poses
from .sync.request import SyncSolveRequest
from .sync.constants import GROUND_SLACK_DEFAULT, KNOWN_3D_SLACK_DEFAULT


# Soft prior: 1 px endpoint-equivalent VP line RMS ≈ this many sync pixels.
DEFAULT_VP_WEIGHT = 4.0
# Absolute VP ceilings only used when no baseline is available.
DEFAULT_MAX_VP_LINE_RMS = 40.0
DEFAULT_MAX_VP_ANGLE_DEG = 10.0
# Relative guardrails: reject trials that make VP worse than the start.
DEFAULT_VP_LINE_SLACK_PX = 2.0
DEFAULT_VP_ANGLE_SLACK_DEG = 1.0
# Search window as a fraction of the starting focal length.
DEFAULT_FX_SPAN = 0.18
_FAILURE_COST = 1.0e6


@dataclass
class MatchLensInput:
    """One match's VP lines + intrinsics for locked-focal trials."""

    match_id: str
    line_bundles: dict[core.AxisId, list[core.LineSegment]]
    intrinsics: core.CameraIntrinsics
    division_lambda: float = 0.0
    brown_conrady: tuple[float, ...] = ()
    origin_image: tuple[float, float] | None = None
    # When True, keep the starting fx (Manual FOV / 1-point / not enough lines).
    freeze_focal: bool = False
    # Exact private calib to keep when freeze_focal (avoids re-orienting).
    base_calibration: core.Calibration | None = None
    # Rebuild orientation from VP lines when fx changes (needs usable lines).
    reorient_from_vp: bool = False


@dataclass
class LensRefineResult:
    """Outcome of the outer lens + sync search."""

    calibrations: dict[str, core.Calibration]
    sync_result: sync_module.SyncSolveResult
    initial_cost: float
    final_cost: float
    initial_sync_rmse: float
    final_sync_rmse: float
    fx_deltas: dict[str, float] = field(default_factory=dict)
    message: str = ""
    improved: bool = False
    cancelled: bool = False
    point_focal_mode: bool = False
    # Absolute 95% local focal intervals in pixels at the stated pick sigma.
    focal_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    refusal_reason: str = ""
    candidate: FocalFitCandidate | None = None


def estimate_refine_evaluation_count(
    free_match_count: int,
    *,
    passes: int = 2,
    coarse_samples: int = 9,
    refine_samples: int = 7,
    couple_pair_limit: int = 3,
    couple_samples: int = 3,
    share_lens: bool = False,
) -> int:
    """Upper bound on sync evaluations (for progress bars)."""
    if share_lens:
        coarse_evaluations = max(int(coarse_samples), 0) - (
            1 if int(coarse_samples) > 0 and int(coarse_samples) % 2 == 1 else 0
        )
        refine_evaluations = max(int(refine_samples), 0) - (
            1 if int(refine_samples) > 0 and int(refine_samples) % 2 == 1 else 0
        )
        return 1 + coarse_evaluations + refine_evaluations
    if free_match_count <= 0:
        return 1
    # The current focal and fine-grid center already have known scores.
    coarse_evaluations = max(int(coarse_samples), 0) - (
        1 if int(coarse_samples) > 0 and int(coarse_samples) % 2 == 1 else 0
    )
    refine_evaluations = max(int(refine_samples), 0) - (
        1 if int(refine_samples) > 0 and int(refine_samples) % 2 == 1 else 0
    )
    per_match = coarse_evaluations + refine_evaluations
    total = 1 + max(1, int(passes)) * int(free_match_count) * per_match
    # Coupled polish: pairwise grids + a global relative-scale probe.
    if free_match_count >= 2:
        pair_budget = min(
            int(couple_pair_limit),
            int(free_match_count) * (int(free_match_count) - 1) // 2,
        )
        pair_evals = pair_budget * max(int(couple_samples) * int(couple_samples) - 1, 0)
        total += pair_evals + 2
    return total


def _shared_landmark_counts(
    free_ids: list[str],
    observations: list[sync_module.SyncObservation],
) -> dict[tuple[str, str], int]:
    """How many landmarks each free-match pair observes together."""
    by_match: dict[str, set[str]] = {match_id: set() for match_id in free_ids}
    for observation in observations:
        if observation.match_id in by_match:
            by_match[observation.match_id].add(observation.landmark_id)
    counts: dict[tuple[str, str], int] = {}
    for index, match_a in enumerate(free_ids):
        for match_b in free_ids[index + 1 :]:
            shared = by_match[match_a] & by_match[match_b]
            counts[(match_a, match_b)] = len(shared)
    return counts


def _ranked_couple_pairs(
    free_ids: list[str],
    observations: list[sync_module.SyncObservation],
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    """Prefer pairs that share landmarks; fall back to list-adjacent pairs."""
    counts = _shared_landmark_counts(free_ids, observations)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    pairs = [pair for pair, count in ranked if count > 0][: max(0, int(limit))]
    if pairs:
        return pairs
    return [
        (free_ids[index], free_ids[index + 1])
        for index in range(len(free_ids) - 1)
    ][: max(0, int(limit))]


def _sample_focals(center: float, span: float, count: int) -> list[float]:
    low = max(center * (1.0 - span), 1.0)
    high = max(center * (1.0 + span), low + 1.0)
    return [float(value) for value in np.linspace(low, high, count)]


def _sample_scales(center: float, span: float, count: int) -> list[float]:
    """Sample relative focal scales; unlike ``_sample_focals`` this may go below 1."""
    low = max(float(center) * (1.0 - float(span)), 1.0e-3)
    high = max(float(center) * (1.0 + float(span)), low + 1.0e-6)
    return [float(value) for value in np.linspace(low, high, count)]


def calibration_at_focal(
    match: MatchLensInput,
    focal_px: float,
) -> core.Calibration:
    """Rebuild private calibration at ``focal_px`` with VP orientation locked."""
    focal = max(float(focal_px), 1.0)
    intrinsics = core.CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=float(match.intrinsics.cx),
        cy=float(match.intrinsics.cy),
        image_width=int(match.intrinsics.image_width),
        image_height=int(match.intrinsics.image_height),
    )
    calibration = core.refine_camera(
        match.line_bundles,
        intrinsics,
        lock_focal=True,
        estimate_principal_point=False,
        estimate_distortion=False,
        initial_division_lambda=float(match.division_lambda),
        initial_brown_conrady=match.brown_conrady,
        initial_rotation=(
            None
            if match.base_calibration is None
            else match.base_calibration.rotation_w2c
        ),
    )
    if match.origin_image is not None:
        calibration.camera_center, _scale = core.apply_origin_and_scale(
            calibration,
            match.origin_image,
        )
    return calibration


def calibration_scaled_keep_pose(
    base: core.Calibration,
    scale: float,
) -> core.Calibration:
    """Copy a calibration with fx/fy multiplied, orientation unchanged."""
    scale = max(float(scale), 1.0e-6)
    source = base.intrinsics
    return core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=max(float(source.fx) * scale, 1.0),
            fy=max(float(source.fy) * scale, 1.0),
            cx=float(source.cx),
            cy=float(source.cy),
            image_width=int(source.image_width),
            image_height=int(source.image_height),
        ),
        rotation_w2c=np.array(base.rotation_w2c, copy=True),
        camera_center=np.array(base.camera_center, copy=True),
        division_lambda=float(base.division_lambda),
        lambda_saturated=bool(base.lambda_saturated),
        brown_conrady=tuple(base.brown_conrady),
    )


def _sync_rmse(
    result: sync_module.SyncSolveResult,
    observations: list[sync_module.SyncObservation] | None = None,
    calibrations: dict[str, core.Calibration] | None = None,
) -> float:
    """Score supported point picks, including cameras recovered after joint BA."""
    if result.success and observations:
        by_match: dict[str, list[sync_module.SyncObservation]] = {}
        for observation in observations:
            if (
                observation.match_id in result.similarities
                and observation.landmark_id in result.landmarks
            ):
                by_match.setdefault(observation.match_id, []).append(observation)
        if by_match:
            squared = []
            for match_id, items in by_match.items():
                calibration = (calibrations or {}).get(match_id)
                if calibration is None:
                    return float("inf")
                projected, valid = sync_module._project_shared_points(
                    np.asarray([result.landmarks[item.landmark_id] for item in items]),
                    calibration, result.similarities[match_id],
                )
                errors = projected - np.asarray([(item.u, item.v) for item in items])
                if not np.all(valid) or not np.isfinite(errors).all():
                    return float("inf")
                squared.extend(np.sum(errors * errors, axis=1))
            return float(np.sqrt(np.mean(squared)))
    # A refused solve may still carry useful residuals while recovering bad
    # starting lenses. Line-only solves retain their existing headline score.
    error = float(result.mean_reprojection_px)
    if not np.isfinite(error) or error < 0.0:
        return float("inf")
    if error > 1.0e-9:
        return error
    return _FAILURE_COST if not result.success else 0.0


def _retains_sync_support(
    candidate: sync_module.SyncSolveResult,
    incumbent: sync_module.SyncSolveResult | None,
) -> bool:
    """Keep an accepted solve's cameras and geometry; permit failed-start recovery."""
    if incumbent is None or not incumbent.success:
        return True
    return (
        candidate.success
        and candidate.similarities.keys() >= incumbent.similarities.keys()
        and candidate.landmarks.keys() >= incumbent.landmarks.keys()
        and candidate.line_segments.keys() >= incumbent.line_segments.keys()
    )


def _vp_terms(
    calibrations: dict[str, core.Calibration],
    matches: dict[str, MatchLensInput],
) -> tuple[float, float, float]:
    """Return (Σ line_rms², max line_rms, max angular residual)."""
    vp_term = 0.0
    max_line_rms = 0.0
    max_angle = 0.0
    for match_id, match in matches.items():
        calibration = calibrations[match_id]
        line_rms = core.vp_line_residual_rms(calibration, match.line_bundles)
        angle = core.vp_angular_residual_degrees(calibration, match.line_bundles)
        vp_term += line_rms * line_rms
        max_line_rms = max(max_line_rms, line_rms)
        max_angle = max(max_angle, angle)
    return vp_term, max_line_rms, max_angle


def _joint_cost(
    calibrations: dict[str, core.Calibration],
    matches: dict[str, MatchLensInput],
    sync_result: sync_module.SyncSolveResult,
    *,
    vp_weight: float,
    max_vp_line_rms: float = DEFAULT_MAX_VP_LINE_RMS,
    max_vp_angle_deg: float = DEFAULT_MAX_VP_ANGLE_DEG,
    baseline_max_line_rms: float | None = None,
    baseline_max_angle: float | None = None,
    vp_line_slack_px: float = DEFAULT_VP_LINE_SLACK_PX,
    vp_angle_slack_deg: float = DEFAULT_VP_ANGLE_SLACK_DEG,
    observations: list[sync_module.SyncObservation] | None = None,
    incumbent: sync_module.SyncSolveResult | None = None,
) -> float:
    if not _retains_sync_support(sync_result, incumbent):
        return float("inf")
    sync_term = _sync_rmse(sync_result, observations, calibrations)
    if not np.isfinite(sync_term):
        return float("inf")
    if sync_term >= _FAILURE_COST:
        return _FAILURE_COST
    vp_term, max_line_rms, max_angle = _vp_terms(calibrations, matches)
    # Guardrails are relative to the starting lenses when available so a messy
    # real plate (line RMS already >12px) can still be searched. Absolute
    # ceilings only apply when there is no baseline.
    if baseline_max_line_rms is None:
        line_limit = float(max_vp_line_rms)
    else:
        line_limit = float(baseline_max_line_rms) + float(vp_line_slack_px)
    if baseline_max_angle is None:
        angle_limit = float(max_vp_angle_deg)
    else:
        angle_limit = float(baseline_max_angle) + float(vp_angle_slack_deg)
    if max_line_rms > line_limit or max_angle > angle_limit:
        return _FAILURE_COST
    return sync_term + float(vp_weight) * vp_term


def _run_sync(
    calibrations: dict[str, core.Calibration],
    match_ids: list[str],
    observations: list[sync_module.SyncObservation],
    line_observations: list[sync_module.SyncLineObservation],
    anchor_id: str,
    known_world: dict,
    known_lines: dict,
    parallel_pairs: list,
    initial_similarities: dict[str, sync_module.SimilarityTransform] | None = None,
    *,
    fixed_similarities: dict[str, sync_module.SimilarityTransform] | None = None,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    ground_slack: float | None = None,
    known_3d_slack: float | None = None,
    mirror_pairs: list | None = None,
    mirror_plane: tuple | None = None,
    mirror_slack: float | None = None,
    mirror_landmark_id: str | None = None,
    plane_groups: list | None = None,
    plane_slack: float | None = None,
    location_match_ids: set[str] | None = None,
    readonly_match_ids: set[str] | None = None,
    cancel_check=None,
    progress_callback=None,
    initial_solution=None,
) -> sync_module.SyncSolveResult:
    sync_matches = [
        sync_module.SyncMatchInput(match_id=match_id, calibration=calibrations[match_id])
        for match_id in match_ids
    ]
    return sync_module.solve_landmark_sync(
        sync_matches,
        observations,
        anchor_id=anchor_id,
        known_world=known_world,
        line_observations=line_observations,
        known_lines=known_lines,
        parallel_pairs=parallel_pairs,
        initial_similarities=initial_similarities,
        **({"initial_solution": initial_solution} if initial_solution is not None else {}),
        fixed_similarities=fixed_similarities,
        lock_rotation=lock_rotation,
        lock_translation=lock_translation,
        ground_slack=ground_slack,
        known_3d_slack=known_3d_slack,
        mirror_pairs=mirror_pairs,
        mirror_plane=mirror_plane,
        mirror_slack=mirror_slack,
        **({"mirror_landmark_id": mirror_landmark_id}
           if mirror_landmark_id is not None else {}),
        plane_groups=plane_groups,
        plane_slack=plane_slack,
        location_match_ids=location_match_ids,
        readonly_match_ids=readonly_match_ids,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def refine_lenses_from_landmarks(
    matches: list[MatchLensInput],
    observations: list[sync_module.SyncObservation],
    *,
    anchor_id: str,
    known_world: dict | None = None,
    line_observations: list[sync_module.SyncLineObservation] | None = None,
    known_lines: dict | None = None,
    parallel_pairs: list | None = None,
    vp_weight: float = DEFAULT_VP_WEIGHT,
    max_vp_line_rms: float = DEFAULT_MAX_VP_LINE_RMS,
    max_vp_angle_deg: float = DEFAULT_MAX_VP_ANGLE_DEG,
    fx_span: float | None = None,
    passes: int = 2,
    coarse_samples: int = 9,
    refine_samples: int = 7,
    couple_pair_limit: int = 3,
    couple_samples: int = 3,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    share_lens: bool = False,
    estimate_focal_from_points: bool = False,
    pick_sigma_px: float = 1.0,
    fixed_similarities: dict[str, sync_module.SimilarityTransform] | None = None,
    ground_slack: float | None = None,
    known_3d_slack: float | None = None,
    mirror_pairs: list | None = None,
    mirror_plane: tuple | None = None,
    mirror_slack: float | None = None,
    mirror_landmark_id: str | None = None,
    plane_groups: list | None = None,
    plane_slack: float | None = None,
    location_match_ids: set[str] | None = None,
    readonly_match_ids: set[str] | None = None,
    cancel_check=None,
    progress_callback=None,
    initial_solution=None,
) -> LensRefineResult:
    """Fit focal from supported joint geometry, or search the ordinary lens objective.

    Coordinate descent moves one unlocked match at a time. The coupled polish
    then jointly varies focals for landmark-sharing pairs (and a global
    relative-scale probe) so multi-camera FOV error can shrink together.

    When ``share_lens`` is true, search one scale applied to every match's fx
    (YAML-only stills keep orientation; stills with VP lines may re-orient).

    Frozen matches (Manual FOV / 1-point / weak VPs) keep their starting fx but
    still contribute VP residual and sync observations — unless ``share_lens``.

    ``cancel_check`` is an optional ``() -> bool`` polled between evaluations.
    ``progress_callback(step, total, label)`` reports progress (may be called
    from a worker thread — keep it bpy-free).

    ``estimate_focal_from_points`` opts into the joint focal/pose/geometry fit,
    with an explicit per-coordinate pixel-noise assumption.
    """
    if not matches:
        raise ValueError("No matches to refine")
    match_map = {item.match_id: item for item in matches}
    match_ids = [item.match_id for item in matches]
    if anchor_id not in match_map:
        raise ValueError("Anchor match is missing from the lens refine set")
    if fx_span is None:
        fx_span = DEFAULT_POINT_FOCAL_SPAN if estimate_focal_from_points else DEFAULT_FX_SPAN

    if estimate_focal_from_points:
        # This path fits private camera calibrations, roots and supported
        # point/line geometry under the same current-evidence constraints.
        initial_cals = {item.match_id: item.base_calibration for item in matches
                        if item.base_calibration is not None}
        if (len(initial_cals) != len(matches) and initial_solution is not None and
                getattr(initial_solution, "calibrations", None)):
            if set(initial_solution.calibrations) == set(match_ids):
                initial_cals = initial_solution.calibrations
        empty_sync = sync_module.SyncSolveResult(
            similarities={}, landmarks={}, mean_reprojection_px=float("inf"),
            per_match_rmse_px={}, per_landmark_rmse_px={}, message="Point focal refused",
            success=False)

        def refusal(reason: str, *, initial=empty_sync, initial_rmse=float("inf"),
                    cancelled=False, candidate=None) -> LensRefineResult:
            return LensRefineResult(
                calibrations=initial_cals, sync_result=initial,
                initial_cost=initial_rmse, final_cost=initial_rmse,
                initial_sync_rmse=initial_rmse, final_sync_rmse=initial_rmse,
                message=reason, improved=False, cancelled=cancelled,
                point_focal_mode=True, refusal_reason=reason, candidate=candidate)

        if len(initial_cals) != len(matches):
            return refusal("Point focal estimation needs a saved private camera solve for every match")
        if len(matches) < (2 if known_world or known_lines or fixed_similarities or
                              any(item.on_ground for item in observations) else 3):
            return refusal("Independent focal fitting needs three free-point views or two referenced cameras")
        if len(matches) > MAX_CAMERAS:
            return refusal(f"Point focal estimation currently supports up to {MAX_CAMERAS} cameras per joint fit (resource limit)")
        point_count = len({item.landmark_id for item in observations})
        if not (known_world or known_lines or line_observations or
                any(item.on_ground for item in observations)) and not 8 <= point_count <= MAX_POINTS:
            return refusal(f"Point focal estimation needs 8–{MAX_POINTS} points")
        for match_id in match_ids:
            count = sum(item.match_id == match_id for item in observations)
            if count < 8 and not (known_world or known_lines or line_observations or
                                  any(item.on_ground for item in observations)):
                return refusal(f"{match_id} has {count} point picks; each camera needs at least eight")
        if not np.isfinite(pick_sigma_px) or pick_sigma_px <= 0:
            return refusal("Pick sigma must be positive and finite")
        if not np.isfinite(fx_span) or not 0 < fx_span < 1:
            return refusal("Focal search span must lie between 0 and 100%")
        point_ids = {item.landmark_id for item in observations} | set(known_world or {})
        picked_views = {point_id: {item.match_id for item in observations
                                   if item.landmark_id == point_id}
                        for point_id in point_ids}
        if mirror_landmark_id is not None:
            if not isinstance(mirror_landmark_id, str) or not mirror_landmark_id:
                return refusal("Mirror reference landmark ID is invalid")
            if mirror_landmark_id not in picked_views:
                return refusal("Mirror reference landmark needs two-view picks")
            if any(mirror_landmark_id in pair for pair in (mirror_pairs or ())):
                return refusal("Mirror reference landmark cannot be a mirror pair member")
        if mirror_pairs and mirror_plane is None:
            return refusal("Point mirror pairs need a mirror plane normal" if
                           mirror_landmark_id is not None else
                           "Point mirror pairs need a supplied Mirror Empty")
        if any(not isinstance(item, sync_module.SyncLineObservation)
               for item in (line_observations or ())):
            return refusal("Invalid line landmarks: expected line strokes")
        line_ids = {item.landmark_id for item in (line_observations or ())} | set(known_lines or {})
        if len(line_ids) > MAX_LINES or len(line_observations or ()) > MAX_LINE_STROKES:
            return refusal(f"Line fit supports up to {MAX_LINES} lines and {MAX_LINE_STROKES} strokes (resource limit)")
        try:
            validate_line_relations(point_ids, line_ids, plane_groups, parallel_pairs,
                                    known_line_ids=set(known_lines or {}))
        except (ValueError, TypeError) as exc:
            return refusal(str(exc))
        if any(set(pair) & line_ids and set(pair) & point_ids for pair in (mirror_pairs or ())):
            return refusal("Mirror relation contains unsupported or mixed point/line members")
        relation_ids = {item[0] for item in (plane_groups or ())} | {
            point_id for pair in (mirror_pairs or ()) for point_id in pair}
        if not relation_ids <= point_ids | line_ids | set(known_lines or {}):
            return refusal("Plane or mirror relation contains an unrepresented landmark")
        line_views = {key: {item.match_id for item in (line_observations or ())
                            if item.landmark_id == key} for key in line_ids}
        if any(key not in (known_lines or {}) and len(views) < 2 and not any(
                key in pair and (
                    (partner := pair[0] if pair[1] == key else pair[1]) in (known_lines or {}) or
                    len(line_views.get(partner, ())) >= 2)
                for pair in (mirror_pairs or ()))
               for key, views in line_views.items()):
            return refusal("Line landmarks need two-view strokes or a reconstructed mirror partner")

        joint_request = SyncSolveRequest(
            matches=[sync_module.SyncMatchInput(key, initial_cals[key])
                     for key in match_ids],
            observations=observations, anchor_id=anchor_id,
            known_world=known_world, line_observations=line_observations,
            known_lines=known_lines, parallel_pairs=parallel_pairs,
            fixed_similarities=fixed_similarities,
            lock_rotation=lock_rotation, lock_translation=lock_translation,
            ground_slack=ground_slack, known_3d_slack=known_3d_slack,
            mirror_pairs=mirror_pairs, mirror_plane=mirror_plane,
            mirror_slack=mirror_slack, mirror_landmark_id=mirror_landmark_id,
            plane_groups=plane_groups, plane_slack=plane_slack,
            location_match_ids=location_match_ids,
            readonly_match_ids=readonly_match_ids)
        seed_result = None
        seed_score = None
        seed_evidence_matches = bool(
            initial_solution is not None and
            getattr(initial_solution, "evidence_sha256", "") ==
            joint_request.evidence_sha256())
        frozen_seed_weights = (
            initial_solution.diagnostics.joint_point_weights
            if seed_evidence_matches and getattr(initial_solution, "diagnostics", None)
            else None)
        if initial_solution is not None and getattr(initial_solution, "diagnostics", None) is not None:
            try:
                from .sync.solve import solution_result_from_seed
            except ImportError:
                solution_result_from_seed = None
            if solution_result_from_seed is not None:
                seed_result = solution_result_from_seed(initial_solution)
            if seed_result is not None and set(seed_result.similarities) >= set(match_ids):
                seed_scorer = JointFitScorer(joint_request, seed_result,
                                             calibrations=initial_cals,
                                             frozen_point_weights=frozen_seed_weights)
                trial = seed_scorer.score(seed_result, calibrations=initial_cals)
                if trial.valid and trial.supported_observations == (
                        len(observations) + len(line_observations or ())):
                    seed_score = trial
        certified_seed = (
            seed_score is not None and seed_evidence_matches and
            (frozen_seed_weights is not None or not observations))
        if cancel_check and cancel_check():
            return refusal("Cancelled", cancelled=True)
        if certified_seed:
            initial = seed_result
        else:
            # For a free, unreferenced point graph, hard plane/mirror geometry
            # at the guessed focal can bias camera registration. These
            # relations enter the common fitter and final scorer unchanged.
            point_only_start = (
                initial_solution is None and not known_world and not known_lines and
                not line_observations and not fixed_similarities and
                not lock_rotation and not lock_translation and
                not readonly_match_ids and
                (location_match_ids is None or
                 set(location_match_ids) == set(match_ids)) and
                not any(item.on_ground for item in observations))
            if progress_callback:
                progress_callback(0, 101, "Registering cameras for point focal estimation")
            try:
                initial = _run_sync(
                    initial_cals, match_ids, observations, line_observations or [],
                    anchor_id, known_world or {}, known_lines or {}, parallel_pairs or [],
                    fixed_similarities=fixed_similarities,
                    lock_rotation=lock_rotation, lock_translation=lock_translation,
                    ground_slack=ground_slack, known_3d_slack=known_3d_slack,
                    plane_groups=None if point_only_start else plane_groups,
                    plane_slack=0.0 if point_only_start else plane_slack,
                    mirror_pairs=None if point_only_start else mirror_pairs,
                    mirror_plane=None if point_only_start else mirror_plane,
                    mirror_slack=0.0 if point_only_start else mirror_slack,
                    mirror_landmark_id=None if point_only_start else mirror_landmark_id,
                    location_match_ids=location_match_ids,
                    readonly_match_ids=readonly_match_ids,
                    initial_solution=initial_solution,
                    cancel_check=cancel_check,
                    progress_callback=(
                        (lambda label: progress_callback(0, 101, label))
                        if progress_callback else None))
            except sync_module.SyncCancelled:
                return refusal("Cancelled", cancelled=True)
        if not initial.success and seed_score is not None:
            initial = seed_result
        if not initial.success:
            return refusal("Initial camera registration did not support every camera and point",
                           initial=initial, initial_rmse=_sync_rmse(initial, observations,
                                                                      initial_cals))
        if set(match_ids) - set(initial.similarities):
            initial, startup_refusal = provisional_poses(
                initial_cals, observations, initial, anchor_id=anchor_id,
                cancel_check=cancel_check, progress_callback=(
                    (lambda _step, _total, label: progress_callback(0, 101, label))
                    if progress_callback else None))
            if startup_refusal:
                if startup_refusal == "Cancelled":
                    return refusal(startup_refusal, initial=initial,
                                   initial_rmse=_sync_rmse(initial, observations, initial_cals),
                                   cancelled=True)
                if seed_score is None:
                    return refusal(startup_refusal, initial=initial,
                                   initial_rmse=_sync_rmse(initial, observations, initial_cals))
                initial = seed_result
        scorer = (JointFitScorer(joint_request, seed_result, calibrations=initial_cals,
                                 frozen_point_weights=(frozen_seed_weights
                                                       if certified_seed else None))
                  if seed_score is not None else
                  JointFitScorer(joint_request, initial, calibrations=initial_cals))
        initial_score = scorer.score(initial, calibrations=initial_cals)
        if seed_score is not None and (not initial_score.valid or
                                       seed_score.objective < initial_score.objective):
            initial = seed_result
            initial_score = seed_score
        initial_rmse = _sync_rmse(initial, observations, initial_cals)
        def point_progress(step: int, total: int, label: str) -> None:
            if progress_callback:
                # A zero-step status (including provisional startup) is
                # activity, not completed numerical progress.
                progress_callback(0 if step <= 0 else step + 1,
                                  total + 1, label)
        outcome = fit_independent_focals(
            initial_cals, scorer.point_observations, initial, anchor_id=anchor_id,
            pick_sigma_px=pick_sigma_px, fx_span=fx_span,
            plane_groups=plane_groups, plane_slack=plane_slack,
            mirror_pairs=mirror_pairs, mirror_plane=mirror_plane,
            mirror_slack=mirror_slack, mirror_landmark_id=mirror_landmark_id,
            line_observations=line_observations, parallel_pairs=parallel_pairs,
            fixed_similarities=fixed_similarities,
            location_match_ids=location_match_ids,
            readonly_match_ids=readonly_match_ids,
            known_world=known_world, known_3d_slack=(KNOWN_3D_SLACK_DEFAULT
                                                      if known_3d_slack is None else known_3d_slack),
            ground_slack=(GROUND_SLACK_DEFAULT if ground_slack is None else ground_slack),
            known_lines=known_lines,
            lock_rotation=lock_rotation, lock_translation=lock_translation,
            share_lens=share_lens,
            frozen_focal_ids=({item.match_id for item in matches if item.freeze_focal}
                              if not share_lens else set()),
            cancel_check=cancel_check,
            progress_callback=point_progress)
        if not outcome.accepted or outcome.sync_result is None:
            if outcome.candidate is not None:
                outcome.candidate.sync_result.joint_point_weights = scorer.effective_point_weights
                outcome.candidate.sync_result.downweighted_landmark_ids = list(
                    scorer.downweighted_landmark_ids)
            return refusal(outcome.reason, initial=initial, initial_rmse=initial_rmse,
                           cancelled=outcome.reason == "Cancelled", candidate=outcome.candidate)
        final_score = scorer.score(outcome.sync_result, calibrations=outcome.calibrations)
        if not final_score.valid:
            return refusal(final_score.reason, initial=initial, initial_rmse=initial_rmse,
                           candidate=outcome.candidate)
        if (initial_score.valid and final_score.objective >= initial_score.objective):
            return refusal("Joint focal objective did not improve", initial=initial,
                           initial_rmse=initial_rmse, candidate=outcome.candidate)
        outcome.sync_result.joint_initial_objective = (
            initial_score.objective if initial_score.valid else None)
        outcome.sync_result.joint_final_objective = final_score.objective
        outcome.sync_result.joint_constraint_gaps = final_score.constraint_gaps
        outcome.sync_result.calibrations = outcome.calibrations
        outcome.sync_result.point_rmse_px = final_score.point_rmse_px
        outcome.sync_result.line_rmse_px = final_score.line_rmse_px
        outcome.sync_result.mean_reprojection_px = (
            final_score.point_rmse_px if observations else final_score.line_rmse_px)
        outcome.sync_result.per_match_point_rmse_px = final_score.per_match_point_rmse_px
        outcome.sync_result.per_match_line_rmse_px = final_score.per_match_line_rmse_px
        outcome.sync_result.per_match_rmse_px = final_score.per_match_rmse_px
        outcome.sync_result.per_landmark_rmse_px = final_score.per_landmark_rmse_px
        outcome.sync_result.downweighted_landmark_ids = list(
            scorer.downweighted_landmark_ids)
        outcome.sync_result.joint_point_weights = scorer.effective_point_weights
        outcome.sync_result.plane_seeded_landmark_ids = [
            key for key in initial.plane_seeded_landmark_ids
            if key in outcome.sync_result.landmarks]
        angles, weak = joint_line_support_diagnostics(
            joint_request, outcome.sync_result, outcome.calibrations)
        outcome.sync_result.line_support_angles_deg = angles
        outcome.sync_result.weak_line_ids = weak
        if weak:
            outcome.sync_result.message += (
                " · weak 3D line support (" + ", ".join(weak[:3]) + ")")
        if progress_callback:
            progress_callback(101, 101, "Point focal estimation complete")
        return LensRefineResult(
            calibrations=outcome.calibrations, sync_result=outcome.sync_result,
            initial_cost=(initial_score.objective if initial_score.valid else float("inf")),
            final_cost=final_score.objective,
            initial_sync_rmse=(initial_score.point_rmse_px if observations else
                               initial_score.line_rmse_px),
            final_sync_rmse=(final_score.point_rmse_px if observations else
                             final_score.line_rmse_px),
            fx_deltas={key: float(outcome.calibrations[key].intrinsics.fx -
                                  initial_cals[key].intrinsics.fx) for key in match_ids},
            message=(f"Landmark focal fit · point RMSE {final_score.point_rmse_px:.2f}px"
                     if observations else
                     f"Landmark focal fit · line endpoint RMS {final_score.line_rmse_px:.2f}px") +
                    f" · 95% intervals at σ={pick_sigma_px:g}px",
            improved=True, point_focal_mode=True,
            focal_intervals=outcome.intervals_px)

    calibrations = {}
    for item in matches:
        # Start from the stored private solve so "no improvement" does not churn R.
        if item.base_calibration is not None:
            calibrations[item.match_id] = item.base_calibration
        else:
            calibrations[item.match_id] = calibration_at_focal(
                item, item.intrinsics.fx
            )

    known_world = known_world or {}
    line_observations = line_observations or []
    known_lines = known_lines or {}
    parallel_pairs = parallel_pairs or []

    free_ids = [item.match_id for item in matches if not item.freeze_focal]
    if share_lens:
        free_ids = [item.match_id for item in matches]
    total_steps = estimate_refine_evaluation_count(
        len(free_ids),
        passes=passes,
        coarse_samples=coarse_samples,
        refine_samples=refine_samples,
        couple_pair_limit=couple_pair_limit,
        couple_samples=couple_samples,
        share_lens=share_lens,
    )
    step = 0

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def _progress(label: str) -> None:
        if progress_callback is not None:
            progress_callback(min(step, total_steps), total_steps, label)

    def evaluate(
        cals: dict[str, core.Calibration],
        incumbent: sync_module.SyncSolveResult | None = None,
    ):
        nonlocal step
        # Sync refusals currently expose identity placeholders rather than
        # estimated camera poses. Retry those lens candidates from registration;
        # explicit pose locks remain available through fixed_similarities.
        initial_similarities = (
            incumbent.similarities
            if incumbent is not None and incumbent.success
            else None
        )
        result = _run_sync(
            cals,
            match_ids,
            observations,
            line_observations,
            anchor_id,
            known_world,
            known_lines,
            parallel_pairs,
            initial_similarities,
            fixed_similarities=fixed_similarities,
            lock_rotation=lock_rotation,
            lock_translation=lock_translation,
            ground_slack=ground_slack,
            known_3d_slack=known_3d_slack,
            mirror_pairs=mirror_pairs,
            mirror_plane=mirror_plane,
            mirror_slack=mirror_slack,
            mirror_landmark_id=mirror_landmark_id,
            plane_groups=plane_groups,
            plane_slack=plane_slack,
            location_match_ids=location_match_ids,
            readonly_match_ids=readonly_match_ids,
        )
        cost = _joint_cost(
            cals,
            match_map,
            result,
            vp_weight=vp_weight,
            max_vp_line_rms=max_vp_line_rms,
            max_vp_angle_deg=max_vp_angle_deg,
            baseline_max_line_rms=baseline_line_rms,
            baseline_max_angle=baseline_angle,
            observations=observations,
            incumbent=incumbent,
        )
        step += 1
        return cost, result

    def _cancelled_result(
        best_cals,
        best_sync,
        initial_cost,
        best_cost,
        initial_sync,
        start_fx,
        free_ids_local,
    ) -> LensRefineResult:
        initial_rmse = _sync_rmse(initial_sync, observations, calibrations)
        final_rmse = _sync_rmse(best_sync, observations, best_cals)
        return LensRefineResult(
            calibrations=best_cals,
            sync_result=best_sync,
            initial_cost=initial_cost,
            final_cost=best_cost,
            initial_sync_rmse=initial_rmse,
            final_sync_rmse=final_rmse,
            fx_deltas={
                match_id: float(best_cals[match_id].intrinsics.fx - start_fx[match_id])
                for match_id in free_ids_local
            },
            message=f"Lens refine cancelled · sync {final_rmse:.1f}px",
            improved=best_cost + 1.0e-3 < initial_cost,
            cancelled=True,
        )

    # Baseline VP quality: relative guardrails compare against this, not a
    # fixed 12px absolute, so messy real plates are still searchable.
    _baseline_vp_term, baseline_line_rms, baseline_angle = _vp_terms(
        calibrations, match_map
    )

    _progress("Scoring initial lenses")
    if _cancelled():
        start_fx = {item.match_id: float(item.intrinsics.fx) for item in matches}
        empty_sync = sync_module.SyncSolveResult(
            similarities={},
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="Cancelled",
            success=False,
        )
        return _cancelled_result(
            calibrations,
            empty_sync,
            _FAILURE_COST,
            _FAILURE_COST,
            empty_sync,
            start_fx,
            free_ids,
        )

    initial_cost, initial_sync = evaluate(calibrations)
    best_cost = initial_cost
    best_sync = initial_sync
    best_cals = {key: value for key, value in calibrations.items()}
    start_fx = {item.match_id: float(item.intrinsics.fx) for item in matches}
    _progress("Initial sync scored")

    def _calibrations_at_scale(scale: float) -> dict[str, core.Calibration]:
        trial: dict[str, core.Calibration] = {}
        for item in matches:
            start = calibrations[item.match_id]
            if item.reorient_from_vp:
                focal = max(float(item.intrinsics.fx) * float(scale), 1.0)
                trial[item.match_id] = calibration_at_focal(item, focal)
            else:
                trial[item.match_id] = calibration_scaled_keep_pose(start, scale)
        return trial

    if share_lens:
        if _cancelled():
            return _cancelled_result(
                best_cals,
                best_sync,
                initial_cost,
                best_cost,
                initial_sync,
                start_fx,
                free_ids,
            )
        local_best_scale = 1.0
        for scale in _sample_scales(1.0, fx_span, coarse_samples):
            if abs(scale - 1.0) <= 1.0e-9:
                continue
            if _cancelled():
                return _cancelled_result(
                    best_cals,
                    best_sync,
                    initial_cost,
                    best_cost,
                    initial_sync,
                    start_fx,
                    free_ids,
                )
            _progress(f"Same lens · scale {scale:.3f}")
            trial = _calibrations_at_scale(scale)
            cost, result = evaluate(
                trial,
                incumbent=best_sync,
            )
            if cost + 1.0e-6 < best_cost:
                best_cost = cost
                best_sync = result
                best_cals = trial
                local_best_scale = scale
        fine_span = fx_span * 0.25
        fine_center = local_best_scale
        for scale in _sample_scales(local_best_scale, fine_span, refine_samples):
            if abs(scale - fine_center) <= 1.0e-9:
                continue
            if _cancelled():
                return _cancelled_result(
                    best_cals,
                    best_sync,
                    initial_cost,
                    best_cost,
                    initial_sync,
                    start_fx,
                    free_ids,
                )
            _progress(f"Same lens refine · scale {scale:.3f}")
            trial = _calibrations_at_scale(scale)
            cost, result = evaluate(
                trial,
                incumbent=best_sync,
            )
            if cost + 1.0e-6 < best_cost:
                best_cost = cost
                best_sync = result
                best_cals = trial
                local_best_scale = scale
        fx_deltas = {
            match_id: float(best_cals[match_id].intrinsics.fx - start_fx[match_id])
            for match_id in free_ids
        }
        improved = best_cost + 1.0e-3 < initial_cost
        initial_rmse = _sync_rmse(initial_sync, observations, calibrations)
        final_rmse = _sync_rmse(best_sync, observations, best_cals)
        percent = (local_best_scale - 1.0) * 100.0
        anchor_fx = float(best_cals[anchor_id].intrinsics.fx)
        start_anchor = start_fx[anchor_id]
        if improved:
            message = (
                f"Same lens ×{local_best_scale:.4f} ({percent:+.1f}%) · "
                f"fx {start_anchor:.0f}→{anchor_fx:.0f}px · "
                f"sync {initial_rmse:.1f}→{final_rmse:.1f}px"
            )
        else:
            message = (
                f"No shared-lens improvement · fx {anchor_fx:.0f}px · "
                f"sync {final_rmse:.1f}px"
            )
        _progress("Finished")
        return LensRefineResult(
            calibrations=best_cals,
            sync_result=best_sync,
            initial_cost=initial_cost,
            final_cost=best_cost,
            initial_sync_rmse=initial_rmse,
            final_sync_rmse=final_rmse,
            fx_deltas=fx_deltas,
            message=message,
            improved=improved,
        )

    if not free_ids:
        return LensRefineResult(
            calibrations=best_cals,
            sync_result=best_sync,
            initial_cost=initial_cost,
            final_cost=best_cost,
            initial_sync_rmse=_sync_rmse(initial_sync, observations, calibrations),
            final_sync_rmse=_sync_rmse(best_sync, observations, best_cals),
            fx_deltas={},
            message="No free focals (all matches Manual FOV / 1-point / locked)",
            improved=False,
        )

    for pass_index in range(max(1, int(passes))):
        for match_id in free_ids:
            if _cancelled():
                return _cancelled_result(
                    best_cals,
                    best_sync,
                    initial_cost,
                    best_cost,
                    initial_sync,
                    start_fx,
                    free_ids,
                )
            current_fx = float(best_cals[match_id].intrinsics.fx)
            candidates = _sample_focals(current_fx, fx_span, coarse_samples)
            # Always evaluate the current fx so we never force a worse step.
            if all(abs(candidate - current_fx) > 0.25 for candidate in candidates):
                candidates.append(current_fx)

            local_best_fx = current_fx
            local_best_cost = best_cost
            local_best_sync = best_sync
            local_best_cals = best_cals

            for focal in candidates:
                # The current complete focal vector was scored before this
                # coordinate step; do not run the same global sync again.
                if abs(focal - current_fx) <= 1.0e-9:
                    continue
                if _cancelled():
                    return _cancelled_result(
                        best_cals,
                        best_sync,
                        initial_cost,
                        best_cost,
                        initial_sync,
                        start_fx,
                        free_ids,
                    )
                _progress(
                    f"Pass {pass_index + 1}/{passes} · {match_id} · "
                    f"fx {focal:.0f}px"
                )
                trial = {key: value for key, value in local_best_cals.items()}
                trial[match_id] = calibration_at_focal(match_map[match_id], focal)
                cost, result = evaluate(
                    trial,
                    incumbent=local_best_sync,
                )
                if cost + 1.0e-6 < local_best_cost:
                    local_best_cost = cost
                    local_best_sync = result
                    local_best_fx = focal
                    local_best_cals = trial

            # Fine pass around the coarse winner.
            fine_span = fx_span * 0.25
            fine_center_fx = local_best_fx
            for focal in _sample_focals(local_best_fx, fine_span, refine_samples):
                # The coarse winner already owns local_best_cost/local_best_sync.
                if abs(focal - fine_center_fx) <= 1.0e-9:
                    continue
                if _cancelled():
                    return _cancelled_result(
                        local_best_cals,
                        local_best_sync,
                        initial_cost,
                        local_best_cost,
                        initial_sync,
                        start_fx,
                        free_ids,
                    )
                _progress(
                    f"Pass {pass_index + 1}/{passes} · {match_id} refine · "
                    f"fx {focal:.0f}px"
                )
                trial = {key: value for key, value in local_best_cals.items()}
                trial[match_id] = calibration_at_focal(match_map[match_id], focal)
                cost, result = evaluate(
                    trial,
                    incumbent=local_best_sync,
                )
                if cost + 1.0e-6 < local_best_cost:
                    local_best_cost = cost
                    local_best_sync = result
                    local_best_cals = trial

            best_cost = local_best_cost
            best_sync = local_best_sync
            best_cals = local_best_cals

    # Coupled polish: move landmark-sharing pairs together, then probe a shared
    # relative scale so multi-camera FOV bias is not stuck in coordinate descent.
    couple_span = float(fx_span) * 0.2
    for match_a, match_b in _ranked_couple_pairs(
        free_ids,
        observations,
        limit=couple_pair_limit,
    ):
        if _cancelled():
            return _cancelled_result(
                best_cals,
                best_sync,
                initial_cost,
                best_cost,
                initial_sync,
                start_fx,
                free_ids,
            )
        center_a = float(best_cals[match_a].intrinsics.fx)
        center_b = float(best_cals[match_b].intrinsics.fx)
        for focal_a in _sample_focals(center_a, couple_span, couple_samples):
            for focal_b in _sample_focals(center_b, couple_span, couple_samples):
                if (
                    abs(focal_a - center_a) <= 1.0e-9
                    and abs(focal_b - center_b) <= 1.0e-9
                ):
                    continue
                if _cancelled():
                    return _cancelled_result(
                        best_cals,
                        best_sync,
                        initial_cost,
                        best_cost,
                        initial_sync,
                        start_fx,
                        free_ids,
                    )
                _progress(
                    f"Coupled · {match_a}/{match_b} · "
                    f"fx {focal_a:.0f}/{focal_b:.0f}px"
                )
                trial = {key: value for key, value in best_cals.items()}
                trial[match_a] = calibration_at_focal(match_map[match_a], focal_a)
                trial[match_b] = calibration_at_focal(match_map[match_b], focal_b)
                cost, result = evaluate(
                    trial,
                    incumbent=best_sync,
                )
                if cost + 1.0e-6 < best_cost:
                    best_cost = cost
                    best_sync = result
                    best_cals = trial

    for scale in (1.0 - couple_span, 1.0 + couple_span):
        if _cancelled():
            return _cancelled_result(
                best_cals,
                best_sync,
                initial_cost,
                best_cost,
                initial_sync,
                start_fx,
                free_ids,
            )
        _progress(f"Coupled · global scale ×{scale:.3f}")
        trial = {key: value for key, value in best_cals.items()}
        for match_id in free_ids:
            focal = max(float(best_cals[match_id].intrinsics.fx) * scale, 1.0)
            trial[match_id] = calibration_at_focal(match_map[match_id], focal)
        cost, result = evaluate(
            trial,
            incumbent=best_sync,
        )
        if cost + 1.0e-6 < best_cost:
            best_cost = cost
            best_sync = result
            best_cals = trial

    fx_deltas = {
        match_id: float(best_cals[match_id].intrinsics.fx - start_fx[match_id])
        for match_id in free_ids
    }
    improved = best_cost + 1.0e-3 < initial_cost
    initial_rmse = _sync_rmse(initial_sync, observations, calibrations)
    final_rmse = _sync_rmse(best_sync, observations, best_cals)
    changed = [
        f"{match_id} Δfx {delta:+.1f}px"
        for match_id, delta in fx_deltas.items()
        if abs(delta) > 0.5
    ]
    if improved:
        if final_rmse <= initial_rmse + 1.0e-3:
            message = (
                f"Lenses refined · sync {initial_rmse:.1f}→{final_rmse:.1f}px"
            )
        else:
            # Joint cost can improve by preserving VP agreement even when the
            # landmark-only RMSE rises slightly; make that tradeoff explicit.
            message = (
                f"Lens/VP fit improved · sync {initial_rmse:.1f}→"
                f"{final_rmse:.1f}px · joint cost {initial_cost:.1f}→"
                f"{best_cost:.1f}"
            )
        if changed:
            message += " · " + ", ".join(changed[:4])
    else:
        message = (
            f"No lens improvement · sync {final_rmse:.1f}px "
            f"(cost {initial_cost:.1f}→{best_cost:.1f})"
        )
        if not best_sync.success or final_rmse > 40.0:
            message += (
                " · re-pick/exclude worst landmarks (Refine cannot fix bad picks)"
            )
        elif baseline_line_rms > 20.0:
            message += (
                f" · VP lines already noisy ({baseline_line_rms:.0f}px RMS)"
            )
    _progress("Finished")
    return LensRefineResult(
        calibrations=best_cals,
        sync_result=best_sync,
        initial_cost=initial_cost,
        final_cost=best_cost,
        initial_sync_rmse=initial_rmse,
        final_sync_rmse=final_rmse,
        fx_deltas=fx_deltas,
        message=message,
        improved=improved,
    )
