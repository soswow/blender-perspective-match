"""Refine per-match focal length from landmark sync + a VP line prior.

Sync keeps intrinsics frozen. This outer loop varies ``fx`` (= ``fy``), rebuilds
orientation from VP lines at each candidate (locked-focal refine), and scores
``sync_rmse + vp_weight * Σ line_rms²``. Coordinate descent — one match at a
time — is followed by a coupled polish that jointly moves landmark-sharing
pairs (and a global relative-scale probe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core
from . import sync as sync_module


# Soft prior: 1 px VP line RMS ≈ this many sync pixels in the joint cost.
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
    origin_image: tuple[float, float] | None = None
    # When True, keep the starting fx (Manual FOV / 1-point / not enough lines).
    freeze_focal: bool = False
    # Exact private calib to keep when freeze_focal (avoids re-orienting).
    base_calibration: core.Calibration | None = None


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


def estimate_refine_evaluation_count(
    free_match_count: int,
    *,
    passes: int = 2,
    coarse_samples: int = 9,
    refine_samples: int = 7,
    couple_pair_limit: int = 3,
    couple_samples: int = 3,
) -> int:
    """Upper bound on sync evaluations (for progress bars)."""
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
    )
    if match.origin_image is not None:
        calibration.camera_center, _scale = core.apply_origin_and_scale(
            calibration,
            match.origin_image,
        )
    return calibration


def _sync_rmse(result: sync_module.SyncSolveResult) -> float:
    if result.mean_reprojection_px > 1.0e-9:
        return float(result.mean_reprojection_px)
    return _FAILURE_COST if not result.success else 0.0


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
) -> float:
    sync_term = _sync_rmse(sync_result)
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
    fx_span: float = DEFAULT_FX_SPAN,
    passes: int = 2,
    coarse_samples: int = 9,
    refine_samples: int = 7,
    couple_pair_limit: int = 3,
    couple_samples: int = 3,
    cancel_check=None,
    progress_callback=None,
) -> LensRefineResult:
    """Search free focals with coordinate descent, then a coupled polish.

    Coordinate descent moves one unlocked match at a time. The coupled polish
    then jointly varies focals for landmark-sharing pairs (and a global
    relative-scale probe) so multi-camera FOV error can shrink together.

    Frozen matches (Manual FOV / 1-point / weak VPs) keep their starting fx but
    still contribute VP residual and sync observations.

    ``cancel_check`` is an optional ``() -> bool`` polled between evaluations.
    ``progress_callback(step, total, label)`` reports progress (may be called
    from a worker thread — keep it bpy-free).
    """
    if not matches:
        raise ValueError("No matches to refine")
    match_map = {item.match_id: item for item in matches}
    match_ids = [item.match_id for item in matches]
    if anchor_id not in match_map:
        raise ValueError("Anchor match is missing from the lens refine set")

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
    total_steps = estimate_refine_evaluation_count(
        len(free_ids),
        passes=passes,
        coarse_samples=coarse_samples,
        refine_samples=refine_samples,
        couple_pair_limit=couple_pair_limit,
        couple_samples=couple_samples,
    )
    step = 0

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def _progress(label: str) -> None:
        if progress_callback is not None:
            progress_callback(min(step, total_steps), total_steps, label)

    def evaluate(
        cals: dict[str, core.Calibration],
        initial_similarities: dict[str, sync_module.SimilarityTransform] | None = None,
    ):
        nonlocal step
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
        initial_rmse = _sync_rmse(initial_sync)
        final_rmse = _sync_rmse(best_sync)
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

    if not free_ids:
        return LensRefineResult(
            calibrations=best_cals,
            sync_result=best_sync,
            initial_cost=initial_cost,
            final_cost=best_cost,
            initial_sync_rmse=_sync_rmse(initial_sync),
            final_sync_rmse=_sync_rmse(best_sync),
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
                    initial_similarities=local_best_sync.similarities,
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
                    initial_similarities=local_best_sync.similarities,
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
                    initial_similarities=best_sync.similarities,
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
            initial_similarities=best_sync.similarities,
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
    initial_rmse = _sync_rmse(initial_sync)
    final_rmse = _sync_rmse(best_sync)
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
