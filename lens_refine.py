"""Refine per-match focal length from landmark sync + a VP angular prior.

Sync keeps intrinsics frozen. This outer loop varies ``fx`` (= ``fy``), rebuilds
orientation from VP lines at each candidate (locked-focal refine), and scores
``sync_rmse + vp_weight * Σ residual²``. Coordinate descent — one match at a
time — keeps the search cheap without SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core
from . import sync as sync_module


# Soft prior: 1° VP residual ≈ this many sync pixels in the joint cost.
DEFAULT_VP_WEIGHT = 8.0
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


def _joint_cost(
    calibrations: dict[str, core.Calibration],
    matches: dict[str, MatchLensInput],
    sync_result: sync_module.SyncSolveResult,
    *,
    vp_weight: float,
) -> float:
    sync_term = _sync_rmse(sync_result)
    if sync_term >= _FAILURE_COST:
        return _FAILURE_COST
    vp_term = 0.0
    for match_id, match in matches.items():
        calibration = calibrations[match_id]
        residual = core.vp_angular_residual_degrees(calibration, match.line_bundles)
        vp_term += residual * residual
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
    )


def _sample_focals(center: float, span: float, count: int) -> list[float]:
    low = max(center * (1.0 - span), 1.0)
    high = max(center * (1.0 + span), low + 1.0)
    return [float(value) for value in np.linspace(low, high, count)]


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
    fx_span: float = DEFAULT_FX_SPAN,
    passes: int = 2,
    coarse_samples: int = 9,
    refine_samples: int = 7,
) -> LensRefineResult:
    """Coordinate-descent search over free focals; returns best calibrations.

    Frozen matches (Manual FOV / 1-point / weak VPs) keep their starting fx but
    still contribute VP residual and sync observations.
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

    def evaluate(cals: dict[str, core.Calibration]):
        result = _run_sync(
            cals,
            match_ids,
            observations,
            line_observations,
            anchor_id,
            known_world,
            known_lines,
            parallel_pairs,
        )
        cost = _joint_cost(cals, match_map, result, vp_weight=vp_weight)
        return cost, result

    initial_cost, initial_sync = evaluate(calibrations)
    best_cost = initial_cost
    best_sync = initial_sync
    best_cals = {key: value for key, value in calibrations.items()}
    start_fx = {item.match_id: float(item.intrinsics.fx) for item in matches}

    free_ids = [item.match_id for item in matches if not item.freeze_focal]
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

    for _pass in range(max(1, int(passes))):
        for match_id in free_ids:
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
                trial = {key: value for key, value in local_best_cals.items()}
                trial[match_id] = calibration_at_focal(match_map[match_id], focal)
                cost, result = evaluate(trial)
                if cost + 1.0e-6 < local_best_cost:
                    local_best_cost = cost
                    local_best_sync = result
                    local_best_fx = focal
                    local_best_cals = trial

            # Fine pass around the coarse winner.
            fine_span = fx_span * 0.25
            for focal in _sample_focals(local_best_fx, fine_span, refine_samples):
                trial = {key: value for key, value in local_best_cals.items()}
                trial[match_id] = calibration_at_focal(match_map[match_id], focal)
                cost, result = evaluate(trial)
                if cost + 1.0e-6 < local_best_cost:
                    local_best_cost = cost
                    local_best_sync = result
                    local_best_cals = trial

            best_cost = local_best_cost
            best_sync = local_best_sync
            best_cals = local_best_cals

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
        message = (
            f"Lenses refined · sync {initial_rmse:.1f}→{final_rmse:.1f}px"
        )
        if changed:
            message += " · " + ", ".join(changed[:4])
    else:
        message = (
            f"No lens improvement · sync {final_rmse:.1f}px "
            f"(cost {initial_cost:.1f}→{best_cost:.1f})"
        )
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
