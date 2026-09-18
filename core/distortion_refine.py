"""Conservative division-distortion polish for a frozen joint fit."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import geometry
from .sync.projection import _project_shared_points
from .sync.types import SyncObservation, SyncSolveResult


MIN_DISTORTION_PICKS = 16
VALIDATION_STRIDE = 4
MIN_NORMALIZED_OUTER_RADIUS = 0.36
MIN_NORMALIZED_RADIUS_SPAN = 0.22
MAX_DIVISION_LAMBDA = 0.35
MAX_DIVISION_LAMBDA_CHANGE = 0.18
MIN_RELATIVE_RMSE_GAIN = 0.04
MIN_SIGMA_RMSE_GAIN = 0.12
_GRID_SAMPLES = 37
_REFINE_STEPS = 18


@dataclass
class DistortionRefineOutcome:
    calibrations: dict[str, geometry.Calibration]
    accepted_match_ids: list[str]
    skipped_reasons: dict[str, str]
    cancelled: bool = False
    diagnostics: dict[str, "DistortionCameraDiagnostic"] | None = None


@dataclass
class DistortionCameraDiagnostic:
    """Evidence and pick-validation scores for one frozen-camera distortion trial."""

    supported_point_picks: int
    start_lambda: float
    validation_point_picks: int = 0
    normalized_outer_radius: float | None = None
    normalized_radius_span: float | None = None
    best_lambda: float | None = None
    start_fit_rmse_px: float | None = None
    best_fit_rmse_px: float | None = None
    start_validation_rmse_px: float | None = None
    best_validation_rmse_px: float | None = None
    start_full_rmse_px: float | None = None
    best_full_rmse_px: float | None = None
    accepted: bool = False
    reason: str = ""


def _weighted_rmse(errors: np.ndarray, weights: np.ndarray) -> float:
    squared = np.sum(np.asarray(errors, dtype=np.float64) ** 2, axis=1)
    return float(np.sqrt(np.sum(weights * squared) / max(float(np.sum(weights)), 1.0e-12)))


def _calibration_at_lambda(
    calibration: geometry.Calibration,
    division_lambda: float,
) -> geometry.Calibration:
    return replace(
        calibration,
        rotation_w2c=np.array(calibration.rotation_w2c, copy=True),
        camera_center=np.array(calibration.camera_center, copy=True),
        division_lambda=float(division_lambda),
        lambda_saturated=False,
    )


def _division_valid_over_image(
    intrinsics: geometry.CameraIntrinsics,
    division_lambda: float,
) -> bool:
    """Require forward and inverse division maps to stay real on the full plate."""
    x_radius = max(abs(float(intrinsics.cx)),
                   abs(float(intrinsics.image_width) - float(intrinsics.cx))) / float(intrinsics.fx)
    y_radius = max(abs(float(intrinsics.cy)),
                   abs(float(intrinsics.image_height) - float(intrinsics.cy))) / float(intrinsics.fy)
    corner_radius_squared = x_radius * x_radius + y_radius * y_radius
    value = float(division_lambda)
    return (1.0 - 4.0 * value * corner_radius_squared > 0.05 and
            1.0 + value * corner_radius_squared > 0.05)


def refine_division_distortion(
    calibrations: dict[str, geometry.Calibration],
    observations: list[SyncObservation],
    result: SyncSolveResult,
    *,
    pick_sigma_px: float,
    cancel_check=None,
) -> DistortionRefineOutcome:
    """Polish one radial coefficient per camera while all geometry stays fixed."""
    output = dict(calibrations)
    accepted: list[str] = []
    skipped: dict[str, str] = {}
    diagnostics: dict[str, DistortionCameraDiagnostic] = {}
    by_match: dict[str, list[SyncObservation]] = {}
    for item in observations:
        if item.match_id in calibrations and item.landmark_id in result.landmarks:
            by_match.setdefault(item.match_id, []).append(item)

    for match_id, calibration in calibrations.items():
        if cancel_check and cancel_check():
            return DistortionRefineOutcome(
                dict(calibrations), [], skipped, cancelled=True, diagnostics=diagnostics)
        items = by_match.get(match_id, [])
        diagnostic = DistortionCameraDiagnostic(
            supported_point_picks=len(items),
            start_lambda=float(calibration.division_lambda),
        )
        diagnostics[match_id] = diagnostic

        def skip(reason: str) -> None:
            skipped[match_id] = reason
            diagnostic.reason = reason

        if geometry.has_brown_conrady(calibration.brown_conrady):
            skip("imported Brown–Conrady distortion is already active")
            continue
        if match_id not in result.similarities:
            skip("camera has no fitted pose")
            continue
        if len(items) < MIN_DISTORTION_PICKS:
            skip(f"needs at least {MIN_DISTORTION_PICKS} supported point picks")
            continue

        observed = np.asarray([(item.u, item.v) for item in items], dtype=np.float64)
        intrinsics = calibration.intrinsics
        radii = np.sqrt(
            ((observed[:, 0] - intrinsics.cx) / intrinsics.fx) ** 2
            + ((observed[:, 1] - intrinsics.cy) / intrinsics.fy) ** 2
        )
        order = np.argsort(radii, kind="stable")
        validation_indices = order[VALIDATION_STRIDE - 1::VALIDATION_STRIDE]
        diagnostic.validation_point_picks = int(len(validation_indices))
        validation_mask = np.zeros(len(items), dtype=bool)
        validation_mask[validation_indices] = True
        fit_mask = ~validation_mask
        radius_span = float(np.quantile(radii, 0.9) - np.quantile(radii, 0.1))
        diagnostic.normalized_outer_radius = float(np.quantile(radii, 0.9))
        diagnostic.normalized_radius_span = radius_span
        if (len(validation_indices) < 4 or
                float(np.quantile(radii, 0.9)) < MIN_NORMALIZED_OUTER_RADIUS or
                radius_span < MIN_NORMALIZED_RADIUS_SPAN or
                float(np.max(radii[fit_mask])) < MIN_NORMALIZED_OUTER_RADIUS or
                float(np.max(radii[validation_mask])) < MIN_NORMALIZED_OUTER_RADIUS):
            skip("point picks do not cover enough of the image radius")
            continue

        points = np.asarray([result.landmarks[item.landmark_id] for item in items])
        pinhole = _calibration_at_lambda(calibration, 0.0)
        ideal, valid = _project_shared_points(
            points, pinhole, result.similarities[match_id])
        if not np.all(valid) or not np.isfinite(ideal).all():
            skip("fitted points do not project in front of the camera")
            continue
        weights = np.asarray([max(float(item.weight), 1.0e-12) for item in items])

        def errors_at(value: float) -> np.ndarray:
            projected = geometry.distort_points(
                ideal, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                float(value), ())
            return projected - observed

        def fit_loss(value: float) -> float:
            if not _division_valid_over_image(intrinsics, value):
                return float("inf")
            errors = errors_at(value)[fit_mask]
            return float(np.sum(weights[fit_mask] * np.sum(errors * errors, axis=1)))

        start = float(calibration.division_lambda)
        if (not np.isfinite(start) or abs(start) > MAX_DIVISION_LAMBDA or
                not _division_valid_over_image(intrinsics, start)):
            skip("existing distortion lies outside the conservative model bound")
            continue
        low = max(-MAX_DIVISION_LAMBDA, start - MAX_DIVISION_LAMBDA_CHANGE)
        high = min(MAX_DIVISION_LAMBDA, start + MAX_DIVISION_LAMBDA_CHANGE)
        if low >= high:
            skip("existing distortion leaves no conservative search range")
            continue
        samples = np.linspace(low, high, _GRID_SAMPLES)
        losses = []
        for value in samples:
            if cancel_check and cancel_check():
                return DistortionRefineOutcome(
                    dict(calibrations), [], skipped, cancelled=True, diagnostics=diagnostics)
            losses.append(fit_loss(float(value)))
        best_index = int(np.argmin(losses))
        bracket_low = float(samples[max(best_index - 1, 0)])
        bracket_high = float(samples[min(best_index + 1, len(samples) - 1)])
        for _ in range(_REFINE_STEPS):
            if cancel_check and cancel_check():
                return DistortionRefineOutcome(
                    dict(calibrations), [], skipped, cancelled=True, diagnostics=diagnostics)
            first = bracket_low + (bracket_high - bracket_low) / 3.0
            second = bracket_high - (bracket_high - bracket_low) / 3.0
            if fit_loss(first) <= fit_loss(second):
                bracket_high = second
            else:
                bracket_low = first
        candidate = 0.5 * (bracket_low + bracket_high)
        diagnostic.best_lambda = candidate
        if cancel_check and cancel_check():
            return DistortionRefineOutcome(
                dict(calibrations), [], skipped, cancelled=True, diagnostics=diagnostics)
        if not _division_valid_over_image(intrinsics, candidate):
            skip("best correction is not invertible across the full image")
            continue
        start_errors = errors_at(start)
        candidate_errors = errors_at(candidate)
        start_fit = _weighted_rmse(start_errors[fit_mask], weights[fit_mask])
        final_fit = _weighted_rmse(candidate_errors[fit_mask], weights[fit_mask])
        start_validation = _weighted_rmse(
            start_errors[validation_mask], weights[validation_mask])
        final_validation = _weighted_rmse(
            candidate_errors[validation_mask], weights[validation_mask])
        start_full = _weighted_rmse(start_errors, weights)
        final_full = _weighted_rmse(candidate_errors, weights)
        diagnostic.start_fit_rmse_px = start_fit
        diagnostic.best_fit_rmse_px = final_fit
        diagnostic.start_validation_rmse_px = start_validation
        diagnostic.best_validation_rmse_px = final_validation
        diagnostic.start_full_rmse_px = start_full
        diagnostic.best_full_rmse_px = final_full
        absolute_gain = max(MIN_SIGMA_RMSE_GAIN * float(pick_sigma_px), 0.05)

        def supported(before: float, after: float) -> bool:
            return (before - after >= absolute_gain and
                    after <= before * (1.0 - MIN_RELATIVE_RMSE_GAIN))

        if not (supported(start_fit, final_fit) and
                supported(start_validation, final_validation) and
                supported(start_full, final_full)):
            skip("validation points do not support a distortion correction")
            continue
        boundary_margin = (high - low) / max(_GRID_SAMPLES - 1, 1)
        if candidate <= low + boundary_margin or candidate >= high - boundary_margin:
            skip("best correction reaches the conservative distortion bound")
            continue
        output[match_id] = _calibration_at_lambda(calibration, candidate)
        accepted.append(match_id)
        diagnostic.accepted = True

    return DistortionRefineOutcome(output, accepted, skipped, diagnostics=diagnostics)
