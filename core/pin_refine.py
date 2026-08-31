"""Single-camera pose / FOV / principal-point refine from Known 3D pins.

Observations are source-image pixels. Distortion stays hard-locked per
hypothesis; VP lines are a soft cost plus a baseline-relative guardrail.
Auto from VPs / Estimate Distortion keep orientation on the VP-line
manifold as FOV / PP change, so verticals are not left behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as core
from .lens_refine import (
    DEFAULT_VP_LINE_SLACK_PX,
    DEFAULT_VP_WEIGHT,
)
from .sync.projection import _log_rodrigues, _rodrigues, project_private_point

MIN_PIN_COUNT = 4
HYPOTHESIS_STORED = "stored_distortion"
HYPOTHESIS_PINHOLE = "pinhole"

_BEHIND_RESIDUAL_PX = 1.0e3
_T_SIGMA = 1.0
_R_SIGMA = np.radians(4.0)
_FX_SIGMA = 0.18
_PP_SIGMA = 200.0
_PIN_HUBER_PX = 40.0
PIN_ACCEPT_RMSE_PX = 25.0
# Alternate Auto from VPs (Known 3D polish) and Solve Sync until joint RMSE
# stops falling. Pixel floor avoids chasing noise after the overlay has settled.
PIN_SYNC_MAX_ROUNDS = 8
PIN_SYNC_MIN_IMPROVE_PX = 0.02
_AXIS_SCORE_WEIGHT = 3.0
_AXIS_VP_SLACK_PX = 1.5
_HFOV_MIN_DEG = 8.0
_HFOV_MAX_DEG = 140.0
_JACOBIAN_STEP = 1.0e-5
_MAX_ITERATIONS = 25
_DAMPING_START = 1.0e-2


@dataclass
class KnownPin:
    """One 3D Empty origin and its source-image pick in the private camera frame."""

    landmark_id: str
    point_private: np.ndarray
    u: float
    v: float
    weight: float = 1.0
    landmark_name: str = ""


@dataclass
class PinMetrics:
    """Reprojection of Known 3D pins through one calibration."""

    rms_px: float
    max_px: float
    per_pin_px: tuple[float, ...] = ()
    behind_ids: tuple[str, ...] = ()


@dataclass
class PinRefineResult:
    """Outcome of a distortion-locked Known 3D camera refine."""

    success: bool
    calibration: core.Calibration
    pin_rms_px: float
    pin_max_px: float
    per_pin_px: tuple[float, ...] = ()
    vp_line_rms_px: float = 0.0
    vp_angle_deg: float = 0.0
    initial_pin_rms_px: float = 0.0
    initial_vp_line_rms_px: float = 0.0
    axis_rms_px: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hypothesis: str = HYPOTHESIS_STORED
    behind_ids: tuple[str, ...] = ()
    message: str = ""
    improved: bool = False


def copy_calibration(calibration: core.Calibration) -> core.Calibration:
    """Deep-copy a calibration, including numpy pose arrays."""
    source = calibration.intrinsics
    return core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=float(source.fx),
            fy=float(source.fy),
            cx=float(source.cx),
            cy=float(source.cy),
            image_width=int(source.image_width),
            image_height=int(source.image_height),
        ),
        rotation_w2c=np.array(calibration.rotation_w2c, dtype=np.float64, copy=True),
        camera_center=np.array(calibration.camera_center, dtype=np.float64, copy=True),
        division_lambda=float(calibration.division_lambda),
        lambda_saturated=bool(calibration.lambda_saturated),
        brown_conrady=tuple(calibration.brown_conrady),
    )


def pin_metrics(pins: list[KnownPin], calibration: core.Calibration) -> PinMetrics:
    """Source-pixel reprojection of each pin; behind-camera pins are listed."""
    squared: list[float] = []
    per_pin: list[float] = []
    behind: list[str] = []
    for pin in pins:
        projected = project_private_point(pin.point_private, calibration)
        if projected is None:
            behind.append(pin.landmark_id)
            per_pin.append(float("inf"))
            continue
        error = float(
            np.hypot(float(projected[0]) - pin.u, float(projected[1]) - pin.v)
        )
        per_pin.append(error)
        squared.append(error * error)
    if not squared:
        return PinMetrics(
            float("inf"),
            float("inf"),
            tuple(per_pin),
            tuple(behind),
        )
    rms = float(np.sqrt(np.mean(squared)))
    finite = [value for value in per_pin if np.isfinite(value)]
    return PinMetrics(
        rms,
        float(max(finite)) if finite else float("inf"),
        tuple(per_pin),
        tuple(behind),
    )


def pin_sync_round_improved(previous_rmse_px: float, next_rmse_px: float) -> bool:
    """True when the next joint RMSE is worth another Known 3D / Sync round."""
    if not np.isfinite(next_rmse_px):
        return False
    if not np.isfinite(previous_rmse_px):
        return True
    return float(previous_rmse_px) - float(next_rmse_px) >= PIN_SYNC_MIN_IMPROVE_PX


def refine_from_known_pins(
    calibration: core.Calibration,
    pins: list[KnownPin],
    line_bundles: dict[core.AxisId, list[core.LineSegment]] | None = None,
    *,
    vp_weight: float = DEFAULT_VP_WEIGHT,
    vp_line_slack_px: float = DEFAULT_VP_LINE_SLACK_PX,
    lock_rotation: bool = False,
    lock_focal: bool = False,
    orient_from_vp: bool = False,
) -> PinRefineResult:
    """Fit pose, focal, and principal point. Distortion is locked per hypothesis.

    ``lock_rotation`` keeps the seed orientation. ``orient_from_vp`` instead
    rebuilds orientation from VP lines at the current K so FOV / PP can move
    without leaving the strokes. ``lock_focal`` keeps Manual FOV / 1-point.
    """
    start = copy_calibration(calibration)
    start.intrinsics.fy = start.intrinsics.fx
    bundles = line_bundles or {"x": [], "y": [], "z": []}
    if len(pins) < MIN_PIN_COUNT:
        metrics = pin_metrics(pins, start)
        return PinRefineResult(
            False,
            start,
            metrics.rms_px,
            metrics.max_px,
            metrics.per_pin_px,
            core.vp_line_residual_rms(start, bundles),
            core.vp_angular_residual_degrees(start, bundles),
            metrics.rms_px,
            core.vp_line_residual_rms(start, bundles),
            _axis_rms_tuple(start, bundles),
            HYPOTHESIS_STORED,
            metrics.behind_ids,
            f"Need at least {MIN_PIN_COUNT} Known 3D point landmarks picked in this still",
        )

    initial_metrics = pin_metrics(pins, start)
    initial_vp_rms = core.vp_line_residual_rms(start, bundles)
    initial_axis = _axis_rms_tuple(start, bundles)
    freeze_rotation = bool(lock_rotation or orient_from_vp)
    has_vp = _has_vp_constraint(bundles)

    candidates: list[PinRefineResult] = []
    stored = _polish_hypothesis(
        start,
        pins,
        bundles,
        hypothesis=HYPOTHESIS_STORED,
        pp_prior=(float(start.intrinsics.cx), float(start.intrinsics.cy)),
        vp_weight=vp_weight if has_vp else 0.0,
        vp_baseline_rms=initial_vp_rms if has_vp else None,
        vp_line_slack_px=vp_line_slack_px,
        lock_rotation=freeze_rotation,
        lock_focal=lock_focal,
        orient_from_vp=orient_from_vp,
    )
    candidates.append(stored)
    if core.has_lens_distortion(start.division_lambda, start.brown_conrady):
        pinhole_seed = _pinhole_seed(
            start,
            bundles,
            lock_rotation=freeze_rotation,
        )
        pinhole_pp = (
            (float(start.intrinsics.cx), float(start.intrinsics.cy))
            if freeze_rotation
            else (
                float(start.intrinsics.image_width) * 0.5,
                float(start.intrinsics.image_height) * 0.5,
            )
        )
        pinhole = _polish_hypothesis(
            pinhole_seed,
            pins,
            bundles,
            hypothesis=HYPOTHESIS_PINHOLE,
            pp_prior=pinhole_pp,
            vp_weight=vp_weight if has_vp else 0.0,
            vp_baseline_rms=None,
            vp_line_slack_px=vp_line_slack_px,
            lock_rotation=freeze_rotation,
            lock_focal=lock_focal,
            orient_from_vp=orient_from_vp,
        )
        candidates.append(pinhole)

    accepted: list[PinRefineResult] = []
    for candidate in candidates:
        if not np.isfinite(candidate.pin_rms_px):
            continue
        if candidate.behind_ids:
            continue
        pinhole_slack = max(float(vp_line_slack_px), 6.0)
        slack = (
            vp_line_slack_px
            if candidate.hypothesis == HYPOTHESIS_STORED
            else pinhole_slack
        )
        if has_vp and not _passes_vp_guardrail(candidate, initial_vp_rms, slack):
            continue
        axis_slack = (
            _AXIS_VP_SLACK_PX
            if candidate.hypothesis == HYPOTHESIS_STORED
            else max(_AXIS_VP_SLACK_PX, 6.0)
        )
        if has_vp and not _passes_axis_guardrail(
            candidate.axis_rms_px, initial_axis, axis_slack
        ):
            continue
        accepted.append(candidate)

    if not accepted:
        failed = min(candidates, key=_joint_score)
        failed.success = False
        failed.initial_pin_rms_px = initial_metrics.rms_px
        failed.initial_vp_line_rms_px = initial_vp_rms
        failed.message = _failure_message(
            candidates,
            pins,
            initial_metrics,
            initial_vp_rms,
        )
        return failed

    winner = min(accepted, key=_joint_score)
    winner.initial_pin_rms_px = initial_metrics.rms_px
    winner.initial_vp_line_rms_px = initial_vp_rms
    winner.improved = winner.pin_rms_px + 1.0e-6 < initial_metrics.rms_px
    if winner.pin_rms_px > PIN_ACCEPT_RMSE_PX:
        winner.success = False
        winner.message = _failure_message(
            candidates,
            pins,
            initial_metrics,
            initial_vp_rms,
        )
        return winner
    winner.success = True
    winner.message = _success_message(winner, initial_metrics.rms_px)
    return winner


def _has_vp_constraint(line_bundles: dict[core.AxisId, list[core.LineSegment]]) -> bool:
    return any(line_bundles.get(axis) for axis in ("x", "y", "z"))


def _axis_rms_tuple(
    calibration: core.Calibration,
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
) -> tuple[float, float, float]:
    mapping = core.vp_line_axis_residuals(calibration, line_bundles)
    return (
        float(mapping.get("x", 0.0)),
        float(mapping.get("y", 0.0)),
        float(mapping.get("z", 0.0)),
    )


def _joint_score(result: PinRefineResult) -> tuple[float, float]:
    """Prefer cameras that do not wreck the worst VP axis just to shave pin RMS."""
    worst_axis = max(result.axis_rms_px) if result.axis_rms_px else result.vp_line_rms_px
    return (
        float(result.pin_rms_px) + _AXIS_SCORE_WEIGHT * float(worst_axis),
        float(result.vp_line_rms_px),
    )


def _passes_axis_guardrail(
    candidate_axis: tuple[float, float, float],
    baseline_axis: tuple[float, float, float],
    slack_px: float,
) -> bool:
    for candidate, baseline in zip(candidate_axis, baseline_axis):
        if float(candidate) > float(baseline) + float(slack_px):
            return False
    return True


def _with_vp_orientation(
    calibration: core.Calibration,
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
) -> core.Calibration:
    """Rebuild R from VP lines at this K; keep translation and intrinsics."""
    working = core.undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
        calibration.brown_conrady,
    )
    rotation = core.rotation_from_orthogonal_lines(
        working,
        calibration.intrinsics,
        initial_rotation=calibration.rotation_w2c,
    )
    if rotation is None:
        return calibration
    updated = copy_calibration(calibration)
    updated.rotation_w2c = np.array(rotation, dtype=np.float64, copy=True)
    return updated


def _pinhole_seed(
    start: core.Calibration,
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
    *,
    lock_rotation: bool = False,
) -> core.Calibration:
    """λ = 0 with the stored pose/PP; re-orient from VP lines when possible."""
    seed = copy_calibration(start)
    seed.division_lambda = 0.0
    seed.lambda_saturated = False
    seed.brown_conrady = ()
    if lock_rotation or not _has_vp_constraint(line_bundles):
        return seed
    if not core.can_solve_orientation(line_bundles, lock_focal=True, vp_mode="3"):
        return seed
    oriented = core.refine_camera(
        line_bundles,
        seed.intrinsics,
        lock_focal=True,
        estimate_principal_point=False,
        estimate_distortion=False,
        initial_division_lambda=0.0,
        initial_rotation=seed.rotation_w2c,
    )
    oriented.camera_center = np.array(start.camera_center, dtype=np.float64, copy=True)
    return oriented


def _huber_scale(residual: float, delta: float = _PIN_HUBER_PX) -> float:
    """Taper huge pin misses so a swapped pair cannot steer the camera."""
    abs_residual = abs(float(residual))
    if abs_residual <= float(delta):
        return float(residual)
    return float(residual) * float(np.sqrt(delta / abs_residual))


def _passes_vp_guardrail(
    candidate: PinRefineResult,
    baseline_rms: float,
    line_slack_px: float,
) -> bool:
    """Keep local VP-line RMSE close to the invocation; ignore far-VP angles."""
    if not np.isfinite(baseline_rms) or baseline_rms >= 90.0:
        return True
    return candidate.vp_line_rms_px <= float(baseline_rms) + float(line_slack_px)


def _worst_pin_text(pins: list[KnownPin], metrics: PinMetrics, *, limit: int = 2) -> str:
    ranked = sorted(
        zip(pins, metrics.per_pin_px),
        key=lambda item: float(item[1]) if np.isfinite(item[1]) else 1.0e9,
        reverse=True,
    )
    parts = []
    for pin, error in ranked[: max(0, int(limit))]:
        name = pin.landmark_name or pin.landmark_id
        if np.isfinite(error):
            parts.append(f"{name} {error:.0f}px")
        else:
            parts.append(f"{name} behind camera")
    return ", ".join(parts)


def _failure_message(
    candidates: list[PinRefineResult],
    pins: list[KnownPin],
    initial_metrics: PinMetrics,
    initial_vp_rms: float,
) -> str:
    trials = "; ".join(
        (
            f"{item.hypothesis.replace('_', ' ')} pin {item.pin_rms_px:.1f}px "
            f"VP {item.vp_line_rms_px:.2f}px"
            for item in candidates
        )
    )
    worst = _worst_pin_text(pins, initial_metrics)
    message = (
        f"Known 3D refine could not fit pins without breaking VP lines "
        f"(start pin {initial_metrics.rms_px:.1f}px, VP {initial_vp_rms:.2f}px). "
        f"{trials}"
    )
    if worst:
        message += f". Worst picks: {worst}"
    return message


def _success_message(result: PinRefineResult, initial_rms: float) -> str:
    label = "pinhole" if result.hypothesis == HYPOTHESIS_PINHOLE else "stored lens"
    message = (
        f"Known 3D refine · HFOV {result.calibration.hfov_degrees:.2f}° · "
        f"pin RMS {result.pin_rms_px:.2f} px · {label}"
    )
    if initial_rms < 1.0 and not result.improved:
        message += " · picks already match this camera"
    return message


def _pack_params(
    calibration: core.Calibration,
    *,
    lock_rotation: bool,
    lock_focal: bool,
) -> np.ndarray:
    values = [*calibration.camera_center.tolist()]
    if not lock_rotation:
        values.extend(_log_rodrigues(calibration.rotation_w2c).tolist())
    if not lock_focal:
        values.append(float(np.log(max(float(calibration.intrinsics.fx), 1.0))))
    values.append(float(calibration.intrinsics.cx))
    values.append(float(calibration.intrinsics.cy))
    return np.array(values, dtype=np.float64)


def _unpack_calibration(
    params: np.ndarray,
    template: core.Calibration,
    *,
    division_lambda: float,
    brown_conrady: tuple[float, ...],
    lock_rotation: bool,
    lock_focal: bool,
) -> core.Calibration:
    width = int(template.intrinsics.image_width)
    height = int(template.intrinsics.image_height)
    index = 3
    if lock_rotation:
        rotation = np.array(template.rotation_w2c, dtype=np.float64, copy=True)
    else:
        rotation = _rodrigues(params[index : index + 3])
        index += 3
    if lock_focal:
        focal = float(template.intrinsics.fx)
    else:
        min_focal = core.focal_from_hfov(_HFOV_MAX_DEG, width)
        max_focal = core.focal_from_hfov(_HFOV_MIN_DEG, width)
        focal = float(np.clip(np.exp(float(params[index])), min_focal, max_focal))
        index += 1
    cx = float(np.clip(params[index], -0.25 * width, 1.25 * width))
    cy = float(np.clip(params[index + 1], -0.25 * height, 1.25 * height))
    return core.Calibration(
        intrinsics=core.CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=cx,
            cy=cy,
            image_width=width,
            image_height=height,
        ),
        rotation_w2c=rotation,
        camera_center=np.array(params[0:3], dtype=np.float64),
        division_lambda=float(division_lambda),
        lambda_saturated=False,
        brown_conrady=tuple(brown_conrady),
    )


def _residual_vector(
    params: np.ndarray,
    template: core.Calibration,
    pins: list[KnownPin],
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
    start_params: np.ndarray,
    pp_prior: tuple[float, float],
    *,
    division_lambda: float,
    brown_conrady: tuple[float, ...],
    vp_weight: float,
    lock_rotation: bool,
    lock_focal: bool,
    orient_from_vp: bool = False,
) -> np.ndarray:
    calibration = _unpack_calibration(
        params,
        template,
        division_lambda=division_lambda,
        brown_conrady=brown_conrady,
        lock_rotation=lock_rotation,
        lock_focal=lock_focal,
    )
    if orient_from_vp:
        calibration = _with_vp_orientation(calibration, line_bundles)
    residuals: list[float] = []
    for pin in pins:
        scale = float(np.sqrt(max(float(pin.weight), 1.0e-12)))
        projected = project_private_point(pin.point_private, calibration)
        if projected is None:
            residuals.extend((_BEHIND_RESIDUAL_PX * scale, _BEHIND_RESIDUAL_PX * scale))
            continue
        residuals.append(_huber_scale(scale * (float(projected[0]) - pin.u)))
        residuals.append(_huber_scale(scale * (float(projected[1]) - pin.v)))
    if vp_weight > 0.0 and _has_vp_constraint(line_bundles):
        axis_map = core.vp_line_axis_residuals(calibration, line_bundles)
        if axis_map:
            residuals.append(
                float(vp_weight)
                * float(np.sqrt(len(axis_map)))
                * float(np.mean(list(axis_map.values())))
            )
    delta = params - start_params
    residuals.extend((delta[0] / _T_SIGMA, delta[1] / _T_SIGMA, delta[2] / _T_SIGMA))
    index = 3
    if not lock_rotation:
        residuals.extend(
            (delta[index] / _R_SIGMA, delta[index + 1] / _R_SIGMA, delta[index + 2] / _R_SIGMA)
        )
        index += 3
    if not lock_focal:
        residuals.append(delta[index] / _FX_SIGMA)
        index += 1
    residuals.append((float(params[index]) - pp_prior[0]) / _PP_SIGMA)
    residuals.append((float(params[index + 1]) - pp_prior[1]) / _PP_SIGMA)
    return np.asarray(residuals, dtype=np.float64)


def _numeric_jacobian(
    params: np.ndarray,
    residual_kwargs: dict,
) -> np.ndarray:
    base = _residual_vector(params, **residual_kwargs)
    jacobian = np.zeros((base.size, params.size), dtype=np.float64)
    for index in range(params.size):
        perturbed = params.copy()
        magnitude = abs(float(params[index]))
        delta = _JACOBIAN_STEP if magnitude < 1.0 else _JACOBIAN_STEP * magnitude
        perturbed[index] += delta
        sample = _residual_vector(perturbed, **residual_kwargs)
        jacobian[:, index] = (sample - base) / delta
    return jacobian


def _polish_hypothesis(
    seed: core.Calibration,
    pins: list[KnownPin],
    line_bundles: dict[core.AxisId, list[core.LineSegment]],
    *,
    hypothesis: str,
    pp_prior: tuple[float, float],
    vp_weight: float,
    vp_baseline_rms: float | None = None,
    vp_line_slack_px: float = DEFAULT_VP_LINE_SLACK_PX,
    lock_rotation: bool = False,
    lock_focal: bool = False,
    orient_from_vp: bool = False,
) -> PinRefineResult:
    template = copy_calibration(seed)
    start_params = _pack_params(
        template, lock_rotation=lock_rotation, lock_focal=lock_focal
    )
    params = start_params.copy()
    residual_kwargs = {
        "template": template,
        "pins": pins,
        "line_bundles": line_bundles,
        "start_params": start_params,
        "pp_prior": pp_prior,
        "division_lambda": float(template.division_lambda),
        "brown_conrady": tuple(template.brown_conrady),
        "vp_weight": float(vp_weight),
        "lock_rotation": bool(lock_rotation),
        "lock_focal": bool(lock_focal),
        "orient_from_vp": bool(orient_from_vp),
    }
    damping = _DAMPING_START
    previous_cost = float("inf")
    for _iteration in range(_MAX_ITERATIONS):
        residuals = _residual_vector(params, **residual_kwargs)
        cost = float(residuals @ residuals)
        if cost < 1.0e-8:
            break
        if abs(previous_cost - cost) / max(previous_cost, 1.0e-12) < 1.0e-8:
            break
        jacobian = _numeric_jacobian(params, residual_kwargs)
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        step_accepted = False
        for _attempt in range(8):
            try:
                delta = np.linalg.solve(
                    gram + damping * np.diag(np.maximum(np.diag(gram), 1.0e-8)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = params + delta
            candidate_residuals = _residual_vector(candidate, **residual_kwargs)
            candidate_cost = float(candidate_residuals @ candidate_residuals)
            if candidate_cost < cost:
                if vp_baseline_rms is not None:
                    trial = _unpack_calibration(
                        candidate,
                        template,
                        division_lambda=float(template.division_lambda),
                        brown_conrady=tuple(template.brown_conrady),
                        lock_rotation=lock_rotation,
                        lock_focal=lock_focal,
                    )
                    if orient_from_vp:
                        trial = _with_vp_orientation(trial, line_bundles)
                    trial_vp = core.vp_line_residual_rms(trial, line_bundles)
                    if trial_vp > float(vp_baseline_rms) + float(vp_line_slack_px):
                        damping *= 10.0
                        continue
                params = candidate
                previous_cost = cost
                damping = max(damping * 0.3, 1.0e-8)
                step_accepted = True
                break
            damping *= 10.0
        if not step_accepted:
            break

    refined = _unpack_calibration(
        params,
        template,
        division_lambda=float(template.division_lambda),
        brown_conrady=tuple(template.brown_conrady),
        lock_rotation=lock_rotation,
        lock_focal=lock_focal,
    )
    if orient_from_vp:
        refined = _with_vp_orientation(refined, line_bundles)
    metrics = pin_metrics(pins, refined)
    return PinRefineResult(
        True,
        refined,
        metrics.rms_px,
        metrics.max_px,
        metrics.per_pin_px,
        core.vp_line_residual_rms(refined, line_bundles),
        core.vp_angular_residual_degrees(refined, line_bundles),
        axis_rms_px=_axis_rms_tuple(refined, line_bundles),
        hypothesis=hypothesis,
        behind_ids=metrics.behind_ids,
    )
