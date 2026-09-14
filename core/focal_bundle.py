"""Independent pinhole focal bundle adjustment for supported landmark graphs.

The anchor camera fixes center; supplied world directions can fit orientation.
Free-scale graphs fix one camera baseline; an off-anchor supplied mirror plane
instead fixes metric scale.
All decisions use image picks, supplied relations and the fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time

import numpy as np

from . import geometry as core
from .focal_constraints import PointFocalConstraints
from .focal_lines import LineChart, canonical_line_point, endpoint_distances
from .focal_line_constraints import LineFocalConstraints, LINE_RELATION_DIRECTION_HARD_SINE
from .focal_optimizer import bounded_lm_step
from .focal_orientation import orientation_basis, overall_rotation
from .focal_point_priors import PointReferenceConstraints
from .focal_projection import project_camera_points, line_endpoint_distances
from .joint_fit_score import (JointFitScorer, joint_line_support_diagnostics,
                              supported_joint_request)
from .focal_startup import provisional_poses
from .sync import SyncObservation, SyncLineObservation, SyncSolveResult, SimilarityTransform, SyncMatchInput
from .sync.request import SyncSolveRequest
from .sync.constants import (MIRROR_PAIR_HARD_GAP, PLANE_HARD_SLACK,
                             GROUND_SLACK_DEFAULT, KNOWN_3D_SLACK_DEFAULT)
from .sync.lines import (_reconstruct_line_from_observations,
                         _finite_segment_from_line_observations,
                         _closest_point_on_line_to_ray)
from .sync.mirrors import _dedupe_mirror_pairs
from .sync.projection import _log_rodrigues, _rodrigues
from .sync.pose import _snap_to_axis_aligned_rotation


DEFAULT_POINT_FOCAL_SPAN = 0.4
# Resource guard for the dense joint fit, not an identifiability limit.
MAX_CAMERAS = 32
MAX_POINTS = 80
MAX_LINES = 24
MAX_LINE_STROKES = 96
MAX_DENSE_JACOBIAN_BYTES = 192 * 1024 * 1024
LINE_MIRROR_DIRECTION_HARD_SINE = 0.01
MAX_ITERATIONS = 400
MAX_SECONDS = 60.0
RELATIVE_COST_TOLERANCE = 1.0e-7
MIN_CANDIDATE_RELATIVE_GAIN = 1.0e-9
FOCAL_BOUND_MARGIN = 1.0e-4
MAX_LOG_METRIC_BASELINE_CHANGE = 10.0
METRIC_BASELINE_BOUND_MARGIN = 1e-4
EPIPOLAR_HINT_MIN_WITHHELD_SIGMA = 8.0
EPIPOLAR_HINT_MAX_FITS = 1000
EPIPOLAR_HINT_MAX_SECONDS = 10.0


@dataclass
class FocalFitCandidate:
    """Lower combined error with physical checks, without calibration claims."""

    calibrations: dict[str, core.Calibration]
    sync_result: SyncSolveResult
    initial_rmse_px: float
    fitted_rmse_px: float
    reason: str
    initial_objective: float | None = None
    fitted_objective: float | None = None


@dataclass
class FocalBundleOutcome:
    accepted: bool
    reason: str
    calibrations: dict[str, core.Calibration] = field(default_factory=dict)
    sync_result: SyncSolveResult | None = None
    intervals_px: dict[str, tuple[float, float]] = field(default_factory=dict)
    initial_rmse_px: float = float("inf")
    fitted_rmse_px: float = float("inf")
    candidate: FocalFitCandidate | None = None
    initial_objective: float | None = None
    fitted_objective: float | None = None
    constraint_gaps: dict[str, float] = field(default_factory=dict)
    support_coverage: dict[str, object] = field(default_factory=dict)


def _estimated_dense_jacobian_bytes(parameter_count: int, point_picks: int,
                                    line_strokes: int, point_count: int,
                                    line_count: int, relation_count: int) -> int:
    """Conservative full pixel-plus-prior Jacobian allocation bound."""
    rows = 2 * (point_picks + line_strokes) + 16 * (
        point_count + line_count + relation_count + 1)
    return rows * parameter_count * 8


def _redundant_hard_line_columns(line_constraints: LineFocalConstraints,
                                 points: np.ndarray, line_charts: list[LineChart],
                                 line_offset: int) -> set[int]:
    """Remove one angular chart coordinate per fitted hard-plane line."""
    geometry = [chart.decode(chart.initial) for chart in line_charts]
    return {
        line_offset + 4 * line + int(np.argmax(np.abs(line_charts[line].seed_basis @ normal)))
        for line, normal in line_constraints.hard_direction_normals(points, geometry).items()
    }


def _coupled_hard_line_columns(line_constraints: LineFocalConstraints,
                               point_offset: int, line_end: int,
                               orientation_end: int) -> set[int]:
    """Identify moving plane support and common-frame columns seen by line pixels."""
    if not line_constraints.hard_plane or not line_constraints.groups:
        return set()
    free_support = {point for normal, members, _lines in line_constraints.groups
                    if normal is None for point in members}
    columns = {point_offset + 3 * point + axis
               for point in free_support for axis in range(3)}
    columns.update(range(line_end, orientation_end))
    return columns


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    spread = np.linalg.norm(points - center, axis=1).mean()
    if spread <= 1.0e-9:
        raise ValueError("Point picks have no image spread")
    if np.linalg.matrix_rank(points - center) < 2:
        raise ValueError("Point picks are collinear")
    scale = math.sqrt(2.0) / spread
    transform = np.array(((scale, 0.0, -scale * center[0]),
                          (0.0, scale, -scale * center[1]), (0.0, 0.0, 1.0)))
    return (points - center) * scale, transform


def _project_with_depth_penalty(q: np.ndarray, focal: float,
                                principal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera points and return exact derivative of the clipped-depth residual."""
    z = q[:, 2]
    safe_z = np.maximum(z, 1.0e-6)
    predicted = focal * q[:, :2] / safe_z[:, None] + principal
    predicted += np.maximum(1.0e-6 - z, 0.0)[:, None] * 1.0e3
    derivative = np.zeros((len(q), 2, 3))
    derivative[:, 0, 0] = focal / safe_z
    derivative[:, 1, 1] = focal / safe_z
    positive = z > 1.0e-6
    derivative[positive, 0, 2] = -focal * q[positive, 0] / z[positive]**2
    derivative[positive, 1, 2] = -focal * q[positive, 1] / z[positive]**2
    derivative[~positive, :, 2] = -1.0e3
    return predicted, derivative


def _line_image_fd_jacobian(
    params: np.ndarray, baseline: np.ndarray, *,
    camera_columns: list[tuple[int, ...]], camera_strokes: list[np.ndarray],
    line_strokes: list[np.ndarray], camera_state, line_charts: list[LineChart],
    line_offset: int, lci: np.ndarray, lli: np.ndarray, luv: np.ndarray,
    line_weights: np.ndarray, pp: np.ndarray, focal: np.ndarray,
    rotations: list[np.ndarray], centers: np.ndarray,
    geometry: list[tuple[np.ndarray, np.ndarray]],
    line_distance=None, geometry_at=None,
) -> np.ndarray:
    """Differentiate each stroke only for its camera and line coordinates.

    camera_state returns the changed focal, rotation or center; other fields are None.
    """
    jac = np.zeros((len(baseline), len(params)))
    for camera, columns in enumerate(camera_columns):
        strokes = camera_strokes[camera]
        if not len(strokes):
            continue
        for column in columns:
            step = 1e-6 * max(1.0, abs(params[column]))
            shifted = params.copy()
            shifted[column] += step
            changed_focal, changed_rotation, changed_center = camera_state(
                shifted, camera, column)
            shifted_focal = focal[camera] if changed_focal is None else changed_focal
            shifted_rotation = rotations[camera] if changed_rotation is None else changed_rotation
            shifted_center = centers[camera] if changed_center is None else changed_center
            for stroke in strokes:
                point, direction = geometry[lli[stroke]]
                changed = line_weights[stroke] * (
                    endpoint_distances(point, direction, shifted_rotation, shifted_center,
                                       shifted_focal, pp[camera], luv[stroke])
                    if line_distance is None else
                    line_distance(point, direction, shifted_rotation, shifted_center,
                                  shifted_focal, camera, luv[stroke]))
                rows = slice(2 * stroke, 2 * stroke + 2)
                jac[rows, column] = (changed - baseline[rows]) / step
    for line, chart in enumerate(line_charts):
        strokes = line_strokes[line]
        for component in range(4):
            column = line_offset + 4 * line + component
            step = 1e-6 * max(1.0, abs(params[column]))
            if geometry_at is None:
                shifted = params[line_offset + 4 * line:line_offset + 4 * line + 4].copy()
                shifted[component] += step
                point, direction = chart.decode(shifted)
            else:
                shifted = params.copy()
                shifted[column] += step
                point, direction = geometry_at(shifted)[line]
            for stroke in strokes:
                camera = lci[stroke]
                changed = line_weights[stroke] * (
                    endpoint_distances(point, direction, rotations[camera], centers[camera],
                                       focal[camera], pp[camera], luv[stroke])
                    if line_distance is None else
                    line_distance(point, direction, rotations[camera], centers[camera],
                                  focal[camera], camera, luv[stroke]))
                rows = slice(2 * stroke, 2 * stroke + 2)
                jac[rows, column] = (changed - baseline[rows]) / step
    return jac


def _coupled_hard_line_image_fd_jacobian(
    params: np.ndarray, baseline: np.ndarray, columns: set[int], *,
    point_values_at, geometry_at, line_residual_at,
) -> np.ndarray:
    """Differentiate line pixels through moving Free-plane support and frame rotation."""
    jac = np.zeros((len(baseline), len(params)))
    for column in sorted(columns):
        step = 1e-6 * max(1.0, abs(params[column]))
        shifted = params.copy()
        shifted[column] += step
        points = point_values_at(shifted)
        changed = line_residual_at(geometry_at(shifted, points))
        jac[:, column] = (changed - baseline) / step
    return jac


def _refine_homography(a: np.ndarray, b: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    """Polish normalized DLT by forward geometric error, avoiding marginal DLT false positives."""
    if abs(h[2, 2]) < 1.0e-12:
        return None
    params = (h / h[2, 2]).reshape(-1)[:8]
    x, y = a.T
    for _ in range(15):
        numerator_u = params[0] * x + params[1] * y + params[2]
        numerator_v = params[3] * x + params[4] * y + params[5]
        denominator = params[6] * x + params[7] * y + 1.0
        if np.any(np.abs(denominator) < 1.0e-8):
            return None
        projected = np.column_stack((numerator_u / denominator, numerator_v / denominator))
        residual = projected - b
        design = np.zeros((2 * len(a), 8))
        inverse = 1.0 / denominator
        design[0::2, 0] = x * inverse
        design[0::2, 1] = y * inverse
        design[0::2, 2] = inverse
        design[1::2, 3] = x * inverse
        design[1::2, 4] = y * inverse
        design[1::2, 5] = inverse
        design[0::2, 6] = -projected[:, 0] * x * inverse
        design[0::2, 7] = -projected[:, 0] * y * inverse
        design[1::2, 6] = -projected[:, 1] * x * inverse
        design[1::2, 7] = -projected[:, 1] * y * inverse
        step = np.linalg.lstsq(design, -residual.ravel(), rcond=None)[0]
        old_cost = float(np.sum(residual * residual))
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = params + fraction * step
            trial_denominator = trial[6] * x + trial[7] * y + 1.0
            if np.any(np.abs(trial_denominator) < 1.0e-8):
                continue
            trial_projected = np.column_stack((
                (trial[0] * x + trial[1] * y + trial[2]) / trial_denominator,
                (trial[3] * x + trial[4] * y + trial[5]) / trial_denominator))
            if float(np.sum((trial_projected - b)**2)) < old_cost:
                params = trial
                accepted = True
                break
        if not accepted or np.linalg.norm(step) < 1.0e-10:
            break
    return np.asarray((*params, 1.0)).reshape(3, 3)


def _homography_forward_error(first: np.ndarray, second: np.ndarray,
                              sigma: float) -> float | None:
    """Return forward-transfer squared error whitened for noise in both images."""
    a, ta = _normalize_points(first)
    b, tb = _normalize_points(second)
    x, y = a.T
    u, v = b.T
    ones = np.ones(len(first))
    zeros = np.zeros(len(first))
    design = np.empty((2 * len(first), 9))
    design[0::2] = np.column_stack((-x, -y, -ones, zeros, zeros, zeros,
                                     u * x, u * y, u))
    design[1::2] = np.column_stack((zeros, zeros, zeros, -x, -y, -ones,
                                     v * x, v * y, v))
    if np.linalg.matrix_rank(design) < 8:
        return None
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    refined = _refine_homography(a, b, vh[-1].reshape(3, 3))
    if refined is None:
        return None
    h = np.linalg.solve(tb, refined @ ta)
    mapped = (h @ np.column_stack((first, ones)).T).T
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 2]) < 1.0e-12):
        return None
    error = mapped[:, :2] / mapped[:, 2, None] - second
    whitened = 0.0
    for index in range(len(first)):
        denominator = mapped[index, 2]
        derivative = (h[:2, :2] - np.outer(mapped[index, :2] / denominator,
                                            h[2, :2])) / denominator
        covariance = sigma * sigma * (np.eye(2) + derivative @ derivative.T)
        whitened += float(error[index] @ np.linalg.solve(covariance, error[index]))
    return whitened


def _fit_fundamental(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Fit a rank-two fundamental matrix from noisy image pairs."""
    a, ta = _normalize_points(first)
    b, tb = _normalize_points(second)
    x, y = a.T
    u, v = b.T
    design = np.column_stack((u*x, u*y, u, v*x, v*y, v, x, y, np.ones(len(a))))
    _, singular, vh = np.linalg.svd(design, full_matrices=True)
    if singular[-2] <= singular[0] * 1.0e-10:
        return None
    model = vh[-1].reshape(3, 3)
    left, values, right = np.linalg.svd(model)
    values[-1] = 0.0
    return tb.T @ ((left * values) @ right) @ ta


def _epipolar_errors(model: np.ndarray, first: np.ndarray,
                     second: np.ndarray) -> np.ndarray:
    """Return first-order symmetric squared transfer errors in pixels."""
    a = np.column_stack((first, np.ones(len(first))))
    b = np.column_stack((second, np.ones(len(second))))
    fa = a @ model.T
    ftb = b @ model
    numerator = np.sum(b * fa, axis=1)
    denominator = np.sum(fa[:, :2]**2, axis=1) + np.sum(ftb[:, :2]**2, axis=1)
    return numerator**2 / np.maximum(denominator, 1.0e-20)


def _epipolar_conflict_pairs(ids: list[str], observations: list[SyncObservation],
                             sigma: float, *, cancel_check=None) -> list[tuple[str, str]]:
    """Find tentative pair conflicts as a bounded refusal diagnostic."""
    picks: dict[str, dict[str, tuple[float, float]]] = {camera_id: {} for camera_id in ids}
    for observation in observations:
        picks[observation.match_id][observation.landmark_id] = (observation.u, observation.v)
    conflicts = []
    pair_count = len(ids) * (len(ids) - 1) // 2
    started = time.monotonic()
    model_fits = 0
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1:]:
            if (cancel_check and cancel_check()) or time.monotonic() - started > EPIPOLAR_HINT_MAX_SECONDS:
                return []
            common = sorted(picks[first_id].keys() & picks[second_id].keys())
            # Leave-one-out fits need enough redundancy beyond the eight-point
            # minimum; thin overlaps rely on the later fit/noise checks.
            if len(common) < 12:
                continue
            first = np.asarray([picks[first_id][key] for key in common], float)
            second = np.asarray([picks[second_id][key] for key in common], float)
            # A sparse leave-one-out model can predict its withheld pick
            # poorly even when one model explains all shared picks. In that
            # case there is no pairwise evidence for blaming correspondence.
            if model_fits >= EPIPOLAR_HINT_MAX_FITS:
                return []
            model_fits += 1
            try:
                full_model = _fit_fundamental(first, second)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if full_model is None:
                continue
            full_errors = _epipolar_errors(full_model, first, second)
            if not np.isfinite(full_errors).all():
                continue
            tail = math.log(100.0 * pair_count)
            full_dof = max(len(common) - 7, 1)
            full_limit = sigma**2 * (full_dof + 2 * math.sqrt(full_dof * tail) + 2 * tail)
            if float(np.sum(full_errors)) <= full_limit:
                continue
            best = None
            for omitted in range(len(common)):
                if ((cancel_check and cancel_check()) or
                    time.monotonic() - started > EPIPOLAR_HINT_MAX_SECONDS or
                    model_fits >= EPIPOLAR_HINT_MAX_FITS):
                    return []
                kept = np.arange(len(common)) != omitted
                model_fits += 1
                try:
                    model = _fit_fundamental(first[kept], second[kept])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if model is None:
                    continue
                errors = _epipolar_errors(model, first, second)
                if not np.isfinite(errors).all():
                    continue
                inlier_sse = float(np.sum(errors[kept]))
                if best is None or inlier_sse < best[0]:
                    best = (inlier_sse, float(errors[omitted]))
            if best is None:
                continue
            # This heuristic compares one withheld Sampson error with an
            # inlier fit. It omits model uncertainty, so its camera-pair hint
            # cannot classify a pick or justify a separate acceptance gate.
            dof = max(len(common) - 8, 1)
            tail = math.log(100.0 * pair_count)
            inlier_limit = sigma**2 * (dof + 2 * math.sqrt(dof * tail) + 2 * tail)
            # Eight-point estimates can have high leverage. Eight assumed
            # coordinate sigmas is a deliberately strong diagnostic heuristic,
            # not a calibrated probability or an acceptance threshold.
            heldout_limit = (EPIPOLAR_HINT_MIN_WITHHELD_SIGMA * sigma)**2
            if best[0] <= inlier_limit and best[1] > heldout_limit:
                conflicts.append((first_id, second_id))
    return conflicts


def _conflict_hint(ids: list[str], observations: list[SyncObservation],
                   sigma: float, *, cancel_check=None) -> str:
    pairs = _epipolar_conflict_pairs(ids, observations, sigma,
                                     cancel_check=cancel_check)
    if not pairs:
        return ""
    first, second = sorted(pairs[0])
    return (f" Check shared picks between {first} and {second}; "
            "they may be inconsistent with the stated pick error.")


def _has_depth_evidence(ids: list[str], observations: list[SyncObservation],
                        sigma: float) -> bool:
    """Require a connected graph of pairs with evidence against a homography."""
    picks: dict[str, dict[str, tuple[float, float]]] = {}
    for observation in observations:
        picks.setdefault(observation.landmark_id, {})[observation.match_id] = (
            observation.u, observation.v)
    pair_count = len(ids) * (len(ids) - 1) // 2
    # Laurent-Massart upper tail for chi-square, Bonferroni across pairs.
    # The transfer covariance propagates pick noise through each homography.
    # DLT is approximate; this conservative bound only accepts strong depth
    # evidence, not a fine statistical classification of near-planar scenes.
    log_inverse_alpha = math.log(100.0 * pair_count)
    strong_edges: dict[str, set[str]] = {camera_id: set() for camera_id in ids}
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            common = [value for value in picks.values() if first in value and second in value]
            if len(common) < 5:
                continue
            first_uv = np.asarray([value[first] for value in common], float)
            second_uv = np.asarray([value[second] for value in common], float)
            try:
                squared = _homography_forward_error(first_uv, second_uv, sigma)
            except (ValueError, np.linalg.LinAlgError):
                continue
            dof = 2 * len(common) - 8
            threshold = dof + 2 * math.sqrt(dof * log_inverse_alpha) + 2 * log_inverse_alpha
            if squared is not None and np.isfinite(squared) and squared > threshold:
                strong_edges[first].add(second)
                strong_edges[second].add(first)
    reached = {ids[0]}
    frontier = [ids[0]]
    while frontier:
        for neighbor in strong_edges[frontier.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached == set(ids)


def _fits_noise_model(raw_residual: np.ndarray, parameter_count: int,
                      sigma: float) -> bool:
    """Check raw coordinate error against the declared independent pixel noise."""
    degrees_freedom = max(raw_residual.size - parameter_count, 1)
    tail = math.log(100.0)
    expected = sigma * sigma * (degrees_freedom +
        2 * math.sqrt(degrees_freedom * tail) + 2 * tail)
    return bool(np.isfinite(raw_residual).all() and
                float(np.sum(raw_residual**2)) <= expected)


def _fits_constrained_noise_model(raw_residual: np.ndarray, pixel_jacobian: np.ndarray,
                                  pixel_influence: np.ndarray, sigma: float) -> bool:
    """Conditional click-noise bound with geometric priors excluded as picks."""
    operator = np.eye(raw_residual.size) - pixel_jacobian @ pixel_influence
    singular = np.linalg.svd(operator, compute_uv=False)
    squared = singular**2
    tail = math.log(100.0)
    limit = sigma**2 * (float(np.sum(squared)) +
        2.0 * math.sqrt(float(np.sum(squared**2)) * tail) +
        2.0 * float(np.max(squared)) * tail)
    return bool(np.isfinite(raw_residual).all() and
                float(raw_residual @ raw_residual) <= limit)


def _active_pixel_jacobian(jacobian: np.ndarray, pixel_rows: int,
                           active_columns: np.ndarray,
                           coordinate_weights: np.ndarray) -> np.ndarray:
    """Use the same free parameter columns for conditional noise and covariance."""
    return jacobian[:pixel_rows, active_columns] / coordinate_weights[:, None]


def fit_independent_focals(
    calibrations: dict[str, core.Calibration],
    observations: list[SyncObservation],
    initial: SyncSolveResult,
    *, anchor_id: str,
    pick_sigma_px: float = 1.0,
    fx_span: float = DEFAULT_POINT_FOCAL_SPAN,
    plane_groups: list[tuple[str, str, int]] | None = None,
    plane_slack: float = 0.0,
    mirror_pairs: list[tuple[str, str]] | None = None,
    mirror_plane: tuple[np.ndarray, np.ndarray] | None = None,
    mirror_slack: float = 0.0,
    mirror_landmark_id: str | None = None,
    line_observations: list[SyncLineObservation] | None = None,
    parallel_pairs: list[tuple[str, str]] | None = None,
    cancel_check=None,
    progress_callback=None,
    diagnostic_callback=None,
    fixed_focals: bool = False,
    fixed_similarities: dict[str, SimilarityTransform] | None = None,
    location_match_ids: set[str] | None = None,
    readonly_match_ids: set[str] | None = None,
    known_world: dict[str, np.ndarray] | None = None,
    known_3d_slack: float = 0.0,
    ground_slack: float = 0.0,
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    lock_rotation: bool = False,
    lock_translation: bool = False,
    share_lens: bool = False,
    frozen_focal_ids: set[str] | None = None,
) -> FocalBundleOutcome:
    """Fit focal/pose/geometry, or the same joint objective with focal frozen."""
    candidate = None

    def refuse(reason: str, *, fitted: float = float("inf"),
               allow_candidate: bool = False) -> FocalBundleOutcome:
        if allow_candidate and candidate is None:
            reason += " No best fit is available: the combined fit did not improve."
        retained = replace(candidate, reason=reason) if allow_candidate and candidate else None
        return FocalBundleOutcome(False, reason, initial_rmse_px=initial_rmse,
                                  fitted_rmse_px=fitted, candidate=retained)

    initial_rmse = float("inf")
    ids = list(calibrations)
    referenced_focal = bool(known_world or known_lines or fixed_similarities or
                            any(item.on_ground for item in observations))
    if len(ids) < (2 if fixed_focals or referenced_focal else 3):
        return refuse("Independent focal fitting needs three free-point views or two referenced cameras")
    if len(ids) > MAX_CAMERAS and not fixed_focals:
        return refuse(f"Point focal estimation currently supports up to {MAX_CAMERAS} cameras per joint fit (resource limit)")
    if anchor_id not in calibrations:
        return refuse("Anchor camera is missing")
    ids.remove(anchor_id)
    ids.insert(0, anchor_id)
    if not np.isfinite(pick_sigma_px) or pick_sigma_px <= 0:
        return refuse("Pick sigma must be positive and finite")
    if not fixed_focals and (not np.isfinite(fx_span) or not 0.0 < fx_span < 1.0):
        return refuse("Focal search span must lie between 0 and 100%")
    if not initial.success or not set(initial.similarities) <= set(ids):
        return refuse("Initial Sync has invalid camera support")
    if not fixed_focals and set(ids) - set(initial.similarities):
        initial, startup_refusal = provisional_poses(
            calibrations, observations, initial, anchor_id=anchor_id,
            cancel_check=cancel_check, progress_callback=progress_callback)
        if startup_refusal:
            return refuse(startup_refusal)
    if not set(ids) <= set(initial.similarities):
        missing = sorted(set(ids) - set(initial.similarities))
        detail = ": " + ", ".join(missing) if missing else ""
        return refuse("Initial Sync must support every camera" + detail)
    for camera_id, locked in (fixed_similarities or {}).items():
        current = initial.similarities.get(camera_id)
        if current is None or (abs(current.scale - locked.scale) > 1e-8 or
                               np.linalg.norm(current.rotation - locked.rotation) > 1e-8 or
                               np.linalg.norm(current.translation - locked.translation) > 1e-8):
            return refuse(f"Locked pose for {camera_id} differs from the Sync seed")
    if lock_rotation:
        for camera_id in ids:
            if camera_id == anchor_id or camera_id in (fixed_similarities or {}):
                continue
            current = initial.similarities[camera_id].rotation
            if np.linalg.norm(current - _snap_to_axis_aligned_rotation(current)) > 1e-7:
                return refuse(f"Root rotation lock for {camera_id} needs an axis-aligned Sync seed")
    if lock_rotation and lock_translation:
        for camera_id in ids[1:]:
            if camera_id in (fixed_similarities or {}):
                continue
            current = initial.similarities[camera_id]
            if (abs(current.scale - 1.0) > 1e-7 or
                    np.linalg.norm(current.rotation - np.eye(3)) > 1e-7 or
                    np.linalg.norm(current.translation) > 1e-7):
                return refuse("Both root locks need the identity-root Sync seed")
    point_ids = sorted({item.landmark_id for item in observations} |
                       set(known_world or {}))
    line_observations = line_observations or []
    if not observations and not line_observations:
        return refuse("Joint fit needs point picks or line strokes")
    if any(not isinstance(item, SyncLineObservation) for item in line_observations):
        return refuse("Line landmarks contain an invalid stroke")
    line_ids = sorted({item.landmark_id for item in line_observations} |
                      set(known_lines or {}))
    if (len(line_ids) > MAX_LINES or len(line_observations) > MAX_LINE_STROKES) and not fixed_focals:
        return refuse(f"Line fit supports up to {MAX_LINES} lines and {MAX_LINE_STROKES} strokes (resource limit)")
    if set(line_ids) & set(point_ids):
        return refuse("A landmark cannot be both a point and a line")
    if len({(item.match_id, item.landmark_id) for item in line_observations}) != len(line_observations):
        return refuse("A free line has duplicate strokes in one camera")
    if any(item.match_id not in calibrations or
           not np.isfinite((item.u1, item.v1, item.u2, item.v2, item.weight)).all() or
           item.weight <= 0 or np.hypot(item.u2-item.u1, item.v2-item.v1) < 1.0
           for item in line_observations):
        return refuse("Line strokes must be finite, nonzero and from included cameras")
    minimum_points = (0 if fixed_focals or known_world or known_lines or
                      any(item.on_ground for item in observations) or
                      line_observations else 8)
    if (len(point_ids) < minimum_points or
            (len(point_ids) > MAX_POINTS and not fixed_focals) or
            not set(point_ids) <= set(initial.landmarks) | set(known_world or {})):
        return refuse(f"Point focal estimation needs 8–{MAX_POINTS} fully reconstructed points")
    if len({(item.match_id, item.landmark_id) for item in observations}) != len(observations):
        return refuse("A free point has duplicate picks in one camera")
    if any(item.match_id not in calibrations or
           not np.isfinite((item.u, item.v, item.weight)).all() or item.weight <= 0
           for item in observations):
        return refuse("Point picks must be finite, free and from included cameras")
    for calibration in calibrations.values():
        k = calibration.intrinsics
        if not np.isfinite((k.fx, k.fy, k.cx, k.cy)).all() or k.fx <= 0 or k.fy <= 0:
            return refuse("Every focal and principal point must be finite")
    if mirror_landmark_id is not None:
        if not isinstance(mirror_landmark_id, str) or not mirror_landmark_id:
            return refuse("Mirror reference landmark ID is invalid")
        if mirror_landmark_id not in point_ids:
            return refuse("Mirror reference landmark needs two-view picks")
    for camera_id in ids:
        count = sum(item.match_id == camera_id for item in observations)
        line_count = sum(item.match_id == camera_id for item in line_observations)
        if fixed_focals and camera_id != anchor_id and not count and not line_count:
            return refuse(f"{camera_id} has no supported image observations")
        if count < (0 if fixed_focals or known_world or known_lines or line_observations or
                    any(item.on_ground for item in observations)
                    else 8):
            return refuse(f"{camera_id} has {count} point picks; each camera needs at least eight")
    if (not fixed_focals and not (known_world or known_lines or line_observations or
                                  any(item.on_ground for item in observations) or plane_groups)
            and not _has_depth_evidence(ids, observations, float(pick_sigma_px))):
        return refuse(
            "Not enough depth evidence for independent FOV fitting. Add views "
            "from different positions and shared points at different depths."
        )

    anchor_cal = calibrations[anchor_id]
    anchor_sim = initial.similarities[anchor_id]
    if (abs(anchor_sim.scale - 1.0) > 1.0e-8 or
        np.linalg.norm(anchor_sim.rotation - np.eye(3)) > 1.0e-8 or
        np.linalg.norm(anchor_sim.translation) > 1.0e-8):
        return refuse("Initial anchor gauge is not fixed")
    global_centers = []
    global_rotations = []
    for camera_id in ids:
        cal = calibrations[camera_id]
        sim = initial.similarities[camera_id]
        if (not np.isfinite(sim.scale) or sim.scale <= 0 or
            not np.isfinite(sim.rotation).all() or not np.isfinite(sim.translation).all() or
            not np.isfinite(cal.rotation_w2c).all() or not np.isfinite(cal.camera_center).all()):
            return refuse("Initial camera pose is invalid")
        global_centers.append(sim.transform_point(cal.camera_center))
        global_rotations.append(cal.rotation_w2c @ sim.rotation.T)
    # A nearly coincident view may be listed first. Use the longest available
    # seed baseline as the unit gauge so input order cannot collapse the chart.
    baseline_candidates = [index for index in range(1, len(ids))
                           if location_match_ids is None or ids[index] in location_match_ids]
    if not baseline_candidates:
        baseline_candidates = list(range(1, len(ids)))
    reference = max(baseline_candidates, key=lambda index: float(
        np.linalg.norm(global_centers[index] - global_centers[0])))
    if reference != 1:
        ids[1], ids[reference] = ids[reference], ids[1]
        global_centers[1], global_centers[reference] = (
            global_centers[reference], global_centers[1])
        global_rotations[1], global_rotations[reference] = (
            global_rotations[reference], global_rotations[1])
    anchor_r = global_rotations[0]
    anchor_c = global_centers[0]
    baseline = float(np.linalg.norm(global_centers[1] - anchor_c))
    if not np.isfinite(baseline) or baseline < 1.0e-8:
        return refuse("Initial camera baseline is zero")
    centers0 = np.asarray([anchor_r @ (center - anchor_c) / baseline
                           for center in global_centers])
    rotations0 = [rotation @ anchor_r.T for rotation in global_rotations]
    points0 = np.asarray([anchor_r @
                          (np.asarray(initial.landmarks[key] if key in initial.landmarks
                                      else known_world[key]) - anchor_c) / baseline
                          for key in point_ids], dtype=float).reshape(-1, 3)
    if not np.isfinite(points0).all():
        return refuse("Initial point geometry is invalid")
    line_set = set(line_ids)
    point_set = set(point_ids)
    if mirror_pairs is not None and not isinstance(mirror_pairs, (tuple, list)):
        return refuse("Mirror pairs contain malformed landmark links")
    all_pairs = mirror_pairs or []
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(key, str) or not key for key in pair)
           for pair in all_pairs):
        return refuse("Mirror pairs contain malformed landmark links")
    if len(_dedupe_mirror_pairs(all_pairs)) != len(all_pairs):
        return refuse("Mirror pairs contain duplicate or self links")
    point_mirrors = [pair for pair in all_pairs if set(pair) <= point_set]
    line_mirrors = [pair for pair in all_pairs if set(pair) <= line_set]
    if len(point_mirrors) + len(line_mirrors) != len(all_pairs):
        return refuse("Mirror relation contains unsupported or mixed point/line members")
    if line_mirrors and mirror_plane is None:
        return refuse("Line mirrors need a supplied mirror plane normal")
    if line_mirrors:
        try:
            supplied_normal = np.asarray(mirror_plane[1], float).reshape(3)
            supplied_origin = np.asarray(mirror_plane[0], float).reshape(3)
        except (ValueError, TypeError, IndexError):
            return refuse("Supplied mirror plane is invalid")
        if (not np.isfinite(supplied_normal).all() or
            not np.isfinite(supplied_origin).all() or
            np.linalg.norm(supplied_normal) < 1e-12):
            return refuse("Supplied mirror plane is invalid")
    point_groups = [item for item in (plane_groups or []) if item[0] in point_set]
    line_groups = [item for item in (plane_groups or []) if item[0] in line_set]
    if len(point_groups) + len(line_groups) != len(plane_groups or []):
        return refuse("Plane relation contains an unsupported landmark")
    line_seeds = {}
    for line_id in line_ids:
        if line_id in (known_lines or {}):
            seed = tuple(np.asarray(end, dtype=float) for end in known_lines[line_id])
            seed = (0.5 * (seed[0] + seed[1]), seed[1] - seed[0])
        elif line_id in initial.line_segments:
            segment = initial.line_segments[line_id]
            seed = (0.5 * (segment[0] + segment[1]), segment[1] - segment[0])
        else:
            strokes = [item for item in line_observations if item.landmark_id == line_id]
            seed = _reconstruct_line_from_observations(
                strokes, initial.similarities,
                {key: SyncMatchInput(key, calibrations[key]) for key in ids})
        if seed is None:
            continue
        seed_point = anchor_r @ (seed[0] - anchor_c) / baseline
        seed_direction = anchor_r @ seed[1]
        if not np.isfinite(seed_point).all() or not np.isfinite(seed_direction).all():
            return refuse("Initial line geometry is invalid")
        line_seeds[line_id] = (seed_point, seed_direction)
    if line_mirrors and mirror_plane is not None:
        normal = anchor_r @ supplied_normal
        normal /= np.linalg.norm(normal)
        householder = np.eye(3) - 2 * np.outer(normal, normal)
        if mirror_landmark_id is not None:
            reference = points0[point_ids.index(mirror_landmark_id)]
            distance = float(normal @ reference)
        else:
            distance = float(normal @ (anchor_r @ (
                np.asarray(mirror_plane[0], float) - anchor_c) / baseline))
        for left, right in line_mirrors:
            if left in line_seeds and right not in line_seeds:
                point, direction = line_seeds[left]
                line_seeds[right] = (householder @ point + 2*distance*normal,
                                     householder @ direction)
            elif right in line_seeds and left not in line_seeds:
                point, direction = line_seeds[right]
                line_seeds[left] = (householder @ point + 2*distance*normal,
                                    householder @ direction)
    if set(line_seeds) != line_set:
        return refuse("Every free line needs two-view strokes or a reconstructed mirror partner")
    line_charts = [LineChart(*line_seeds[key]) for key in line_ids]
    try:
        reference_constraints = PointReferenceConstraints.from_inputs(
            point_ids, anchor_rotation=anchor_r, anchor_center=anchor_c,
            baseline_world=baseline, known_world=known_world,
            known_3d_slack=known_3d_slack,
            ground_landmark_ids=sorted({item.landmark_id for item in observations
                                        if item.on_ground}),
            ground_slack=ground_slack)
        points0 = reference_constraints.project_hard(points0)
        constraints = PointFocalConstraints.from_inputs(
            point_ids, anchor_rotation=anchor_r, anchor_center=anchor_c,
            baseline_world=baseline, plane_groups=point_groups,
            plane_slack=0.0 if plane_slack is None else plane_slack,
            mirror_pairs=point_mirrors, mirror_plane=mirror_plane,
            mirror_slack=0.0 if mirror_slack is None else mirror_slack,
            mirror_landmark_id=mirror_landmark_id,
            extra_mirror_pairs=bool(line_mirrors))
        line_constraints = LineFocalConstraints.from_inputs(
            point_ids, line_ids, plane_groups=plane_groups, parallel_pairs=parallel_pairs,
            anchor_rotation=anchor_r, plane_spring=constraints.plane_spring,
            baseline_world=baseline, hard_plane=constraints.hard_plane,
            known_line_ids=set(known_lines or {}),
            known_line_positions={key: anchor_r @ (
                0.5 * (np.asarray(segment[0]) + np.asarray(segment[1])) -
                anchor_c) / baseline for key, segment in (known_lines or {}).items()})
    except (ValueError, TypeError, IndexError) as exc:
        return refuse(str(exc))
    direction = centers0[1] / np.linalg.norm(centers0[1])
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    tangent_a = np.cross(direction, axis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(direction, tangent_a)
    tangents = np.vstack((tangent_a, tangent_b))
    ncam = len(ids)
    has_geometric_priors = (constraints.active or line_constraints.active or
                            reference_constraints.active or bool(known_lines))
    npoint = len(point_ids)
    scale_columns = int(constraints.free_baseline or bool(known_world) or
                        bool(known_lines) or
                        (bool(reference_constraints.ground_indices) and
                         abs(anchor_c[2]) > 1e-8))
    point_offset = ncam + 5 + scale_columns + 6 * (ncam - 2)
    point_end = point_offset + 3 * npoint
    line_offset = point_end
    line_end = line_offset + 4 * len(line_ids)
    rotation_basis = ([] if (reference_constraints.active or known_lines or
                             fixed_similarities or lock_rotation or lock_translation)
                      else orientation_basis(
                          constraints, line_constraints, points0,
                          [chart.decode(chart.initial) for chart in line_charts]))
    orientation_end = line_end + len(rotation_basis)
    mirror_offset_column = orientation_end if constraints.free_mirror_offset else None
    x = np.concatenate((np.zeros(ncam), _log_rodrigues(rotations0[1]),
                        np.zeros(2), *([np.zeros(1)] if scale_columns else []),
                        *[np.concatenate((_log_rodrigues(rotations0[i]),
                                          centers0[i])) for i in range(2, ncam)],
                        points0.ravel(),
                        *[chart.initial for chart in line_charts],
                        np.zeros(len(rotation_basis)),
                        *([np.zeros(1)] if constraints.free_mirror_offset else [])))
    if lock_translation:
        for camera in range(2, ncam):
            if ids[camera] not in (fixed_similarities or {}):
                x[ncam + 5 + scale_columns + 6 * (camera - 2) + 3] = 0.0
    estimated_bytes = _estimated_dense_jacobian_bytes(
        len(x), len(observations), len(line_observations),
        len(point_ids), len(line_ids),
        len(plane_groups or ()) + len(mirror_pairs or ()) +
        len(parallel_pairs or ()))
    if estimated_bytes > MAX_DENSE_JACOBIAN_BYTES:
        return refuse("Joint graph exceeds the bounded dense-Jacobian memory limit")
    camera_index = {key: index for index, key in enumerate(ids)}
    point_index = {key: index for index, key in enumerate(point_ids)}
    ci = np.asarray([camera_index[item.match_id] for item in observations], int)
    pi = np.asarray([point_index[item.landmark_id] for item in observations], int)
    uv = np.asarray([(item.u, item.v) for item in observations], float)
    weights = np.sqrt(np.asarray([item.weight for item in observations], float))
    line_index = {key: index for index, key in enumerate(line_ids)}
    lci = np.asarray([camera_index[item.match_id] for item in line_observations], int)
    lli = np.asarray([line_index[item.landmark_id] for item in line_observations], int)
    luv = np.asarray([((item.u1, item.v1), (item.u2, item.v2))
                      for item in line_observations], float).reshape(-1, 2, 2)
    line_weights = np.sqrt(np.asarray([item.weight for item in line_observations], float))
    fit_weights = weights.copy()
    fit_line_weights = line_weights.copy()
    camera_strokes = [np.flatnonzero(lci == camera) for camera in range(ncam)]
    line_strokes = [np.flatnonzero(lli == line) for line in range(len(line_ids))]
    camera_columns = [(camera,) for camera in range(ncam)]
    if share_lens:
        camera_columns = [(0,) for _ in range(ncam)]
    camera_columns[1] += tuple(range(ncam, ncam + 5 + scale_columns))
    for camera in range(2, ncam):
        start = ncam + 5 + scale_columns + 6 * (camera - 2)
        camera_columns[camera] += tuple(range(start, start + 6))
    base_fx = np.asarray([calibrations[key].intrinsics.fx for key in ids], float)
    base_fy = np.asarray([calibrations[key].intrinsics.fy for key in ids], float)
    general_projection = fixed_focals or any(
        calibrations[key].has_distortion or
        not np.isclose(base_fx[index], base_fy[index], rtol=1e-6)
        for index, key in enumerate(ids))
    pp = np.asarray([(calibrations[key].intrinsics.cx, calibrations[key].intrinsics.cy)
                     for key in ids], float)
    lower, upper = np.log((1.0 - fx_span, 1.0 + fx_span))
    parameter_lower = np.full(len(x), -np.inf)
    parameter_upper = np.full(len(x), np.inf)
    parameter_lower[:ncam], parameter_upper[:ncam] = lower, upper
    if scale_columns:
        parameter_lower[ncam + 5] = -MAX_LOG_METRIC_BASELINE_CHANGE
        parameter_upper[ncam + 5] = MAX_LOG_METRIC_BASELINE_CHANGE
    # Keep one residual/parameter chart for both focal policies. A fixed lens
    # has no column in the solved system, rather than a zero-width bound that
    # would make the covariance and active-set step singular.
    active = np.ones(len(x), dtype=bool)
    if fixed_focals:
        active[:ncam] = False
    elif share_lens:
        active[1:ncam] = False
    else:
        for camera_id in frozen_focal_ids or ():
            if camera_id not in camera_index:
                return refuse(f"Frozen focal camera is missing: {camera_id}")
            active[camera_index[camera_id]] = False
    for camera_id in (fixed_similarities or {}):
        if camera_id not in camera_index:
            return refuse(f"Locked camera is missing: {camera_id}")
        camera = camera_index[camera_id]
        if camera == 0:
            continue
        if camera == 1:
            active[ncam:ncam + 5 + scale_columns] = False
        else:
            start = ncam + 5 + scale_columns + 6 * (camera - 2)
            active[start:start + 6] = False
    for point in reference_constraints.hard_xyz:
        start = point_offset + 3 * point
        active[start:start + 3] = False
    hard_z_pivots = {}
    for point, (normal, _offset) in reference_constraints.hard_z.items():
        pivot = int(np.argmax(np.abs(normal)))
        hard_z_pivots[point] = pivot
        active[point_offset + 3 * point + pivot] = False
    for line_id in known_lines or {}:
        if line_id not in line_index:
            continue
        start = line_offset + 4 * line_index[line_id]
        active[start:start + 4] = False
    for column in _redundant_hard_line_columns(
            line_constraints, points0, line_charts, line_offset):
        active[column] = False
    point_derivatives = [np.eye(3) for _ in point_ids]
    for point in reference_constraints.hard_xyz:
        point_derivatives[point] = np.zeros((3, 3))
    for point, (normal, _offset) in reference_constraints.hard_z.items():
        point_derivatives[point] = np.eye(3) - np.outer(normal, normal)
    if lock_rotation:
        active[ncam:ncam + 3] = False
        for camera in range(2, ncam):
            start = ncam + 5 + scale_columns + 6 * (camera - 2)
            active[start:start + 3] = False
    if lock_translation:
        # A fixed Empty translation still permits rotation about that Empty's
        # origin. Its optical center therefore follows the root rotation and
        # the private camera center; it is not an independent 3D coordinate.
        active[ncam + 3:ncam + 5] = False
        for camera in range(2, ncam):
            start = ncam + 5 + scale_columns + 6 * (camera - 2)
            active[start + 4:start + 6] = False
        for camera_id in ids[1:]:
            if camera_id not in (fixed_similarities or {}) and np.linalg.norm(
                    initial.similarities[camera_id].translation) > 1.0e-7:
                return refuse("Root translation lock needs a zero-translation Sync seed")
    if lock_rotation and lock_translation:
        active[ncam:ncam + 5 + scale_columns] = False
        for camera in range(2, ncam):
            start = ncam + 5 + scale_columns + 6 * (camera - 2)
            active[start:start + 6] = False
    unrestricted_active = active.copy()
    fit_only_cameras = [camera for camera, camera_id in enumerate(ids)
                        if camera != 0 and (
                            camera_id in (readonly_match_ids or ()) or
                            (location_match_ids is not None and
                             camera_id not in location_match_ids))]
    for camera in fit_only_cameras:
        active[[column for column in camera_columns[camera]
                if not (share_lens and column == 0)]] = False
    active_columns = np.flatnonzero(active)
    block_mode = fixed_focals and (len(ids) > MAX_CAMERAS or
                                   len(point_ids) > MAX_POINTS or
                                   len(line_ids) > MAX_LINES or
                                   len(line_observations) > MAX_LINE_STROKES)
    main_blocks = ([active_columns[start:start + 48]
                    for start in range(0, len(active_columns), 48)]
                   if block_mode else [active_columns])
    start_time = time.monotonic()

    def intrinsics_at(camera: int, focal_value: float) -> core.CameraIntrinsics:
        source = calibrations[ids[camera]].intrinsics
        return core.CameraIntrinsics(
            float(focal_value), float(base_fy[camera] * focal_value / base_fx[camera]),
            source.cx, source.cy, source.image_width, source.image_height)

    def root_scale_at(camera: int, params: np.ndarray) -> float:
        original = float(initial.similarities[ids[camera]].scale)
        if ids[camera] in (fixed_similarities or {}) or lock_rotation and lock_translation:
            return original
        if camera == 1:
            return original * (math.exp(float(params[ncam + 5]))
                               if scale_columns else 1.0)
        column = ncam + 5 + scale_columns + 6 * (camera - 2) + 3
        return original * math.exp(float(params[column]))

    def translated_root_center(camera: int, rotation_internal: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        camera_id = ids[camera]
        source = calibrations[camera_id]
        similarity = initial.similarities[camera_id]
        root_rotation = (rotation_internal @ anchor_r).T @ source.rotation_w2c
        translation = (similarity.translation if camera_id in (fixed_similarities or {})
                       else np.zeros(3))
        world_center = translation + root_scale_at(camera, params) * (
            root_rotation @ source.camera_center)
        return anchor_r @ (world_center - anchor_c) / baseline

    def root_scale_center_derivative(camera: int, rotation_internal: np.ndarray,
                                     params: np.ndarray) -> np.ndarray:
        source = calibrations[ids[camera]]
        root_rotation = (rotation_internal @ anchor_r).T @ source.rotation_w2c
        return (anchor_r @ (root_scale_at(camera, params) *
                            (root_rotation @ source.camera_center)) / baseline)

    def projected_line(point, line_direction, rotation, center, focal_value,
                       camera, endpoints):
        source = calibrations[ids[camera]]
        return line_endpoint_distances(
            point * baseline, line_direction, rotation, center * baseline,
            intrinsics_at(camera, focal_value), endpoints,
            division_lambda=source.division_lambda,
            brown_conrady=source.brown_conrady)

    def decode(params: np.ndarray):
        focal_exponents = (np.full(ncam, params[0]) if share_lens else params[:ncam])
        focal = base_fx * np.exp(focal_exponents)
        raw = direction + params[ncam + 3] * tangents[0] + params[ncam + 4] * tangents[1]
        radius = math.exp(float(params[ncam + 5])) if scale_columns else 1.0
        first_center = radius * raw / np.linalg.norm(raw)
        rotations = [np.eye(3), _rodrigues(params[ncam:ncam + 3])]
        centers = [np.zeros(3), first_center]
        for camera in range(2, ncam):
            offset = ncam + 5 + scale_columns + 6 * (camera - 2)
            rotations.append(_rodrigues(params[offset:offset + 3]))
            centers.append(params[offset + 3:offset + 6])
        if lock_translation:
            for camera in range(1, ncam):
                if ids[camera] not in (fixed_similarities or {}):
                    centers[camera] = translated_root_center(camera, rotations[camera], params)
        points = reference_constraints.project_hard(
            params[point_offset:point_end].reshape(npoint, 3))
        return focal, rotations, np.asarray(centers), points

    def camera_state(params: np.ndarray, camera: int, column: int):
        if column == (0 if share_lens else camera):
            return base_fx[camera] * math.exp(float(params[column])), None, None
        if camera == 1:
            if column < ncam + 3:
                rotation = _rodrigues(params[ncam:ncam + 3])
                return (None, rotation,
                        translated_root_center(camera, rotation, params) if lock_translation else None)
            if lock_translation:
                return None, None, translated_root_center(
                    camera, _rodrigues(params[ncam:ncam + 3]), params)
            raw = direction + params[ncam + 3] * tangents[0] + params[ncam + 4] * tangents[1]
            radius = math.exp(float(params[ncam + 5])) if scale_columns else 1.0
            return None, None, radius * raw / np.linalg.norm(raw)
        offset = ncam + 5 + scale_columns + 6 * (camera - 2)
        if column < offset + 3:
            rotation = _rodrigues(params[offset:offset + 3])
            return (None, rotation,
                    translated_root_center(camera, rotation, params) if lock_translation else None)
        if lock_translation:
            return None, None, translated_root_center(
                camera, _rodrigues(params[offset:offset + 3]), params)
        return None, None, params[offset + 3:offset + 6]

    def line_geometry(params: np.ndarray, points: np.ndarray):
        geometry = [chart.decode(params[line_offset + 4*i:line_offset + 4*i + 4])
                    for i, chart in enumerate(line_charts)]
        for key, segment in (known_lines or {}).items():
            index = line_index[key]
            first, second = [np.asarray(end, float) for end in segment]
            direction = anchor_r @ (second - first)
            geometry[index] = (
                anchor_r @ (0.5 * (first + second) - anchor_c) / baseline,
                direction / np.linalg.norm(direction))
        if line_constraints.hard_plane and line_constraints.groups:
            rotation = frame_rotation(params)
            constrained = line_constraints.constrain_hard_directions(
                points @ rotation.T,
                [(rotation @ point, rotation @ direction) for point, direction in geometry])
            geometry = [(rotation.T @ point, rotation.T @ direction)
                        for point, direction in constrained]
        return geometry

    def frame_rotation(params):
        return overall_rotation(params[line_end:orientation_end], rotation_basis)

    def prior_geometry(params, points, geometry):
        rotation = frame_rotation(params)
        return points @ rotation.T, [(rotation @ p, rotation @ d) for p, d in geometry]

    def line_image_residual(focal, rotations, centers, geometry):
        result = np.empty((len(luv), 2))
        for index in range(len(luv)):
            camera = lci[index]
            point, direction = geometry[lli[index]]
            result[index] = fit_line_weights[index] * (
                projected_line(point, direction, rotations[camera], centers[camera],
                               focal[camera], camera, luv[index]) if general_projection else
                endpoint_distances(point, direction, rotations[camera], centers[camera],
                                   focal[camera], pp[camera], luv[index]))
        return result.ravel()

    def line_prior(params: np.ndarray, points: np.ndarray, geometry):
        relations = line_constraints.residual(points, geometry)
        if not line_mirrors:
            return relations
        normal = constraints.mirror_normal
        assert normal is not None
        householder = np.eye(3) - 2 * np.outer(normal, normal)
        distance = (float(normal @ points[constraints.mirror_reference_index])
                    if constraints.mirror_reference_index is not None else constraints.mirror_distance)
        if mirror_offset_column is not None:
            distance += float(params[mirror_offset_column])
        result = list(relations)
        for left_id, right_id in line_mirrors:
            left_p, left_d = geometry[line_index[left_id]]
            right_p, right_d = geometry[line_index[right_id]]
            left_p = canonical_line_point(left_p, left_d)
            right_p = canonical_line_point(right_p, right_d)
            reflected_p = householder @ left_p + 2 * distance * normal
            reflected_d = householder @ left_d
            if reflected_d @ right_d < 0:
                reflected_d = -reflected_d
            # Two perpendicular position coordinates and two direction
            # coordinates: translation along either line is no constraint.
            result.extend((constraints.mirror_pair_spring *
                           np.cross(right_d, reflected_p - right_p))[:].tolist())
            result.extend((200.0 * np.cross(right_d, reflected_d)).tolist())
        return np.asarray(result)

    def residual_and_jacobian(params: np.ndarray, *, jacobian: bool):
        focal, rotations, centers, points = decode(params)
        residual = np.empty((len(uv), 2))
        jac = np.zeros((2 * len(uv), len(params))) if jacobian else None
        depths = np.empty(len(uv))
        for camera in range(ncam):
            selected = np.flatnonzero(ci == camera)
            if not len(selected):
                continue
            point = points[pi[selected]]
            offset_xyz = point - centers[camera]
            q = (rotations[camera] @ offset_xyz.T).T
            z = q[:, 2]
            depths[selected] = z
            source = calibrations[ids[camera]]
            predicted, dq, focal_j = project_camera_points(
                q, intrinsics_at(camera, focal[camera]),
                division_lambda=source.division_lambda,
                brown_conrady=source.brown_conrady,
                jacobian=jacobian)
            residual[selected] = (predicted - uv[selected]) * fit_weights[selected, None]
            if jac is None:
                continue
            dq *= fit_weights[selected, None, None]
            rows = np.column_stack((2 * selected, 2 * selected + 1)).ravel()
            jac[rows, 0 if share_lens else camera] = (
                focal_j * fit_weights[selected, None]).ravel()
            point_j = np.einsum('nij,jk->nik', dq, rotations[camera])
            for local, point_id in enumerate(pi[selected]):
                jac[2 * selected[local]:2 * selected[local] + 2,
                    point_offset + 3 * point_id:point_offset + 3 * point_id + 3] = (
                        point_j[local] @ point_derivatives[point_id])
            if camera:
                if camera == 1:
                    rotation_start = ncam
                    center_j = []
                    raw = direction + params[ncam + 3] * tangents[0] + params[ncam + 4] * tangents[1]
                    norm = np.linalg.norm(raw)
                    unit = raw / norm
                    radius = float(np.linalg.norm(centers[1]))
                    for tangent in tangents:
                        center_j.append(radius * (tangent - unit * (unit @ tangent)) / norm)
                    if scale_columns:
                        center_j.append(centers[1])
                    if lock_translation:
                        center_j = [np.zeros(3), np.zeros(3)] + (
                            [root_scale_center_derivative(camera, rotations[camera], params)]
                            if scale_columns else [])
                    center_start = ncam + 3
                else:
                    rotation_start = ncam + 5 + scale_columns + 6 * (camera - 2)
                    center_j = np.eye(3)
                    if lock_translation:
                        center_j = np.asarray((
                            root_scale_center_derivative(camera, rotations[camera], params),
                            np.zeros(3), np.zeros(3)))
                    center_start = rotation_start + 3
                jac[rows, center_start:center_start + len(center_j)] = (
                    -np.einsum('nij,kj->nik', point_j, np.asarray(center_j))).reshape(-1, len(center_j))
                for component in range(3):
                    trial = params[rotation_start:rotation_start + 3].copy()
                    step = 1.0e-6 * max(1.0, abs(trial[component]))
                    trial[component] += step
                    trial_rotation = _rodrigues(trial)
                    if lock_translation and ids[camera] not in (fixed_similarities or {}):
                        trial_center = translated_root_center(camera, trial_rotation, params)
                        trial_q = (trial_rotation @ (point - trial_center).T).T
                        trial_predicted, _, _ = project_camera_points(
                            trial_q, intrinsics_at(camera, focal[camera]),
                            division_lambda=source.division_lambda,
                            brown_conrady=source.brown_conrady, jacobian=False)
                        jac[rows, rotation_start + component] = (
                            (trial_predicted - predicted) * fit_weights[selected, None] /
                            step).ravel()
                    else:
                        derivative = ((trial_rotation - rotations[camera]) @ offset_xyz.T).T / step
                        jac[rows, rotation_start + component] = np.einsum(
                            'nij,nj->ni', dq, derivative).ravel()
        pixel_residual = residual.ravel()
        geometry = line_geometry(params, points)
        line_residual = line_image_residual(focal, rotations, centers, geometry)
        if jacobian and len(line_residual):
            line_jac = _line_image_fd_jacobian(
                params, line_residual, camera_columns=camera_columns,
                camera_strokes=camera_strokes, line_strokes=line_strokes,
                camera_state=camera_state, line_charts=line_charts,
                line_offset=line_offset, lci=lci, lli=lli, luv=luv,
                line_weights=fit_line_weights, pp=pp, focal=focal,
                rotations=rotations, centers=centers, geometry=geometry,
                line_distance=projected_line if general_projection else None,
                geometry_at=(lambda shifted: line_geometry(shifted, points))
                if line_constraints.hard_plane and line_constraints.groups else None)
            if line_constraints.hard_plane and line_constraints.groups:
                # Free plane normals depend on their fitted support points;
                # world-axis planes depend on the common rotation parameters.
                coupled_columns = _coupled_hard_line_columns(
                    line_constraints, point_offset, line_end, orientation_end)
                line_jac += _coupled_hard_line_image_fd_jacobian(
                    params, line_residual, coupled_columns,
                    point_values_at=lambda shifted: decode(shifted)[3],
                    geometry_at=line_geometry,
                    line_residual_at=lambda changed: line_image_residual(
                        focal, rotations, centers, changed))
            jac = np.vstack((jac, line_jac))
        pixel_residual = np.concatenate((pixel_residual, line_residual))
        if not has_geometric_priors:
            return pixel_residual, jac, depths
        offset = float(params[mirror_offset_column]) if mirror_offset_column is not None else 0.0
        base_frame = frame_rotation(params)
        constrained_points = points @ base_frame.T
        constrained_lines = [(base_frame @ p, base_frame @ d) for p, d in geometry]
        prior_residual, prior_jacobian = constraints.residual_and_jacobian(
            constrained_points, point_offset=point_offset, parameter_count=len(params),
            mirror_offset=offset, mirror_offset_column=mirror_offset_column,
            jacobian=jacobian)
        reference_residual, reference_jacobian = reference_constraints.residual_and_jacobian(
            points, point_offset=point_offset, parameter_count=len(params),
            jacobian=jacobian)
        if len(reference_residual):
            prior_residual = np.concatenate((prior_residual, reference_residual))
            if jacobian:
                prior_jacobian = np.vstack((prior_jacobian, reference_jacobian))
        if jacobian and len(rotation_basis):
            point_block = prior_jacobian[:, point_offset:point_end].reshape(-1, npoint, 3)
            prior_jacobian[:, point_offset:point_end] = (
                point_block @ base_frame).reshape(-1, 3 * npoint)
            for column in range(line_end, orientation_end):
                step = 1e-6 * max(1.0, abs(params[column]))
                shifted = params.copy()
                shifted[column] += step
                changed_points = points @ frame_rotation(shifted).T
                changed, _ = constraints.residual_and_jacobian(
                    changed_points, point_offset=point_offset, parameter_count=len(params),
                    mirror_offset=offset, mirror_offset_column=mirror_offset_column,
                    jacobian=False)
                prior_jacobian[:, column] = (changed - prior_residual) / step
        if jacobian:
            for point, transform in enumerate(point_derivatives):
                if point in reference_constraints.hard_xyz or point in reference_constraints.hard_z:
                    start = point_offset + 3 * point
                    prior_jacobian[:, start:start + 3] = (
                        prior_jacobian[:, start:start + 3] @ transform)
        mirror_line_residual = line_prior(params, constrained_points, constrained_lines)
        if jacobian and len(mirror_line_residual):
            mirror_jac = np.zeros((len(mirror_line_residual), len(params)))
            relevant = set(range(line_offset, orientation_end))
            for point in line_constraints.point_indices:
                relevant.update(range(point_offset + 3 * point, point_offset + 3 * point + 3))
            if constraints.mirror_reference_index is not None:
                start = point_offset + 3 * constraints.mirror_reference_index
                relevant.update(range(start, start + 3))
            if mirror_offset_column is not None:
                relevant.add(mirror_offset_column)
            for column in sorted(relevant):
                step = 1e-6 * max(1.0, abs(params[column]))
                shifted = params.copy()
                shifted[column] += step
                if column < orientation_end:
                    shifted_points = decode(shifted)[3] if column < line_offset else points
                    changed_points, changed_lines = prior_geometry(
                        shifted, shifted_points, line_geometry(shifted, shifted_points))
                else:
                    changed_points, changed_lines = constrained_points, constrained_lines
                mirror_jac[:, column] = (
                    line_prior(shifted, changed_points, changed_lines) - mirror_line_residual) / step
        else:
            mirror_jac = None
        return (np.concatenate((pixel_residual, prior_residual, mirror_line_residual)),
                np.vstack((jac, prior_jacobian, mirror_jac)) if jacobian and mirror_jac is not None
                else np.vstack((jac, prior_jacobian)) if jacobian else None, depths)

    try:
        initial_residual, _, _ = residual_and_jacobian(x, jacobian=False)
        point_rows = 2 * len(uv)
        pixel_rows = point_rows + 2 * len(luv)
        coordinate_weights = np.concatenate((np.repeat(weights, 2), np.repeat(line_weights, 2)))
        initial_raw = initial_residual[:point_rows].reshape(-1, 2) / weights[:, None]
        initial_line_raw = initial_residual[point_rows:pixel_rows].reshape(-1, 2) / line_weights[:, None]
        initial_measurements = (initial_raw if len(initial_raw) else initial_line_raw)
        initial_rmse = float(np.sqrt(np.mean(np.sum(initial_measurements**2, axis=1))))
        if fit_only_cameras:
            fit_weights = weights * ~np.isin(ci, fit_only_cameras)
            fit_line_weights = line_weights * ~np.isin(lci, fit_only_cameras)
        damping = 1.0e-3
        converged = False
        iterations = 0
        stop_reason = "iteration_limit"
        relative_gains = []
        sweep_start_cost = None
        iteration_limit = MAX_ITERATIONS * (len(main_blocks) if block_mode else 1)
        for iteration in range(iteration_limit):
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            if time.monotonic() - start_time > MAX_SECONDS:
                stop_reason = "time_limit"
                break
            if progress_callback:
                progress_callback(iteration, iteration_limit,
                                  "Estimating focal from landmarks" if line_ids else
                                  "Estimating focal from points")
            residual, jac, _ = residual_and_jacobian(x, jacobian=True)
            cost = float(residual @ residual)
            if not main_blocks or not len(main_blocks[0]):
                converged = True
                stop_reason = "no_free_parameters"
                break
            iteration_columns = main_blocks[iteration % len(main_blocks)]
            if block_mode and iteration % len(main_blocks) == 0:
                sweep_start_cost = cost
            active_jac = jac[:, iteration_columns]
            column_scale = np.maximum(np.linalg.norm(active_jac, axis=0), 1.0e-8)
            scaled = active_jac / column_scale
            gradient = scaled.T @ residual
            if np.linalg.norm(gradient, ord=np.inf) < 1.0e-7:
                if block_mode:
                    iterations = iteration + 1
                    continue
                converged = True
                iterations = iteration
                stop_reason = "small_gradient"
                break
            accepted_step = False
            for _trial in range(12):
                if cancel_check and cancel_check():
                    return refuse("Cancelled")
                if time.monotonic() - start_time > MAX_SECONDS:
                    stop_reason = "time_limit"
                    break
                try:
                    step = bounded_lm_step(
                        active_jac, residual, x[iteration_columns],
                        parameter_lower[iteration_columns], parameter_upper[iteration_columns], damping,
                        cancel_check=lambda: (bool(cancel_check and cancel_check()) or
                                              time.monotonic() - start_time > MAX_SECONDS))
                except InterruptedError:
                    if cancel_check and cancel_check():
                        return refuse("Cancelled")
                    stop_reason = "time_limit"
                    break
                trial_x = x.copy()
                trial_x[iteration_columns] = np.clip(
                    x[iteration_columns] + step,
                    parameter_lower[iteration_columns], parameter_upper[iteration_columns])
                trial_residual, _, _ = residual_and_jacobian(trial_x, jacobian=False)
                trial_cost = float(trial_residual @ trial_residual)
                if np.isfinite(trial_cost) and trial_cost < cost:
                    relative_gain = (cost - trial_cost) / max(cost, 1.0)
                    x = trial_x
                    damping = max(damping / 3.0, 1.0e-9)
                    accepted_step = True
                    relative_gains.append(relative_gain)
                    if relative_gain < RELATIVE_COST_TOLERANCE and not block_mode:
                        converged = True
                        stop_reason = "small_improvement"
                    break
                damping *= 10.0
            iterations = iteration + 1
            if stop_reason == "time_limit":
                break
            if block_mode:
                if (iteration + 1) % len(main_blocks) == 0 and sweep_start_cost is not None:
                    current, _, _ = residual_and_jacobian(x, jacobian=False)
                    sweep_gain = (sweep_start_cost - float(current @ current)) / max(sweep_start_cost, 1.0)
                    if sweep_gain < RELATIVE_COST_TOLERANCE:
                        converged = True
                        stop_reason = "small_improvement"
                        break
                continue
            if converged or not accepted_step:
                converged = converged or not accepted_step and np.linalg.norm(gradient, ord=np.inf) < 1.0e-5
                if not accepted_step:
                    stop_reason = "small_gradient" if converged else "no_improving_step"
                break
        # Fit Only stills can refine their own optical pose (and independent
        # focal) against the frozen cloud. Their picks never enter a step that
        # moves shared points, lines, or another camera.
        for camera in fit_only_cameras:
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            camera_active = np.asarray([column for column in camera_columns[camera]
                                        if unrestricted_active[column] and
                                        not (share_lens and column == 0)], dtype=int)
            if not len(camera_active):
                continue
            fit_weights = weights * (ci == camera)
            fit_line_weights = line_weights * (lci == camera)
            if not np.any(fit_weights) and not np.any(fit_line_weights):
                continue
            camera_damping = 1.0e-3
            for _ in range(24):
                if cancel_check and cancel_check():
                    return refuse("Cancelled")
                if time.monotonic() - start_time > MAX_SECONDS:
                    stop_reason = "time_limit"
                    break
                camera_residual, camera_jac, _ = residual_and_jacobian(x, jacobian=True)
                camera_cost = float(camera_residual @ camera_residual)
                local_jac = camera_jac[:, camera_active]
                if not np.any(local_jac) or np.linalg.norm(
                        local_jac.T @ camera_residual, ord=np.inf) < 1.0e-7:
                    break
                try:
                    camera_step = bounded_lm_step(
                        local_jac, camera_residual, x[camera_active],
                        parameter_lower[camera_active], parameter_upper[camera_active],
                        camera_damping,
                        cancel_check=lambda: bool(cancel_check and cancel_check()) or
                        time.monotonic() - start_time > MAX_SECONDS)
                except InterruptedError:
                    if cancel_check and cancel_check():
                        return refuse("Cancelled")
                    stop_reason = "time_limit"
                    break
                trial = x.copy()
                trial[camera_active] = np.clip(
                    x[camera_active] + camera_step,
                    parameter_lower[camera_active], parameter_upper[camera_active])
                trial_residual, _, _ = residual_and_jacobian(trial, jacobian=False)
                if float(trial_residual @ trial_residual) < camera_cost:
                    x = trial
                    camera_damping = max(camera_damping / 3.0, 1.0e-9)
                    if ((camera_cost - float(trial_residual @ trial_residual)) /
                            max(camera_cost, 1.0) < RELATIVE_COST_TOLERANCE):
                        break
                else:
                    camera_damping *= 10.0
                    if camera_damping > 1.0e9:
                        break
        fit_weights = weights
        fit_line_weights = line_weights
        endpoint_residual, _, endpoint_depths = residual_and_jacobian(x, jacobian=False)
        point_raw = endpoint_residual[:point_rows].reshape(-1, 2) / weights[:, None]
        endpoint_measurements = (point_raw if len(point_raw) else
                                 endpoint_residual[point_rows:pixel_rows].reshape(-1, 2) /
                                 line_weights[:, None])
        endpoint_rmse = float(np.sqrt(np.mean(np.sum(endpoint_measurements**2, axis=1))))
        world_from_internal = anchor_r.T @ frame_rotation(x)
        if diagnostic_callback is not None:
            # Capture the numerical endpoint even when a later acceptance gate
            # refuses it. This is diagnostic data, never an applicable result.
            endpoint_focal, endpoint_rotations, endpoint_centers, endpoint_points = decode(x)
            line_raw = (endpoint_residual[point_rows:pixel_rows].reshape(-1, 2) /
                        line_weights[:, None])
            prior_rows = endpoint_residual[pixel_rows:]

            def rms(rows: np.ndarray) -> float | None:
                return float(np.sqrt(np.mean(rows * rows))) if rows.size and np.isfinite(rows).all() else None

            diagnostic_callback({
                "iterations": iterations,
                "converged": bool(converged),
                "stop_reason": stop_reason,
                "relative_cost_tolerance": RELATIVE_COST_TOLERANCE,
                "initial_objective": float(initial_residual @ initial_residual),
                "final_objective": float(endpoint_residual @ endpoint_residual),
                "recent_relative_improvements": relative_gains[-10:],
                "orientation_parameters": len(rotation_basis),
                "world_rotation": (world_from_internal @ anchor_r).tolist(),
                "focal_bound_hits": {
                    camera_id: ("wider_fov" if x[0 if share_lens else i] <= lower + FOCAL_BOUND_MARGIN else
                                "narrower_fov" if x[0 if share_lens else i] >= upper - FOCAL_BOUND_MARGIN else None)
                    for i, camera_id in enumerate(ids)},
                "focal_px": {camera_id: float(endpoint_focal[i])
                             for i, camera_id in enumerate(ids)},
                "cameras": {
                    camera_id: {
                        "rotation_w2c": (endpoint_rotations[i] @ world_from_internal.T).tolist(),
                        "center": (world_from_internal @ (endpoint_centers[i] * baseline) + anchor_c).tolist(),
                    } for i, camera_id in enumerate(ids)},
                "points": {
                    point_id: (world_from_internal @ (endpoint_points[i] * baseline) + anchor_c).tolist()
                    for i, point_id in enumerate(point_ids)},
                "point_rmse_px": rms(np.linalg.norm(point_raw, axis=1)),
                "line_rmse_px": rms(np.linalg.norm(line_raw, axis=1)),
                "weighted_point_residual_norm": float(np.linalg.norm(endpoint_residual[:point_rows])),
                "weighted_line_residual_norm": float(np.linalg.norm(endpoint_residual[point_rows:pixel_rows])),
                "prior_residual_norm": float(np.linalg.norm(prior_rows)),
                "depth_valid": bool(np.isfinite(endpoint_depths).all() and
                                    np.all(endpoint_depths > 0)),
            })
        calibration_refusal = ""
        bounded = [ids[i] + (" (wider FOV)" if x[0 if share_lens else i] <= lower + FOCAL_BOUND_MARGIN else " (narrower FOV)")
                   for i in range(ncam)
                   if unrestricted_active[0 if share_lens else i] and
                   (x[0 if share_lens else i] <= lower + FOCAL_BOUND_MARGIN or
                    x[0 if share_lens else i] >= upper - FOCAL_BOUND_MARGIN)]
        if bounded:
            calibration_refusal = (
                "Fitted focal reached the search bound for " + ", ".join(bounded) +
                f"; candidate point RMSE {endpoint_rmse:.2f}px. "
                "Check starting FOVs and image calibration before widening Lens Search %.")
        elif stop_reason == "time_limit" and not fixed_focals:
            calibration_refusal = (
                f"Point focal fit reached its time limit; candidate point RMSE {endpoint_rmse:.2f}px.")
        elif not converged and not fixed_focals:
            stopped = (f"reached the {iteration_limit}-iteration limit" if stop_reason == "iteration_limit"
                       else f"no improving step found at iteration {iterations}")
            calibration_refusal = (
                f"Point focal fit did not converge: {stopped}; "
                f"candidate point RMSE {endpoint_rmse:.2f}px.")
        fitted_residual, jac, depths = residual_and_jacobian(x, jacobian=True)
        end_raw = fitted_residual[:point_rows].reshape(-1, 2) / weights[:, None]
        all_raw = fitted_residual[:pixel_rows] / coordinate_weights
        fitted_measurements = (end_raw if len(end_raw) else
                               fitted_residual[point_rows:pixel_rows].reshape(-1, 2) /
                               line_weights[:, None])
        fitted_rmse = float(np.sqrt(np.mean(np.sum(fitted_measurements**2, axis=1))))
        if (not np.isfinite(fitted_rmse) or not np.isfinite(fitted_residual).all() or not np.isfinite(x).all() or
                not np.isfinite(depths).all() or np.any(depths <= 0)):
            return refuse("Fitted scene has points behind a camera", fitted=fitted_rmse)
        if scale_columns and abs(x[ncam + 5]) >= (
                MAX_LOG_METRIC_BASELINE_CHANGE - METRIC_BASELINE_BOUND_MARGIN):
            return refuse("Metric scale reached its numerical bound; check Mirror Empty and anchor placement",
                          fitted=fitted_rmse)
        if reference_constraints.active:
            chart_points = decode(x)[3]
            for point, target in reference_constraints.hard_xyz.items():
                if np.linalg.norm(chart_points[point] - target) * baseline > 1.0e-7:
                    return refuse("Fitted Known 3D point moved from its hard reference",
                                  fitted=fitted_rmse)
            for point, (normal, offset) in reference_constraints.hard_z.items():
                if abs(float(normal @ chart_points[point]) - offset) * baseline > 1.0e-7:
                    return refuse("Fitted On Ground point left the hard ground plane",
                                  fitted=fitted_rmse)
        if constraints.active:
            _focal, _rotations, _centers, final_points = decode(x)
            final_points, final_geometry = prior_geometry(x, final_points, line_geometry(x, final_points))
            plane_max, mirror_max = constraints.world_gaps(
                final_points, mirror_offset=(float(x[mirror_offset_column])
                                             if mirror_offset_column is not None else 0.0))
            if plane_max > PLANE_HARD_SLACK:
                return refuse("Fitted points violate a hard Is in Plane relation", fitted=fitted_rmse)
            if mirror_max > MIRROR_PAIR_HARD_GAP:
                return refuse("Fitted points violate a supplied point mirror relation", fitted=fitted_rmse)
            if line_mirrors:
                normal = constraints.mirror_normal
                assert normal is not None
                householder = np.eye(3) - 2 * np.outer(normal, normal)
                distance = (float(normal @ final_points[constraints.mirror_reference_index])
                            if constraints.mirror_reference_index is not None else constraints.mirror_distance)
                if mirror_offset_column is not None:
                    distance += float(x[mirror_offset_column])
                for left_id, right_id in line_mirrors:
                    left_p, left_d = final_geometry[line_index[left_id]]
                    right_p, right_d = final_geometry[line_index[right_id]]
                    left_p = canonical_line_point(left_p, left_d)
                    right_p = canonical_line_point(right_p, right_d)
                    reflected_p = householder @ left_p + 2 * distance * normal
                    reflected_d = householder @ left_d
                    position_gap = baseline * np.linalg.norm(
                        np.cross(right_d, reflected_p - right_p))
                    direction_gap = np.linalg.norm(np.cross(right_d, reflected_d))
                    if (position_gap > MIRROR_PAIR_HARD_GAP or
                            direction_gap > LINE_MIRROR_DIRECTION_HARD_SINE):
                        return refuse("Fitted lines violate a supplied mirror relation", fitted=fitted_rmse)
        if len(luv) or line_constraints.active:
            final_geometry = line_geometry(x, decode(x)[3])
            focal_check, rotations_check, centers_check, _points = decode(x)
            constrained_points, constrained_geometry = prior_geometry(x, _points, final_geometry)
            plane_gap, direction_gap = line_constraints.world_gaps(constrained_points, constrained_geometry)
            if plane_gap > PLANE_HARD_SLACK:
                return refuse("Fitted lines violate a hard Is in Plane relation", fitted=fitted_rmse)
            if direction_gap > LINE_RELATION_DIRECTION_HARD_SINE:
                return refuse("Fitted line directions violate Is in Plane or Is Parallel To", fitted=fitted_rmse)
            for index, endpoints in enumerate(luv):
                camera = lci[index]
                line_point, line_direction = final_geometry[lli[index]]
                for endpoint in endpoints:
                    source = calibrations[ids[camera]]
                    current_k = intrinsics_at(camera, focal_check[camera])
                    ideal = core.undistort_points(
                        np.asarray(endpoint, float).reshape(1, 2),
                        current_k.fx, current_k.fy, current_k.cx, current_k.cy,
                        source.division_lambda, source.brown_conrady)[0]
                    ray_camera = np.array((
                        (ideal[0] - current_k.cx) / current_k.fx,
                        (ideal[1] - current_k.cy) / current_k.fy, 1.0))
                    ray = rotations_check[camera].T @ ray_camera
                    support = _closest_point_on_line_to_ray(
                        line_point, line_direction, centers_check[camera], ray)
                    if (rotations_check[camera] @ (support - centers_check[camera]))[2] <= 0:
                        return refuse("Fitted line stroke has geometry behind a camera", fitted=fitted_rmse)
        # Reject serious camera-specific regression even if total cost improves.
        start_per_camera = initial_raw
        end_per_camera = end_raw
        for camera in range(ncam):
            chosen = ci == camera
            if not np.any(chosen):
                continue
            start_sse = float(np.sum(start_per_camera[chosen]**2))
            end_sse = float(np.sum(end_per_camera[chosen]**2))
            slack = pick_sigma_px**2 * (2 * len(start_per_camera[chosen]) +
                2 * math.sqrt(2 * len(start_per_camera[chosen]) * math.log(100.0)) +
                2 * math.log(100.0))
            if end_sse > start_sse + slack:
                return refuse("A camera's point fit deteriorated", fitted=fitted_rmse)
        focal, rotations, centers, points = decode(x)
        # Persist anchor orientation in its private calibration so subsequent
        # Sync keeps the corrected frame with an identity anchor root.
        result_sims = {anchor_id: SimilarityTransform()}
        result_cals = {}
        result_points = {point_ids[index]: world_from_internal @ (point * baseline) + anchor_c
                         for index, point in enumerate(points)}
        for camera, camera_id in enumerate(ids):
            source = calibrations[camera_id]
            source_k = source.intrinsics
            result_cals[camera_id] = core.Calibration(
                intrinsics=core.CameraIntrinsics(float(focal[camera]),
                                                 float(base_fy[camera] * focal[camera] / base_fx[camera]),
                                                 source_k.cx, source_k.cy,
                                                 source_k.image_width, source_k.image_height),
                rotation_w2c=np.array(world_from_internal.T if camera == 0 and len(rotation_basis) else
                                      source.rotation_w2c, copy=True),
                camera_center=np.array(source.camera_center, copy=True),
                division_lambda=source.division_lambda,
                brown_conrady=source.brown_conrady)
            if camera == 0:
                continue
            global_rotation = rotations[camera] @ world_from_internal.T
            global_center = world_from_internal @ (centers[camera] * baseline) + anchor_c
            sim_rotation = global_rotation.T @ source.rotation_w2c
            old_scale = float(initial.similarities[camera_id].scale)
            fitted_scale = (root_scale_at(camera, x)
                            if lock_translation and camera_id not in (fixed_similarities or {})
                            else old_scale)
            result_sims[camera_id] = SimilarityTransform(
                scale=fitted_scale, rotation=sim_rotation,
                translation=(np.zeros(3) if lock_translation and
                             camera_id not in (fixed_similarities or {}) else
                             global_center - fitted_scale *
                             (sim_rotation @ source.camera_center)))
        result_lines = {}
        geometry = line_geometry(x, points)
        match_inputs = {key: SyncMatchInput(key, result_cals[key]) for key in ids}
        for index, line_id in enumerate(line_ids):
            if line_id in (known_lines or {}):
                result_lines[line_id] = tuple(np.asarray(end, float).copy()
                                              for end in known_lines[line_id])
                result_points[line_id] = 0.5 * (result_lines[line_id][0] +
                                                result_lines[line_id][1])
                continue
            local_point, local_direction = geometry[index]
            world_point = world_from_internal @ (local_point * baseline) + anchor_c
            world_direction = world_from_internal @ local_direction
            segment = _finite_segment_from_line_observations(
                world_point, world_direction,
                [item for item in line_observations if item.landmark_id == line_id],
                result_sims, match_inputs)
            result_lines[line_id] = segment
            result_points[line_id] = 0.5 * (segment[0] + segment[1])
        line_raw = fitted_residual[point_rows:pixel_rows].reshape(-1, 2) / line_weights[:, None]
        per_camera = {}
        for camera, camera_id in enumerate(ids):
            chosen = ci == camera
            if np.any(chosen):
                per_camera[camera_id] = float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
            else:
                line_chosen = lci == camera
                per_camera[camera_id] = (
                    float(np.sqrt(np.mean(np.sum(line_raw[line_chosen]**2, axis=1))))
                    if np.any(line_chosen) else 0.0)
        per_point = {}
        for point, point_id in enumerate(point_ids):
            chosen = pi == point
            per_point[point_id] = (float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
                                   if np.any(chosen) else 0.0)
        for line_id, index in line_index.items():
            chosen = lli == index
            per_point[line_id] = (float(np.sqrt(np.mean(np.sum(line_raw[chosen]**2, axis=1))))
                                  if np.any(chosen) else 0.0)
        sync_result = SyncSolveResult(
            similarities=result_sims, landmarks=result_points,
            line_segments=result_lines,
            mean_reprojection_px=fitted_rmse, per_match_rmse_px=per_camera,
            per_landmark_rmse_px=per_point,
            message=f"Independent focals fitted from points and lines in {iterations} iterations"
                    if line_ids else f"Independent point focals fitted in {iterations} iterations",
            bundle_adjusted=True)
        sync_result.joint_mirror_offset_m = (
            float(x[mirror_offset_column]) * baseline
            if mirror_offset_column is not None else 0.0)
        if not all(np.isfinite(segment).all() for segment in result_lines.values()):
            return refuse("Fitted line endpoints are not finite", fitted=fitted_rmse)
        initial_objective = float(initial_residual @ initial_residual)
        fitted_objective = float(fitted_residual @ fitted_residual)
        if (np.isfinite(initial_objective) and np.isfinite(fitted_objective) and
                initial_objective - fitted_objective >
                MIN_CANDIDATE_RELATIVE_GAIN * max(initial_objective, 1.0)):
            candidate = FocalFitCandidate(
                result_cals, sync_result, initial_rmse, fitted_rmse, "",
                initial_objective, fitted_objective)
        if calibration_refusal:
            hint = ("" if stop_reason == "time_limit" else
                    _conflict_hint(ids, observations, pick_sigma_px, cancel_check=cancel_check))
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse(calibration_refusal + hint, fitted=fitted_rmse, allow_candidate=True)
        main_pixel_mask = np.concatenate((
            np.repeat(~np.isin(ci, fit_only_cameras), 2),
            np.repeat(~np.isin(lci, fit_only_cameras), 2)))
        main_raw = all_raw.copy()
        main_raw[~main_pixel_mask] = 0.0
        # Residual count and model dimensions make this a noise-aware fit test.
        if not has_geometric_priors and not _fits_noise_model(
                main_raw[main_pixel_mask], len(active_columns), pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Landmark fit is inconsistent with the stated pick noise." + hint,
                          fitted=fitted_rmse, allow_candidate=True)
        # Local covariance after fixing anchor and baseline gauges. Report
        # unbounded focal directions rather than mistaking a pseudoinverse for
        # evidence when the scene is rank deficient.
        if fixed_focals:
            return FocalBundleOutcome(True, "", result_cals, sync_result, {},
                                      initial_rmse, fitted_rmse,
                                      initial_objective=initial_objective,
                                      fitted_objective=fitted_objective)
        if fit_only_cameras:
            fit_weights = weights * ~np.isin(ci, fit_only_cameras)
            fit_line_weights = line_weights * ~np.isin(lci, fit_only_cameras)
            _main_residual, main_jac, _main_depths = residual_and_jacobian(
                x, jacobian=True)
            fit_weights = weights
            fit_line_weights = line_weights
        else:
            main_jac = jac
        active_jac = main_jac[:, active_columns]
        column_scale = np.maximum(np.linalg.norm(active_jac, axis=0), 1.0e-8)
        u, singular, vh = np.linalg.svd(active_jac / column_scale, full_matrices=False)
        tolerance = np.finfo(float).eps * max(active_jac.shape) * singular[0]
        null = singular <= tolerance
        if active_jac.shape[0] < active_jac.shape[1]:
            return refuse("Too few point observations for independent focal estimation", fitted=fitted_rmse, allow_candidate=True)
        if np.any(null):
            return refuse("The point geometry does not determine a full 3D scene", fitted=fitted_rmse, allow_candidate=True)
        # Only the weighted image rows receive independent click noise. Hard
        # constraints and soft springs affect the fit, never the pick count.
        pixel_influence = ((vh[~null].T / singular[~null]) @
                           u[:pixel_rows, ~null].T) * coordinate_weights / column_scale[:, None]
        if has_geometric_priors and not _fits_constrained_noise_model(
                main_raw, _active_pixel_jacobian(main_jac, pixel_rows,
                                                 active_columns, coordinate_weights),
                pixel_influence, pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Landmark fit is inconsistent with the stated pick noise." + hint,
                          fitted=fitted_rmse, allow_candidate=True)
        intervals = {}
        for camera, camera_id in enumerate(ids):
            if camera in fit_only_cameras and not share_lens and \
                    unrestricted_active[camera]:
                local_columns = np.asarray([
                    column for column in camera_columns[camera]
                    if unrestricted_active[column]], dtype=int)
                local_mask = np.concatenate((np.repeat(ci == camera, 2),
                                             np.repeat(lci == camera, 2)))
                local_jac = jac[:pixel_rows, local_columns][local_mask]
                if local_jac.shape[0] < local_jac.shape[1]:
                    return refuse("Fit Only focal has too few independent picks",
                                  fitted=fitted_rmse, allow_candidate=True)
                local_scale = np.maximum(np.linalg.norm(local_jac, axis=0), 1.0e-8)
                local_u, local_singular, local_vh = np.linalg.svd(
                    local_jac / local_scale, full_matrices=False)
                local_tolerance = (np.finfo(float).eps * max(local_jac.shape) *
                                   local_singular[0])
                if np.any(local_singular <= local_tolerance):
                    return refuse("Fit Only focal uncertainty is unbounded",
                                  fitted=fitted_rmse, allow_candidate=True)
                local_focal = int(np.flatnonzero(local_columns == camera)[0])
                local_influence = ((local_vh.T / local_singular) @
                                   local_u.T) * coordinate_weights[local_mask] / \
                                  local_scale[:, None]
                sigma_log = (pick_sigma_px *
                             np.linalg.norm(local_influence[local_focal]))
                if not np.isfinite(sigma_log) or sigma_log >= 10:
                    return refuse("Fit Only focal uncertainty is too broad to use",
                                  fitted=fitted_rmse, allow_candidate=True)
                intervals[camera_id] = (
                    float(focal[camera] * math.exp(-1.96 * sigma_log)),
                    float(focal[camera] * math.exp(1.96 * sigma_log)))
                continue
            if not active[0 if share_lens else camera]:
                intervals[camera_id] = (float(focal[camera]), float(focal[camera]))
                continue
            reduced_camera = int(np.flatnonzero(active_columns ==
                                               (0 if share_lens else camera))[0])
            if np.any(np.abs(vh[null, reduced_camera]) > 1.0e-6):
                return refuse("Focal uncertainty is unbounded", fitted=fitted_rmse, allow_candidate=True)
            # Weighted fitting with homoscedastic raw pixel noise uses sandwich
            # covariance. Sync weights change the estimator, not the noise of
            # an independent click: influence = (W^1/2 J)^+ W^1/2.
            influence = pixel_influence[reduced_camera]
            sigma_log = pick_sigma_px * np.linalg.norm(influence)
            if not np.isfinite(sigma_log):
                return refuse("Focal uncertainty is unbounded", fitted=fitted_rmse, allow_candidate=True)
            if sigma_log >= 10:
                return refuse("Focal uncertainty is too broad to use", fitted=fitted_rmse, allow_candidate=True)
            intervals[camera_id] = (float(focal[camera] * math.exp(-1.96 * sigma_log)),
                                    float(focal[camera] * math.exp(1.96 * sigma_log)))
        return FocalBundleOutcome(True, "", result_cals, sync_result, intervals,
                                  initial_rmse, fitted_rmse)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return refuse(f"Point focal fit failed: {exc}")


def refine_fixed_focals(
    request: SyncSolveRequest,
    initial: SyncSolveResult,
    *,
    cancel_check=None,
    progress_callback=None,
    diagnostic_callback=None,
    frozen_point_weights: list[tuple[str, str, float]] | None = None,
) -> FocalBundleOutcome:
    """Polish a complete Sync seed with the joint fit's focal columns frozen."""
    work_request, coverage = supported_joint_request(request, initial)
    included_cameras = {item.match_id for item in work_request.matches}
    if request.anchor_id not in included_cameras or len(included_cameras) < 2:
        return FocalBundleOutcome(False, "Joint continuation needs an anchor and a second supported camera",
                                  support_coverage=coverage)
    seed_lines = {key: value for key, value in initial.line_segments.items()
                  if key in {item.landmark_id for item in work_request.line_observations or ()} |
                  set(work_request.known_lines or {})}
    for key, segment in (work_request.known_lines or {}).items():
        seed_lines.setdefault(key, segment)
    seed_landmarks = {key: value for key, value in initial.landmarks.items()
                      if key in {item.landmark_id for item in work_request.observations} |
                      set(seed_lines) | set(work_request.known_world or {})}
    for key, point in (work_request.known_world or {}).items():
        seed_landmarks.setdefault(key, point)
    seed = replace(
        initial,
        similarities={key: value for key, value in initial.similarities.items()
                      if key in included_cameras},
        landmarks=seed_landmarks,
        line_segments=seed_lines)
    scorer = JointFitScorer(work_request, seed,
                            frozen_point_weights=frozen_point_weights)
    if scorer.weight_refusal:
        return FocalBundleOutcome(False,
                                  f"Joint continuation needs complete supported geometry: {scorer.weight_refusal}",
                                  support_coverage=coverage)
    incumbent_score = scorer.score(seed)
    if not incumbent_score.valid and incumbent_score.reason != "Hard world relation is violated":
        return FocalBundleOutcome(False,
                                  f"Joint continuation cannot score the incumbent: {incumbent_score.reason}",
                                  support_coverage=coverage)
    outcome = fit_independent_focals(
        {item.match_id: item.calibration for item in work_request.matches},
        scorer.point_observations, seed, anchor_id=work_request.anchor_id,
        plane_groups=work_request.plane_groups, plane_slack=work_request.plane_slack,
        mirror_pairs=work_request.mirror_pairs, mirror_plane=work_request.mirror_plane,
        mirror_slack=work_request.mirror_slack,
        mirror_landmark_id=work_request.mirror_landmark_id,
        line_observations=work_request.line_observations,
        parallel_pairs=work_request.parallel_pairs,
        cancel_check=cancel_check, progress_callback=progress_callback,
        diagnostic_callback=diagnostic_callback, fixed_focals=True,
        fixed_similarities=work_request.fixed_similarities,
        location_match_ids=work_request.location_match_ids,
        readonly_match_ids=work_request.readonly_match_ids,
        known_world=work_request.known_world,
        known_3d_slack=(KNOWN_3D_SLACK_DEFAULT if work_request.known_3d_slack is None
                        else work_request.known_3d_slack),
        ground_slack=(GROUND_SLACK_DEFAULT if work_request.ground_slack is None
                      else work_request.ground_slack),
        known_lines=work_request.known_lines,
        lock_rotation=work_request.lock_rotation,
        lock_translation=work_request.lock_translation,
    )
    outcome.support_coverage = coverage
    if not outcome.accepted or outcome.sync_result is None:
        outcome.initial_objective = (incumbent_score.objective
                                     if incumbent_score.valid else None)
        return outcome
    final_score = scorer.score(outcome.sync_result,
                               calibrations=outcome.calibrations)
    if not final_score.valid:
        return FocalBundleOutcome(
            False, f"Joint result cannot represent current evidence: {final_score.reason}",
            initial_rmse_px=outcome.initial_rmse_px,
            fitted_rmse_px=outcome.fitted_rmse_px,
            initial_objective=(incumbent_score.objective if incumbent_score.valid else None),
            constraint_gaps=final_score.constraint_gaps,
            support_coverage=coverage)
    # Published geometry is authoritative. Its full current-evidence score can
    # differ slightly from the private chart's cost after reconstruction.
    if (incumbent_score.valid and
            final_score.objective > incumbent_score.objective +
            1.0e-9 * max(incumbent_score.objective, 1.0)):
        return FocalBundleOutcome(
            False, "Joint result worsened the current weighted point, line and prior objective",
            initial_rmse_px=outcome.initial_rmse_px,
            fitted_rmse_px=outcome.fitted_rmse_px,
            initial_objective=incumbent_score.objective,
            fitted_objective=final_score.objective,
            constraint_gaps=final_score.constraint_gaps,
            support_coverage=coverage)
    outcome.initial_objective = incumbent_score.objective if incumbent_score.valid else None
    outcome.fitted_objective = final_score.objective
    outcome.constraint_gaps = final_score.constraint_gaps
    outcome.initial_rmse_px = (incumbent_score.point_rmse_px if work_request.observations
                               else incumbent_score.line_rmse_px)
    outcome.fitted_rmse_px = (final_score.point_rmse_px if work_request.observations
                              else final_score.line_rmse_px)
    fitted = outcome.sync_result
    fitted.similarities = {**initial.similarities, **fitted.similarities}
    fitted.landmarks = {**initial.landmarks, **fitted.landmarks}
    fitted.line_segments = {**initial.line_segments, **fitted.line_segments}
    outcome.calibrations = ({item.match_id: item.calibration for item in request.matches} |
                            outcome.calibrations)
    fitted.mean_reprojection_px = (final_score.point_rmse_px if work_request.observations
                                   else final_score.line_rmse_px)
    fitted.per_match_rmse_px = {**initial.per_match_rmse_px, **final_score.per_match_rmse_px}
    fitted.per_landmark_rmse_px = {**initial.per_landmark_rmse_px,
                                   **final_score.per_landmark_rmse_px}
    fitted.point_rmse_px = final_score.point_rmse_px
    fitted.line_rmse_px = final_score.line_rmse_px
    fitted.per_match_point_rmse_px = {
        **getattr(initial, "per_match_point_rmse_px", {}),
        **final_score.per_match_point_rmse_px}
    fitted.per_match_line_rmse_px = {
        **getattr(initial, "per_match_line_rmse_px", {}),
        **final_score.per_match_line_rmse_px}
    fitted.plane_seeded_landmark_ids = [
        key for key in initial.plane_seeded_landmark_ids if key in fitted.landmarks]
    angles, weak = joint_line_support_diagnostics(
        work_request, fitted, outcome.calibrations)
    fitted.line_support_angles_deg = angles
    fitted.weak_line_ids = weak
    if weak:
        fitted.message += " · weak 3D line support (" + ", ".join(weak[:3]) + ")"
    fitted.joint_initial_objective = outcome.initial_objective
    fitted.joint_final_objective = outcome.fitted_objective
    fitted.joint_constraint_gaps = final_score.constraint_gaps
    fitted.joint_support_coverage = coverage
    fitted.downweighted_landmark_ids = scorer.downweighted_landmark_ids
    fitted.joint_point_weights = scorer.effective_point_weights
    fitted.calibrations = outcome.calibrations
    return outcome
