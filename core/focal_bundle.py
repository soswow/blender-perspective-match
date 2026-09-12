"""Independent pinhole focal bundle adjustment for supported point graphs.

The anchor camera fixes orientation and center. Free-scale graphs fix one camera
baseline; an off-anchor supplied mirror plane instead fixes metric scale.
All decisions use image picks, supplied relations and the fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from . import geometry as core
from .focal_constraints import PointFocalConstraints
from .sync import SyncObservation, SyncSolveResult, SimilarityTransform
from .sync.constants import MIRROR_PAIR_HARD_GAP, PLANE_HARD_SLACK
from .sync.projection import _log_rodrigues, _rodrigues


DEFAULT_POINT_FOCAL_SPAN = 0.4
# Resource guard for the dense joint fit, not an identifiability limit.
MAX_CAMERAS = 32
MAX_POINTS = 80
MAX_ITERATIONS = 100
MAX_SECONDS = 30.0
MAX_LOG_METRIC_BASELINE_CHANGE = 10.0
METRIC_BASELINE_BOUND_MARGIN = 1e-4
EPIPOLAR_HINT_MIN_WITHHELD_SIGMA = 8.0
EPIPOLAR_HINT_MAX_FITS = 1000
EPIPOLAR_HINT_MAX_SECONDS = 10.0


@dataclass
class FocalBundleOutcome:
    accepted: bool
    reason: str
    calibrations: dict[str, core.Calibration] = field(default_factory=dict)
    sync_result: SyncSolveResult | None = None
    intervals_px: dict[str, tuple[float, float]] = field(default_factory=dict)
    initial_rmse_px: float = float("inf")
    fitted_rmse_px: float = float("inf")


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
    cancel_check=None,
    progress_callback=None,
) -> FocalBundleOutcome:
    """Refine all free point focals, poses and 3D in one fixed anchor gauge."""
    def refuse(reason: str, *, fitted: float = float("inf")) -> FocalBundleOutcome:
        return FocalBundleOutcome(False, reason, initial_rmse_px=initial_rmse,
                                  fitted_rmse_px=fitted)

    initial_rmse = float("inf")
    ids = list(calibrations)
    if len(ids) < 3:
        return refuse("Point focal estimation needs at least three cameras")
    if len(ids) > MAX_CAMERAS:
        return refuse(f"Point focal estimation currently supports up to {MAX_CAMERAS} cameras per joint fit (resource limit)")
    if anchor_id not in calibrations:
        return refuse("Anchor camera is missing")
    ids.remove(anchor_id)
    ids.insert(0, anchor_id)
    if not np.isfinite(pick_sigma_px) or pick_sigma_px <= 0:
        return refuse("Pick sigma must be positive and finite")
    if not np.isfinite(fx_span) or not 0.0 < fx_span < 1.0:
        return refuse("Focal search span must lie between 0 and 100%")
    if not initial.success or set(initial.similarities) != set(ids):
        return refuse("Initial Sync must support every camera")
    point_ids = sorted({item.landmark_id for item in observations})
    if not (8 <= len(point_ids) <= MAX_POINTS) or set(initial.landmarks) != set(point_ids):
        return refuse(f"Point focal estimation needs 8–{MAX_POINTS} fully reconstructed points")
    if len({(item.match_id, item.landmark_id) for item in observations}) != len(observations):
        return refuse("A free point has duplicate picks in one camera")
    if any(item.match_id not in calibrations or item.on_ground or
           not np.isfinite((item.u, item.v, item.weight)).all() or item.weight <= 0
           for item in observations):
        return refuse("Point picks must be finite, free and from included cameras")
    for calibration in calibrations.values():
        k = calibration.intrinsics
        if (not np.isfinite((k.fx, k.fy, k.cx, k.cy)).all() or k.fx <= 0 or
            not np.isclose(k.fx, k.fy, rtol=1.0e-6) or calibration.has_distortion):
            return refuse("Point focal estimation needs square pixels and zero distortion")
    if any(len({item.match_id for item in observations if item.landmark_id == point_id}) < 2
           for point_id in point_ids):
        return refuse("Every point needs at least two picks")
    if mirror_landmark_id is not None:
        if not isinstance(mirror_landmark_id, str) or not mirror_landmark_id:
            return refuse("Mirror reference landmark ID is invalid")
        if mirror_landmark_id not in point_ids:
            return refuse("Mirror reference landmark needs two-view picks")
    for camera_id in ids:
        count = sum(item.match_id == camera_id for item in observations)
        if count < 8:
            return refuse(f"{camera_id} has {count} point picks; each camera needs at least eight")
    if not _has_depth_evidence(ids, observations, float(pick_sigma_px)):
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
    reference = max(range(1, len(ids)), key=lambda index: float(
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
    points0 = np.asarray([anchor_r @ (initial.landmarks[key] - anchor_c) / baseline
                          for key in point_ids])
    if not np.isfinite(points0).all():
        return refuse("Initial point geometry is invalid")
    try:
        constraints = PointFocalConstraints.from_inputs(
            point_ids, anchor_rotation=anchor_r, anchor_center=anchor_c,
            baseline_world=baseline, plane_groups=plane_groups,
            plane_slack=0.0 if plane_slack is None else plane_slack,
            mirror_pairs=mirror_pairs, mirror_plane=mirror_plane,
            mirror_slack=0.0 if mirror_slack is None else mirror_slack,
            mirror_landmark_id=mirror_landmark_id)
    except (ValueError, TypeError, IndexError) as exc:
        return refuse(str(exc))
    direction = centers0[1] / np.linalg.norm(centers0[1])
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    tangent_a = np.cross(direction, axis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(direction, tangent_a)
    tangents = np.vstack((tangent_a, tangent_b))
    ncam = len(ids)
    npoint = len(point_ids)
    scale_columns = int(constraints.free_baseline)
    point_offset = ncam + 5 + scale_columns + 6 * (ncam - 2)
    point_end = point_offset + 3 * npoint
    mirror_offset_column = point_end if constraints.free_mirror_offset else None
    x = np.concatenate((np.zeros(ncam), _log_rodrigues(rotations0[1]),
                        np.zeros(2), *([np.zeros(1)] if scale_columns else []),
                        *[np.concatenate((_log_rodrigues(rotations0[i]),
                                          centers0[i])) for i in range(2, ncam)],
                        points0.ravel(),
                        *([np.zeros(1)] if constraints.free_mirror_offset else [])))
    camera_index = {key: index for index, key in enumerate(ids)}
    point_index = {key: index for index, key in enumerate(point_ids)}
    ci = np.asarray([camera_index[item.match_id] for item in observations], int)
    pi = np.asarray([point_index[item.landmark_id] for item in observations], int)
    uv = np.asarray([(item.u, item.v) for item in observations], float)
    weights = np.sqrt(np.asarray([item.weight for item in observations], float))
    base_fx = np.asarray([calibrations[key].intrinsics.fx for key in ids], float)
    pp = np.asarray([(calibrations[key].intrinsics.cx, calibrations[key].intrinsics.cy)
                     for key in ids], float)
    lower, upper = np.log((1.0 - fx_span, 1.0 + fx_span))
    start_time = time.monotonic()

    def decode(params: np.ndarray):
        focal = base_fx * np.exp(params[:ncam])
        raw = direction + params[ncam + 3] * tangents[0] + params[ncam + 4] * tangents[1]
        radius = math.exp(float(params[ncam + 5])) if scale_columns else 1.0
        first_center = radius * raw / np.linalg.norm(raw)
        rotations = [np.eye(3), _rodrigues(params[ncam:ncam + 3])]
        centers = [np.zeros(3), first_center]
        for camera in range(2, ncam):
            offset = ncam + 5 + scale_columns + 6 * (camera - 2)
            rotations.append(_rodrigues(params[offset:offset + 3]))
            centers.append(params[offset + 3:offset + 6])
        return focal, rotations, np.asarray(centers), params[point_offset:point_end].reshape(npoint, 3)

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
            predicted, dq = _project_with_depth_penalty(q, focal[camera], pp[camera])
            ratio = q[:, :2] / np.maximum(z, 1.0e-6)[:, None]
            residual[selected] = (predicted - uv[selected]) * weights[selected, None]
            if jac is None:
                continue
            dq *= weights[selected, None, None]
            rows = np.column_stack((2 * selected, 2 * selected + 1)).ravel()
            jac[rows, camera] = (focal[camera] * ratio * weights[selected, None]).ravel()
            point_j = np.einsum('nij,jk->nik', dq, rotations[camera])
            for local, point_id in enumerate(pi[selected]):
                jac[2 * selected[local]:2 * selected[local] + 2,
                    point_offset + 3 * point_id:point_offset + 3 * point_id + 3] = point_j[local]
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
                    center_start = ncam + 3
                else:
                    rotation_start = ncam + 5 + scale_columns + 6 * (camera - 2)
                    center_j = np.eye(3)
                    center_start = rotation_start + 3
                jac[rows, center_start:center_start + len(center_j)] = (
                    -np.einsum('nij,kj->nik', point_j, np.asarray(center_j))).reshape(-1, len(center_j))
                for component in range(3):
                    trial = params[rotation_start:rotation_start + 3].copy()
                    step = 1.0e-6 * max(1.0, abs(trial[component]))
                    trial[component] += step
                    derivative = ((_rodrigues(trial) - rotations[camera]) @ offset_xyz.T).T / step
                    jac[rows, rotation_start + component] = np.einsum(
                        'nij,nj->ni', dq, derivative).ravel()
        pixel_residual = residual.ravel()
        if not constraints.active:
            return pixel_residual, jac, depths
        offset = float(params[mirror_offset_column]) if mirror_offset_column is not None else 0.0
        prior_residual, prior_jacobian = constraints.residual_and_jacobian(
            points, point_offset=point_offset, parameter_count=len(params),
            mirror_offset=offset, mirror_offset_column=mirror_offset_column,
            jacobian=jacobian)
        return (np.concatenate((pixel_residual, prior_residual)),
                np.vstack((jac, prior_jacobian)) if jacobian else None, depths)

    try:
        initial_residual, _, _ = residual_and_jacobian(x, jacobian=False)
        pixel_rows = 2 * len(uv)
        coordinate_weights = np.repeat(weights, 2)
        initial_raw = initial_residual[:pixel_rows].reshape(-1, 2) / weights[:, None]
        initial_rmse = float(np.sqrt(np.mean(np.sum(initial_raw**2, axis=1))))
        damping = 1.0e-3
        converged = False
        iterations = 0
        for iteration in range(MAX_ITERATIONS):
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            if time.monotonic() - start_time > MAX_SECONDS:
                return refuse("Point focal fit reached its time limit")
            if progress_callback:
                progress_callback(iteration, MAX_ITERATIONS, "Estimating focal from points")
            residual, jac, _ = residual_and_jacobian(x, jacobian=True)
            cost = float(residual @ residual)
            column_scale = np.maximum(np.linalg.norm(jac, axis=0), 1.0e-8)
            scaled = jac / column_scale
            gradient = scaled.T @ residual
            if np.linalg.norm(gradient, ord=np.inf) < 1.0e-7:
                converged = True
                iterations = iteration
                break
            accepted_step = False
            for _trial in range(12):
                if cancel_check and cancel_check():
                    return refuse("Cancelled")
                if time.monotonic() - start_time > MAX_SECONDS:
                    return refuse("Point focal fit reached its time limit")
                # Augmented least squares is better conditioned than normal equations.
                system = np.vstack((scaled, math.sqrt(damping) * np.eye(len(x))))
                target = np.concatenate((-residual, np.zeros(len(x))))
                step_scaled = np.linalg.lstsq(system, target, rcond=None)[0]
                trial_x = x + step_scaled / column_scale
                trial_x[:ncam] = np.clip(trial_x[:ncam], lower, upper)
                if scale_columns:
                    trial_x[ncam + 5] = np.clip(
                        trial_x[ncam + 5], -MAX_LOG_METRIC_BASELINE_CHANGE,
                        MAX_LOG_METRIC_BASELINE_CHANGE)
                trial_residual, _, _ = residual_and_jacobian(trial_x, jacobian=False)
                trial_cost = float(trial_residual @ trial_residual)
                if np.isfinite(trial_cost) and trial_cost < cost:
                    relative_gain = (cost - trial_cost) / max(cost, 1.0)
                    x = trial_x
                    damping = max(damping / 3.0, 1.0e-9)
                    accepted_step = True
                    if relative_gain < 1.0e-9:
                        converged = True
                    break
                damping *= 10.0
            iterations = iteration + 1
            if converged or not accepted_step:
                converged = converged or not accepted_step and np.linalg.norm(gradient, ord=np.inf) < 1.0e-5
                break
        if not converged:
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Point focal fit did not converge." + hint)
        fitted_residual, jac, depths = residual_and_jacobian(x, jacobian=True)
        end_raw = fitted_residual[:pixel_rows].reshape(-1, 2) / weights[:, None]
        fitted_rmse = float(np.sqrt(np.mean(np.sum(end_raw**2, axis=1))))
        if not np.isfinite(fitted_rmse) or np.any(depths <= 0):
            return refuse("Fitted scene has points behind a camera", fitted=fitted_rmse)
        if np.any(x[:ncam] <= lower + 1.0e-4) or np.any(x[:ncam] >= upper - 1.0e-4):
            return refuse("Fitted focal reached the search bound; widen Lens Search %", fitted=fitted_rmse)
        if scale_columns and abs(x[ncam + 5]) >= (
                MAX_LOG_METRIC_BASELINE_CHANGE - METRIC_BASELINE_BOUND_MARGIN):
            return refuse("Metric scale reached its numerical bound; check Mirror Empty and anchor placement",
                          fitted=fitted_rmse)
        if constraints.active:
            _focal, _rotations, _centers, final_points = decode(x)
            plane_max, mirror_max = constraints.world_gaps(
                final_points, mirror_offset=(float(x[mirror_offset_column])
                                             if mirror_offset_column is not None else 0.0))
            if plane_max > PLANE_HARD_SLACK:
                return refuse("Fitted points violate a hard Is in Plane relation", fitted=fitted_rmse)
            if mirror_max > MIRROR_PAIR_HARD_GAP:
                return refuse("Fitted points violate a supplied point mirror relation", fitted=fitted_rmse)
        # Residual count and model dimensions make this a noise-aware fit test.
        if not constraints.active and not _fits_noise_model(end_raw, len(x), pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Point fit is inconsistent with the stated pick noise." + hint,
                          fitted=fitted_rmse)
        # Reject serious camera-specific regression even if total cost improves.
        start_per_camera = initial_raw
        end_per_camera = end_raw
        for camera in range(ncam):
            chosen = ci == camera
            start_sse = float(np.sum(start_per_camera[chosen]**2))
            end_sse = float(np.sum(end_per_camera[chosen]**2))
            slack = pick_sigma_px**2 * (2 * len(start_per_camera[chosen]) +
                2 * math.sqrt(2 * len(start_per_camera[chosen]) * math.log(100.0)) +
                2 * math.log(100.0))
            if end_sse > start_sse + slack:
                return refuse("A camera's point fit deteriorated", fitted=fitted_rmse)
        # Local covariance after fixing anchor and baseline gauges. Report
        # unbounded focal directions rather than mistaking a pseudoinverse for
        # evidence when the scene is rank deficient.
        column_scale = np.maximum(np.linalg.norm(jac, axis=0), 1.0e-8)
        u, singular, vh = np.linalg.svd(jac / column_scale, full_matrices=False)
        tolerance = np.finfo(float).eps * max(jac.shape) * singular[0]
        null = singular <= tolerance
        if jac.shape[0] < jac.shape[1]:
            return refuse("Too few point observations for independent focal estimation", fitted=fitted_rmse)
        if np.any(null):
            return refuse("The point geometry does not determine a full 3D scene", fitted=fitted_rmse)
        # Only the weighted image rows receive independent click noise. Hard
        # constraints and soft springs affect the fit, never the pick count.
        pixel_influence = ((vh[~null].T / singular[~null]) @
                           u[:pixel_rows, ~null].T) * coordinate_weights / column_scale[:, None]
        if constraints.active and not _fits_constrained_noise_model(
                end_raw.ravel(), jac[:pixel_rows] / coordinate_weights[:, None],
                pixel_influence, pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Point fit is inconsistent with the stated pick noise." + hint,
                          fitted=fitted_rmse)
        focal, rotations, centers, points = decode(x)
        intervals = {}
        for camera, camera_id in enumerate(ids):
            if np.any(np.abs(vh[null, camera]) > 1.0e-6):
                return refuse("Focal uncertainty is unbounded", fitted=fitted_rmse)
            # Weighted fitting with homoscedastic raw pixel noise uses sandwich
            # covariance. Sync weights change the estimator, not the noise of
            # an independent click: influence = (W^1/2 J)^+ W^1/2.
            influence = pixel_influence[camera]
            sigma_log = pick_sigma_px * np.linalg.norm(influence)
            if not np.isfinite(sigma_log):
                return refuse("Focal uncertainty is unbounded", fitted=fitted_rmse)
            if sigma_log >= 10:
                return refuse("Focal uncertainty is too broad to use", fitted=fitted_rmse)
            intervals[camera_id] = (float(focal[camera] * math.exp(-1.96 * sigma_log)),
                                    float(focal[camera] * math.exp(1.96 * sigma_log)))
        # Preserve private poses; the fitted world poses go into the existing
        # root similarities, and points return to the original anchor world.
        result_sims = {anchor_id: SimilarityTransform()}
        result_cals = {}
        result_points = {point_ids[index]: anchor_r.T @ (point * baseline) + anchor_c
                         for index, point in enumerate(points)}
        for camera, camera_id in enumerate(ids):
            source = calibrations[camera_id]
            source_k = source.intrinsics
            result_cals[camera_id] = core.Calibration(
                intrinsics=core.CameraIntrinsics(float(focal[camera]), float(focal[camera]),
                                                 source_k.cx, source_k.cy,
                                                 source_k.image_width, source_k.image_height),
                rotation_w2c=np.array(source.rotation_w2c, copy=True),
                camera_center=np.array(source.camera_center, copy=True),
                division_lambda=0.0, brown_conrady=())
            if camera == 0:
                continue
            global_rotation = rotations[camera] @ anchor_r
            global_center = anchor_r.T @ (centers[camera] * baseline) + anchor_c
            sim_rotation = global_rotation.T @ source.rotation_w2c
            old_scale = float(initial.similarities[camera_id].scale)
            result_sims[camera_id] = SimilarityTransform(
                scale=old_scale, rotation=sim_rotation,
                translation=global_center - old_scale * (sim_rotation @ source.camera_center))
        per_camera = {}
        for camera, camera_id in enumerate(ids):
            chosen = ci == camera
            per_camera[camera_id] = float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
        per_point = {}
        for point, point_id in enumerate(point_ids):
            chosen = pi == point
            per_point[point_id] = float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
        sync_result = SyncSolveResult(
            similarities=result_sims, landmarks=result_points,
            mean_reprojection_px=fitted_rmse, per_match_rmse_px=per_camera,
            per_landmark_rmse_px=per_point,
            message=f"Independent point focals fitted in {iterations} iterations",
            bundle_adjusted=True)
        return FocalBundleOutcome(True, "", result_cals, sync_result, intervals,
                                  initial_rmse, fitted_rmse)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return refuse(f"Point focal fit failed: {exc}")
