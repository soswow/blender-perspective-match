"""Independent pinhole focal bundle adjustment for supported landmark graphs.

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
from .focal_lines import LineChart, endpoint_distances
from .focal_line_constraints import LineFocalConstraints, LINE_RELATION_DIRECTION_HARD_SINE
from .focal_optimizer import bounded_lm_step
from .focal_orientation import orientation_basis, overall_rotation
from .focal_startup import provisional_poses
from .sync import SyncObservation, SyncLineObservation, SyncSolveResult, SimilarityTransform, SyncMatchInput
from .sync.constants import MIRROR_PAIR_HARD_GAP, PLANE_HARD_SLACK
from .sync.lines import (_reconstruct_line_from_observations,
                         _finite_segment_from_line_observations,
                         _closest_point_on_line_to_ray)
from .sync.mirrors import _dedupe_mirror_pairs
from .sync.projection import _log_rodrigues, _rodrigues


DEFAULT_POINT_FOCAL_SPAN = 0.4
# Resource guard for the dense joint fit, not an identifiability limit.
MAX_CAMERAS = 32
MAX_POINTS = 80
MAX_LINES = 24
MAX_LINE_STROKES = 96
LINE_MIRROR_DIRECTION_HARD_SINE = 0.01
MAX_ITERATIONS = 200
MAX_SECONDS = 30.0
FOCAL_BOUND_MARGIN = 1.0e-4
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
) -> FocalBundleOutcome:
    """Fit focal/pose/geometry; optional diagnostics capture an unvalidated endpoint."""
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
    if not initial.success or not set(initial.similarities) <= set(ids):
        missing = sorted(set(ids) - set(initial.similarities))
        detail = ": " + ", ".join(missing) if missing else ""
        return refuse("Initial Sync must support every camera" + detail)
    point_ids = sorted({item.landmark_id for item in observations})
    line_observations = line_observations or []
    if any(not isinstance(item, SyncLineObservation) for item in line_observations):
        return refuse("Line landmarks contain an invalid stroke")
    line_ids = sorted({item.landmark_id for item in line_observations})
    if len(line_ids) > MAX_LINES or len(line_observations) > MAX_LINE_STROKES:
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

    initial, startup_refusal = provisional_poses(
        calibrations, observations, initial, anchor_id=anchor_id,
        cancel_check=cancel_check, progress_callback=progress_callback)
    if startup_refusal:
        return refuse(startup_refusal)

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
            baseline_world=baseline, hard_plane=constraints.hard_plane)
    except (ValueError, TypeError, IndexError) as exc:
        return refuse(str(exc))
    direction = centers0[1] / np.linalg.norm(centers0[1])
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    tangent_a = np.cross(direction, axis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(direction, tangent_a)
    tangents = np.vstack((tangent_a, tangent_b))
    ncam = len(ids)
    has_geometric_priors = constraints.active or line_constraints.active
    npoint = len(point_ids)
    scale_columns = int(constraints.free_baseline)
    point_offset = ncam + 5 + scale_columns + 6 * (ncam - 2)
    point_end = point_offset + 3 * npoint
    line_offset = point_end
    line_end = line_offset + 4 * len(line_ids)
    rotation_basis = orientation_basis(
        constraints, line_constraints, points0, [chart.decode(chart.initial) for chart in line_charts])
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
    base_fx = np.asarray([calibrations[key].intrinsics.fx for key in ids], float)
    pp = np.asarray([(calibrations[key].intrinsics.cx, calibrations[key].intrinsics.cy)
                     for key in ids], float)
    lower, upper = np.log((1.0 - fx_span, 1.0 + fx_span))
    parameter_lower = np.full(len(x), -np.inf)
    parameter_upper = np.full(len(x), np.inf)
    parameter_lower[:ncam], parameter_upper[:ncam] = lower, upper
    if scale_columns:
        parameter_lower[ncam + 5] = -MAX_LOG_METRIC_BASELINE_CHANGE
        parameter_upper[ncam + 5] = MAX_LOG_METRIC_BASELINE_CHANGE
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

    def line_geometry(params: np.ndarray):
        return [chart.decode(params[line_offset + 4*i:line_offset + 4*i + 4])
                for i, chart in enumerate(line_charts)]

    def frame_rotation(params):
        return overall_rotation(params[line_end:orientation_end], rotation_basis)

    def prior_geometry(params, points, geometry):
        rotation = frame_rotation(params)
        return points @ rotation.T, [(rotation @ p, rotation @ d) for p, d in geometry]

    def line_image_residual(params: np.ndarray, focal, rotations, centers, geometry):
        result = np.empty((len(luv), 2))
        for index in range(len(luv)):
            camera = lci[index]
            point, direction = geometry[lli[index]]
            result[index] = line_weights[index] * endpoint_distances(
                point, direction, rotations[camera], centers[camera],
                focal[camera], pp[camera], luv[index])
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
        geometry = line_geometry(params)
        line_residual = line_image_residual(params, focal, rotations, centers, geometry)
        if jacobian and len(line_residual):
            line_jac = np.zeros((len(line_residual), len(params)))
            relevant = set(range(ncam))
            for camera in set(lci):
                if camera == 1:
                    relevant.update(range(ncam, ncam + 5 + scale_columns))
                elif camera > 1:
                    start = ncam + 5 + scale_columns + 6*(camera-2)
                    relevant.update(range(start, start+6))
            relevant.update(range(line_offset, line_end))
            for column in sorted(relevant):
                step = 1e-6 * max(1.0, abs(params[column]))
                shifted = params.copy()
                shifted[column] += step
                f, r, c, _p = decode(shifted)
                line_jac[:, column] = (
                    line_image_residual(shifted, f, r, c, line_geometry(shifted)) - line_residual) / step
            jac = np.vstack((jac, line_jac))
        pixel_residual = np.concatenate((pixel_residual, line_residual))
        if not has_geometric_priors:
            return pixel_residual, jac, depths
        offset = float(params[mirror_offset_column]) if mirror_offset_column is not None else 0.0
        constrained_points, constrained_lines = prior_geometry(params, points, geometry)
        prior_residual, prior_jacobian = constraints.residual_and_jacobian(
            constrained_points, point_offset=point_offset, parameter_count=len(params),
            mirror_offset=offset, mirror_offset_column=mirror_offset_column,
            jacobian=jacobian)
        if jacobian and len(rotation_basis):
            point_block = prior_jacobian[:, point_offset:point_end].reshape(-1, npoint, 3)
            prior_jacobian[:, point_offset:point_end] = (
                point_block @ frame_rotation(params)).reshape(-1, 3 * npoint)
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
                shifted_points = shifted[point_offset:point_end].reshape(npoint, 3)
                changed_points, changed_lines = prior_geometry(
                    shifted, shifted_points, line_geometry(shifted))
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
                progress_callback(iteration, MAX_ITERATIONS,
                                  "Estimating focal from landmarks" if line_ids else
                                  "Estimating focal from points")
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
                try:
                    step = bounded_lm_step(
                        jac, residual, x, parameter_lower, parameter_upper, damping,
                        cancel_check=lambda: (bool(cancel_check and cancel_check()) or
                                              time.monotonic() - start_time > MAX_SECONDS))
                except InterruptedError:
                    return refuse("Cancelled" if cancel_check and cancel_check() else
                                  "Point focal fit reached its time limit")
                trial_x = np.clip(x + step, parameter_lower, parameter_upper)
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
        endpoint_residual, _, endpoint_depths = residual_and_jacobian(x, jacobian=False)
        point_raw = endpoint_residual[:point_rows].reshape(-1, 2) / weights[:, None]
        endpoint_rmse = float(np.sqrt(np.mean(np.sum(point_raw**2, axis=1))))
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
                "orientation_parameters": len(rotation_basis),
                "world_rotation": (world_from_internal @ anchor_r).tolist(),
                "focal_bound_hits": {
                    camera_id: ("wider_fov" if x[i] <= lower + FOCAL_BOUND_MARGIN else
                                "narrower_fov" if x[i] >= upper - FOCAL_BOUND_MARGIN else None)
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
        bounded = [ids[i] + (" (wider FOV)" if x[i] <= lower + FOCAL_BOUND_MARGIN else " (narrower FOV)")
                   for i in range(ncam)
                   if x[i] <= lower + FOCAL_BOUND_MARGIN or x[i] >= upper - FOCAL_BOUND_MARGIN]
        if bounded:
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Fitted focal reached the search bound for " + ", ".join(bounded) +
                          f"; candidate point RMSE {endpoint_rmse:.2f}px. "
                          "Check starting FOVs and image calibration before widening Lens Search %." + hint,
                          fitted=endpoint_rmse)
        if not converged:
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse(f"Point focal fit did not converge; candidate point RMSE {endpoint_rmse:.2f}px." + hint,
                          fitted=endpoint_rmse)
        fitted_residual, jac, depths = residual_and_jacobian(x, jacobian=True)
        end_raw = fitted_residual[:point_rows].reshape(-1, 2) / weights[:, None]
        all_raw = fitted_residual[:pixel_rows] / coordinate_weights
        fitted_rmse = float(np.sqrt(np.mean(np.sum(end_raw**2, axis=1))))
        if not np.isfinite(fitted_rmse) or np.any(depths <= 0):
            return refuse("Fitted scene has points behind a camera", fitted=fitted_rmse)
        if scale_columns and abs(x[ncam + 5]) >= (
                MAX_LOG_METRIC_BASELINE_CHANGE - METRIC_BASELINE_BOUND_MARGIN):
            return refuse("Metric scale reached its numerical bound; check Mirror Empty and anchor placement",
                          fitted=fitted_rmse)
        if constraints.active:
            _focal, _rotations, _centers, final_points = decode(x)
            final_points, final_geometry = prior_geometry(x, final_points, line_geometry(x))
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
                    reflected_p = householder @ left_p + 2 * distance * normal
                    reflected_d = householder @ left_d
                    position_gap = baseline * np.linalg.norm(
                        np.cross(right_d, reflected_p - right_p))
                    direction_gap = np.linalg.norm(np.cross(right_d, reflected_d))
                    if (position_gap > MIRROR_PAIR_HARD_GAP or
                            direction_gap > LINE_MIRROR_DIRECTION_HARD_SINE):
                        return refuse("Fitted lines violate a supplied mirror relation", fitted=fitted_rmse)
        if len(luv):
            final_geometry = line_geometry(x)
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
                    ray_camera = np.array((
                        (endpoint[0] - pp[camera, 0]) / focal_check[camera],
                        (endpoint[1] - pp[camera, 1]) / focal_check[camera], 1.0))
                    ray = rotations_check[camera].T @ ray_camera
                    support = _closest_point_on_line_to_ray(
                        line_point, line_direction, centers_check[camera], ray)
                    if (rotations_check[camera] @ (support - centers_check[camera]))[2] <= 0:
                        return refuse("Fitted line stroke has geometry behind a camera", fitted=fitted_rmse)
        # Residual count and model dimensions make this a noise-aware fit test.
        if not has_geometric_priors and not _fits_noise_model(all_raw, len(x), pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Landmark fit is inconsistent with the stated pick noise." + hint,
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
        if has_geometric_priors and not _fits_constrained_noise_model(
                all_raw, jac[:pixel_rows] / coordinate_weights[:, None],
                pixel_influence, pick_sigma_px):
            hint = _conflict_hint(ids, observations, pick_sigma_px,
                                  cancel_check=cancel_check)
            if cancel_check and cancel_check():
                return refuse("Cancelled")
            return refuse("Landmark fit is inconsistent with the stated pick noise." + hint,
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
                intrinsics=core.CameraIntrinsics(float(focal[camera]), float(focal[camera]),
                                                 source_k.cx, source_k.cy,
                                                 source_k.image_width, source_k.image_height),
                rotation_w2c=np.array(world_from_internal.T if camera == 0 and len(rotation_basis) else
                                      source.rotation_w2c, copy=True),
                camera_center=np.array(source.camera_center, copy=True),
                division_lambda=0.0, brown_conrady=())
            if camera == 0:
                continue
            global_rotation = rotations[camera] @ world_from_internal.T
            global_center = world_from_internal @ (centers[camera] * baseline) + anchor_c
            sim_rotation = global_rotation.T @ source.rotation_w2c
            old_scale = float(initial.similarities[camera_id].scale)
            result_sims[camera_id] = SimilarityTransform(
                scale=old_scale, rotation=sim_rotation,
                translation=global_center - old_scale * (sim_rotation @ source.camera_center))
        result_lines = {}
        geometry = line_geometry(x)
        match_inputs = {key: SyncMatchInput(key, result_cals[key]) for key in ids}
        for index, line_id in enumerate(line_ids):
            local_point, local_direction = geometry[index]
            world_point = world_from_internal @ (local_point * baseline) + anchor_c
            world_direction = world_from_internal @ local_direction
            segment = _finite_segment_from_line_observations(
                world_point, world_direction,
                [item for item in line_observations if item.landmark_id == line_id],
                result_sims, match_inputs)
            result_lines[line_id] = segment
            result_points[line_id] = 0.5 * (segment[0] + segment[1])
        per_camera = {}
        for camera, camera_id in enumerate(ids):
            chosen = ci == camera
            per_camera[camera_id] = float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
        per_point = {}
        for point, point_id in enumerate(point_ids):
            chosen = pi == point
            per_point[point_id] = float(np.sqrt(np.mean(np.sum(end_per_camera[chosen]**2, axis=1))))
        line_raw = fitted_residual[point_rows:pixel_rows].reshape(-1, 2) / line_weights[:, None]
        for line_id, index in line_index.items():
            chosen = lli == index
            per_point[line_id] = float(np.sqrt(np.mean(np.sum(line_raw[chosen]**2, axis=1))))
        sync_result = SyncSolveResult(
            similarities=result_sims, landmarks=result_points,
            line_segments=result_lines,
            mean_reprojection_px=fitted_rmse, per_match_rmse_px=per_camera,
            per_landmark_rmse_px=per_point,
            message=f"Independent focals fitted from points and lines in {iterations} iterations"
                    if line_ids else f"Independent point focals fitted in {iterations} iterations",
            bundle_adjusted=True)
        return FocalBundleOutcome(True, "", result_cals, sync_result, intervals,
                                  initial_rmse, fitted_rmse)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return refuse(f"Point focal fit failed: {exc}")
