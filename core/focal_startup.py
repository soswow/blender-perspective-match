"""Tentative missing-camera poses for joint focal fitting, never Sync acceptance."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from .sync.types import SimilarityTransform
from .sync.pose import _pnp_similarity, _planar_homography_similarities
from .sync.projection import _project_shared_points

MAX_SEED_SECONDS = 5.0
SEED_PNP_ITERATIONS = 30
MIN_SEED_POINTS = 8


def _dlt_seed(world, pixels, calibration):
    """Calibrated linear pose seed with normalized world coordinates."""
    origin = world.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((world - origin)**2, axis=1))))
    if scale < 1e-10:
        return None
    xyz = (world - origin) / scale
    if np.linalg.matrix_rank(xyz) < 3:
        return None
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    k = calibration.intrinsics
    uv = (pixels - [k.cx, k.cy]) / [k.fx, k.fy]
    rows = np.zeros((2 * len(xyz), 12))
    rows[0::2, :4] = homogeneous
    rows[1::2, 4:8] = homogeneous
    rows[0::2, 8:] = -uv[:, 0, None] * homogeneous
    rows[1::2, 8:] = -uv[:, 1, None] * homogeneous
    _, _, vh = np.linalg.svd(rows, full_matrices=False)
    camera = vh[-1].reshape(3, 4)
    if np.linalg.det(camera[:, :3]) < 0:
        camera *= -1
    u, singular, v = np.linalg.svd(camera[:, :3])
    rotation = u @ v
    magnitude = float(singular.mean())
    if magnitude < 1e-12 or np.linalg.det(rotation) < 0:
        return None
    center = origin - scale * rotation.T @ (camera[:, 3] / magnitude)
    transform = rotation.T @ calibration.rotation_w2c
    return SimilarityTransform(1., transform, center - transform @ calibration.camera_center)


def provisional_poses(calibrations, observations, initial, *, anchor_id,
                      cancel_check=None, progress_callback=None):
    """Complete a partial point startup without certifying its provisional poses."""
    missing = sorted(set(calibrations) - set(initial.similarities))
    if not missing:
        return initial, ''
    if anchor_id not in initial.similarities or len(initial.similarities) < 2:
        return initial, 'Initial Sync needs an anchor and another registered camera'
    poses = dict(initial.similarities)
    deadline = time.monotonic() + MAX_SEED_SECONDS
    for key in missing:
        if cancel_check and cancel_check():
            return initial, 'Cancelled'
        if time.monotonic() > deadline:
            return initial, 'Provisional camera startup reached its time limit'
        if progress_callback:
            progress_callback(0, 1, f'Preparing provisional camera: {key}')
        picks = [o for o in observations if o.match_id == key and o.landmark_id in initial.landmarks]
        if len(picks) < MIN_SEED_POINTS:
            return initial, f'{key} needs at least eight reconstructed point picks for provisional startup'
        world = np.asarray([initial.landmarks[o.landmark_id] for o in picks])
        if world.shape != (len(picks), 3) or not np.isfinite(world).all():
            return initial, 'Initial point geometry is invalid'
        pixels = np.asarray([(o.u, o.v) for o in picks])
        weights = np.asarray([o.weight for o in picks])
        cal = calibrations[key]
        candidates = []
        linear = _dlt_seed(world, pixels, cal)
        if linear is not None:
            candidates.append(linear)
        else:
            candidates.extend(_planar_homography_similarities(world, pixels, cal, weights=weights))
        # A registered view supplies a second orientation basin, not a fixed pose.
        shared = {o.landmark_id for o in picks}
        nearby = max(initial.similarities, key=lambda camera: sum(
            o.match_id == camera and o.landmark_id in shared for o in observations))
        other, sim = calibrations[nearby], initial.similarities[nearby]
        rotation = (other.rotation_w2c @ sim.rotation.T).T @ cal.rotation_w2c
        center = sim.transform_point(other.camera_center)
        candidates.append(SimilarityTransform(1., rotation, center - rotation @ cal.camera_center))
        best, best_cost = None, float('inf')
        for candidate in candidates:
            if cancel_check and cancel_check():
                return initial, 'Cancelled'
            if time.monotonic() > deadline:
                return initial, 'Provisional camera startup reached its time limit'
            fitted = _pnp_similarity(key, world, pixels, cal, initial=candidate,
                                    weights=weights, max_iterations=SEED_PNP_ITERATIONS)
            if fitted is None:
                continue
            projected, valid = _project_shared_points(world, cal, fitted)
            cost = float(np.sum(weights[:, None] * (projected - pixels)**2))
            if np.all(valid) and np.isfinite(cost) and cost < best_cost:
                best, best_cost = fitted, cost
        if best is None:
            return initial, f'No front-facing provisional pose for {key}; check its shared picks and starting FOV'
        poses[key] = best
    # The caller must jointly fit every camera, focal and point and run all its
    # acceptance checks. No pose or point here is written to the Blender scene.
    return replace(initial, similarities=poses), ''
