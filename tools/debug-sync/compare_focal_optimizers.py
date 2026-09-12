#!/usr/bin/env python3
"""Compare SciPy TRF on the exact initialized point-FOV residual problem.

This is a private numerical probe. It captures a production closure through a
one-shot trace before the first objective evaluation; no Sync is rerun and no
result is applied. The capture is intentionally tied to the current function's
local variable names and rejects a changed source hash when requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares


class CapturedProblem(Exception):
    pass


def load_problem(inputs_path: Path, startup_path: Path, expected_hash: str | None):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.synthetic_sync.solver import load_core
    core, sync = load_core()
    from match_perspective.core import focal_bundle
    source = Path(focal_bundle.__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_hash and source_hash != expected_hash:
        raise RuntimeError(f"source hash changed: {source_hash}")
    inputs = json.loads(inputs_path.read_text())
    startup = json.loads(startup_path.read_text())
    calibrations = {}
    for match in inputs['matches']:
        base = match['base_calibration']
        calibrations[match['match_id']] = core.Calibration(
            core.CameraIntrinsics(**base['intrinsics']), np.array(base['rotation_w2c']),
            np.array(base['camera_center']), division_lambda=base.get('division_lambda', 0.),
            brown_conrady=tuple(base.get('brown_conrady', ())))
    initial = sync.SyncSolveResult(
        similarities={k: sync.SimilarityTransform(v['scale'], np.array(v['rotation']),
                                                    np.array(v['translation']))
                      for k, v in startup['similarities'].items()},
        landmarks={k: np.array(v) for k, v in startup['landmarks'].items()},
        mean_reprojection_px=startup['mean_reprojection_px'],
        per_match_rmse_px=startup['per_match_rmse_px'],
        per_landmark_rmse_px=startup['per_landmark_rmse_px'],
        message=startup['message'], success=startup['success'])
    fields = ('anchor_id', 'fx_span', 'pick_sigma_px', 'plane_groups', 'plane_slack',
              'mirror_pairs', 'mirror_plane', 'mirror_slack', 'mirror_landmark_id',
              'parallel_pairs')
    captured = {}
    def trace(frame, event, arg):
        if frame.f_code is not focal_bundle.fit_independent_focals.__code__:
            return None
        if event == 'line':
            local = frame.f_locals
            needed = ('residual_and_jacobian', 'decode', 'x', 'lower', 'upper',
                      'ncam', 'ids', 'point_ids', 'uv', 'luv', 'weights',
                      'line_weights', 'anchor_r', 'anchor_c', 'baseline',
                      'scale_columns', 'frame_rotation')
            if all(k in local for k in needed) and frame.f_lineno >= focal_bundle.fit_independent_focals.__code__.co_firstlineno:
                captured.update({k: local[k] for k in needed})
                raise CapturedProblem
        return trace
    sys.settrace(trace)
    try:
        focal_bundle.fit_independent_focals(
            calibrations, [sync.SyncObservation(**p) for p in inputs['observations']], initial,
            line_observations=[sync.SyncLineObservation(**p) for p in inputs['line_observations']],
            **{k: inputs[k] for k in fields if k in inputs})
    except CapturedProblem:
        pass
    finally:
        sys.settrace(None)
    if not captured:
        raise RuntimeError('capture failed before numerical evaluation; verify startup and local names')
    x = captured['x']
    if x.ndim != 1 or len(x) <= captured['ncam'] or len(captured['ids']) != captured['ncam']:
        raise RuntimeError('captured problem has unexpected dimensions')
    captured['source_sha256'] = source_hash
    captured['core_source_sha256'] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / 'core').rglob('*.py'))}
    captured['inputs_sha256'] = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    captured['startup_sha256'] = hashlib.sha256(startup_path.read_bytes()).hexdigest()
    return captured, calibrations


def report(problem, calibrations, result, bounds, seconds):
    x = result.x
    residual, _, depths = problem['residual_and_jacobian'](x, jacobian=False)
    npoint = 2 * len(problem['uv'])
    nline = 2 * len(problem['luv'])
    weights = problem['weights']
    line_weights = problem['line_weights']
    point_raw = residual[:npoint].reshape(-1, 2) / weights[:, None]
    line_raw = residual[npoint:npoint+nline].reshape(-1, 2) / line_weights[:, None]
    focal, rotations, centers, points = problem['decode'](x)
    world_from_internal = problem['anchor_r'].T @ problem['frame_rotation'](x)
    ids = problem['ids']
    return {
        'success': bool(result.success), 'status': int(result.status), 'message': result.message,
        'seconds': seconds, 'nfev': int(result.nfev), 'njev': int(result.njev),
        'weighted_loss': float(residual @ residual),
        'point_rmse_px': float(np.sqrt(np.mean(np.sum(point_raw**2, axis=1)))),
        'line_rmse_px': float(np.sqrt(np.mean(np.sum(line_raw**2, axis=1)))) if len(line_raw) else None,
        'prior_norm': float(np.linalg.norm(residual[npoint+nline:])),
        'depth_valid': bool(np.isfinite(depths).all() and np.all(depths > 0)),
        'minimum_depth': float(np.min(depths)),
        'bound_hits': {key: ('lower' if x[i] - bounds[0][i] < 1e-4 else
                            'upper' if bounds[1][i] - x[i] < 1e-4 else None)
                       for i, key in enumerate(ids)},
        'fov_degrees': {key: math.degrees(2*math.atan(calibrations[key].intrinsics.image_width/(2*focal[i])))
                        for i, key in enumerate(ids)},
        'focal_px': {key: float(focal[i]) for i, key in enumerate(ids)},
        'world_rotation': (world_from_internal @ problem['anchor_r']).tolist(),
        'cameras': {key: {'rotation_w2c': (rotations[i] @ world_from_internal.T).tolist(),
                          'center': (world_from_internal @ (centers[i]*problem['baseline']) + problem['anchor_c']).tolist()}
                    for i, key in enumerate(ids)},
        'points': {key: (world_from_internal @ (points[i]*problem['baseline']) + problem['anchor_c']).tolist()
                   for i, key in enumerate(problem['point_ids'])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('startup', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bounds', choices=('span', 'broad'), default='span')
    parser.add_argument('--method', choices=('trf', 'projected'), default='trf')
    parser.add_argument('--seconds', type=float, default=45.)
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if not 0 < args.seconds <= 45:
        parser.error('seconds must be in (0, 45]')
    problem, calibrations = load_problem(args.inputs, args.startup, args.source_sha256)
    x0 = problem['x'].copy()
    ncam = problem['ncam']
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lo[:ncam], hi[:ncam] = problem['lower'], problem['upper']
    if args.bounds == 'broad':
        for i, key in enumerate(problem['ids']):
            intr = calibrations[key].intrinsics
            lo[i] = math.log(intr.image_width/(2*math.tan(math.radians(130/2)))/intr.fx)
            hi[i] = math.log(intr.image_width/(2*math.tan(math.radians(5/2)))/intr.fx)
    if problem['scale_columns']:
        lo[ncam+5], hi[ncam+5] = -10., 10.
    x0 = np.maximum(lo+1e-12, np.minimum(hi-1e-12, x0))
    cache = {}
    started = time.monotonic()
    class Deadline(Exception):
        pass
    def evaluate(x, jacobian):
        if time.monotonic() - started > args.seconds:
            raise Deadline
        key = x.tobytes()
        if cache.get('key') != key or (jacobian and cache.get('jac') is None):
            r, j, _ = problem['residual_and_jacobian'](x, jacobian=jacobian)
            cache.update(key=key, residual=r, jac=j)
        return cache['jac' if jacobian else 'residual']
    class Endpoint:
        pass
    try:
        if args.method == 'trf':
            result = least_squares(lambda x: evaluate(x, False), x0,
                                   jac=lambda x: evaluate(x, True), bounds=(lo, hi),
                                   method='trf', x_scale='jac', max_nfev=200,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
        else:
            from match_perspective.core.focal_optimizer import bounded_lm_step
            x = x0.copy()
            damping = 1e-3
            nfev = njev = 0
            result = Endpoint()
            result.success = False
            result.status = 0
            result.message = 'iteration limit'
            for iteration in range(100):
                residual = evaluate(x, False)
                jac = evaluate(x, True)
                nfev += 1
                njev += 1
                cost = float(residual @ residual)
                accepted = False
                for _ in range(12):
                    step = bounded_lm_step(
                        jac, residual, x, lo, hi, damping,
                        cancel_check=lambda: time.monotonic() - started > args.seconds)
                    trial_x = np.maximum(lo, np.minimum(hi, x + step))
                    trial_residual = evaluate(trial_x, False)
                    nfev += 1
                    trial_cost = float(trial_residual @ trial_residual)
                    if np.isfinite(trial_cost) and trial_cost < cost:
                        x = trial_x
                        accepted = True
                        damping = max(damping/3, 1e-9)
                        if (cost-trial_cost)/max(cost, 1) < 1e-9:
                            result.success = True
                            result.status = 2
                            result.message = 'relative gain below 1e-9'
                        break
                    damping *= 10
                if result.success or not accepted:
                    if not accepted:
                        result.message = 'no accepted step'
                    break
            result.x = x
            result.nfev = nfev
            result.njev = njev
        output = report(problem, calibrations, result, (lo, hi), time.monotonic()-started)
    except (Deadline, InterruptedError):
        # SciPy has no general wall-time callback in older versions. Preserve
        # the last evaluated endpoint if deadline interrupts an iteration.
        result = Endpoint()
        result.x = np.frombuffer(cache['key'], dtype=x0.dtype).copy()
        result.success = False
        result.status = 0
        result.message = f'{args.seconds:g}-second diagnostic deadline'
        result.nfev = result.njev = 0
        output = report(problem, calibrations, result, (lo, hi), time.monotonic()-started)
    initial_residual = problem['residual_and_jacobian'](x0, jacobian=False)[0]
    output.update(method='scipy_trf' if args.method == 'trf' else 'numpy_projected_lm', bounds=args.bounds,
                  source_sha256=problem['source_sha256'],
                  core_source_sha256=problem['core_source_sha256'],
                  inputs_sha256=problem['inputs_sha256'],
                  startup_sha256=problem['startup_sha256'],
                  initial_weighted_loss=float(initial_residual @ initial_residual))
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ('cameras', 'points')}, indent=2))


if __name__ == '__main__':
    main()
