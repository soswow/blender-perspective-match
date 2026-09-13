#!/usr/bin/env python3
"""Read-only pose and image-overlap diagnostics for saved focal input/startup JSON.

The saved 3D startup cloud is conditional on its lens and pose estimates. A
large reprojection error against that cloud does not independently identify a
bad image pick. No Sync solve or Blender process is started.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


def joint_report(inputs, startup, *, max_iterations=None):
    """Run one production bundle from saved startup; never rerun Sync or apply."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.synthetic_sync.solver import load_core
    core, sync = load_core()
    from match_perspective.core import focal_bundle
    from match_perspective.core.sync.request import json_values
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
              'mirror_pairs', 'mirror_plane', 'mirror_slack', 'mirror_landmark_id', 'parallel_pairs')
    started = time.monotonic()
    endpoints = []
    timeout_seconds = focal_bundle.MAX_SECONDS + 15.
    iterations = focal_bundle.MAX_ITERATIONS if max_iterations is None else max_iterations
    with patch.object(focal_bundle, 'MAX_ITERATIONS', iterations):
        result = focal_bundle.fit_independent_focals(
            calibrations, [sync.SyncObservation(**p) for p in inputs['observations']], initial,
            line_observations=[sync.SyncLineObservation(**p) for p in inputs['line_observations']],
            **{k: inputs[k] for k in fields if k in inputs},
            cancel_check=lambda: time.monotonic() - started > timeout_seconds,
            diagnostic_callback=endpoints.append)
    return dict(seconds=time.monotonic() - started, outcome=json_values(result),
                focal_span=inputs.get('fx_span'), max_iterations=iterations,
                timeout_seconds=timeout_seconds, bundle_max_seconds=focal_bundle.MAX_SECONDS,
                endpoint=endpoints[-1] if endpoints else None,
                full_sync_calls=0, bundle_calls=1, applied=False)


def _points(inputs):
    groups = defaultdict(dict)
    for row in inputs['observations']:
        groups[row['match_id']][row['landmark_id']] = np.array((row['u'], row['v']), dtype=np.float64)
    return groups


def _camera_matrix(match, scale=1.0):
    k = match['intrinsics']
    return np.array(((k['fx'] * scale, 0, k['cx']), (0, k['fy'] * scale, k['cy']), (0, 0, 1)), dtype=np.float64)


def _pnp(world, image, k):
    import cv2
    if len(world) < 4:
        return None
    ok, r, t, inliers = cv2.solvePnPRansac(
        world, image, k, None, flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=300, reprojectionError=12.0, confidence=0.99,
    )
    if not ok:
        ok, r, t = cv2.solvePnP(world, image, k, None, flags=cv2.SOLVEPNP_EPNP)
        inliers = None
    if not ok:
        return None
    if inliers is not None and len(inliers) >= 4:
        cv2.solvePnPRefineLM(world[inliers[:, 0]], image[inliers[:, 0]], k, None, r, t)
    projected = cv2.projectPoints(world, r, t, k, None)[0].reshape(-1, 2)
    error = np.linalg.norm(projected - image.reshape(-1, 2), axis=1)
    depth = (cv2.Rodrigues(r)[0] @ world.T + t).T[:, 2]
    return {
        'rotation_vector': [float(x) for x in r[:, 0]],
        'translation_vector': [float(x) for x in t[:, 0]],
        'inliers_12px': int(np.count_nonzero(error < 12)),
        'median_px': round(float(np.median(error)), 2),
        'rmse_px': round(float(np.sqrt(np.mean(error ** 2))), 2),
        'positive_depth': int(np.count_nonzero(depth > 0)),
        'errors_px': [round(float(x), 2) for x in error],
    }


def load(inputs_path, startup_path):
    """Load saved sidecars without preparing or running a Sync solve."""
    return json.loads(Path(inputs_path).read_text()), json.loads(Path(startup_path).read_text())


def similarity_from_pnp(match, trial, scale=1.0):
    """Convert a shared-world PnP camera into this match's private-world similarity."""
    import cv2
    rotation_fit = cv2.Rodrigues(np.asarray(trial['rotation_vector'], dtype=np.float64))[0]
    base = match['base_calibration']
    rotation_base = np.asarray(base['rotation_w2c'], dtype=np.float64)
    center_base = np.asarray(base['camera_center'], dtype=np.float64)
    rotation = rotation_fit.T @ rotation_base
    center_shared = -rotation_fit.T @ np.asarray(trial['translation_vector'], dtype=np.float64)
    translation = center_shared - scale * rotation @ center_base
    return {'scale': float(scale), 'rotation': rotation.tolist(), 'translation': translation.tolist()}


def _pair_consistency(a, b):
    import cv2
    ids = sorted(set(a) & set(b))
    if len(ids) < 8:
        return {'shared': len(ids)}
    p = np.array([a[x] for x in ids])
    q = np.array([b[x] for x in ids])
    f, mask = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, 3.0, 0.99, 2000)
    if f is None or f.shape != (3, 3) or mask is None:
        return {'shared': len(ids), 'fundamental_fit': False}
    left = np.column_stack((p, np.ones(len(p))))
    right = np.column_stack((q, np.ones(len(q))))
    fl = (f @ left.T).T
    ftr = (f.T @ right.T).T
    numerator = np.abs(np.sum(right * fl, axis=1))
    distance = 0.5 * numerator * (
        1 / np.maximum(np.linalg.norm(fl[:, :2], axis=1), 1e-9)
        + 1 / np.maximum(np.linalg.norm(ftr[:, :2], axis=1), 1e-9)
    )
    return {
        'shared': len(ids), 'ransac_inliers_3px': int(mask.sum()),
        'median_symmetric_epipolar_px': round(float(np.median(distance)), 2),
        'outlier_ids': [id for id, keep in zip(ids, mask[:, 0]) if not keep],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, help='Saved .inputs.json sidecar')
    parser.add_argument('startup', type=Path, help='Saved .startup.json sidecar')
    parser.add_argument('--match-id', help='Camera to probe; defaults to a sole omitted camera')
    parser.add_argument('--out', type=Path, help='Write JSON report here')
    parser.add_argument('--joint', action='store_true', help='Run one joint fit from saved startup instead of diagnostic scans')
    parser.add_argument('--span-percent', type=float, help='Override focal range for this read-only joint trial')
    parser.add_argument('--max-iterations', type=int,
                        help='Diagnostic joint-fit iteration cap (1–500); retains the time limit')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    inputs, startup = load(args.inputs, args.startup)
    if args.max_iterations is not None and (not args.joint or not 1 <= args.max_iterations <= 500):
        parser.error('--max-iterations requires --joint and a value from 1 to 500')
    if args.span_percent is not None:
        if not args.joint or not 1 <= args.span_percent <= 80:
            parser.error('--span-percent requires --joint and a value from 1 to 80')
        inputs['fx_span'] = args.span_percent / 100.
    if args.joint:
        output = json.dumps(joint_report(inputs, startup, max_iterations=args.max_iterations), indent=2)
        if args.out:
            args.out.write_text(output + '\n')
        else:
            print(output)
        return
    matches = {row['match_id']: row for row in inputs['matches']}
    missing = sorted(set(matches) - set(startup['similarities']))
    match_id = args.match_id or (missing[0] if len(missing) == 1 else None)
    if not match_id or match_id not in matches:
        parser.error('Specify --match-id when there is not exactly one omitted camera')
    if any(float(match.get('division_lambda', 0.0)) != 0.0 or match.get('brown_conrady')
           for match in matches.values()):
        parser.error('This probe currently requires undistorted input cameras')
    observations = _points(inputs)
    ids = sorted(set(observations[match_id]) & set(startup['landmarks']))
    world = np.array([startup['landmarks'][x] for x in ids], dtype=np.float64)
    image = np.array([observations[match_id][x] for x in ids], dtype=np.float64)
    scales = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0)
    trials = []
    for scale in scales:
        fit = _pnp(world, image, _camera_matrix(matches[match_id], scale))
        if fit:
            trials.append({'focal_scale': scale, **fit,
                           'similarity_seed': similarity_from_pnp(matches[match_id], fit)})
    pairs = {
        other: _pair_consistency(observations[match_id], observations[other])
        for other in matches if other != match_id
    }
    report = {
        'match_id': match_id, 'omitted_camera_ids': missing,
        'picked_points': len(observations[match_id]), 'startup_3d_correspondences': len(ids),
        'startup_rmse_px': startup['mean_reprojection_px'],
        'point_ids_in_order': ids, 'focal_trials': trials,
        'raw_2d_pairs': pairs,
        'interpretation_limit': 'PnP uses the saved guessed-lens 3D cloud. Fundamental RANSAC can fit seven of eight points and is weak evidence with few shared picks. Neither certifies true FOV or proves a specific pick wrong.',
    }
    output = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(output + '\n')
    else:
        print(output)


if __name__ == '__main__':
    main()
