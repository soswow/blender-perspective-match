"""Generated line-FOV Blender preparation, bundle, apply and stale-input checks.

One bundle, with an explicit oracle point/pose start to isolate native integration.
Optional --fresh also runs one real point-based registration.
No user files are opened and no blend is saved. Output folder must be new.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync import focal_line_constraints, focal_lines
from tools.synthetic_sync.blender_case import blender_pixels, register_extension
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.verify_focal_constraints_blender import build


def check(out, fresh=False, partial=False, relations=False):
    from match_perspective import properties, scene
    from match_perspective.core import lens_refine, sync

    case = focal_line_constraints.generate() if relations else focal_lines.generate()
    (out / 'case.json').write_text(json.dumps(case, indent=2))
    build(case, out)
    space = properties.workspace(bpy.context)
    if not relations:
        space.mirror_origin = 'LANDMARK'
        space.mirror_landmark = case['request']['mirror_landmark_id']
    prep = scene.collect_lens_refine_inputs(bpy.context)
    assert len(prep.line_observations) == len(case['request']['line_observations'])
    if relations:
        assert set(map(tuple, case['request']['plane_groups'])) == set(map(tuple, prep.plane_groups))
        assert set(map(tuple, case['request']['parallel_pairs'])) == set(map(tuple, prep.parallel_pairs))
    else:
        assert tuple(case['request']['mirror_pairs'][-1]) in [tuple(p) for p in prep.mirror_pairs]
    true_cameras = {c['id']: c for c in case['truth']['cameras']}
    similarities = {}
    for item in prep.lens_inputs:
        cal = item.base_calibration
        true = true_cameras[item.match_id]
        rotation = np.asarray(true['rotation']).T @ cal.rotation_w2c
        similarities[item.match_id] = sync.SimilarityTransform(
            1., rotation, np.asarray(true['center']) - rotation @ cal.camera_center)
    similarities[prep.anchor_id] = sync.SimilarityTransform()
    if partial:
        del similarities[case['request']['cameras'][-1]['id']]
    initial = sync.SyncSolveResult(
        similarities=similarities,
        landmarks={k: np.asarray(p) for k, p in case['truth']['points'].items()},
        mean_reprojection_px=0., per_match_rmse_px={}, per_landmark_rmse_px={},
        message='Oracle point/pose start; line geometry absent', success=True)
    # The point initializer is isolated; line reconstruction and focal fitting
    # use real production code and the prepared RNA stroke coordinates.
    start_options = dict(wraps=lens_refine._run_sync) if fresh else dict(return_value=initial)
    with patch.object(lens_refine, '_run_sync', **start_options) as startup:
        result = scene.run_lens_refine(prep)
    assert startup.call_count == 1
    assert result.improved, result.refusal_reason
    scene.apply_lens_refine_result(bpy.context, result, prep)
    landmarks = {p.item_id: p for p in space.landmarks}
    for key, ends in result.sync_result.line_segments.items():
        landmark = landmarks[key]
        assert landmark.has_line_segment and landmark.has_position
        np.testing.assert_allclose(landmark.position, ends[0], atol=1e-5)
        np.testing.assert_allclose(landmark.position_b, ends[1], atol=1e-5)
        helper = scene.landmark_viewport_object(landmark)
        assert helper is not None and helper.type == 'MESH'
    anchor = np.asarray(true_cameras[prep.anchor_id]['center'])
    points = case['truth']['points']
    truth_points = np.asarray(list(points.values())) - anchor
    fitted_points = np.asarray([result.sync_result.landmarks[k] for k in points]) - anchor
    scale = np.sum(truth_points * fitted_points) / np.sum(truth_points**2)
    withheld = np.asarray(list(case['truth']['holdouts'].values()))
    errors = []
    for camera in case['truth']['cameras']:
        root = prep.root_by_name[camera['id']]
        pixels = blender_pixels(root.pm_session.camera_object, camera,
                                anchor + scale * (withheld - anchor))
        expected = project(withheld, camera)[0]
        errors.extend(np.linalg.norm(pixels - expected, axis=1))
    assert max(errors) < .02, max(errors)
    # Edits to a stroke must invalidate the prior job, including line-only edits.
    current_prep = scene.collect_lens_refine_inputs(bpy.context)
    line = next(p for p in space.landmarks if p.kind == 'LINE')
    pick = line.observations[0]
    # Test the actual line endpoint RNA field below, not the point coordinate.
    endpoint = pick.x2, pick.y2
    pick.x2 = endpoint[0] + 1.
    assert scene.collect_lens_refine_inputs(bpy.context).source_request_sha256 != current_prep.source_request_sha256
    with patch.object(scene, '_apply_sync_solve_result', side_effect=AssertionError('Stale result applied')):
        try:
            scene.apply_lens_refine_result(bpy.context, result, current_prep)
        except scene.StaleSyncResult:
            pass
        else:
            raise AssertionError('Changed line stroke did not invalidate the job')
    pick.x2, pick.y2 = endpoint
    if relations:
        for relation, edit in (
            ('plane', lambda item: setattr(item, 'plane_group', '3')),
            ('parallel', lambda item: setattr(item, 'parallel_to', 'NONE')),
        ):
            item = landmarks['free_edge_a']
            old = item.plane_group if relation == 'plane' else item.parallel_to
            edit(item)
            assert scene.collect_lens_refine_inputs(bpy.context).source_request_sha256 != current_prep.source_request_sha256
            with patch.object(scene, '_apply_sync_solve_result', side_effect=AssertionError('Stale result applied')):
                try:
                    scene.apply_lens_refine_result(bpy.context, result, current_prep)
                except scene.StaleSyncResult:
                    pass
                else:
                    raise AssertionError(f'Changed {relation} relation did not invalidate the job')
            if relation == 'plane':
                item.plane_group = old
            else:
                item.parallel_to = old
    report = dict(passed=True, bundles=1, fresh_registration=fresh, partial_startup=partial,
                  withheld_max_px=float(max(errors)), lines=len(result.sync_result.line_segments),
                  relations=relations)
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print('Line FOV Blender PASS:', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--partial-startup', action='store_true')
    parser.add_argument('--relations', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.fresh and args.partial_startup:
        parser.error('--partial-startup uses the prepared oracle start, not --fresh')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    register_extension()
    check(out, args.fresh, args.partial_startup, args.relations)


if __name__ == '__main__':
    main()
