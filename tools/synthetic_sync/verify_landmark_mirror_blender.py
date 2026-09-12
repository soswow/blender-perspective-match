"""Generated live mirror reference through Blender RNA, solve, apply and reopen.

Reserve one Sync and (in focal mode) one bundle per fresh run. Preparation and
reopening never solve. Only generated files are saved, in a new output folder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync import focal_constraints
from tools.synthetic_sync.blender_case import assert_equivalent, register_extension
from tools.synthetic_sync.solver import environment, fingerprint, result_record
from tools.synthetic_sync.verify_focal_constraints_blender import (
    build, check_accuracy, native_check, state as constraint_state, write_json,
)


def make_case(mode):
    from tools.synthetic_sync.focal_live_reference import generate

    case = generate()
    # Change coordinates so the same geometry uses the world YZ mirror plane.
    # This tests the primary workflow without an orientation object at all.
    normal = np.asarray(case['request']['mirror_plane'][1], float)
    tangent = np.cross(normal, (0.0, 0.0, 1.0))
    tangent /= np.linalg.norm(tangent)
    turn = np.vstack((normal, tangent, np.cross(normal, tangent)))
    for collection in ('points', 'holdouts'):
        case['truth'][collection] = {
            key: (turn @ point).tolist() for key, point in case['truth'][collection].items()}
    for cameras in (case['truth']['cameras'], case['request']['cameras']):
        for camera in cameras:
            camera['center'] = (turn @ camera['center']).tolist()
            camera['rotation'] = (np.asarray(camera['rotation']) @ turn.T).tolist()
    for plane in (case['truth']['mirror_plane'], case['request']['mirror_plane']):
        plane[0] = (turn @ plane[0]).tolist()
        plane[1] = [1.0, 0.0, 0.0]
    if mode == 'sync':
        for camera, true in zip(case['request']['cameras'], case['truth']['cameras']):
            camera['fx'], camera['fy'] = true['fx'], true['fy']
    case['name'] = f'landmark-mirror-{mode}'
    return case


def state():
    from match_perspective import properties
    workspace = properties.workspace(bpy.context)
    return dict(base=constraint_state(), origin=workspace.mirror_origin,
                reference=workspace.mirror_landmark_id, plane=workspace.mirror_plane)


def assess(case, record):
    effective = deepcopy(case)
    effective['request']['mirror_plane'][0] = record['landmarks'][case['request']['mirror_landmark_id']]
    return focal_constraints.assess(effective, record, validated=True)


def prepare_checks(case):
    from match_perspective import properties, scene
    workspace = properties.workspace(bpy.context)
    reference = next(p for p in workspace.landmarks if p.item_id == workspace.mirror_landmark_id)
    assert workspace.mirror_landmark == reference.item_id
    before = scene.collect_sync_request(bpy.context).to_record()
    assert before['inputs']['mirror_landmark_id'] == reference.item_id
    np.testing.assert_allclose(before['inputs']['mirror_plane'], [[0, 0, 0], [1, 0, 0]])
    # A cached reconstruction must not become a numerical input or a fixed pin.
    saved = tuple(reference.position), reference.has_position
    reference.position, reference.has_position = (800.0, -100.0, 300.0), True
    assert before == scene.collect_sync_request(bpy.context).to_record()
    reference.position, reference.has_position = saved
    old_name = reference.name
    reference.name = 'Renamed plane reference'
    assert workspace.mirror_landmark == reference.item_id
    reference.name = old_name
    # Object translation is irrelevant in live mode; orientation is not.
    obj = bpy.data.objects.new('Optional mirror orientation', None)
    bpy.context.scene.collection.objects.link(obj)
    workspace.mirror_object = obj
    obj.location = (50.0, 60.0, -70.0)
    bpy.context.view_layer.update()
    assert before == scene.collect_sync_request(bpy.context).to_record()
    obj.rotation_euler.z = 0.1
    bpy.context.view_layer.update()
    assert before['sha256'] != scene.collect_sync_request(bpy.context).to_record()['sha256']
    workspace.mirror_object = None
    bpy.data.objects.remove(obj, do_unlink=True)
    # A deleted/nonpoint selection must remain identifiable, not fall back.
    original_id = workspace.mirror_landmark_id
    workspace.mirror_landmark_id = 'missing-reference'
    assert workspace.mirror_landmark == 'missing-reference'
    try:
        scene.collect_sync_request(bpy.context)
    except ValueError as error:
        assert 'point landmark' in str(error)
    else:
        raise AssertionError('Missing reference silently accepted')
    workspace.mirror_landmark_id = original_id
    assert before == scene.collect_sync_request(bpy.context).to_record()
    return before


def fresh(case, out, mode, prepare_only):
    from match_perspective import properties, scene
    from match_perspective.core import lens_refine, sync
    from match_perspective.core.sync.request import json_values

    build(case, out)
    workspace = properties.workspace(bpy.context)
    orientation = workspace.mirror_object
    workspace.mirror_object = None
    if orientation:
        bpy.data.objects.remove(orientation, do_unlink=True)
    workspace.mirror_origin = 'LANDMARK'
    workspace.mirror_plane = 'YZ'
    workspace.mirror_landmark_id = case['request']['mirror_landmark_id']
    write_json(out / 'request.json', prepare_checks(case))
    prep = scene.prepare_lens_refine(bpy.context) if mode == 'focal' else scene.prepare_diagnose_sync(bpy.context)
    assert prep.mirror_landmark_id == workspace.mirror_landmark_id
    write_json(out / 'prepared.json', json_values(prep.solver_kwargs()))
    if prepare_only:
        return dict(passed=True, numerical_solves=0)
    before = state()
    calls = {'sync': 0, 'bundle': 0}
    original_sync, original_bundle = sync.solve_landmark_sync, lens_refine.fit_independent_focals

    def counted_sync(*args, **kwargs):
        calls['sync'] += 1
        assert calls['sync'] == 1
        write_json(out / 'sync-input.json', json_values(kwargs))
        return original_sync(*args, **kwargs)

    def counted_bundle(*args, **kwargs):
        calls['bundle'] += 1
        assert calls['bundle'] == 1
        return original_bundle(*args, **kwargs)

    # Reserve before work; preserve source/input/decision even on failure.
    write_json(out / 'reservation.json', dict(sync=1, bundle=int(mode == 'focal')))
    with patch.object(sync, 'solve_landmark_sync', side_effect=counted_sync), \
         patch.object(lens_refine, 'fit_independent_focals', side_effect=counted_bundle):
        result = (scene.run_lens_refine(prep) if mode == 'focal' else
                  sync.solve_landmark_sync(**prep.solver_kwargs()))
    assert_equivalent(before, state(), 'numerical isolation')
    joint = result.sync_result if mode == 'focal' else result
    write_json(out / 'decision.json', dict(calls=calls, message=result.message,
                                          success=joint.success,
                                          refusal=getattr(result, 'refusal_reason', '')))
    if mode == 'focal':
        assert result.improved, result.refusal_reason
    assert joint.success, joint.message
    cameras = deepcopy(case['request']['cameras'])
    if mode == 'focal':
        for camera in cameras:
            k = result.calibrations[camera['id']].intrinsics
            camera.update(fx=k.fx, fy=k.fy)
    record = result_record(joint, cameras)
    record['joint_result'] = json_values(joint)
    write_json(out / 'record.json', record)
    # A new reference while a job is in flight invalidates its result.
    reference_id = workspace.mirror_landmark_id
    workspace.mirror_landmark_id = 'base_00'
    edited = state()
    try:
        if mode == 'focal':
            scene.apply_lens_refine_result(bpy.context, result, prep)
        else:
            scene.apply_diagnose_sync_result(bpy.context, prep, joint)
    except scene.StaleSyncResult:
        pass
    else:
        raise AssertionError('Changed mirror reference accepted a stale job')
    assert_equivalent(edited, state(), 'stale reference refusal')
    workspace.mirror_landmark_id = reference_id
    if mode == 'focal':
        with patch.object(scene, 'solve_and_apply_sync', side_effect=AssertionError('Unexpected re-solve')):
            scene.apply_lens_refine_result(bpy.context, result, prep)
    else:
        scene.apply_diagnose_sync_result(bpy.context, prep, joint)
        scene._apply_sync_solve_result(bpy.context, joint, prep.matches)
    assert workspace.mirror_object is None
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'solved.blend'), check_existing=False)
    write_json(out / 'solved-state.json', state())
    return check_saved(case, out, record)


def check_saved(case, out, record):
    assessment = assess(case, record)
    check_accuracy(case, assessment)
    assessment['native_blender'] = native_check(case, record, assessment)
    assessment['passed'] = True
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sync', 'focal'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--reopen', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = args.out.resolve()
    register_extension()
    if args.reopen:
        case = json.loads((out / 'case.json').read_text())
        saved = out / 'solved.blend'
        digest = hashlib.sha256(saved.read_bytes()).hexdigest()
        bpy.ops.wm.open_mainfile(filepath=str(saved), load_ui=False)
        assert_equivalent(json.loads((out / 'solved-state.json').read_text()), state(), 'reopened state')
        result = check_saved(case, out, json.loads((out / 'record.json').read_text()))
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    else:
        out.mkdir(parents=True, exist_ok=False)
        case = make_case(args.mode)
        write_json(out / 'case.json', case)
        write_json(out / 'environment.json', environment(ROOT))
        with tarfile.open(out / 'source.tar.xz', 'w:xz') as archive:
            for folder in ('core', 'scene', 'properties', 'ui', 'tools/synthetic_sync'):
                for path in sorted((ROOT / folder).glob('**/*.py')):
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
        result = fresh(case, out, args.mode, args.prepare_only)
    write_json(out / ('reopened-assessment.json' if args.reopen else 'assessment.json'), result)
    print(f'Landmark mirror Blender PASS: {args.mode}', flush=True)


if __name__ == '__main__':
    main()
