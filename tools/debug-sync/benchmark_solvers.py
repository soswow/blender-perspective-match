#!/usr/bin/env python3
"""Bounded, read-only Blender Sync/lens benchmarks with exact inputs and profiles.

Run in factory-startup Blender. Every numerical Sync and focal bundle is ledgered;
use an outer process timeout as well. Captures may contain private project geometry.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import fields, MISSING
from contextlib import ExitStack
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import pstats
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.synthetic_sync.budget import ExperimentBudget



def compare_results(before, after):
    """Compare complete JSON results, retaining exact and numerical differences."""
    differences = []
    largest = 0.0
    numeric_count = 0

    def visit(left, right, path):
        nonlocal largest, numeric_count
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                differences.append(path + ': keys differ')
            for key in left:
                if key not in right:
                    continue
                visit(left[key], right[key], path + '/' + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(path + ': lengths differ')
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + '/' + str(index))
        elif type(left) in (int, float) and type(right) in (int, float):
            if left == right or (math.isnan(left) and math.isnan(right)):
                return
            numeric_count += 1
            largest = max(largest, abs(left-right))
            if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
                differences.append(path + ': numbers differ')
        elif type(left) is not type(right) or left != right:
            differences.append(path + ': values differ')

    visit(before, after, '')
    return dict(exact=(not differences and numeric_count == 0),
                close=not differences, changed_numbers=numeric_count,
                max_absolute_difference=largest, differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_options = parser.add_mutually_exclusive_group(required=True)
    source_options.add_argument('--blend', type=Path)
    source_options.add_argument('--request', type=Path, help='Captured Sync inputs, or lens inputs with --startup for a bundle-only replay')
    parser.add_argument('--startup', type=Path, help='Saved Sync result for --operation joint')
    parser.add_argument('--operation', choices=('sync', 'lens', 'joint'), required=True)
    parser.add_argument('--out', type=Path, required=True, help='New private artifact directory')
    parser.add_argument('--compare', type=Path, help='Prior artifact directory; compare exact inputs and complete results')
    parser.add_argument('--profile', action='store_true', help='cProfile adds overhead; measure clean timings separately')
    parser.add_argument('--serial-pairs', action='store_true', help='Diagnostic control: execute the same pair jobs in input order without a thread pool')
    parser.add_argument('--profile-stage', choices=('all', 'bundle'), default='all')
    parser.add_argument('--max-calls', type=int, default=2, help='Total inner Sync plus bundle calls')
    parser.add_argument('--seconds', type=float, default=240, help='Cumulative numerical budget')
    parser.add_argument('--per-call-seconds', type=float, default=180)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.operation == 'joint' and (not args.request or not args.startup):
        parser.error('joint requires --request lens-inputs.json and --startup sync-result.json')
    if args.operation == 'lens' and not args.blend:
        parser.error('lens requires --blend')
    if args.startup and args.operation != 'joint':
        parser.error('--startup requires --operation joint')
    args.out.mkdir(parents=True, exist_ok=False)
    source = (args.blend or args.request).expanduser().resolve(strict=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    import bpy
    from probe_cameras import _load_extension
    _load_extension()
    from match_perspective import scene
    from match_perspective.core import sync, lens_refine, focal_bundle
    from match_perspective.core.sync.request import json_values
    from tools.synthetic_sync.solver import environment

    def save(name, value):
        (args.out / name).write_text(json.dumps(json_values(value), indent=2) + '\n')

    sources = {}
    for folder in ('core', 'scene', 'properties'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            relative = path.relative_to(ROOT)
            sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            target = args.out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in (Path(__file__), Path(__file__).with_name('probe_focal_startup.py'),
                 ROOT / 'tools/synthetic_sync/budget.py'):
        target = args.out / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = environment()
    try:
        import cv2
        runtime['opencv'] = cv2.__version__
    except ImportError:
        runtime['opencv'] = None
    runtime['threads'] = {key: os.environ.get(key) for key in (
        'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    runtime['blender'] = bpy.app.version_string
    runtime['gil_enabled'] = getattr(sys, '_is_gil_enabled', lambda: True)()
    metadata = dict(source_sha256=sources, environment=runtime, operation=args.operation,
                    profile=args.profile, profile_stage=args.profile_stage,
                    serial_pairs=args.serial_pairs, input_file_sha256=source_hash,
                    startup_sha256=hashlib.sha256(args.startup.read_bytes()).hexdigest() if args.startup else None)
    save('metadata.json', metadata)
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=str(source))
    started = time.perf_counter()
    if args.blend:
        prep = (scene.prepare_diagnose_sync(bpy.context) if args.operation == 'sync'
                else scene.prepare_lens_refine(bpy.context))
        kwargs = prep.solver_kwargs()
    elif args.operation == 'sync':
        from match_perspective.core.sync.request import SyncSolveRequest, request_fingerprint
        record = json.loads(source.read_text())
        if 'format' not in record:
            defaults = {f.name: f.default for f in fields(SyncSolveRequest) if f.default is not MISSING}
            values = {**defaults, **{f.name: record[f.name] for f in fields(SyncSolveRequest) if f.name in record}}
            record = dict(format='perspective-match-sync-request', version=4, inputs=values,
                          sha256=request_fingerprint(values))
        prep = SyncSolveRequest.from_record(record)
        kwargs = prep.solver_kwargs()
    else:
        kwargs = json.loads(source.read_text())
        startup = json.loads(args.startup.read_text())
        save('startup.json', startup)
    preparation_seconds = time.perf_counter() - started
    captured_inputs = json_values(kwargs)
    save('inputs.json', captured_inputs)
    calls = []
    profiler = cProfile.Profile() if args.profile else None
    with ExperimentBudget(args.out / 'ledger.jsonl', metadata=metadata,
                          max_calls=args.max_calls, wall_seconds=args.seconds,
                          per_call_seconds=args.per_call_seconds) as budget:
        def instrument(function, label):
            def run(*values, **options):
                index = len(calls) + 1
                bound = inspect.signature(function).bind(*values, **options)
                evidence = {k: v for k, v in bound.arguments.items() if not callable(v)}
                input_file = f'{index:03d}-{label}-inputs.json'
                save(input_file, evidence)
                row = dict(index=index, label=label, input_file=input_file)
                calls.append(row)
                print(f'Starting {index}: {label}', flush=True)
                endpoints = []
                if label == 'bundle':
                    original_callback = options.get('diagnostic_callback')
                    def capture_endpoint(value):
                        endpoints.append(value)
                        if original_callback:
                            original_callback(value)
                    options['diagnostic_callback'] = capture_endpoint
                with budget.attempt(label, json_values(evidence)) as attempt:
                    call_start = time.perf_counter()
                    if profiler and args.profile_stage == 'bundle' and label == 'bundle':
                        profiler.enable()
                    try:
                        result = function(*values, **options)
                    finally:
                        if profiler and args.profile_stage == 'bundle' and label == 'bundle':
                            profiler.disable()
                    row['seconds'] = time.perf_counter() - call_start
                    save(f'{index:03d}-{label}-result.json', result)
                    if endpoints:
                        save(f'{index:03d}-{label}-endpoint.json', endpoints)
                    attempt.complete(row)
                print(f'Finished {label}: {row["seconds"]:.3f}s', flush=True)
                return result
            return run

        started = time.perf_counter()
        try:
            with ExitStack() as contexts, patch.object(sync, 'solve_landmark_sync', instrument(sync.solve_landmark_sync, 'sync')), \
                 patch.object(lens_refine, 'fit_independent_focals', instrument(lens_refine.fit_independent_focals, 'bundle')):
                if args.serial_pairs:
                    from match_perspective.core.sync import pose
                    contexts.enter_context(patch.object(pose, '_map_pair_jobs', lambda fn, items: list(map(fn, items))))
                if args.operation == 'joint':
                    contexts.enter_context(patch.object(focal_bundle, 'fit_independent_focals',
                        instrument(focal_bundle.fit_independent_focals, 'bundle')))
                if profiler and args.profile_stage == 'all':
                    profiler.enable()
                if args.operation == 'sync':
                    result = sync.solve_landmark_sync(**kwargs, use_pose_cache=False)
                elif args.operation == 'lens':
                    result = scene.run_lens_refine(prep)
                else:
                    from probe_focal_startup import joint_report
                    report = joint_report(kwargs, startup)
                    save('joint-report.json', report)
                    result = report['outcome']
                if profiler:
                    profiler.disable()
                numerical_seconds = time.perf_counter() - started
                save('result.json', result)
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(args.out / 'profile.pstats'))
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(70)
                (args.out / 'profile.txt').write_text(stream.getvalue())
            save('calls.json', calls)
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                raise RuntimeError('Source file changed during benchmark')
    save('summary.json', dict(preparation_seconds=preparation_seconds,
         numerical_seconds=numerical_seconds, calls=calls, source_file_unchanged=True,
         applied=False, profile=args.profile))
    if args.compare:
        previous_inputs = json.loads((args.compare / 'inputs.json').read_text())
        input_comparison = compare_results(previous_inputs, captured_inputs)
        if not input_comparison['exact']:
            raise ValueError('Comparison requires identical prepared inputs')
        previous = json.loads((args.compare / 'result.json').read_text())
        save('comparison.json', compare_results(previous, json_values(result)))
    print(f'Total numerical time: {numerical_seconds:.3f}s', flush=True)


if __name__ == '__main__':
    main()
