"""Lens-input report with optional bounded fit; never apply or save the blend.

blender --factory-startup --disable-autoexec -b --python this.py -- --blend scene.blend
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_cameras import _load_extension


def report():
    from match_perspective import properties, scene
    prep = scene.collect_lens_refine_inputs(bpy.context)
    space = properties.workspace(bpy.context)
    names = {p.item_id: p.name for p in space.landmarks}
    kinds = {p.item_id: p.kind for p in space.landmarks}
    views = defaultdict(set)
    for pick in prep.observations:
        views[pick.landmark_id].add(pick.match_id)
    relations = {p[0] for p in (prep.plane_groups or [])}
    relations.update(p for pair in (prep.mirror_pairs or []) for p in pair)
    counts = Counter(p.match_id for p in prep.observations)
    images = []
    for root in properties.iter_match_roots():
        settings = root.pm_session
        images.append(dict(match_id=root.name,
            source_path=bpy.path.abspath(settings.image_path) if settings.image_path else None,
            stored_size=[settings.image_width, settings.image_height],
            source_width=settings.source_image_width,
            principal_point=[settings.cx, settings.cy],
            undistorted=bool(settings.view_undistorted)))
    return dict(same_lens=prep.share_lens, point_focal=prep.estimate_focal_from_points,
        images=images,
        cameras=[dict(name=m.match_id, picks=counts[m.match_id], focal=m.intrinsics.fx,
                      vp_lines=sum(len(v) for v in m.line_bundles.values())) for m in prep.lens_inputs],
        points=len(views), known_points=len(prep.known_world),
        ground_picks=sum(p.on_ground for p in prep.observations),
        line_picks=len(prep.line_observations),
        plane_groups=[(names.get(p, p), axis, bucket) for p, axis, bucket in (prep.plane_groups or [])],
        mirror_pairs=[(names.get(a, a), names.get(b, b)) for a, b in (prep.mirror_pairs or [])],
        parallel_pairs=[(names.get(a, a), names.get(b, b)) for a, b in (prep.parallel_pairs or [])],
        line_landmarks=sorted({names.get(p.landmark_id, p.landmark_id)
                                           for p in prep.line_observations}),
        constrained_landmarks=[dict(id=p, name=names.get(p, p), kind=kinds.get(p), views=sorted(views[p]))
                            for p in sorted(relations)],
        insufficient_points=[dict(id=p, name=names.get(p, p), views=sorted(views[p]))
                             for p in sorted(set(views) | relations)
                             if kinds.get(p) == 'POINT' and len(views[p]) < 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', required=True)
    parser.add_argument('--out')
    parser.add_argument('--fit-seconds', type=float, default=0,
                        help='Optionally run one fit, with cooperative timeout; never apply it')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).expanduser().resolve()))
    if not 0 <= args.fit_seconds <= 600:
        parser.error('--fit-seconds must be between 0 and 600')
    result = report()

    def emit():
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.out:
            Path(args.out).write_text(text + '\n', encoding='utf-8')

    emit()
    if args.fit_seconds:
        from match_perspective import scene
        from match_perspective.core import lens_refine
        from match_perspective.core.sync.request import json_values
        prep = scene.collect_lens_refine_inputs(bpy.context)
        started = time.monotonic()
        result['fit'] = dict(status='started', budget_seconds=args.fit_seconds,
                             input_sha256=prep.source_request_sha256)
        emit()
        if args.out:
            Path(args.out + '.inputs.json').write_text(json.dumps(json_values(prep.solver_kwargs()), indent=2))
        original_sync = lens_refine._run_sync

        def capture_startup(*values, **options):
            initial = original_sync(*values, **options)
            result['fit']['registered_cameras'] = sorted(initial.similarities)
            result['fit']['startup_message'] = initial.message
            if args.out:
                Path(args.out + '.startup.json').write_text(json.dumps(json_values(initial), indent=2))
            emit()
            return initial

        with patch.object(lens_refine, '_run_sync', side_effect=capture_startup):
            fitted = scene.run_lens_refine(
                prep, cancel_check=lambda: time.monotonic() - started > args.fit_seconds)
        result['fit'].update(status='completed', seconds=time.monotonic() - started,
            accepted=fitted.improved, refusal=fitted.refusal_reason,
            message=fitted.message, focal_intervals=fitted.focal_intervals)
        candidate = fitted.candidate
        result['fit']['candidate'] = (dict(
            initial_rmse_px=candidate.initial_rmse_px,
            fitted_rmse_px=candidate.fitted_rmse_px, reason=candidate.reason,
            cameras=sorted(candidate.calibrations)) if candidate else None)
        if args.out:
            Path(args.out + '.fit.json').write_text(
                json.dumps(json_values(fitted), indent=2) + '\n', encoding='utf-8')
        emit()


if __name__ == '__main__':
    main()
