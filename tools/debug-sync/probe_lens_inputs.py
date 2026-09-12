"""Read-only lens-input eligibility report; never solve or save the blend.

blender --factory-startup --disable-autoexec -b --python this.py -- --blend scene.blend
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

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
    return dict(same_lens=prep.share_lens, point_focal=prep.estimate_focal_from_points,
        cameras=[dict(name=m.match_id, picks=counts[m.match_id], focal=m.intrinsics.fx,
                      vp_lines=sum(len(v) for v in m.line_bundles.values())) for m in prep.lens_inputs],
        points=len(views), known_points=len(prep.known_world),
        ground_picks=sum(p.on_ground for p in prep.observations),
        line_picks=len(prep.line_observations),
        unsupported_line_landmarks=sorted({names.get(p.landmark_id, p.landmark_id)
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
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).expanduser().resolve()))
    text = json.dumps(report(), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
