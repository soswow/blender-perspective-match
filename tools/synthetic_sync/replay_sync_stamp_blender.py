"""Replay an archived Refine apply and trace why its solution seed is not stamped."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flush-before-stamp", action="store_true",
                        help="Test the proposed dependency graph ordering fix")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source))

    import bpy
    import numpy as np
    from tools.synthetic_sync.blender_case import register_extension
    from tools.synthetic_sync.verify_focal_constraints_blender import build

    register_extension()
    from match_perspective import properties, scene
    from match_perspective.core import focal_bundle, lens_refine, sync
    from match_perspective.core.lens_refine import LensRefineResult
    from match_perspective.core.sync.request import (
        _decode_calibration, _decode_similarity,
    )

    case = json.loads(args.case.read_text())
    report = json.loads(args.result.read_text())
    values = dict(report["result"])
    values["calibrations"] = {
        key: _decode_calibration(value)
        for key, value in values["calibrations"].items()}
    values["similarities"] = {
        key: _decode_similarity(value)
        for key, value in values["similarities"].items()}
    values["landmarks"] = {
        key: np.asarray(value, float)
        for key, value in values["landmarks"].items()}
    values["line_segments"] = {
        key: tuple(np.asarray(end, float) for end in segment)
        for key, segment in values["line_segments"].items()}
    allowed = {field.name for field in fields(sync.SyncSolveResult)}
    result = sync.SyncSolveResult(**{
        key: value for key, value in values.items() if key in allowed})
    args.out.mkdir(parents=True, exist_ok=False)
    build(case, args.out)
    prep = scene.prepare_lens_refine(bpy.context)
    refined = LensRefineResult(
        calibrations=result.calibrations, sync_result=result,
        initial_cost=0.0, final_cost=0.0,
        initial_sync_rmse=0.0, final_sync_rmse=result.mean_reprojection_px,
        message="Saved Refine result apply replay", improved=True,
        point_focal_mode=True,
    )

    trace = []
    original = scene._live_sync_solution_seed
    original_stamp = scene._stamp_sync_solution

    def traced(context, matches, evidence, diagnostics=None):
        seed, geometry = original(context, matches, evidence, diagnostics)
        roots = list(properties.iter_match_roots())
        anchor = properties.anchor_root(context)
        trace.append(dict(
            evidence_present=bool(evidence), seed_available=seed is not None,
            geometry_present=bool(geometry),
            match_ids=[item.match_id for item in matches],
            anchor=anchor.name if anchor else None,
            roots=[dict(name=root.name,
                        sync_last_ok=bool(root.pm_session.sync_last_ok),
                        sync_is_applied=bool(root.pm_session.sync_is_applied),
                        camera_drifted=scene.matched_camera_has_drifted(root.pm_session))
                   for root in roots],
            positioned_landmarks=sum(item.has_position for item in
                                     properties.workspace(context).landmarks),
        ))
        return seed, geometry

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Saved application replay must not run a numerical solver")

    scene._live_sync_solution_seed = traced
    if args.flush_before_stamp:
        def flushed_stamp(context, result):
            context.view_layer.update()
            return original_stamp(context, result)
        scene._stamp_sync_solution = flushed_stamp
    try:
        with patch.object(scene, "run_solve_sync", forbidden), \
             patch.object(scene, "run_lens_refine", forbidden), \
             patch.object(sync, "solve_landmark_sync", forbidden), \
             patch.object(focal_bundle, "fit_independent_focals", forbidden), \
             patch.object(lens_refine, "fit_independent_focals", forbidden):
            _refined, applied = scene.apply_lens_refine_result(
                bpy.context, refined, prep)
    finally:
        scene._live_sync_solution_seed = original
        scene._stamp_sync_solution = original_stamp
    request = scene.collect_sync_request(bpy.context)
    space = properties.workspace(bpy.context)
    output = dict(
        archived_source_sha256=report["source_sha256"],
        scene_source_sha256=hashlib.sha256((source / "scene" / "__init__.py").read_bytes()).hexdigest(),
        applied=applied is not None and applied.success,
        certified=(request.initial_solution is not None and
                   request.initial_solution.evidence_sha256 == request.evidence_sha256()),
        stored_evidence=bool(space.sync_solution_evidence_sha256),
        stored_geometry=bool(space.sync_solution_geometry_sha256),
        trace=trace,
    )
    args.out.joinpath("stamp-trace.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["certified"]:
        raise AssertionError("Saved Refine application did not certify its live seed")


if __name__ == "__main__":
    main()
