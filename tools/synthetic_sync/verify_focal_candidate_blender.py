"""Generated Blender best-fit apply, stale-input and rollback checks; no solves."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import create_scene, register_extension
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.verify_jobs import HeadlessContext, InlineWorker
from tools.synthetic_sync.verify_point_focal_apply import state


def check(out: Path) -> dict:
    from match_perspective import properties, scene
    from match_perspective.core import sync
    from match_perspective.ui import operators

    context = HeadlessContext()
    space = properties.workspace(context)
    space.share_lens = False
    space.estimate_focal_from_points = True
    for root in properties.iter_match_roots():
        root.pm_session.lines.clear()
        root.pm_session.origin_is_set = False
    prep = scene.prepare_lens_refine(context)
    calibrations = {item.match_id: deepcopy(item.base_calibration) for item in prep.lens_inputs}
    for index, calibration in enumerate(calibrations.values()):
        calibration.intrinsics.fx *= 1.04 + .01 * index
        calibration.intrinsics.fy = calibration.intrinsics.fx
    similarities = {
        match_id: sync.SimilarityTransform(
            translation=np.zeros(3) if match_id == prep.anchor_id else
                        np.array((index + 1., .25, -.1)))
        for index, match_id in enumerate(calibrations)
    }
    points = {item.item_id: np.array((index * .2, .3, 1.))
              for index, item in enumerate(space.landmarks)}
    fitted = sync.SyncSolveResult(
        similarities=similarities, landmarks=points,
        mean_reprojection_px=2.1, per_match_rmse_px={key: 2.1 for key in calibrations},
        per_landmark_rmse_px={key: 2.1 for key in points},
        message="Generated provisional geometry", success=True)
    candidate = SimpleNamespace(
        calibrations=calibrations, sync_result=fitted,
        initial_rmse_px=2., fitted_rmse_px=2.1,
        initial_objective=100., fitted_objective=20.,
        reason="Point focal fit did not converge")
    result = SimpleNamespace(
        cancelled=False, point_focal_mode=True, refusal_reason=candidate.reason,
        improved=False, message=candidate.reason, candidate=candidate,
        calibrations={}, sync_result=None, focal_intervals={})
    before = state(context)
    _, unselected = scene.apply_lens_refine_result(context, result, prep)
    assert unselected is None and state(context) == before
    operator = SimpleNamespace(report=lambda *_: None)
    # Exercise the real job ownership path: the active finished worker, rather
    # than a discarded or cancelled job, publishes the explicit-use option.
    with patch.object(scene, "run_lens_refine", return_value=result), \
         patch.object(operators, "threading", SimpleNamespace(
             Thread=InlineWorker, Event=threading.Event)):
        job = SimpleNamespace(_timer=None, report=lambda *_: None)
        job._finish_job = lambda context, *, cancelled: operators.PM_OT_refine_lenses._finish_job(
            job, context, cancelled=cancelled)
        assert operators.PM_OT_refine_lenses.invoke(job, context, None) == {"RUNNING_MODAL"}
        assert operators.PM_OT_refine_lenses.modal(
            job, context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
    assert operators.lens_best_fit_is_available(context)
    assert operators.lens_best_fit_label() == "Use Best Fit (2.10 px)"
    operators._diagnose_sync_running = True
    assert not operators.lens_best_fit_is_available(context)
    operators._diagnose_sync_running = False
    operators._pin_sync_running = True
    assert not operators.lens_best_fit_is_available(context)
    operators._pin_sync_running = False

    # The explicit operator must reject a changed numerical input and retire
    # the pending result without touching cameras or landmarks.
    prior = space.focal_pick_sigma_px
    space.focal_pick_sigma_px = prior + .25
    changed = state(context)
    assert operators.PM_OT_use_best_focal_fit.execute(operator, context) == {"CANCELLED"}
    assert state(context) == changed and not operators.lens_best_fit_is_available(context)
    space.focal_pick_sigma_px = prior

    # A failure after writes must restore every camera, landmark and plate.
    injected = RuntimeError("Generated candidate apply failure")
    operators._lens_best_fit = {"prep": prep, "result": result}
    with patch.object(scene, "sync_landmark_empties", side_effect=injected):
        try:
            scene.apply_lens_refine_result(context, result, prep, use_candidate=True)
        except RuntimeError as error:
            assert error is injected
        else:
            raise AssertionError("Candidate apply swallowed injected error")
    assert state(context) == before
    assert operators.lens_best_fit_is_available(context)

    with patch.object(scene, "solve_and_apply_sync",
                      side_effect=AssertionError("Candidate apply reran Sync")):
        assert bpy.ops.perspective_match.use_best_focal_fit() == {"FINISHED"}
    assert not operators.lens_best_fit_is_available(context)
    assert "Provisional fit applied" in space.sync_status
    assert "calibration not validated" in space.sync_status
    assert "point RMSE 2.00 → 2.10px" in space.sync_status
    assert "combined fit improved, although point RMSE increased" in space.sync_status
    assert candidate.reason in space.sync_status
    assert "95%" not in space.sync_status
    for root in properties.iter_match_roots():
        assert abs(root.pm_session.fx - calibrations[root.name].intrinsics.fx) < 1e-4
    for item in space.landmarks:
        assert item.has_position and np.allclose(item.position, points[item.item_id])

    operators._lens_best_fit = {"prep": prep, "result": result}
    operators.reset_sync_background_jobs()
    assert not operators.lens_best_fit_is_available(context)
    with patch.object(scene, "run_lens_refine", return_value=result):
        assert operators.PM_OT_refine_lenses.execute(operator, context) == {"FINISHED"}
    assert operators.lens_best_fit_is_available(context)
    operators.reset_sync_background_jobs()
    assert not operators.lens_best_fit_is_available(context)
    # Background Blender has no Undo context. The registered operator ran,
    # but its UNDO flag cannot be exercised until an interactive window test.
    return dict(passed=True, numerical_solves=0, candidate_rmse_px=2.1,
                cameras=len(calibrations), landmarks=len(points),
                undo_checked=False, headless_undo_available=bool(bpy.ops.ed.undo.poll()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reload", action="store_true",
                        help="Reload the extension before the controlled candidate job")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=False)
    register_extension()
    create_scene(generate("free_scale", seed=0, noise_px=0.0), args.out, False)
    if args.reload:
        import match_perspective
        match_perspective.reload_addon()
    report = check(args.out)
    report["reloaded_before_job"] = args.reload
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Best-fit Blender PASS:", report)


if __name__ == "__main__":
    main()
