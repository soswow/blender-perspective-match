"""Check point-FOV Blender input/apply ownership with generated data and no solves."""

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


def state(context):
    from match_perspective import properties

    space = properties.workspace(context)
    return dict(
        roots={root.name: [list(row) for row in root.matrix_world]
               for root in properties.iter_match_roots()},
        sessions={root.name: dict(
            fx=root.pm_session.fx, fy=root.pm_session.fy,
            sync_scale=root.pm_session.sync_scale,
            sync_translation=list(root.pm_session.sync_translation),
            sync_is_applied=root.pm_session.sync_is_applied,
            sync_last_ok=root.pm_session.sync_last_ok,
            lens=root.pm_session.camera_object.data.lens,
            camera_matrix=[list(row) for row in root.pm_session.camera_object.matrix_local],
            plate=root.pm_session.undistorted_image.name
                  if root.pm_session.undistorted_image else None,
            view_undistorted=root.pm_session.view_undistorted,
            undistorted_path=root.pm_session.undistorted_path,
        ) for root in properties.iter_match_roots()},
        landmarks={item.item_id: dict(
            position=list(item.position), position_b=list(item.position_b),
            has_position=item.has_position,
            has_line_segment=item.has_line_segment,
            rmse=item.rmse_px,
        ) for item in space.landmarks},
        scene_camera=context.scene.camera.name if context.scene.camera else None,
        images={image.name: image.use_fake_user for image in bpy.data.images},
    )


def check(out):
    from match_perspective import core, properties, scene
    from match_perspective.core import sync
    from match_perspective.ui import operators

    context = HeadlessContext()
    space = properties.workspace(context)
    assert "estimate_focal_from_points" in space.bl_rna.properties
    assert "focal_pick_sigma_px" in space.bl_rna.properties
    assert "point_focal_span_percent" in space.bl_rna.properties
    space.share_lens = False
    space.estimate_focal_from_points = True
    space.focal_pick_sigma_px = 1.25
    assert space.point_focal_span_percent == 40.0
    for root in properties.iter_match_roots():
        root.pm_session.lines.clear()
        root.pm_session.origin_is_set = False
    # An unsupported ground constraint must reach the numerical refusal
    # without any automatic origin write during preparation.
    space.landmarks[0].on_ground = True
    cached_root = properties.iter_match_roots()[0]
    cached = bpy.data.images.new("Generated cached undistorted plate", width=64, height=64)
    cached_root.pm_session.undistorted_image = cached
    cached_root.pm_session.undistorted_width = 64
    cached_root.pm_session.undistorted_height = 64
    cached_root.pm_session.undistorted_path = str(out / "generated-cache.png")
    cached_root.pm_session.view_undistorted = True
    scene._apply_camera_background(cached_root.pm_session)
    with patch.object(scene, "ensure_origins_from_ground_landmarks",
                      side_effect=AssertionError("Point mode prepared an origin")):
        prep = scene.prepare_lens_refine(context)
    assert prep.estimate_focal_from_points and prep.pick_sigma_px == 1.25
    assert prep.fx_span == 0.4
    assert all(not item.freeze_focal and not item.reorient_from_vp
               for item in prep.lens_inputs)
    assert all(not root.pm_session.origin_is_set for root in properties.iter_match_roots())
    assert prep.solver_kwargs()["estimate_focal_from_points"] is True
    assert prep.solver_kwargs()["pick_sigma_px"] == 1.25
    space.share_lens = True
    shared_prep = scene.collect_lens_refine_inputs(context)
    assert not shared_prep.estimate_focal_from_points
    assert abs(shared_prep.fx_span - 0.18) < 1e-6
    assert space.estimate_focal_from_points, "Hidden point preference was lost"
    space.share_lens = False

    initial = state(context)
    calibrations = {item.match_id: deepcopy(item.base_calibration)
                    for item in prep.lens_inputs}
    intervals = {}
    for index, (match_id, calibration) in enumerate(calibrations.items()):
        calibration.intrinsics.fx *= 1.08 + index * 0.02
        calibration.intrinsics.fy = calibration.intrinsics.fx
        intervals[match_id] = (calibration.intrinsics.fx * 0.95,
                               calibration.intrinsics.fx * 1.05)
    anchor_id = prep.anchor_id
    similarities = {match_id: sync.SimilarityTransform(
        scale=1.0 if match_id == anchor_id else 1.15,
        translation=np.zeros(3) if match_id == anchor_id
                    else np.array([index + 2.0, -0.25, 0.5]),
    ) for index, match_id in enumerate(calibrations)}
    landmark_ids = [item.item_id for item in space.landmarks][:4]
    assert landmark_ids
    fitted_points = {item_id: np.array([index + 1.25, 2.5, -0.75])
                     for index, item_id in enumerate(landmark_ids)}
    fitted_sync = sync.SyncSolveResult(
        similarities=similarities, landmarks=fitted_points,
        mean_reprojection_px=0.3,
        per_match_rmse_px={key: 0.3 for key in calibrations},
        per_landmark_rmse_px={key: 0.2 for key in fitted_points},
        message="Accepted generated joint fit", success=True,
    )
    result = SimpleNamespace(
        cancelled=False, point_focal_mode=True, refusal_reason="",
        calibrations=calibrations, sync_result=fitted_sync,
        focal_intervals=intervals, improved=True,
        message="Point FOV fit accepted",
    )
    refusal = SimpleNamespace(
        cancelled=False, point_focal_mode=True,
        refusal_reason="Unsupported generated constraint", improved=False,
        message="Point FOV refused: unsupported generated constraint",
        calibrations={}, sync_result=None, focal_intervals={},
    )

    for edit in ("focal_pick_sigma_px", "estimate_focal_from_points",
                 "point_focal_span_percent"):
        original = getattr(space, edit)
        setattr(space, edit, False if edit == "estimate_focal_from_points"
                else original + 0.25)
        before = state(context)
        try:
            scene.apply_lens_refine_result(context, result, prep)
        except scene.StaleSyncResult:
            pass
        else:
            raise AssertionError(f"{edit} did not stale the job")
        assert before == state(context), f"{edit} stale apply changed state"
        setattr(space, edit, original)

    before = state(context)
    with patch.object(scene, "solve_and_apply_sync",
                      side_effect=AssertionError("Refusal ran Sync")):
        _, refused_sync = scene.apply_lens_refine_result(context, refusal, prep)
    assert refused_sync is None and before == state(context)

    wrong_mode = deepcopy(refusal)
    wrong_mode.point_focal_mode = False
    try:
        scene.apply_lens_refine_result(context, wrong_mode, prep)
    except ValueError as error:
        assert "another focal mode" in str(error)
    else:
        raise AssertionError("Point request accepted a legacy result")
    assert before == state(context)

    injected = RuntimeError("Generated post-apply failure")
    actual_sync_empties = scene.sync_landmark_empties
    empty_calls = 0

    def fail_after_landmarks(*args, **kwargs):
        nonlocal empty_calls
        empty_calls += 1
        actual_sync_empties(*args, **kwargs)
        if empty_calls == 1:
            raise injected

    before = state(context)
    with patch.object(scene, "solve_and_apply_sync",
                      side_effect=AssertionError("Point apply reran Sync")), \
            patch.object(scene, "sync_landmark_empties", side_effect=fail_after_landmarks):
        try:
            scene.apply_lens_refine_result(context, result, prep)
        except RuntimeError as error:
            assert error is injected
        else:
            raise AssertionError("Injected post-apply failure was swallowed")
    assert before == state(context), "Point apply failure did not restore snapshot"

    with patch.object(scene, "solve_and_apply_sync",
                      side_effect=AssertionError("Point apply reran Sync")):
        _, applied = scene.apply_lens_refine_result(context, result, prep)
    assert applied is fitted_sync
    for root in properties.iter_match_roots():
        match_id = root.name
        assert abs(root.pm_session.fx - calibrations[match_id].intrinsics.fx) < 1e-4
        assert np.allclose(root.pm_session.sync_translation,
                           similarities[match_id].translation, atol=1e-5)
        expected = sync._metric_scale_similarity(
            similarities[match_id], scene.calibration_from_settings(root.pm_session))
        assert np.allclose(root.matrix_world, scene._similarity_to_matrix(expected), atol=1e-5)
    for item_id, point in fitted_points.items():
        item = next(item for item in space.landmarks if item.item_id == item_id)
        assert item.has_position and np.allclose(item.position, point, atol=1e-5)
    assert "approximate local 95%" in space.sync_status
    assert all(match_id in space.sync_status for match_id in intervals)

    # Exercise the registered operator's real invoke/finish plumbing with a
    # stubbed refusal. No numerical entry point is called in this check.
    callbacks = []
    def pending_progress(_prep, **kwargs):
        callback = kwargs['progress_callback']
        callbacks.append(callback)
        callback(0, 101, 'Registering camera pair')
        return refusal

    with patch.object(scene, "run_lens_refine", side_effect=pending_progress), \
            patch.object(operators, "threading", SimpleNamespace(
                Thread=InlineWorker, Event=threading.Event)):
        operator = SimpleNamespace(_timer=None, report=lambda *_: None)
        status = operators.PM_OT_refine_lenses.invoke(operator, context, None)
        assert status == {"RUNNING_MODAL"}
        assert operators.lens_refine_startup_label() == 'Registering camera pair'
        operator._result_box['done'] = False
        operators.PM_OT_refine_lenses.modal(operator, context, SimpleNamespace(type='TIMER'))
        assert 'Registering camera pair' in space.sync_status and 'elapsed' in space.sync_status
        assert '0/101' not in space.sync_status
        callbacks[0](2, 101, 'Fitting landmarks')
        assert operators.lens_refine_startup_label() == 'Fitting landmarks · 1 iteration'
        operators.PM_OT_refine_lenses.modal(operator, context, SimpleNamespace(type='TIMER'))
        assert 'Fitting landmarks · 1 iteration' in space.sync_status
        assert '2/101' not in space.sync_status
        operator._result_box['done'] = True
        status = operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=False)
        assert status == {"FINISHED"}
    space.share_lens = True
    legacy_prep = scene.collect_lens_refine_inputs(context)
    assert not legacy_prep.estimate_focal_from_points
    before = state(context)
    try:
        scene.apply_lens_refine_result(context, result, legacy_prep)
    except ValueError as error:
        assert "another focal mode" in str(error)
    else:
        raise AssertionError("Legacy request accepted a point result")
    assert before == state(context)
    return dict(passed=True, numerical_solves=0, cameras=len(calibrations),
                fitted_landmarks=len(fitted_points),
                blender=bpy.app.version_string)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=False)
    register_extension()
    case = generate("free_scale", seed=0, noise_px=0.0)
    create_scene(case, args.out, False)
    report = check(args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Point FOV Blender apply PASS; zero numerical solves", flush=True)


if __name__ == "__main__":
    main()
