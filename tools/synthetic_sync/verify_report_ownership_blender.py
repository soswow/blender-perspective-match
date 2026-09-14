"""Generated report ownership checks with substituted numerical results; zero solves."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import create_scene, register_extension
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.verify_jobs import HeadlessContext


def check(out: Path) -> dict:
    from match_perspective import properties, scene
    from match_perspective.core import sync
    from match_perspective.ui import operators, sync_report

    context = HeadlessContext()
    workspace = properties.workspace(context)
    request = scene.collect_sync_request(context)
    ids = [item.match_id for item in request.matches]
    assert len(ids) >= 2

    def result(mean: float, *, success: bool = True):
        value = sync.SyncSolveResult(
            similarities={key: sync.SimilarityTransform() for key in ids},
            landmarks={}, mean_reprojection_px=mean,
            per_match_rmse_px={key: mean for key in ids},
            per_landmark_rmse_px={}, message=f"Generated result {mean:.2f}",
            success=success,
        )
        value.point_rmse_px = mean
        value.line_rmse_px = mean + 4
        value.per_match_point_rmse_px = {key: mean for key in ids}
        value.per_match_line_rmse_px = {key: mean + 4 for key in ids}
        return value

    messages = []
    operator = SimpleNamespace(report=lambda *args: messages.append(args))
    url_open = Mock(side_effect=AssertionError("Report auto-opened"))
    operator_bpy = SimpleNamespace(
        app=bpy.app, data=bpy.data,
        ops=SimpleNamespace(wm=SimpleNamespace(url_open=url_open)),
    )
    accepted = result(1.25)
    with patch.object(scene, "run_solve_sync", return_value=accepted) as solve, \
         patch.object(operators, "bpy", operator_bpy):
        assert bpy.ops.perspective_match.solve_sync("EXEC_DEFAULT") == {"FINISHED"}
    assert solve.call_count == 1
    assert url_open.call_count == 0
    first_report = sync_report._last_report
    assert first_report is not None and first_report.applied
    assert first_report.operation == "Solve Sync"
    assert first_report.point_rmse_px == 1.25
    assert first_report.line_rmse_px == 5.25
    assert sync_report.last_report_path().is_file()
    assert not operators.last_report_is_stale(context)

    # The result is bound to post-application numerical inputs. A pick edit
    # changes the report state without running a new solve.
    pick = next(item.observations[0] for item in workspace.landmarks if item.observations)
    old_x = pick.x
    pick.x += 1
    assert operators._refresh_report_staleness(context)
    assert operators.last_report_is_stale(context)
    url_open = Mock(return_value={"FINISHED"})
    operator_bpy.ops.wm.url_open = url_open
    with patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_open_sync_report.execute(operator, context) == {"FINISHED"}
    assert url_open.call_count == 1
    assert "Stale: scene inputs or cameras changed" in sync_report.last_report_path().read_text()
    pick.x = old_x
    assert not operators._refresh_report_staleness(context)

    # Refine publishes its returned final Sync result. The numerical and
    # application callbacks are substituted; no second Sync solve is allowed.
    refined = result(2.75)
    refine_prep = scene.prepare_lens_refine(context)
    refine = SimpleNamespace(
        cancelled=False, improved=True, refusal_reason="", candidate=None,
        calibrations={}, message="Generated focal result", sync_result=refined,
    )
    with patch.object(scene, "prepare_lens_refine", return_value=refine_prep), \
         patch.object(scene, "run_lens_refine", return_value=refine) as lens, \
         patch.object(scene, "apply_lens_refine_result", return_value=(refine, refined)) as apply, \
         patch.object(scene, "run_solve_sync", side_effect=AssertionError("Refine report re-solved Sync")), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_refine_lenses.execute(operator, context) == {"FINISHED"}
    assert lens.call_count == apply.call_count == 1
    assert sync_report._last_report.operation == "Refine Lenses"
    assert sync_report._last_report.applied
    assert sync_report._last_report.point_rmse_px == 2.75

    refused_refine = SimpleNamespace(
        cancelled=False, improved=False, refusal_reason="Focal interval too wide",
        candidate=None, calibrations={}, message="Focal interval too wide", sync_result=None,
    )
    with patch.object(scene, "prepare_lens_refine", return_value=refine_prep), \
         patch.object(scene, "run_lens_refine", return_value=refused_refine), \
         patch.object(scene, "apply_lens_refine_result", return_value=(refused_refine, None)), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_refine_lenses.execute(operator, context) == {"FINISHED"}
    assert not sync_report._last_report.applied
    assert sync_report._last_report.point_rmse_px is None
    assert sync_report._last_report.enabled_matches == len(refine_prep.lens_inputs)
    assert "Not applied to this scene" in sync_report.last_report_path().read_text()
    shutil.copyfile(sync_report.last_report_path(), out / "refused-focal-report.html")

    legacy_refusal = result(9.0, success=False)
    lens_calibrations = {
        root.name: scene.calibration_from_settings(root.pm_session)
        for root in properties.iter_match_roots()
    }
    legacy_refine = SimpleNamespace(
        cancelled=False, improved=False, refusal_reason="", candidate=None,
        calibrations=lens_calibrations, message="Generated lens step",
        sync_result=legacy_refusal,
    )
    with patch.object(scene, "prepare_lens_refine", return_value=refine_prep), \
         patch.object(scene, "run_lens_refine", return_value=legacy_refine), \
         patch.object(scene, "apply_lens_refine_result", return_value=(legacy_refine, legacy_refusal)), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_refine_lenses.execute(operator, context) == {"FINISHED"}
    assert not sync_report._last_report.applied
    assert sync_report._last_report.application_state == "Lens settings applied; Sync geometry not applied"
    assert "Lens settings applied; Sync geometry not applied" in sync_report.last_report_path().read_text()

    # Investigation is a separate report. It does not replace the visible
    # Solve status or a landmark error with independent numerical estimates.
    prior_status = workspace.sync_status
    landmark = workspace.landmarks[0]
    landmark.rmse_px = 19.0
    prepare = scene.prepare_diagnose_sync
    def readonly_prep(ctx, *, prepare_scene):
        assert prepare_scene is False
        return prepare(ctx)
    with patch.object(scene, "prepare_diagnose_sync", side_effect=readonly_prep), \
         patch.object(scene, "run_diagnose_sync", return_value=result(3.25)) as investigate, \
         patch.object(scene, "apply_diagnose_sync_result", side_effect=lambda _ctx, _prep, value: value), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_diagnose_sync.execute(operator, context) == {"FINISHED"}
    assert investigate.call_count == 1
    assert workspace.sync_status == prior_status
    assert landmark.rmse_px == 19.0
    assert sync_report._last_report.operation == "Investigate Problems"
    assert not sync_report._last_report.applied

    # Apply a fitted point-FOV result through the real scene path, then check
    # the HTML reports the calibrations actually written to the cameras.
    workspace.share_lens = False
    workspace.estimate_focal_from_points = True
    for root in properties.iter_match_roots():
        root.pm_session.lines.clear()
    focal_prep = scene.prepare_lens_refine(context)
    calibrations = {
        item.match_id: deepcopy(item.base_calibration)
        for item in focal_prep.lens_inputs
    }
    for index, calibration in enumerate(calibrations.values()):
        calibration.intrinsics.fx *= 1.03 + index * .01
        calibration.intrinsics.fy = calibration.intrinsics.fx
    fitted = result(4.5)
    fitted.landmarks = {
        item.item_id: np.array((index * .2, .3, 1.0))
        for index, item in enumerate(workspace.landmarks)
    }
    focal = SimpleNamespace(
        cancelled=False, improved=True, refusal_reason="", candidate=None,
        point_focal_mode=True, calibrations=calibrations, sync_result=fitted,
        message="Generated accepted focal fit", focal_intervals={},
    )
    with patch.object(scene, "run_lens_refine", return_value=focal) as lens, \
         patch.object(scene, "solve_and_apply_sync", side_effect=AssertionError("Focal report re-solved Sync")), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_refine_lenses.execute(operator, context) == {"FINISHED"}
    assert lens.call_count == 1
    assert sync_report._last_report.applied
    for root in properties.iter_match_roots():
        live = scene.calibration_from_settings(root.pm_session)
        assert abs(live.intrinsics.fx - calibrations[root.name].intrinsics.fx) < 1e-4
        assert any(f"{live.hfov_degrees:.2f}° HFOV" in item
                   for item in sync_report._last_report.focal_summary)
    assert sync_report._last_report_request_sha256 == scene.collect_sync_request(context).to_record()["sha256"]
    assert url_open.call_count == 1
    shutil.copyfile(sync_report.last_report_path(), out / "accepted-focal-report.html")

    # A free mirror origin can shift the returned result and the anchor's
    # private camera after the numerical job. The report must key off that
    # post-application state rather than its captured starting request.
    workspace.mirror_origin = "LANDMARK"
    workspace.mirror_plane = "YZ"
    workspace.mirror_object = None
    reference = next(item for item in workspace.landmarks if item.kind == "POINT")
    workspace.mirror_landmark_id = reference.item_id
    for item in workspace.landmarks:
        item.on_ground = False
    anchor = properties.anchor_root(context)
    anchor.pm_session.origin_is_set = False
    for root in properties.iter_match_roots():
        if root != anchor:
            root.pm_session.sync_role = "SOLVE"
    workspace.lock_translation = False
    shifted = result(6.0)
    shifted.landmarks = {
        item.item_id: np.array((1.0 + index * .2, .4, -2.0 + index * .1))
        for index, item in enumerate(workspace.landmarks)
    }
    assert scene._free_mirror_origin_offset(context, shifted, anchor) is not None
    center_before = scene.calibration_from_settings(anchor.pm_session).camera_center.copy()
    original_apply = scene.apply_solve_sync_result
    applied_values = []
    def capture_apply(*args):
        value = original_apply(*args)
        applied_values.append(value)
        return value
    with patch.object(scene, "run_solve_sync", return_value=shifted), \
         patch.object(scene, "apply_solve_sync_result", side_effect=capture_apply) as apply, \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_solve_sync.execute(operator, context) == {"FINISHED"}
    assert apply.call_count == 1
    applied = applied_values[0]
    assert applied is not shifted
    assert not np.allclose(center_before, scene.calibration_from_settings(anchor.pm_session).camera_center)
    np.testing.assert_allclose(applied.landmarks[reference.item_id][:2], (0, 0), atol=1e-7)
    assert sync_report._last_report.point_rmse_px == 6.0
    assert sync_report._last_report_request_sha256 == scene.collect_sync_request(context).to_record()["sha256"]
    assert not operators.last_report_is_stale(context)
    shutil.copyfile(sync_report.last_report_path(), out / "anchor-shift-report.html")
    last_path = sync_report.last_report_path()
    with patch.object(scene, "run_solve_sync", return_value=result(8.0)), \
         patch.object(operators, "_write_sync_report", side_effect=OSError("Generated disk failure")), \
         patch.object(operators, "bpy", operator_bpy):
        assert operators.PM_OT_solve_sync.execute(operator, context) == {"FINISHED"}
    assert sync_report.last_report_path() == last_path
    assert all(abs(root.pm_session.sync_rmse_px - 8.0) < 1e-6
               for root in properties.iter_match_roots())
    assert any("Could not write sync report" in args[1] for args in messages)
    return {"passed": True, "numerical_solves": 0, "reports": 7,
            "samples": ["refused-focal-report.html", "accepted-focal-report.html",
                        "anchor-shift-report.html"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=False)
    register_extension()
    create_scene(generate("free_scale", seed=0, noise_px=0.0), args.out, False)
    evidence = check(args.out)
    (args.out / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Report ownership Blender PASS:", evidence)


if __name__ == "__main__":
    main()
