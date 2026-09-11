"""Compare blocking and background job inputs through generated Blender state."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
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
from tools.synthetic_sync.blender_case import assert_equivalent, blender_pixels, create_scene, register_extension
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.planes import plane_case
from tools.synthetic_sync.scenarios import write_case
from tools.synthetic_sync.solver import environment, result_record
from tools.synthetic_sync.verify_requests import state_case


class InlineWorker:
    """Execute the real worker callback without relying on thread timing."""

    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class HeadlessContext:
    """Use real scene/RNA, replacing only modal window-manager plumbing."""

    window_manager = SimpleNamespace(
        windows=(),
        progress_begin=lambda *_: None, progress_end=lambda: None,
        event_timer_add=lambda *_args, **_kwargs: object(),
        event_timer_remove=lambda *_: None, modal_handler_add=lambda *_: None,
    )

    def __getattr__(self, name):
        return getattr(bpy.context, name)


def verify_lens_inputs(out):
    from match_perspective import scene
    from match_perspective.core import lens_refine
    from match_perspective.core.sync.request import json_values
    from match_perspective.ui import operators

    prep = scene.prepare_lens_refine(bpy.context)
    assert prep.plane_groups and prep.plane_slack > 0
    assert prep.fixed_similarities and prep.readonly_match_ids
    signature = inspect.signature(lens_refine.refine_lenses_from_landmarks)
    captured = []

    def capture(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if captured:
            assert callable(bound.arguments["cancel_check"])
            assert callable(bound.arguments["progress_callback"])
            assert not bound.arguments["cancel_check"]()
        values = {key: value for key, value in bound.arguments.items()
                  if key not in {"cancel_check", "progress_callback"}}
        captured.append(json_values(values))
        return SimpleNamespace(cancelled=True, message="Captured job inputs")

    context = HeadlessContext()
    operator = SimpleNamespace(_timer=None, report=lambda *_: None)
    with patch.object(lens_refine, "refine_lenses_from_landmarks", side_effect=capture), \
            patch.object(scene, "apply_lens_refine_result", return_value=(None, None)), \
            patch.object(operators, "threading", SimpleNamespace(Thread=InlineWorker, Event=threading.Event)):
        scene.refine_lenses_and_sync(context)
        try:
            status = operators.PM_OT_refine_lenses.invoke(operator, context, None)
            if status != {"RUNNING_MODAL"}:
                raise AssertionError(f"Background invocation failed: {status}")
            error = operators._lens_refine_result_box.get("error")
            if error is not None:
                raise error
        finally:
            operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=True)
    if len(captured) != 2:
        raise AssertionError(f"Expected two numerical entry calls, got {len(captured)}")
    for label, values in zip(("blocking", "background"), captured):
        (out / (label+"-lens-input.json")).write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")
    # Check every prepared numerical field independently of route-to-route parity.
    expected = {key: value for key, value in vars(prep).items()
                if key not in {"root_by_name", "source_scene_uid", "source_request_sha256"}}
    expected["matches"] = expected.pop("lens_inputs")
    expected = json_values(expected)
    for label, values in zip(("blocking", "background"), captured):
        for key, value in expected.items():
            if type(value) is not type(values[key]):
                raise AssertionError(f"{label}.{key}: prepared {value!r}, received {values[key]!r}")
            assert_equivalent(value, values[key], f"{label}.{key}")
    assert_equivalent(captured[0], captured[1], "lens routes")
    print("Lens inputs PASS: blocking and background preserve every prepared field", flush=True)


def verify_diagnose_ownership(case, out):
    """Finish real prepared jobs after controlled edits, without thread races."""
    from match_perspective import properties, scene
    from match_perspective.ui import operators

    workspace = properties.workspace(bpy.context)
    landmark = next(item for item in workspace.landmarks if item.plane_axis == "FREE")
    root = next(item for item in properties.iter_match_roots() if item.name == "view_1")
    pick = landmark.observations[0]
    results, violations = [], []
    for action in ("unchanged", "unrelated_object", "plane", "role", "pick", "missing_anchor", "other_scene"):
        pending = []

        class DeferredWorker(InlineWorker):
            def start(self):
                pending.append(self.target)

        folder = out / action
        folder.mkdir()
        context = HeadlessContext()
        operator = SimpleNamespace(_timer=None, report=lambda *_: None)
        before = dict(plane_axis=landmark.plane_axis, role=root.pm_session.sync_role,
                      x=pick.x, anchor=workspace.anchor_root, scene=context.scene)
        unrelated = None
        other_scene = None
        try:
            with patch.object(operators, "threading", SimpleNamespace(Thread=DeferredWorker, Event=threading.Event)), \
                    patch.object(operators, "_report_exception", wraps=operators._report_exception) as failure, \
                    patch.object(operators, "_write_diagnose_report",
                                 return_value=(SimpleNamespace(severity="success"), None, False)) as publish, \
                    patch.object(operators.sync_report, "compact_status", return_value="Captured report"):
                status = operators.PM_OT_diagnose_sync.invoke(operator, context, None)
                if status != {"RUNNING_MODAL"} or len(pending) != 1:
                    raise AssertionError(f"Diagnose invocation failed: {status}")
                prepared = operator._prep.to_record()
                if action == "plane":
                    landmark.plane_axis = "NONE"
                elif action == "role":
                    root.pm_session.sync_role = "FIT_ONLY"
                elif action == "pick":
                    pick.x += 25
                elif action == "unrelated_object":
                    unrelated = bpy.data.objects.new("Unrelated modeling edit", None)
                    bpy.context.scene.collection.objects.link(unrelated)
                    unrelated.location.x = 2
                elif action == "missing_anchor":
                    workspace.anchor_root = None
                elif action == "other_scene":
                    other_scene = bpy.data.scenes.new("Another generated scene")
                    bpy.context.window.scene = other_scene
                try:
                    current = scene.collect_sync_request(context).to_record()
                except ValueError as error:
                    current = {"error": str(error)}
                changed = prepared["sha256"] != current.get("sha256")
                landmark.rmse_px = 73.0
                pending[0]()
                error = operators._diagnose_sync_result_box.get("error")
                if error is not None:
                    raise error
                result = operators._diagnose_sync_result_box["result"]
                assessment = evaluate(case, result_record(result, case["request"]["cameras"]))
                if not assessment["passed"]:
                    raise AssertionError(assessment["violations"])
                with patch.object(scene, "ensure_ground_frame_from_landmarks", side_effect=AssertionError("Apply ran ground preparation")) as ground_prepare, \
                        patch.object(scene, "ensure_origins_from_ground_landmarks", side_effect=AssertionError("Apply ran origin preparation")) as origin_prepare:
                    status = operators.PM_OT_diagnose_sync._finish_job(operator, context, cancelled=False)
                if ground_prepare.called or origin_prepare.called:
                    raise AssertionError("Diagnose application attempted preparation")
                rejected = (status == {"CANCELLED"} and not publish.called and landmark.rmse_px == 73.0
                            and failure.called and isinstance(failure.call_args.args[1], scene.StaleSyncResult))
                accepted = status == {"FINISHED"} and publish.called and landmark.rmse_px != 73.0
                expected_changed = action not in {"unchanged", "unrelated_object"}
                if changed != expected_changed or not (rejected if expected_changed else accepted):
                    violations.append(f"{action}: changed={changed}, status={status}, published={publish.called}, rmse={landmark.rmse_px}")
                results.append(dict(action=action, inputs_changed=changed, rejected=rejected, accepted=accepted,
                                    status=sorted(status), published=publish.called, rmse=landmark.rmse_px,
                                    message=properties.workspace(context).sync_status))
                (folder / "prepared.json").write_text(json.dumps(prepared, indent=2)+"\n")
                (folder / "current.json").write_text(json.dumps(current, indent=2)+"\n")
                (folder / "assessment.json").write_text(json.dumps(assessment, indent=2)+"\n")
        finally:
            if other_scene is not None:
                bpy.context.window.scene = before["scene"]
                bpy.data.scenes.remove(other_scene)
            workspace.anchor_root = before["anchor"]
            landmark.plane_axis = before["plane_axis"]
            root.pm_session.sync_role = before["role"]
            pick.x = before["x"]
            if unrelated is not None:
                bpy.data.objects.remove(unrelated, do_unlink=True)
        print(action, results[-1], flush=True)
    (out / "ownership.json").write_text(json.dumps(dict(results=results, violations=violations), indent=2)+"\n")
    if violations:
        raise AssertionError("; ".join(violations))


def verify_lens_ownership(case, out, *, apply_failures=False):
    """Replay one real lens-search result across controlled edits and actual apply."""
    from match_perspective import properties, scene
    from match_perspective.core.sync.request import json_values
    from match_perspective.ui import operators

    workspace = properties.workspace(bpy.context)
    workspace.share_lens = True
    workspace.lens_refine_span_percent = 12.5
    source = out / "input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(source), check_existing=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    prepared = scene.prepare_lens_refine(bpy.context)
    wanted = json_values(prepared.solver_kwargs())
    refined = scene.run_lens_refine(prepared)
    if not refined.improved or not refined.sync_result.success:
        raise AssertionError("The positive control needs a real, improved lens result")
    print("Real lens search completed", refined.initial_sync_rmse, refined.final_sync_rmse, flush=True)
    (out / "lens-input.json").write_text(json.dumps(wanted, indent=2)+"\n")
    (out / "refined-calibrations.json").write_text(json.dumps(json_values(refined.calibrations), indent=2)+"\n")
    results, violations = [], []
    controls = {"unchanged", "unrelated_object", "active_match"}
    actions = (("unchanged", "mid_camera_error", "late_sync_error", "after_sync_error", "after_plate_error", "solver_refusal") if apply_failures else
               ("unchanged", "unrelated_object", "active_match", "plane", "role", "pick", "focal",
                "search", "origin", "vp_stroke", "live_camera", "root_pose", "missing_camera", "other_scene"))
    for action in actions:
        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
        workspace = properties.workspace(bpy.context)
        root = next(item for item in properties.iter_match_roots() if item.name == "view_1")
        landmark = next(item for item in workspace.landmarks if item.plane_axis == "FREE")
        context = HeadlessContext()
        operator = SimpleNamespace(_timer=None, report=lambda *_: None)
        pending = []
        if apply_failures:
            settings = root.pm_session
            cached = bpy.data.images.new("Generated cached plate", width=settings.image_width, height=settings.image_height)
            settings.undistorted_image = cached
            settings.undistorted_width = settings.image_width
            settings.undistorted_height = settings.image_height
            settings.undistorted_path = str((out / "generated-cache.png").resolve())
            settings.view_undistorted = True
            scene._apply_camera_background(settings)

        class DeferredWorker(InlineWorker):
            def start(self):
                pending.append(self.target)

        def completed_search(prep, **_kwargs):
            assert_equivalent(wanted, json_values(prep.solver_kwargs()), "replayed lens input")
            return deepcopy(refined)

        def applied_state():
            return json_values(dict(
                cameras={item.name: scene.calibration_from_settings(item.pm_session) for item in properties.iter_match_roots()},
                camera_objects={item.name: dict(
                                data={key: getattr(item.pm_session.camera_object.data, key) for key in (
                                    "type", "lens", "sensor_fit", "sensor_width", "sensor_height", "shift_x", "shift_y")},
                                matrix=[list(row) for row in item.pm_session.camera_object.matrix_local])
                                if item.pm_session.camera_object is not None else None
                                for item in properties.iter_match_roots()},
                session_state={item.name: {key: (list(getattr(item.pm_session, key)) if key in {"origin_image", "sync_rotation", "sync_translation"}
                    else getattr(item.pm_session, key)) for key in (
                    "origin_is_set", "origin_image", "sync_is_applied", "sync_scale", "sync_rotation", "sync_translation",
                    "sync_rmse_px", "sync_last_ok", "fov_xy", "fov_zy", "fov_zx", "residual_degrees", "vp_line_rms_px",
                )} for item in properties.iter_match_roots()},
                plates={item.name: dict(
                    image=item.pm_session.undistorted_image.name if item.pm_session.undistorted_image else None,
                    **{key: getattr(item.pm_session, key) for key in ("view_undistorted", "undistorted_path",
                        "undistorted_width", "undistorted_height", "undistorted_offset_x", "undistorted_offset_y")},
                ) for item in properties.iter_match_roots()},
                images={item.name: item.use_fake_user for item in bpy.data.images},
                scene_camera=context.scene.camera.name if context.scene.camera else None,
                render_size=[context.scene.render.resolution_x, context.scene.render.resolution_y,
                             context.scene.render.resolution_percentage],
                transforms={item.name: [list(row) for row in item.matrix_world] for item in properties.iter_match_roots()},
                landmarks={item.item_id: [list(item.position), list(item.position_b), item.has_position,
                           item.has_line_segment, item.rmse_px] for item in workspace.landmarks},
            ))

        with patch.object(operators, "threading", SimpleNamespace(Thread=DeferredWorker, Event=threading.Event)), \
                patch.object(operators, "_report_exception", wraps=operators._report_exception) as failure, \
                patch.object(scene, "run_lens_refine", side_effect=completed_search):
            status = operators.PM_OT_refine_lenses.invoke(operator, context, None)
            if status != {"RUNNING_MODAL"} or len(pending) != 1:
                raise AssertionError(f"Lens invocation failed: {status}")
            if action == "plane":
                landmark.plane_axis = "NONE"
            elif action == "role":
                root.pm_session.sync_role = "FIT_ONLY"
            elif action == "pick":
                landmark.observations[0].x += 25
            elif action == "focal":
                root.pm_session.fx *= 1.02
            elif action == "search":
                workspace.lens_refine_span_percent = 9
            elif action == "origin":
                root.pm_session.origin_image[0] += 3
            elif action == "vp_stroke":
                stroke = root.pm_session.lines.add()
                stroke.axis = "z"
                stroke.x1, stroke.y1, stroke.x2, stroke.y2 = 100, 200, 130, 300
            elif action == "live_camera":
                root.pm_session.camera_object.data.lens *= 1.02
            elif action == "root_pose":
                matrix = root.matrix_world.copy()
                matrix.translation.x += .1
                root.matrix_world = matrix
            elif action == "missing_camera":
                bpy.data.objects.remove(root.pm_session.camera_object, do_unlink=True)
            elif action == "other_scene":
                bpy.context.window.scene = bpy.data.scenes.new("Another generated scene")
            elif action == "active_match":
                scene.set_active_match(context, root)
            elif action == "unrelated_object":
                unrelated = bpy.data.objects.new("Unrelated modeling edit", None)
                bpy.context.scene.collection.objects.link(unrelated)
            before = applied_state()
            pending[0]()
            error = operators._lens_refine_result_box.get("error")
            if error is not None:
                raise error
            # Valid apply invokes Solve Sync, which can prepare origins. A stale
            # result must be rejected before any such preparation or write.
            expected_error = action in {"mid_camera_error", "late_sync_error", "after_sync_error", "after_plate_error"}
            expected_partial = action == "solver_refusal"
            expected_rejection = action not in controls and not expected_error and not expected_partial
            injected = RuntimeError("Injected application failure")
            if expected_error or expected_partial:
                with ExitStack() as faults:
                    if action == "mid_camera_error":
                        faults.enter_context(patch.object(scene, "_update_diagnostics", side_effect=injected))
                    elif action == "after_sync_error":
                        original = scene.solve_and_apply_sync

                        def fail_after_apply(*args, **kwargs):
                            original(*args, **kwargs)
                            raise injected

                        faults.enter_context(patch.object(scene, "solve_and_apply_sync", side_effect=fail_after_apply))
                    elif action == "after_plate_error":
                        from match_perspective.scene import distortion
                        original = distortion.rebuild_undistorted_plates

                        def fail_after_plates(*args, **kwargs):
                            original(*args, **kwargs)
                            raise injected

                        faults.enter_context(patch.object(distortion, "rebuild_undistorted_plates", side_effect=fail_after_plates))
                    elif expected_partial:
                        refusal = deepcopy(refined.sync_result)
                        refusal.success, refusal.message = False, "Injected numerical refusal"
                        faults.enter_context(patch.object(scene, "solve_and_apply_sync", side_effect=scene.SyncSolveRejected(refusal)))
                    else:
                        faults.enter_context(patch.object(scene, "solve_and_apply_sync", side_effect=injected))
                    status = operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=False)
            elif expected_rejection:
                with patch.object(scene, "ensure_origins_from_ground_landmarks", side_effect=AssertionError("Stale apply prepared origins")) as origin_prepare:
                    status = operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=False)
                if origin_prepare.called:
                    raise AssertionError("Stale lens application attempted preparation")
            else:
                status = operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=False)
            after = applied_state()
            rejected = (status == {"CANCELLED"} and before == after and failure.called
                        and isinstance(failure.call_args.args[1], scene.StaleSyncResult))
            passed = rejected if expected_rejection else status == {"FINISHED"}
            if expected_error:
                passed = (status == {"CANCELLED"} and before == after and failure.called
                          and failure.call_args.args[1] is injected
                          and workspace.lens_refine_progress == 0.0
                          and "restored" in workspace.sync_status)
            if expected_partial:
                passed = passed and before != after and "Injected numerical refusal" in workspace.sync_status
            worst = None
            if action in controls:
                errors = []
                for camera in case["truth"]["cameras"]:
                    points = [p["position"] for p in case["truth"]["checks"] if camera["id"] in p["views"]]
                    obj = bpy.data.objects[camera["id"]].pm_session.camera_object
                    delta = blender_pixels(obj, camera, points)-project(points, camera)[0]
                    errors.append(float(np.sqrt(np.mean(np.sum(delta*delta, axis=1)))))
                worst = max(errors)
                passed = passed and worst <= case["expectation"]["holdout_rmse_px"]
            row = dict(action=action, passed=passed, status=sorted(status), application_changed_state=before != after,
                       worst_withheld_rms_px=worst, message=properties.workspace(context).sync_status)
            results.append(row)
            if not passed:
                violations.append(action)
            (out / (action+"-application.json")).write_text(json.dumps(dict(before=before, after=after, result=row), indent=2)+"\n")
            print(row, flush=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise AssertionError("Generated source was modified during replay")
    (out / "lens-ownership.json").write_text(json.dumps(dict(results=results, violations=violations), indent=2)+"\n")
    if violations:
        raise AssertionError("Lens ownership failed: " + ", ".join(violations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ownership", action="store_true", help="Check Diagnose results after controlled evidence edits")
    parser.add_argument("--lens-ownership", action="store_true", help="Check real lens-result application after controlled edits")
    parser.add_argument("--apply-failures", action="store_true", help="With --lens-ownership, check controlled failures during application")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2)+"\n")
    register_extension()
    if args.ownership and args.lens_ownership:
        parser.error("Choose one job ownership experiment")
    if args.apply_failures and not args.lens_ownership:
        parser.error("--apply-failures requires --lens-ownership")
    case = plane_case("free_tilted") if args.ownership or args.lens_ownership else state_case("fit_only")
    if args.lens_ownership:
        for point in case["request"]["points"]:
            point["known"] = list(case["truth"]["points"][point["id"]])
        for camera in case["request"]["cameras"]:
            camera["fx"] /= .875
            camera["fy"] /= .875
    write_case(case, args.out / "case.json")
    create_scene(case, args.out, False)
    if args.ownership:
        verify_diagnose_ownership(case, args.out)
        return 0
    if args.lens_ownership:
        verify_lens_ownership(case, args.out, apply_failures=args.apply_failures)
        return 0
    from match_perspective import properties
    for share_lens in (True, False):
        properties.workspace(bpy.context).share_lens = share_lens
        folder = args.out / ("same-lens" if share_lens else "per-match")
        folder.mkdir()
        verify_lens_inputs(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
