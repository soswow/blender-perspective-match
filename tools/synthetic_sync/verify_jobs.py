"""Compare blocking and background job inputs through generated Blender state."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import assert_equivalent, create_scene, register_extension
from tools.synthetic_sync.evaluation import evaluate
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
    expected = {key: value for key, value in vars(prep).items() if key != "root_by_name"}
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
                with patch.object(scene, "ensure_ground_frame_from_landmarks", side_effect=AssertionError("Apply ran ground preparation")), \
                        patch.object(scene, "ensure_origins_from_ground_landmarks", side_effect=AssertionError("Apply ran origin preparation")):
                    status = operators.PM_OT_diagnose_sync._finish_job(operator, context, cancelled=False)
                rejected = status == {"CANCELLED"} and not publish.called and landmark.rmse_px == 73.0
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ownership", action="store_true", help="Check Diagnose results after controlled evidence edits")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2)+"\n")
    register_extension()
    case = plane_case("free_tilted") if args.ownership else state_case("fit_only")
    write_case(case, args.out / "case.json")
    create_scene(case, args.out, False)
    if args.ownership:
        verify_diagnose_ownership(case, args.out)
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
