"""Check job ownership across a real generated-file reload and a new invocation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
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
from tools.synthetic_sync.verify_jobs import HeadlessContext, InlineWorker


def verify(case, out):
    from match_perspective import properties, scene
    from match_perspective.core.sync.request import json_values
    from match_perspective.ui import operators

    source = out / "input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(source), check_existing=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    results = []
    for label, cls, prepare, run_name, prefix in (
        ("diagnose", operators.PM_OT_diagnose_sync, scene.prepare_diagnose_sync, "run_diagnose_sync", "_diagnose_sync"),
        ("solve", operators.PM_OT_solve_sync, scene.prepare_diagnose_sync, "run_solve_sync", "_solve_sync"),
        ("lens", operators.PM_OT_refine_lenses, scene.prepare_lens_refine, "run_lens_refine", "_lens_refine"),
    ):
        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
        prep = prepare(bpy.context)
        wanted = json_values(prep.solver_kwargs())
        result = getattr(scene, run_name)(prep)
        numerical = result if label != "lens" else result.sync_result
        assessment = evaluate(case, result_record(numerical, case["request"]["cameras"]))
        if not assessment["passed"]:
            raise AssertionError(assessment["violations"])
        (out / (label+"-numerical.json")).write_text(json.dumps(assessment, indent=2)+"\n")
        for action in ("unchanged", "reload_pending", "reload_completed", "late_cancel", "late_modal", "cancel_pending"):
            bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
            callbacks = []

            class DeferredWorker(InlineWorker):
                def start(self):
                    callbacks.append(self.target)

            def completed(prepared, **_kwargs):
                assert_equivalent(wanted, json_values(prepared.solver_kwargs()), "reloaded job input")
                # A worker can finish before seeing cancellation. Return the real
                # completed numerical result to exercise that worst interleaving.
                return deepcopy(result)

            context = HeadlessContext()

            def operator():
                item = SimpleNamespace(_timer=None, report=lambda *_: None)
                item._finish_job = lambda ctx, **kw: cls._finish_job(item, ctx, **kw)
                return item

            old = operator()
            with patch.object(operators, "threading", SimpleNamespace(Thread=DeferredWorker, Event=threading.Event)), \
                    patch.object(scene, run_name, side_effect=completed), \
                    patch.object(operators, "_write_diagnose_report", return_value=(SimpleNamespace(severity="success"), None, False)) as publish, \
                    patch.object(operators.sync_report, "compact_status", return_value="Captured report"):
                assert cls.invoke(old, context, None) == {"RUNNING_MODAL"}
                old_cancel = getattr(operators, prefix+"_cancel")
                if action in {"unchanged", "reload_completed"}:
                    callbacks[0]()
                row = dict(job=label, action=action)
                if action == "unchanged":
                    current = old
                else:
                    if action == "cancel_pending":
                        cls.cancel(old, context)
                    else:
                        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
                    row.update(old_cancelled=old_cancel.is_set(), running_after_retirement=getattr(operators, prefix+"_running"),
                               can_start=cls.poll(context))
                    if not row["can_start"]:
                        # Retain the failing evidence and clean up this test's old
                        # job so another control can run on the same checkout.
                        callbacks[0]()
                        cls._finish_job(old, context, cancelled=True)
                        row["passed"] = False
                        results.append(row)
                        print(row, flush=True)
                        continue
                    current = operator()
                    assert cls.invoke(current, context, None) == {"RUNNING_MODAL"}
                    current_box = getattr(operators, prefix+"_result_box")
                    current_cancel = getattr(operators, prefix+"_cancel")
                    before_status = properties.workspace(context).sync_status
                    if action != "reload_completed":
                        callbacks[0]()
                    if action == "late_cancel":
                        try:
                            cls.cancel(old, context)
                        except Exception as error:
                            row["old_cancel_error"] = f"{type(error).__name__}: {error}"
                    elif action == "late_modal":
                        cls.modal(old, context, SimpleNamespace(type="TIMER", value="NOTHING"))
                    else:
                        cls._finish_job(old, context, cancelled=True)
                    row["new_job_untouched"] = (
                        getattr(operators, prefix+"_result_box") is current_box
                        and getattr(operators, prefix+"_running")
                        and getattr(operators, prefix+"_cancel") is current_cancel
                        and not current_cancel.is_set() and not publish.called
                        and before_status == properties.workspace(context).sync_status
                    )
                    callbacks[1]()
                status = cls.modal(current, context, SimpleNamespace(type="TIMER", value="NOTHING"))
                row["status"] = sorted(status)
                row["finished"] = status == {"FINISHED"}
                row["passed"] = row["finished"] and (action == "unchanged" or (
                    row["old_cancelled"] and not row["running_after_retirement"] and row["new_job_untouched"]
                    and "old_cancel_error" not in row))
                if label in {"lens", "solve"} and row["finished"]:
                    errors = []
                    for camera in case["truth"]["cameras"]:
                        points = [p["position"] for p in case["truth"]["checks"] if camera["id"] in p["views"]]
                        obj = bpy.data.objects[camera["id"]].pm_session.camera_object
                        delta = blender_pixels(obj, camera, points)-project(points, camera)[0]
                        errors.append(float(np.sqrt(np.mean(np.sum(delta*delta, axis=1)))))
                    row["worst_withheld_rms_px"] = max(errors)
                    row["passed"] = row["passed"] and max(errors) <= case["expectation"]["holdout_rmse_px"]
                if label == "diagnose":
                    row["passed"] = row["passed"] and publish.call_count == 1
                results.append(row)
                print(row, flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    (out / "reload.json").write_text(json.dumps(results, indent=2)+"\n")
    failures = [row["job"]+":"+row["action"] for row in results if not row["passed"]]
    if failures:
        raise AssertionError("Job reload failed: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2)+"\n")
    register_extension()
    case = plane_case("free_tilted")
    for point in case["request"]["points"]:
        point["known"] = list(case["truth"]["points"][point["id"]])
    write_case(case, args.out / "case.json")
    create_scene(case, args.out, False)
    verify(case, args.out)


if __name__ == "__main__":
    main()
