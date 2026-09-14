"""Check point-FOV modal progress against Blender RNA without numerical solves.

Run with Blender --factory-startup -b --python tools/synthetic_sync/verify_lens_progress.py.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import register_extension


class WindowManager:
    def __init__(self):
        self.calls = []
        self.windows = ()

    def progress_begin(self, *args):
        self.calls.append(("begin", args))

    def progress_update(self, value):
        self.calls.append(("update", value))

    def progress_end(self):
        self.calls.append(("end",))

    def event_timer_add(self, *_args, **_kwargs):
        return object()

    def modal_handler_add(self, *_args):
        pass

    def event_timer_remove(self, *_args):
        pass


class Context:
    def __init__(self):
        self.window_manager = WindowManager()

    def __getattr__(self, name):
        return getattr(bpy.context, name)


class DeferredThread:
    pending = None

    def __init__(self, **kwargs):
        self.target = kwargs["target"]
        DeferredThread.pending = self

    def start(self):
        pass


register_extension()
from match_perspective import properties, scene
from match_perspective.ui import operators
from match_perspective.core import lens_refine

context = Context()
workspace = properties.workspace(context)
operator = SimpleNamespace(_timer=None, report=lambda *_args: None)
prep = SimpleNamespace(estimate_focal_from_points=True)
with patch.object(scene, "prepare_lens_refine", return_value=prep), patch.object(
    operators.threading, "Thread", DeferredThread
):
    assert operators.PM_OT_refine_lenses.invoke(operator, context, None) == {"RUNNING_MODAL"}

event = SimpleNamespace(type="TIMER", value="NOTHING")
try:
    checks = (
        (0, 101, "Registering cameras: testing bridges", "testing bridges"),
        (0, 101, "Preparing provisional cameras", "provisional cameras"),
        (1, 101, "Estimating focal from points", "Estimating focal from points"),
        (3, 101, "Estimating focal from points", "2 iterations"),
        (101, 101, "Point focal estimation complete", "complete"),
    )

    def run(_prep, *, progress_callback, **_kwargs):
        for step, total, label, expected in checks:
            progress_callback(step, total, label)
            assert operators.PM_OT_refine_lenses.modal(operator, context, event) == {"PASS_THROUGH"}
            activity = operators.lens_refine_startup_label()
            assert expected in activity, (expected, activity)
            assert expected in workspace.sync_status, (expected, workspace.sync_status)
            assert workspace.lens_refine_progress == 0.0
        return SimpleNamespace(cancelled=True, message="Cancelled")

    with patch.object(scene, "run_lens_refine", side_effect=run):
        # The deferred worker uses the same callback closure as a live job.
        DeferredThread.pending.target()
    assert not context.window_manager.calls, context.window_manager.calls
finally:
    operator._result_box["result"] = SimpleNamespace(cancelled=True, message="Cancelled")
    operator._result_box["done"] = True
    operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=True)

print("Point-FOV Blender progress PASS", flush=True)

# The ordinary focal search still uses a determinate bar. Its worker may
# report a different total from the initial estimate, so normalize updates.
context.window_manager.calls.clear()
operator = SimpleNamespace(_timer=None, report=lambda *_args: None)
prep = SimpleNamespace(estimate_focal_from_points=False, share_lens=True,
                       lens_inputs=[SimpleNamespace()])
with patch.object(scene, "prepare_lens_refine", return_value=prep), patch.object(
    lens_refine, "estimate_refine_evaluation_count", return_value=20
), patch.object(operators.threading, "Thread", DeferredThread):
    assert operators.PM_OT_refine_lenses.invoke(operator, context, None) == {"RUNNING_MODAL"}

try:
    def run_search(_prep, *, progress_callback, **_kwargs):
        for step, expected in ((0, 0.0), (10, 5.0), (40, 20.0)):
            progress_callback(step, 40, "Scoring lenses")
            assert operators.PM_OT_refine_lenses.modal(operator, context, event) == {"PASS_THROUGH"}
            assert context.window_manager.calls[-1] == ("update", expected)
            assert workspace.lens_refine_progress == step / 40
        return SimpleNamespace(cancelled=True, message="Cancelled")

    with patch.object(scene, "run_lens_refine", side_effect=run_search):
        DeferredThread.pending.target()
finally:
    operators.PM_OT_refine_lenses._finish_job(operator, context, cancelled=True)

assert context.window_manager.calls == [
    ("begin", (0, 20)), ("update", 0.0), ("update", 5.0),
    ("update", 20.0), ("end",)
], context.window_manager.calls
print("Determinate Blender progress PASS", flush=True)

# Diagnose used to keep WindowManager progress at 0/6 during registration.
context.window_manager.calls.clear()
operator = SimpleNamespace(_timer=None, report=lambda *_args: None)
with patch.object(scene, "prepare_diagnose_sync", return_value=SimpleNamespace()), patch.object(
    operators.threading, "Thread", DeferredThread
):
    assert operators.PM_OT_diagnose_sync.invoke(operator, context, None) == {"RUNNING_MODAL"}

try:
    checks = (
        (0, 6, "Registering cameras: testing bridges", "testing bridges"),
        (0, 6, "Bundle adjustment", "Bundle adjustment"),
        (2, 6, "Checking Landmark A", "2/6"),
        (6, 6, "Complete", "Complete"),
    )

    def run_diagnose(_prep, *, progress_callback, **_kwargs):
        for step, total, label, expected in checks:
            progress_callback(step, total, label)
            assert operators.PM_OT_diagnose_sync.modal(operator, context, event) == {"PASS_THROUGH"}
            activity = operators.diagnose_sync_activity_label()
            assert expected in activity, (expected, activity)
            assert f"{expected}" in workspace.sync_status
            assert "elapsed" in workspace.sync_status
        return SimpleNamespace()

    with patch.object(scene, "run_diagnose_sync", side_effect=run_diagnose):
        DeferredThread.pending.target()
    assert not context.window_manager.calls, context.window_manager.calls
finally:
    operator._result_box["cancelled"] = True
    operator._result_box["done"] = True
    operators.PM_OT_diagnose_sync._finish_job(operator, context, cancelled=True)

assert not context.window_manager.calls, context.window_manager.calls
print("Diagnose Blender progress PASS", flush=True)

context.window_manager.calls.clear()
operator = SimpleNamespace(_timer=None, report=lambda *_args: None)
with patch.object(scene, "prepare_diagnose_sync", return_value=SimpleNamespace()), patch.object(
    operators.threading, "Thread", DeferredThread
):
    assert operators.PM_OT_solve_sync.invoke(operator, context, None) == {"RUNNING_MODAL"}

try:
    checks = (
        "Registering cameras: testing bridges",
        "Registering cameras: fitting next camera (3/8 posed)",
        "Bundle adjustment",
        "Retrying skipped cameras",
    )

    def run_solve(_prep, *, progress_callback, **_kwargs):
        for label in checks:
            progress_callback(label)
            assert operators.PM_OT_solve_sync.modal(operator, context, event) == {"PASS_THROUGH"}
            activity = operators.solve_sync_activity_label()
            assert label in activity, (label, activity)
            assert label in workspace.sync_status
            assert "elapsed" in workspace.sync_status
        return SimpleNamespace(success=True, message="Captured")

    with patch.object(scene, "run_solve_sync", side_effect=run_solve):
        DeferredThread.pending.target()
    assert not context.window_manager.calls, context.window_manager.calls
finally:
    operator._result_box["cancelled"] = True
    operator._result_box["done"] = True
    operators.PM_OT_solve_sync._finish_job(operator, context, cancelled=True)

assert not context.window_manager.calls, context.window_manager.calls
print("Solve Sync Blender progress PASS", flush=True)
