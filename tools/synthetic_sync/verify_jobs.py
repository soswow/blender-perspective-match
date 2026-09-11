"""Compare blocking and background job inputs through generated Blender state."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import assert_equivalent, create_scene, register_extension
from tools.synthetic_sync.scenarios import write_case
from tools.synthetic_sync.solver import environment
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
            patch.object(operators.threading, "Thread", InlineWorker):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2)+"\n")
    register_extension()
    case = state_case("fit_only")
    write_case(case, args.out / "case.json")
    create_scene(case, args.out, False)
    from match_perspective import properties
    for share_lens in (True, False):
        properties.workspace(bpy.context).share_lens = share_lens
        folder = args.out / ("same-lens" if share_lens else "per-match")
        folder.mkdir()
        verify_lens_inputs(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
