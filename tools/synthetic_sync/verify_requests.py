"""Prove product/probe/captured-request parity through generated Blender state."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import fields
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/debug-sync")]
from tools.synthetic_sync.blender_case import assert_equivalent, create_scene, register_extension, run_case
from tools.synthetic_sync.constraints import constraint_case
from tools.synthetic_sync.scenarios import generate, write_case
from tools.sync_snapshot import write_snapshot


def state_case(name):
    if name == "auto_origin":
        return generate("ground")
    case = constraint_case("mirror_points")
    request = case["request"]
    request["fixed_similarities"] = generate("locked_bridge")["request"]["fixed_similarities"]
    request.update(lock_rotation=name == "global_locks", lock_translation=name == "global_locks",
                   ground_slack=0.08, known_3d_slack=0.05, mirror_slack=0.04)
    request["mirror_plane"][0][0] = 0.02
    if name == "fit_only":
        request["location_match_ids"] = ["view_0", "view_1"]
        request["readonly_match_ids"] = ["view_2"]
    for point in request["points"][:3]:
        point["known"] = list(case["truth"]["points"][point["id"]])
        point["known"][2] += 0.03  # Imperfect CAD: slack actually has a role.
    return case


def verify_state(name, out):
    from match_perspective import properties, scene
    from match_perspective.core import sync
    from match_perspective.core.sync.request import SyncSolveRequest, json_values
    out.mkdir(parents=True, exist_ok=False)
    case = state_case(name)
    write_case(case, out / "case.json")
    create_scene(case, out, False)
    if name == "constraints":
        # A programmer error must not pass the harness as an intentional refusal.
        rejected_case = deepcopy(case)
        rejected_case["expectation"]["outcome"] = "reject"
        fault_out = out / "exception-contract"
        fault_out.mkdir()
        with patch.object(scene, "solve_and_apply_sync", side_effect=ValueError("Injected implementation error")):
            try:
                run_case(rejected_case, fault_out)
            except ValueError as error:
                assert str(error) == "Injected implementation error"
            else:
                raise AssertionError("Harness accepted an implementation error as a useful refusal")
    workspace = properties.workspace(bpy.context)
    if name == "auto_origin":
        for root in properties.iter_match_roots():
            if root.get("synthetic_match_id") == "view_2":
                root.pm_session.origin_is_set = False
    else:
        workspace.landmarks[0].observations[0].confidence = "HIGH"
        workspace.landmarks[1].sync_weight = 2.5
    source = out / "input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(source), check_existing=False)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = scene.prepare_diagnose_sync(bpy.context)
    snapshot = write_snapshot(expected, out / "request.json", source_name=source.name)
    if name == "auto_origin" and not expected.auto_origin_notes:
        raise AssertionError("Case did not exercise automatic origin preparation")
    if name != "auto_origin":
        assert expected.fixed_similarities
        assert expected.lock_rotation == (name == "global_locks")
        assert expected.lock_translation == (name == "global_locks")
        assert any(o.weight == 4 and not o.protect_outlier for o in expected.observations)
        assert any(o.protect_outlier for o in expected.observations)
        assert expected.readonly_match_ids == ({"view_2"} if name == "fit_only" else set())
        assert expected.location_match_ids == ({"view_0", "view_1"} if name == "fit_only" else {"view_0", "view_1", "view_2"})
    reference_request = SyncSolveRequest.from_record(snapshot)
    reference = sync.solve_landmark_sync(**reference_request.solver_kwargs(), use_pose_cache=False)
    reference_values = json_values(reference)
    (out / "reference-result.json").write_text(json.dumps(reference_values, indent=2, allow_nan=False)+"\n")
    expected_values = snapshot["inputs"]
    actual_solve, actual_leave = sync.solve_landmark_sync, sync.leave_one_out_landmark_report
    calls, errors, results = [], [], []
    request_fields = {item.name for item in fields(SyncSolveRequest)}

    def check(arguments):
        values = {key: value for key, value in arguments.items() if key in request_fields}
        request = SyncSolveRequest(**values)
        found = request.to_record()["inputs"]
        try:
            assert_equivalent(expected_values, found)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            raise
        return request

    def checked_solve(*args, **kwargs):
        arguments = inspect.signature(actual_solve).bind(*args, **kwargs)
        arguments.apply_defaults()
        calls.append("solve")
        check(arguments.arguments)
        result = actual_solve(*args, **kwargs)
        values = json_values(result)
        for key in ("success", "similarities", "landmarks", "line_segments", "per_match_rmse_px"):
            assert_equivalent(reference_values[key], values[key], "result."+key)
        results.append(values)
        return result

    def checked_leave(*args, **kwargs):
        arguments = inspect.signature(actual_leave).bind(*args, **kwargs)
        arguments.apply_defaults()
        calls.append("leave_one_out")
        check(arguments.arguments)
        return []  # Verify forwarding; numerical leave-one-out already has its own suite.

    sync.solve_landmark_sync, sync.leave_one_out_landmark_report = checked_solve, checked_leave
    try:
        for operation in ("solve", "diagnose", "probe_graph", "probe_resected", "dump_sync"):
            calls.clear()
            bpy.ops.wm.open_mainfile(filepath=str(source))
            with (out / (operation+".log")).open("w") as stream, redirect_stdout(stream):
                if operation == "solve":
                    try:
                        scene.solve_and_apply_sync(bpy.context)
                    except scene.SyncSolveRejected as error:
                        if reference.success or not results or reference.message not in str(error):
                            raise
                elif operation == "diagnose":
                    scene.run_diagnose_sync(scene.prepare_diagnose_sync(bpy.context))
                else:
                    probe = importlib.import_module(operation)
                    probe._load_extension = lambda: sys.modules["match_perspective"]
                    arguments = ["--blend", str(source)]
                    if operation == "probe_graph":
                        arguments += ["--leave-one-out"]
                    elif operation == "probe_resected":
                        arguments += ["--match", expected.matches[1].match_id]
                    code = probe.main(arguments)
                    assert code == 0, f"{operation}: exit {code}"
            assert not errors, f"{operation}: {errors}"
            assert "solve" in calls, f"{operation}: did not call Sync"
            if operation == "probe_graph":
                assert "leave_one_out" in calls
            print(f"PASS {name}: {operation} request and result parity", flush=True)
    finally:
        sync.solve_landmark_sync, sync.leave_one_out_landmark_report = actual_solve, actual_leave
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash, "Probe modified source .blend"
    (out / "parity.json").write_text(json.dumps(dict(passed=True, source_sha256=before_hash,
        request_sha256=snapshot["sha256"], compared_solves=len(results)), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", choices=("constraints", "global_locks", "auto_origin", "fit_only", "all"), default="all")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    if args.state == "all":
        for name in ("constraints", "global_locks", "auto_origin", "fit_only"):
            subprocess.run([bpy.app.binary_path, "--factory-startup", "--disable-autoexec", "-b",
                "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
                "--out", str(args.out.resolve()), "--state", name], check=True)
    else:
        register_extension()
        verify_state(args.state, args.out.resolve() / args.state)


if __name__ == "__main__":
    main()
