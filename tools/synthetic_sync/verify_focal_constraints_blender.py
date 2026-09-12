"""Verify point-FOV constraints through real Blender preparation/apply/reopen.

Fresh runs allow one initial Sync and one focal bundle; reserve those calls in
the parent verification budget. Preparation and reopening never solve. Only
generated scenes are saved, and reopened files are checked for unchanged bytes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
from unittest.mock import patch

import bpy
from mathutils import Matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import (
    assert_equivalent, blender_pixels, create_scene, register_extension,
)
from tools.synthetic_sync.focal_constraints import CASE_NAMES, assess, validate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.solver import environment, fingerprint, result_record
from tools.synthetic_sync.verify_point_focal_numerical_blender import (
    check_result_contract, configure_point_mode, product_state,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state():
    from match_perspective import properties

    workspace = properties.workspace(bpy.context)
    mirror = workspace.mirror_object
    return dict(product=product_state(bpy.context),
                mirror_matrix=None if mirror is None else [list(row) for row in mirror.matrix_world],
                mirror_slack=workspace.mirror_slack, plane_slack=workspace.plane_slack,
                groups=[(p.item_id, p.plane_axis, p.plane_group, p.mirror_of_id)
                        for p in workspace.landmarks])


def build(case, out):
    """Adapt the scaffold to the existing native-camera/blank-image builder."""
    from match_perspective import properties

    adapted = deepcopy(case)
    truth = adapted["truth"]
    truth["mesh"] = dict(vertices=list(truth["points"].values()), faces=[])
    truth["checks"] = [dict(id=key, position=point,
                            views=[c["id"] for c in truth["cameras"]])
                       for key, point in truth["holdouts"].items()]
    # The older builder supports only X-normal mirrors. Create our arbitrary
    # rigid Empty after it has populated the same landmark relations.
    adapted["request"]["mirror_plane"] = None
    create_scene(adapted, out, render=False)
    configure_point_mode(bpy.context)
    workspace = properties.workspace(bpy.context)
    workspace.focal_pick_sigma_px = case["pick_sigma_px"]
    if case["request"]["mirror_plane"] is not None:
        origin, normal = np.asarray(case["request"]["mirror_plane"], float)
        normal /= np.linalg.norm(normal)
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = np.cross(normal, axis)
        tangent /= np.linalg.norm(tangent)
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((normal, tangent, np.cross(normal, tangent)))
        matrix[:3, 3] = origin
        mirror = bpy.data.objects.new("Mirror evidence", None)
        bpy.context.scene.collection.objects.link(mirror)
        mirror.matrix_world = Matrix(matrix)
        workspace.mirror_object = mirror
        workspace.mirror_plane = "YZ"
    bpy.context.view_layer.update()
    bpy.context.scene["focal_constraint_case"] = fingerprint(case)


def native_check(case, record, assessment):
    """Compare evaluated native cameras to independent projections and holdouts."""
    from match_perspective import properties, scene

    workspace = properties.workspace(bpy.context)
    roots = {root.name: root for root in properties.iter_match_roots()}
    anchor = case["request"]["anchor_id"]
    truth_anchor = next(c for c in case["truth"]["cameras"] if c["id"] == anchor)
    center = np.asarray(truth_anchor["center"], float)
    point_ids = list(case["truth"]["holdouts"])
    points = np.asarray([case["truth"]["holdouts"][key] for key in point_ids], float)
    if case["expectation"]["scale_gauge"] == "free":
        points = center + assessment["alignment_scale"] * (points - center)
    errors = []
    for camera in case["truth"]["cameras"]:
        fitted = record["cameras"][camera["id"]]
        native = blender_pixels(roots[camera["id"]].pm_session.camera_object, camera, points)
        predicted = project(points, fitted)[0]
        assert np.allclose(native, predicted, atol=0.003, rtol=0), camera["id"]
        wanted = np.asarray([case["truth"]["holdout_pixels"][key][camera["id"]]
                             for key in point_ids])
        errors.extend(np.linalg.norm(native - wanted, axis=1).tolist())
    for item in workspace.landmarks:
        assert item.has_position
        assert np.allclose(item.position, record["landmarks"][item.item_id], atol=1e-5)
        helper = scene.landmark_viewport_object(item)
        assert helper is not None
        assert np.allclose(helper.matrix_world.translation, item.position, atol=1e-5)
    assert np.allclose(roots[anchor].matrix_world, np.eye(4), atol=1e-5)
    native_rmse = float(np.sqrt(np.mean(np.square(errors))))
    assert abs(native_rmse - assessment["withheld_rmse_px"]) < 0.003
    return dict(withheld_rmse_px=native_rmse, withheld_max_px=max(errors))


def check_accuracy(case, assessment):
    """Enforce exact controls; retain separate noisy/reference-bias accuracy flags."""
    flags = []
    if max(abs(v) for v in assessment["focal_relative"].values()) > 0.02:
        flags.append("Focal error exceeds 2%")
    if assessment["withheld_rmse_px"] > 1.0:
        flags.append("Withheld RMS exceeds 1px")
    assessment["accuracy_flags"] = flags
    exact_picks = all(np.allclose(
        [o["u"], o["v"]], case["truth"]["oracle_pixels"][o["landmark_id"]][o["match_id"]],
        rtol=0, atol=1e-10) for o in case["request"]["observations"])
    if exact_picks:
        assert not flags, assessment
    assert np.isfinite(assessment["withheld_rmse_px"]), assessment
    if case["request"]["plane_groups"] and case["request"]["plane_slack"] == 0:
        assert assessment["plane_rms_world"] < 0.001, assessment
    if case["request"]["mirror_pairs"]:
        gap = (assessment["mirror_max_gap_shifted_world"] if case["request"]["mirror_slack"]
               else assessment["mirror_max_gap_world"])
        assert gap < 0.01, assessment


def fresh(case, out, prepare_only):
    from match_perspective import properties, scene
    from match_perspective.core import sync, lens_refine
    from match_perspective.core.sync.request import json_values

    build(case, out)
    before = state()
    prep = scene.prepare_lens_refine(bpy.context)
    assert_equivalent(before, state(), "constraint preparation")
    for name in ("plane_groups", "plane_slack", "mirror_pairs", "mirror_plane", "mirror_slack"):
        assert_equivalent(case["request"][name] or ([] if name.endswith("groups") or name.endswith("pairs") else None),
                          getattr(prep, name) or ([] if name.endswith("groups") or name.endswith("pairs") else None), name)
    write_json(out / "request.json", dict(input_sha256=prep.source_request_sha256,
                                          case_sha256=fingerprint(case),
                                          solver_kwargs=json_values(prep.solver_kwargs())))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "input.blend"), check_existing=False)
    if prepare_only:
        return dict(passed=True, numerical_solves=0)
    sync_calls = bundle_calls = 0
    original_sync = sync.solve_landmark_sync
    original_bundle = lens_refine.fit_independent_focals

    def counted_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        assert sync_calls == 1, "Exceeded one initial Sync"
        return original_sync(*args, **kwargs)

    def counted_bundle(*args, **kwargs):
        nonlocal bundle_calls
        bundle_calls += 1
        assert bundle_calls == 1, "Exceeded one focal bundle"
        return original_bundle(*args, **kwargs)

    started = time.monotonic()
    with patch.object(sync, "solve_landmark_sync", side_effect=counted_sync), \
         patch.object(lens_refine, "fit_independent_focals", side_effect=counted_bundle):
        result = scene.run_lens_refine(prep)
    write_json(out / "decision.json", dict(accepted=result.improved, message=result.message,
                                           refusal=result.refusal_reason, sync_calls=sync_calls,
                                           bundle_calls=bundle_calls, elapsed_s=time.monotonic()-started))
    assert_equivalent(before, state(), "constraint numerical worker")
    check_result_contract(case, prep, result)
    cameras = [dict(c, fx=result.calibrations[c["id"]].intrinsics.fx,
                    fy=result.calibrations[c["id"]].intrinsics.fy) for c in case["request"]["cameras"]]
    record = result_record(result.sync_result, cameras, calibrations=result.calibrations)
    record["focal_intervals"] = json_values(result.focal_intervals)
    record["joint_result"] = json_values(result.sync_result)
    write_json(out / "record.json", record)
    assessment = assess(case, record)
    write_json(out / "assessment.json", assessment)
    check_accuracy(case, assessment)
    # A live edit to constraint settings must invalidate an otherwise valid job.
    workspace = properties.workspace(bpy.context)
    previous = workspace.plane_slack
    workspace.plane_slack = previous + 0.01
    edited = state()
    try:
        scene.apply_lens_refine_result(bpy.context, result, prep)
    except scene.StaleSyncResult:
        pass
    else:
        raise AssertionError("Edited constraint accepted a stale result")
    assert_equivalent(edited, state(), "stale constraint refusal")
    workspace.plane_slack = previous
    mirror_before = state()["mirror_matrix"]
    with patch.object(scene, "solve_and_apply_sync", side_effect=AssertionError("Joint fit re-solved")):
        _, applied = scene.apply_lens_refine_result(bpy.context, result, prep)
    assert applied is result.sync_result
    if prep.plane_groups or prep.mirror_pairs:
        assert "fixed anchor frame and supplied constraints" in workspace.sync_status
    assert_equivalent(mirror_before, state()["mirror_matrix"], "Mirror Empty stayed put")
    # Preserve applied generated state even if the independent checker fails,
    # so a checker correction can use --reopen without another numerical fit.
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "solved.blend"), check_existing=False)
    write_json(out / "solved-state.json", state())
    assessment["native_blender"] = native_check(case, record, assessment)
    assessment["passed"] = True
    write_json(out / "assessment.json", assessment)
    return assessment


def reopen(case, out):
    saved = out / "solved.blend"
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(saved.resolve()), load_ui=False)
    assert bpy.context.scene["focal_constraint_case"] == fingerprint(case)
    assert_equivalent(json.loads((out / "solved-state.json").read_text()), state(), "reopened state")
    record = json.loads((out / "record.json").read_text())
    assessment = assess(case, record)
    check_accuracy(case, assessment)
    assessment["native_blender"] = native_check(case, record, assessment)
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    assessment["passed"] = True
    write_json(out / "reopened-assessment.json", assessment)
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASE_NAMES)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--reopen", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = args.out.resolve()
    if not args.reopen:
        out.mkdir(parents=True, exist_ok=False)
        source = ROOT / "tools/synthetic_sync/cases/focal-constraints" / (args.case + ".json")
        (out / "case.json").write_bytes(source.read_bytes())
        write_json(out / "environment.json", environment(ROOT))
        with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
            for folder in ("core", "scene", "properties", "tools/synthetic_sync"):
                for path in sorted((ROOT / folder).glob("**/*.py")):
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    case = json.loads((out / "case.json").read_text())
    validate(case)
    register_extension()
    result = reopen(case, out) if args.reopen else fresh(case, out, args.prepare_only)
    assert result["passed"]
    print(f"Constrained point FOV Blender PASS: {args.case}", flush=True)


if __name__ == "__main__":
    main()
