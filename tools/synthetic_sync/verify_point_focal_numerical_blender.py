"""Exercise real point-FOV fit and Blender apply on frozen generated cases.

Run each case in a fresh factory-startup Blender process. ``--reopen`` reads a
previously generated solved.blend and never calls either numerical solver.
This tool has no experiment ledger: reserve its one Sync and one point-bundle
call per fresh case in the parent experiment budget before running it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import (
    assert_equivalent, blender_pixels, create_scene, register_extension,
)
from tools.synthetic_sync.evaluation import alignment
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import environment, fingerprint, result_record


CASES = {
    "no-vp-mixed-guessedK": "accept",
    "no-vp-shared-guessedK": "accept",
    "no-vp-weak-baseline-guessedK": "refuse",
    "no-vp-pure-rotation-guessedK": "refuse",
}


def product_state(context) -> dict:
    """Capture the camera, plate, root, and landmark values an apply may change."""
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
            origin_is_set=root.pm_session.origin_is_set,
            origin_image=list(root.pm_session.origin_image),
            camera_lens=root.pm_session.camera_object.data.lens,
            camera_matrix=[list(row) for row in root.pm_session.camera_object.matrix_local],
            view_undistorted=root.pm_session.view_undistorted,
            undistorted_image=(root.pm_session.undistorted_image.name
                               if root.pm_session.undistorted_image else None),
            undistorted_path=root.pm_session.undistorted_path,
            undistorted_width=root.pm_session.undistorted_width,
            undistorted_height=root.pm_session.undistorted_height,
            undistorted_offset_x=root.pm_session.undistorted_offset_x,
            undistorted_offset_y=root.pm_session.undistorted_offset_y,
        ) for root in properties.iter_match_roots()},
        landmarks={item.item_id: dict(
            has_position=item.has_position,
            position=list(item.position),
            has_line_segment=item.has_line_segment,
            position_b=list(item.position_b),
            rmse_px=item.rmse_px,
        ) for item in space.landmarks},
        scene_camera=context.scene.camera.name if context.scene.camera else None,
        render_size=(context.scene.render.resolution_x,
                     context.scene.render.resolution_y,
                     context.scene.render.resolution_percentage),
        images={image.name: image.use_fake_user for image in bpy.data.images},
    )


def configure_point_mode(context) -> None:
    from match_perspective import properties

    space = properties.workspace(context)
    space.share_lens = False
    space.estimate_focal_from_points = True
    assert abs(space.point_focal_span_percent - 40.0) < 1e-6
    assert abs(space.focal_pick_sigma_px - 1.0) < 1e-6
    for root in properties.iter_match_roots():
        settings = root.pm_session
        settings.lines.clear()
        settings.origin_is_set = False
        assert settings.camera_control == "MATCHED"


def check_prepared(case, prep) -> None:
    """Confirm RNA gives the intended frozen input without a truth pose leak."""
    wanted = {camera["id"]: camera for camera in case["request"]["cameras"]}
    assert prep.estimate_focal_from_points and not prep.share_lens
    assert abs(prep.fx_span - 0.4) < 1e-6 and abs(prep.pick_sigma_px - 1.0) < 1e-6
    assert set(wanted) == {item.match_id for item in prep.lens_inputs}
    assert not prep.known_world and not prep.line_observations and not prep.known_lines
    assert not prep.fixed_similarities and not prep.plane_groups
    assert not prep.mirror_pairs and not prep.parallel_pairs
    assert not prep.readonly_match_ids
    for item in prep.lens_inputs:
        stored = wanted[item.match_id]
        calibration = item.base_calibration
        k = calibration.intrinsics
        assert not item.freeze_focal and not item.reorient_from_vp
        assert all(not lines for lines in item.line_bundles.values())
        assert item.origin_image is None
        assert np.allclose([k.fx, k.fy, k.cx, k.cy],
                           [stored[key] for key in ("fx", "fy", "cx", "cy")],
                           atol=1e-5, rtol=2e-6)
        assert np.allclose(calibration.camera_center, stored["center"], atol=2e-6)
        assert np.allclose(calibration.rotation_w2c, stored["rotation"], atol=2e-6)


def make_record(case, result) -> dict:
    """Resolve fitted root similarities through the unchanged private poses."""
    cameras = []
    for stored in case["request"]["cameras"]:
        calibration = result.calibrations[stored["id"]]
        cameras.append(dict(stored, fx=float(calibration.intrinsics.fx),
                            fy=float(calibration.intrinsics.fy)))
    return result_record(result.sync_result, cameras)


def check_result_contract(case, prep, result) -> None:
    assert result.point_focal_mode and not result.cancelled
    assert result.improved and not result.refusal_reason
    assert result.sync_result is not None and result.sync_result.success
    assert set(result.calibrations) == {item.match_id for item in prep.lens_inputs}
    assert set(result.focal_intervals) == set(result.calibrations)
    originals = {item.match_id: item.base_calibration for item in prep.lens_inputs}
    for match_id, calibration in result.calibrations.items():
        old = originals[match_id]
        k = calibration.intrinsics
        low, high = result.focal_intervals[match_id]
        assert np.isfinite([k.fx, k.fy, low, high]).all()
        assert k.fx > 0 and abs(k.fx - k.fy) < 1e-5
        assert 0 < low <= k.fx <= high
        assert np.allclose(calibration.camera_center, old.camera_center, atol=1e-8)
        assert np.allclose(calibration.rotation_w2c, old.rotation_w2c, atol=1e-8)
        assert k.cx == old.intrinsics.cx and k.cy == old.intrinsics.cy
        assert k.image_width == old.intrinsics.image_width
        assert k.image_height == old.intrinsics.image_height


def check_blender_application(case, record, *, expect_state=None) -> dict:
    """Check native evaluated cameras against independent withheld projections."""
    from match_perspective import properties, scene

    space = properties.workspace(bpy.context)
    for item_id, point in record["landmarks"].items():
        landmark = next((item for item in space.landmarks if item.item_id == item_id), None)
        assert landmark is not None and landmark.has_position
        assert np.allclose(landmark.position, point, atol=1e-5, rtol=1e-6)
        helper = scene.landmark_viewport_object(landmark)
        assert helper is not None
        assert np.allclose(helper.matrix_world.translation, point, atol=1e-5, rtol=1e-6)
    transform = alignment(case, record)
    scale, rotation, translation = transform
    native = {}
    for truth in case["truth"]["cameras"]:
        match_id = truth["id"]
        root = next(root for root in properties.iter_match_roots() if root.name == match_id)
        points = np.asarray([item["position"] for item in case["truth"]["checks"]
                             if match_id in item["views"]], dtype=float)
        assert len(points) >= 6
        local = ((points - translation) @ rotation) / scale
        scene.set_active_match(bpy.context, root, record_history=False)
        actual_uv = blender_pixels(root.pm_session.camera_object, truth, local)
        expected_uv, _depth = project(points, truth)
        errors = np.linalg.norm(actual_uv - expected_uv, axis=1)
        rmse = float(np.sqrt(np.mean(errors**2)))
        native[match_id] = dict(holdout_rmse_px=rmse,
                                max_error_px=float(errors.max()), count=len(points))
        assert np.isfinite(errors).all()
        assert rmse <= case["expectation"]["holdout_rmse_px"]
    if expect_state is not None:
        assert_equivalent(expect_state, product_state(bpy.context), "reopened state")
    return native


def run_fresh(case, out: Path, *, prepare_only: bool = False) -> dict:
    from match_perspective import properties, scene
    from match_perspective.core import sync
    from match_perspective.core.sync.request import json_values

    create_scene(case, out, False)
    configure_point_mode(bpy.context)
    bpy.context.scene["point_focal_case_sha256"] = fingerprint(case["request"])
    before_prep = product_state(bpy.context)
    prep = scene.prepare_lens_refine(bpy.context)
    assert_equivalent(before_prep, product_state(bpy.context), "read-only point preparation")
    check_prepared(case, prep)
    (out / "request.json").write_text(json.dumps(dict(
        solver_kwargs=json_values(prep.solver_kwargs()),
        source_request_sha256=prep.source_request_sha256,
        frozen_case_sha256=hashlib.sha256((out / "case.json").read_bytes()).hexdigest(),
    ), indent=2, allow_nan=False) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "input.blend"), check_existing=False)
    if prepare_only:
        report = dict(passed=True, classification="prepared_only",
                      numerical_solves=0, numerical_bundles=0)
        (out / "prepared-only.json").write_text(
            json.dumps(report, indent=2) + "\n")
        return report
    before = product_state(bpy.context)
    sync_calls = 0
    actual_sync = sync.solve_landmark_sync

    def counted_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls > 1:
            raise AssertionError("Point job exceeded its one initial Sync call")
        return actual_sync(*args, **kwargs)

    started = time.monotonic()
    with patch.object(sync, "solve_landmark_sync", side_effect=counted_sync):
        result = scene.run_lens_refine(prep)
    elapsed = time.monotonic() - started
    summary = dict(
        point_focal_mode=result.point_focal_mode, improved=result.improved,
        message=result.message, refusal_reason=result.refusal_reason,
        focal_intervals=json_values(result.focal_intervals),
        fitted_calibrations=json_values(result.calibrations),
        sync_calls=sync_calls, elapsed_s=elapsed,
    )
    (out / "numerical-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    assert sync_calls <= 1, f"Point job attempted {sync_calls} Sync calls"
    assert_equivalent(before, product_state(bpy.context), "numerical worker isolation")
    assert result.point_focal_mode and not result.cancelled

    def write_evidence(record, assessment):
        (out / "record.json").write_text(json.dumps(dict(
            result=record, **summary,
        ), indent=2, allow_nan=False) + "\n")
        (out / "assessment.json").write_text(
            json.dumps(assessment, indent=2, allow_nan=False) + "\n")

    expected = CASES[case["name"]]
    if expected == "refuse":
        record = dict(success=False, message=result.message or result.refusal_reason, cameras={},
                      landmarks={}, line_segments={}, reported_rmse_px=None)
        geometry = assess(case, record)
        assessment = dict(passed=False, classification=geometry["classification"],
                          numerical_geometry=geometry, unchanged_state=False)
        write_evidence(record, assessment)
        assert not result.improved and result.refusal_reason
        with patch.object(scene, "solve_and_apply_sync",
                          side_effect=AssertionError("Refusal ran Solve Sync")):
            _, applied = scene.apply_lens_refine_result(bpy.context, result, prep)
        assert applied is None
        assert_equivalent(before, product_state(bpy.context), "refused apply")
        status = properties.workspace(bpy.context).sync_status
        assert status.strip() and status == (result.message or result.refusal_reason)
        assessment["passed"] = geometry["classification"] == "useful_refusal"
        assessment["unchanged_state"] = True
    else:
        record = make_record(case, result)
        assessment = assess(case, record)
        assessment["passed"] = False  # Native Blender apply has not been checked yet.
        write_evidence(record, assessment)
        assert sync_calls == 1, f"Expected one initial Sync, got {sync_calls}"
        check_result_contract(case, prep, result)
        assert assessment["classification"] == "accurate_acceptance", assessment
        with patch.object(scene, "solve_and_apply_sync",
                          side_effect=AssertionError("Point apply reran Solve Sync")):
            _, applied = scene.apply_lens_refine_result(bpy.context, result, prep)
        assert applied is result.sync_result
        assert "approximate local 95%" in properties.workspace(bpy.context).sync_status
        assessment["native_blender"] = check_blender_application(case, record)
        assessment["passed"] = bool(assessment["independent_geometry"]["passed"]
                                     and assessment["focal_within_limit"])
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "solved.blend"), check_existing=False)
        (out / "solved-state.json").write_text(
            json.dumps(product_state(bpy.context), indent=2, allow_nan=False) + "\n")
    write_evidence(record, assessment)
    assert assessment["passed"], assessment
    return assessment


def run_reopen(case, out: Path) -> dict:
    assert CASES[case["name"]] == "accept"
    saved = out / "solved.blend"
    assert saved.is_file()
    recorded_case_sha = json.loads((out / "request.json").read_text())[
        "frozen_case_sha256"]
    assert hashlib.sha256((out / "case.json").read_bytes()).hexdigest() == recorded_case_sha
    before_sha = hashlib.sha256(saved.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(saved.resolve()), load_ui=False)
    assert bpy.context.scene.get("point_focal_case_sha256") == fingerprint(case["request"])
    record = json.loads((out / "record.json").read_text())["result"]
    expected_state = json.loads((out / "solved-state.json").read_text())
    assessment = assess(case, record)
    assert assessment["classification"] == "accurate_acceptance", assessment
    assessment["native_blender"] = check_blender_application(
        case, record, expect_state=expected_state)
    assessment["passed"] = bool(assessment["independent_geometry"]["passed"]
                                 and assessment["focal_within_limit"])
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == before_sha
    assert assessment["passed"], assessment
    (out / "reopened-assessment.json").write_text(
        json.dumps(assessment, indent=2, allow_nan=False) + "\n")
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reopen", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check Blender RNA and saved generated input without a numerical solve")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.reopen and args.prepare_only:
        parser.error("Choose either --reopen or --prepare-only")
    out = args.out.resolve()
    if args.reopen:
        assert out.is_dir(), "Reopen the original generated output directory"
    else:
        out.mkdir(parents=True, exist_ok=False)
    case_path = ROOT / "tools" / "synthetic_sync" / "cases" / (args.case + ".json")
    case = read_case(case_path)
    validate_fixture(case)
    if not args.reopen:
        (out / "case.json").write_bytes(case_path.read_bytes())
        (out / "environment.json").write_text(
            json.dumps(environment(ROOT), indent=2) + "\n")
    register_extension()
    assessment = (run_reopen(case, out) if args.reopen
                  else run_fresh(case, out, prepare_only=args.prepare_only))
    print(f"Point FOV Blender {'REOPEN ' if args.reopen else ''}PASS: "
          f"{case['name']} ({assessment['classification']})", flush=True)


if __name__ == "__main__":
    main()
