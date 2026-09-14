"""Generated zero-solve Sync seed, provenance, investigation and reopen checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.blender_case import create_scene, register_extension
from tools.synthetic_sync.scenarios import generate


def check_joint_selection(context) -> dict:
    """Inject outcomes to check the scene gate without a numerical fit."""
    from match_perspective import properties, scene
    from match_perspective.core import focal_bundle, sync
    from match_perspective.core.sync.solve import solution_result_from_seed

    prep = scene.prepare_diagnose_sync(context, prepare_scene=False)
    fake_module = types.ModuleType("match_perspective.core.joint_fit_score")
    coverage = dict(
        requested_point_picks=len(prep.observations),
        fitted_point_picks=len(prep.observations),
        requested_line_strokes=len(prep.line_observations or ()),
        fitted_line_strokes=len(prep.line_observations or ()),
        skipped_camera_ids=[], skipped_point_ids=[], skipped_line_ids=[],
        skipped_relation_ids=[],
    )

    def supported(request, _result):
        return request, coverage

    class Scorer:
        def __init__(self, request, _initial, *, calibrations=None,
                     frozen_point_weights=None):
            self.request = request
            self.effective_point_weights = [
                (item.match_id, item.landmark_id, item.weight)
                for item in request.observations]
            self.downweighted_landmark_ids = []

        def score(self, result, *, calibrations=None):
            missing = any(
                item.landmark_id not in result.line_segments
                for item in self.request.line_observations or ()
            )
            return types.SimpleNamespace(
                valid=not missing,
                objective=result.joint_final_objective or 10.0,
                point_rmse_px=4.0,
                line_rmse_px=2.0,
                per_match_rmse_px={"a": 3.0},
                per_match_point_rmse_px={"a": 4.0},
                per_match_line_rmse_px={"a": 2.0},
                per_landmark_rmse_px={"point": 4.0},
                supported_observations=(
                    len(self.request.observations) +
                    len(self.request.line_observations or ()) - int(missing)
                ),
                constraint_gaps={},
                reason="Line support was lost" if missing else "",
            )

    fake_module.JointFitScorer = Scorer
    fake_module.supported_joint_request = supported
    mode = ["worse_objective"]
    fixed_calls = [0]

    def fixed(_request, initial, **_kwargs):
        fixed_calls[0] += 1
        if mode[0] in {"refused", "cancelled"}:
            return types.SimpleNamespace(
                accepted=False, sync_result=None, calibrations={},
                support_coverage=coverage,
                reason="Cancelled" if mode[0] == "cancelled" else "No improvement",
            )
        candidate = deepcopy(initial)
        candidate.mean_reprojection_px = 0.001 if mode[0] == "worse_objective" else 5.0
        candidate.joint_final_objective = 12.0 if mode[0] == "worse_objective" else 9.0
        if mode[0] == "lost_support":
            candidate.line_segments = {}
        if mode[0] == "invalid_success":
            candidate.success = False
        return types.SimpleNamespace(
            accepted=True, sync_result=candidate,
            calibrations=deepcopy(initial.calibrations),
            support_coverage=coverage, reason="accepted",
        )

    def no_registration(**_kwargs):
        raise AssertionError("A certified applied seed must skip registration")

    module_name = fake_module.__name__
    old_module = sys.modules.get(module_name)
    old_fixed = getattr(focal_bundle, "refine_fixed_focals", None)
    old_registration = sync.solve_landmark_sync
    sys.modules[module_name] = fake_module
    focal_bundle.refine_fixed_focals = fixed
    sync.solve_landmark_sync = no_registration
    try:
        worse = scene.run_solve_sync(prep)
        assert worse.joint_final_objective == 10.0
        assert "worsened" in worse.joint_refusal_reason
        mode[0] = "better_objective"
        better = scene.run_solve_sync(prep)
        assert better.joint_final_objective == 9.0
        assert better.mean_reprojection_px == 5.0
        mode[0] = "lost_support"
        lost = scene.run_solve_sync(prep)
        assert lost.joint_final_objective == 10.0
        assert "Line support" in lost.joint_refusal_reason
        mode[0] = "invalid_success"
        invalid = scene.run_solve_sync(prep)
        assert invalid.success and invalid.joint_final_objective == 10.0

        registration_calls = [0]

        def cold_registration(**_kwargs):
            registration_calls[0] += 1
            return solution_result_from_seed(prep.initial_solution)

        sync.solve_landmark_sync = cold_registration
        mode[0] = "better_objective"
        cold_prep = deepcopy(prep)
        cold_prep.initial_solution = None
        cold = scene.run_solve_sync(cold_prep)
        assert cold.success and cold.joint_final_objective == 9.0
        assert registration_calls[0] == 1 and fixed_calls[0] == 5
        changed_prep = deepcopy(prep)
        changed_prep.observations[0].u += 0.5
        changed = scene.run_solve_sync(changed_prep)
        assert changed.success and changed.joint_final_objective == 9.0
        assert registration_calls[0] == 2 and fixed_calls[0] == 6
        mode[0] = "refused"
        refused = scene.run_solve_sync(cold_prep)
        assert refused.success and refused.joint_refusal_reason == "No improvement"
        assert refused.point_rmse_px == 4.0 and refused.line_rmse_px == 2.0
        assert refused.joint_final_objective == 10.0
        mode[0] = "cancelled"
        try:
            scene.run_solve_sync(prep)
        except sync.SyncCancelled:
            pass
        else:
            raise AssertionError("Cancelled fit returned a result for application")
        mode[0] = "better_objective"
        before_status = properties.workspace(context).sync_status
        diagnosed = scene.run_diagnose_sync(prep)
        assert diagnosed.common_leave_one_out is not None
        assert len(diagnosed.common_leave_one_out.items) <= 5
        assert properties.workspace(context).sync_status == before_status
    finally:
        sync.solve_landmark_sync = old_registration
        if old_fixed is None:
            delattr(focal_bundle, "refine_fixed_focals")
        else:
            focal_bundle.refine_fixed_focals = old_fixed
        if old_module is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = old_module
    return {"regressing_candidate_retained": True,
            "combined_gain_accepted": True, "support_loss_retained": True,
            "invalid_success_rejected": True,
            "cold_and_changed_run_common_fit": True,
            "investigation_uses_common_fit": True,
            "cold_refusal_metrics_and_cancellation": True}


def check(out: Path) -> dict:
    from match_perspective import properties, scene
    from match_perspective.core import sync

    case = generate("mixed_lines", seed=0, noise_px=0.0)
    create_scene(case, out, False)
    context = bpy.context
    space = properties.workspace(context)
    request = scene.collect_sync_request(context)
    assert request.initial_solution is None
    match_ids = {item.match_id for item in request.matches}
    point_truth = {
        name: np.asarray(point, dtype=np.float64)
        for name, point in case["truth"]["points"].items()
    }
    line_truth = {
        name: tuple(np.asarray(end, dtype=np.float64) for end in segment)
        for name, segment in case["truth"]["lines"].items()
    }
    geometry = {**point_truth, **{
        name: 0.5 * (ends[0] + ends[1]) for name, ends in line_truth.items()
    }}
    fitted_calibrations = {
        item.match_id: deepcopy(item.calibration) for item in request.matches
    }
    for index, calibration in enumerate(fitted_calibrations.values()):
        angle = 0.0001 * (index + 1)
        rotation = np.array((
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ))
        calibration.rotation_w2c = rotation @ calibration.rotation_w2c
        calibration.camera_center += np.array((0.001, -0.002, 0.003))
    accepted = sync.SyncSolveResult(
        similarities={name: sync.SimilarityTransform() for name in match_ids},
        landmarks=geometry,
        line_segments=line_truth,
        mean_reprojection_px=0.0,
        per_match_rmse_px={name: 0.0 for name in match_ids},
        per_landmark_rmse_px={name: 0.0 for name in geometry},
        message="Generated accepted result",
        calibrations=fitted_calibrations,
        joint_final_objective=10.0,
        joint_point_weights=[
            (item.match_id, item.landmark_id, float(item.weight))
            for item in request.observations],
    )
    scene._apply_sync_solve_result(context, accepted, request.matches)
    joint_selection = check_joint_selection(context)
    applied = scene.collect_sync_request(context)
    seed = applied.initial_solution
    assert seed is not None and seed.evidence_sha256 == applied.evidence_sha256()
    assert set(seed.similarities) == match_ids
    assert set(seed.line_segments) == set(line_truth)
    assert set(point_truth) <= set(seed.landmarks)
    for name, point in point_truth.items():
        np.testing.assert_allclose(seed.landmarks[name], point, atol=1e-6)
    for name, ends in line_truth.items():
        np.testing.assert_allclose(seed.line_segments[name], ends, atol=1e-6)
    for match in applied.matches:
        np.testing.assert_allclose(
            seed.calibrations[match.match_id].camera_center,
            fitted_calibrations[match.match_id].camera_center, atol=1e-6,
        )
        np.testing.assert_allclose(
            seed.calibrations[match.match_id].rotation_w2c,
            fitted_calibrations[match.match_id].rotation_w2c, atol=1e-6,
        )

    # Investigate validates ownership without replacing applied overlay errors.
    prep = scene.prepare_diagnose_sync(context, prepare_scene=False)
    before_status = "Applied result remains visible"
    space.sync_status = before_status
    landmark = next(item for item in space.landmarks if item.item_id in geometry)
    landmark.rmse_px = 3.25
    trial = sync.SyncSolveResult(
        similarities={}, landmarks={}, mean_reprojection_px=99.0,
        per_match_rmse_px={}, per_landmark_rmse_px={landmark.item_id: 99.0},
        message="Generated investigation", success=False,
    )
    scene.apply_diagnose_sync_result(context, prep, trial)
    assert landmark.rmse_px == 3.25 and space.sync_status == before_status
    try:
        scene.apply_solve_sync_result(context, prep, trial)
    except scene.SyncSolveRejected:
        pass
    else:
        raise AssertionError("Refused Solve was applied")
    assert landmark.rmse_px == 3.25 and space.sync_status == before_status

    # Edited evidence may warm-start, but cannot claim to be the same incumbent.
    pick = next(item for item in landmark.observations if item.is_set)
    saved_x = pick.x
    pick.x += 1.0
    changed = scene.collect_sync_request(context)
    assert changed.initial_solution is not None
    assert changed.evidence_sha256() != changed.initial_solution.evidence_sha256
    pick.x = saved_x
    root = next(root for root in properties.iter_match_roots() if root.name != applied.anchor_id)
    old_role = root.pm_session.sync_role
    root.pm_session.sync_role = "FIT_ONLY"
    changed = scene.collect_sync_request(context)
    assert changed.initial_solution is not None
    assert changed.evidence_sha256() != changed.initial_solution.evidence_sha256
    root.pm_session.sync_role = old_role
    old_plane = landmark.plane_axis
    landmark.plane_axis = "Z"
    changed = scene.collect_sync_request(context)
    assert changed.initial_solution is not None
    assert changed.evidence_sha256() != changed.initial_solution.evidence_sha256
    landmark.plane_axis = old_plane

    # Moving a registered root or its managed camera invalidates the seed.
    root_matrix = root.matrix_world.copy()
    root.location.x += 0.25
    context.view_layer.update()
    assert scene.collect_sync_request(context).initial_solution is None
    root.matrix_world = root_matrix
    context.view_layer.update()
    camera = root.pm_session.camera_object
    camera_location = camera.location.copy()
    camera.location.x += 0.25
    context.view_layer.update()
    assert scene.collect_sync_request(context).initial_solution is None
    camera.location = camera_location
    context.view_layer.update()
    assert scene.collect_sync_request(context).initial_solution is not None

    before_reopen = scene.collect_sync_request(context).to_record()
    path = out / "continuation-reopen.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(path))
    reopened = scene.collect_sync_request(bpy.context)
    assert reopened.initial_solution is not None
    assert reopened.evidence_sha256() == reopened.initial_solution.evidence_sha256
    assert reopened.to_record() == before_reopen
    for match_id, calibration in fitted_calibrations.items():
        np.testing.assert_allclose(
            reopened.initial_solution.calibrations[match_id].rotation_w2c,
            calibration.rotation_w2c, atol=1e-6,
        )
    # Simulate an existing saved file created before provenance fields existed.
    old_space = properties.workspace(bpy.context)
    old_space.sync_solution_evidence_sha256 = ""
    old_space.sync_solution_geometry_sha256 = ""
    old_space.sync_solution_result_json = ""
    legacy_path = out / "legacy-continuation-reopen.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(legacy_path), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(legacy_path))
    unproven = scene.collect_sync_request(bpy.context).initial_solution
    assert unproven is not None and unproven.evidence_sha256 == ""
    assert unproven.diagnostics is None
    moved_root = next(root for root in properties.iter_match_roots()
                      if root.name != reopened.anchor_id)
    original_matrix = moved_root.matrix_world.copy()
    moved_root.location.x += 0.25
    bpy.context.view_layer.update()
    assert scene.collect_sync_request(bpy.context).initial_solution is None
    moved_root.matrix_world = original_matrix
    bpy.context.view_layer.update()
    assert scene.collect_sync_request(bpy.context).initial_solution is not None
    scene.clear_sync_transforms(bpy.context)
    assert scene.collect_sync_request(bpy.context).initial_solution is None
    create_scene(generate("free_scale", seed=0, noise_px=0.0), out, False)
    no_ground_space = properties.workspace(bpy.context)
    no_ground_space.share_lens = False
    no_ground_space.estimate_focal_from_points = True
    for root in properties.iter_match_roots():
        root.pm_session.origin_is_set = False
    no_ground_calibrations = {
        item.match_id: deepcopy(item.calibration)
        for item in scene.collect_sync_request(bpy.context).matches
    }
    no_ground_prep = scene.prepare_lens_refine(bpy.context)
    assert no_ground_prep.initial_solution is None
    assert all(not root.pm_session.origin_is_set for root in properties.iter_match_roots())
    for item in scene.collect_sync_request(bpy.context).matches:
        np.testing.assert_allclose(
            item.calibration.rotation_w2c,
            no_ground_calibrations[item.match_id].rotation_w2c, atol=1e-9,
        )
    # A calibrated Ground startup now receives the same frame and origin
    # preparation in independent FOV mode; no-ground above stayed untouched.
    create_scene(generate("ground", seed=0, noise_px=0.0), out, False)
    ground_space = properties.workspace(bpy.context)
    ground_space.share_lens = False
    ground_space.estimate_focal_from_points = True
    for root in properties.iter_match_roots():
        root.pm_session.origin_is_set = False
        root.pm_session.lock_focal = True
    ground_prep = scene.prepare_lens_refine(bpy.context)
    assert ground_prep.initial_solution is None
    assert any(root.pm_session.origin_is_set for root in properties.iter_match_roots())
    ground_again = scene.collect_lens_refine_inputs(bpy.context)
    assert ground_again.source_request_sha256 == ground_prep.source_request_sha256
    for match in ground_again.lens_inputs:
        np.testing.assert_allclose(
            match.base_calibration.rotation_w2c,
            scene.calibration_from_settings(next(
                root.pm_session for root in properties.iter_match_roots()
                if root.name == match.match_id
            )).rotation_w2c,
            atol=1e-9,
        )
    return {
        "joint_selection": joint_selection,
        "passed": True, "numerical_solves": 0,
        "registered_cameras": len(match_ids),
        "points": len(point_truth), "lines": len(line_truth),
        "ground_fov_prepared": True, "no_ground_fov_untouched": True,
        "unproven_upgrade_seed": True,
        "reopen_sha256": before_reopen["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=False)
    register_extension()
    report = check(args.out)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Sync continuation Blender PASS:", report)


if __name__ == "__main__":
    main()
