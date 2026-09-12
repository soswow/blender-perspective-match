"""One generated Blender point-FOV fit, apply and reload with a tilted anchor.

Uses an exact anchor-gauge starting cloud to isolate the joint fit. Opens no user
file, performs no Sync registration, and runs at most one focal bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync import focal_orientation
from tools.synthetic_sync.blender_case import blender_pixels, register_extension
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.verify_focal_constraints_blender import build


def run(out: Path, *, constrained: bool) -> dict:
    from match_perspective import properties, scene
    from match_perspective.core import lens_refine, sync

    case = focal_orientation.generate(constrained=constrained)
    (out / "case.json").write_text(json.dumps(case, indent=2))
    build(case, out)
    prep = scene.collect_lens_refine_inputs(bpy.context)
    assert set(map(tuple, prep.plane_groups)) == set(map(tuple, case["request"]["plane_groups"]))
    assert set(map(tuple, prep.mirror_pairs)) == set(map(tuple, case["request"]["mirror_pairs"]))
    points, target_cameras = focal_orientation.anchor_gauge_start(case)
    similarities = {}
    for item in prep.lens_inputs:
        cal = item.base_calibration
        target = target_cameras[item.match_id]
        transform = np.asarray(target["rotation"]).T @ cal.rotation_w2c
        similarities[item.match_id] = sync.SimilarityTransform(
            1., transform, np.asarray(target["center"]) - transform @ cal.camera_center)
    similarities[prep.anchor_id] = sync.SimilarityTransform()
    initial = sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value) for key, value in points.items()},
        mean_reprojection_px=0., per_match_rmse_px={}, per_landmark_rmse_px={},
        message="Exact tilted-anchor gauge start", success=True)
    with patch.object(lens_refine, "_run_sync", return_value=initial) as startup:
        result = scene.run_lens_refine(prep)
    assert startup.call_count == 1
    assert result.improved, result.refusal_reason
    assert np.linalg.norm(result.sync_result.similarities[prep.anchor_id].rotation - np.eye(3)) < 1e-8
    true = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    center = np.asarray(true[prep.anchor_id]["center"])
    fitted_points = {key: np.asarray(value) for key, value in result.sync_result.landmarks.items()}
    if constrained:
        ids = [key for key in case["truth"]["points"]]
        a = np.asarray([case["truth"]["points"][key] for key in ids]) - center
        b = np.asarray([fitted_points[key] for key in ids]) - center
        scale = float(np.sum(a * b) / np.sum(a * a))
        assert scale > 0
        assert np.linalg.norm(np.asarray(result.calibrations[prep.anchor_id].rotation_w2c) -
                              np.asarray(true[prep.anchor_id]["rotation"])) < .001
        assert np.linalg.norm(np.asarray(result.calibrations[prep.anchor_id].rotation_w2c) -
                              np.asarray(case["request"]["cameras"][0]["rotation"])) > .15
        assert np.allclose(result.calibrations[prep.anchor_id].camera_center, center, atol=1e-7)
        x = [fitted_points[key][0] for key, axis, _ in prep.plane_groups if axis == "X"]
        assert np.ptp(x) < 1e-4
        origin, normal = (np.asarray(v) for v in prep.mirror_plane)
        normal /= np.linalg.norm(normal)
        for left, right in prep.mirror_pairs:
            a, b = fitted_points[left], fitted_points[right]
            reflection = a - 2 * normal * float(normal @ (a - origin))
            assert np.linalg.norm(reflection - b) < 1e-4
    else:
        scale = 1.0
        assert np.allclose(result.calibrations[prep.anchor_id].rotation_w2c,
                           case["request"]["cameras"][0]["rotation"], atol=1e-8)
    scene.apply_lens_refine_result(bpy.context, result, prep)
    roots = prep.root_by_name
    if constrained:
        # The fitted common rotation lives in the anchor camera calibration;
        # the anchor root remains the identity for subsequent Sync calls.
        assert np.allclose(roots[prep.anchor_id].matrix_world, np.eye(4), atol=1e-5)
        recollected = scene.collect_lens_refine_inputs(bpy.context)
        next_anchor = next(item.base_calibration for item in recollected.lens_inputs
                           if item.match_id == prep.anchor_id)
        assert np.allclose(next_anchor.rotation_w2c,
                           result.calibrations[prep.anchor_id].rotation_w2c, atol=1e-7)
    holdouts = np.asarray(list(case["truth"]["holdouts"].values()))
    if constrained:
        holdouts = center + scale * (holdouts - center)
    else:
        holdouts = center + (holdouts - center) @ focal_orientation.rotation().T
    errors = []
    for camera_id, camera in true.items():
        actual = blender_pixels(roots[camera_id].pm_session.camera_object, camera, holdouts)
        wanted = project(list(case["truth"]["holdouts"].values()), camera)[0]
        errors.extend(np.linalg.norm(actual - wanted, axis=1))
    assert max(errors) < .01, max(errors)
    # A changed world prior invalidates a result prepared against the old one.
    if constrained:
        landmark = next(item for item in properties.workspace(bpy.context).landmarks
                        if item.plane_axis == "X" and item.plane_group != "NONE")
        previous = landmark.plane_group
        landmark.plane_group = "3" if previous != "3" else "4"
        try:
            scene.apply_lens_refine_result(bpy.context, result, prep)
        except scene.StaleSyncResult:
            pass
        else:
            raise AssertionError("Edited plane accepted stale orientation result")
        landmark.plane_group = previous
    # Reopening this generated result must preserve the corrected private
    # anchor calibration and every evaluated camera without another fit.
    before_matrices = {
        camera_id: np.asarray(roots[camera_id].pm_session.camera_object.matrix_world).copy()
        for camera_id in true
    }
    before_pixels = {
        camera_id: blender_pixels(roots[camera_id].pm_session.camera_object, camera, holdouts)
        for camera_id, camera in true.items()
    }
    saved = out / "solved.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(saved), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(saved), load_ui=False)
    reopened_roots = {root.name: root for root in properties.iter_match_roots()}
    assert set(reopened_roots) == set(true)
    for camera_id, camera in true.items():
        camera_object = reopened_roots[camera_id].pm_session.camera_object
        assert np.allclose(camera_object.matrix_world, before_matrices[camera_id], atol=1e-7)
        assert np.allclose(blender_pixels(camera_object, camera, holdouts),
                           before_pixels[camera_id], atol=.003)
    reopened = scene.collect_lens_refine_inputs(bpy.context)
    reopened_anchor = next(item.base_calibration for item in reopened.lens_inputs
                           if item.match_id == prep.anchor_id)
    assert np.allclose(reopened_anchor.rotation_w2c,
                       result.calibrations[prep.anchor_id].rotation_w2c, atol=1e-7)
    assert np.allclose(reopened_anchor.camera_center,
                       result.calibrations[prep.anchor_id].camera_center, atol=1e-7)
    assert np.allclose(reopened_roots[prep.anchor_id].matrix_world, np.eye(4), atol=1e-7)
    report = dict(passed=True, constrained=constrained, bundle_calls=1,
                  withheld_max_px=float(max(errors)), reopened=True)
    (out / "result.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--free-control", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    register_extension()
    print("Orientation Blender PASS:", run(out, constrained=not args.free_control))


if __name__ == "__main__":
    main()
