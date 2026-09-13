"""Apply a saved point-FOV candidate in memory and check evaluated projections.

Blender: --factory-startup --disable-autoexec -b --python this.py -- \
  --blend SOURCE.blend --inputs CAPTURE.inputs.json --result JOINT.json --out REPORT.json

No numerical solve runs and the source blend is never saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import blender_pixels
from probe_cameras import _load_extension


def _candidate(saved):
    from match_perspective.core import geometry, sync
    from match_perspective.core.focal_bundle import FocalFitCandidate

    raw = saved["outcome"]
    values = raw.get("candidate")
    if raw.get("accepted") or not values:
        raise ValueError("Saved result has no refused, usable point-FOV candidate")
    calibrations = {}
    for match_id, item in values["calibrations"].items():
        calibrations[match_id] = geometry.Calibration(
            geometry.CameraIntrinsics(**item["intrinsics"]),
            np.asarray(item["rotation_w2c"], float),
            np.asarray(item["camera_center"], float),
            division_lambda=float(item.get("division_lambda", 0.)),
            brown_conrady=tuple(item.get("brown_conrady", ())))
    raw_sync = values["sync_result"]
    solved = sync.SyncSolveResult(
        similarities={key: sync.SimilarityTransform(
            float(item["scale"]), np.asarray(item["rotation"], float),
            np.asarray(item["translation"], float))
            for key, item in raw_sync["similarities"].items()},
        landmarks={key: np.asarray(value, float)
                   for key, value in raw_sync["landmarks"].items()},
        mean_reprojection_px=float(raw_sync["mean_reprojection_px"]),
        per_match_rmse_px=raw_sync["per_match_rmse_px"],
        per_landmark_rmse_px=raw_sync["per_landmark_rmse_px"],
        message=raw_sync["message"], success=bool(raw_sync["success"]),
        line_segments={key: tuple(np.asarray(point, float) for point in segment)
                       for key, segment in raw_sync.get("line_segments", {}).items()},
        bundle_adjusted=bool(raw_sync.get("bundle_adjusted", False)))
    return FocalFitCandidate(
        calibrations, solved, float(values["initial_rmse_px"]),
        float(values["fitted_rmse_px"]), values["reason"],
        values.get("initial_objective"), values.get("fitted_objective")), raw["reason"]


def _project(calibration, similarity, points):
    """Project world points through the candidate's private camera and root."""
    center = (similarity.scale * similarity.rotation @ calibration.camera_center +
              similarity.translation)
    world_to_camera = calibration.rotation_w2c @ similarity.rotation.T
    local = (np.asarray(points, float) - center) @ world_to_camera.T
    if np.any(local[:, 2] <= 0):
        raise AssertionError("A candidate pick has nonpositive depth")
    k = calibration.intrinsics
    pixels = local[:, :2] / local[:, 2, None]
    return pixels * (k.fx, k.fy) + (k.cx, k.cy)


def verify(inputs, saved, *, through_operator=False):
    from match_perspective import properties, scene
    from match_perspective.core.sync.request import json_values

    prep = scene.collect_lens_refine_inputs(bpy.context)
    current = json_values(prep.solver_kwargs())
    if current != inputs:
        raise ValueError("Live numerical lens inputs differ from the saved capture")
    candidate, refusal = _candidate(saved)
    expected_ids = {item.match_id for item in prep.lens_inputs}
    if set(candidate.calibrations) != expected_ids or set(candidate.sync_result.similarities) != expected_ids:
        raise ValueError("Candidate camera support differs from current inputs")
    if not candidate.sync_result.success:
        raise ValueError("Candidate has no complete landmark fit")
    result = SimpleNamespace(
        cancelled=False, point_focal_mode=True, improved=False,
        refusal_reason=refusal, message=refusal, candidate=candidate,
        calibrations={}, sync_result=None, focal_intervals={})
    if through_operator:
        from match_perspective.ui import operators

        def reuse(prepared, **kwargs):
            if json_values(prepared.solver_kwargs()) != inputs:
                raise ValueError("Operator preparation changed the captured inputs")
            return result

        with patch.object(scene, "run_lens_refine", side_effect=reuse):
            assert bpy.ops.perspective_match.refine_lenses() == {"FINISHED"}
        assert operators.lens_best_fit_is_available(bpy.context)
        assert "Use Best Fit is available" in properties.workspace(bpy.context).sync_status
        assert bpy.ops.perspective_match.use_best_focal_fit() == {"FINISHED"}
        assert not operators.lens_best_fit_is_available(bpy.context)
    else:
        _, applied = scene.apply_lens_refine_result(
            bpy.context, result, prep, use_candidate=True)
        assert applied is candidate.sync_result
    space = properties.workspace(bpy.context)
    points = candidate.sync_result.landmarks
    by_camera = {}
    for pick in prep.observations:
        by_camera.setdefault(pick.match_id, []).append(pick)
    errors, native_differences = [], []
    for match_id, picks in by_camera.items():
        calibration = candidate.calibrations[match_id]
        similarity = candidate.sync_result.similarities[match_id]
        xyz = np.asarray([points[item.landmark_id] for item in picks], float)
        expected = _project(calibration, similarity, xyz)
        camera = prep.root_by_name[match_id].pm_session.camera_object
        k = calibration.intrinsics
        native = blender_pixels(camera, {"width": k.image_width, "height": k.image_height}, xyz)
        native_differences.extend(np.linalg.norm(native - expected, axis=1))
        observed = np.asarray([(item.u, item.v) for item in picks])
        errors.extend(np.linalg.norm(native - observed, axis=1))
        stored_fx = scene.calibration_from_settings(prep.root_by_name[match_id].pm_session).intrinsics.fx
        # Blender persists this RNA value as float32.
        if abs(stored_fx - k.fx) > max(1e-3, 2e-7 * k.fx):
            raise AssertionError(f"Applied camera focal differs for {match_id}: {stored_fx} vs {k.fx}")
    if not native_differences or max(native_differences) > .02:
        raise AssertionError(f"Evaluated camera projection mismatch: {max(native_differences):.5f}px")
    landmark_items = {item.item_id: item for item in space.landmarks}
    for point_id, position in points.items():
        item = landmark_items.get(point_id)
        segment = candidate.sync_result.line_segments.get(point_id)
        target = segment[0] if segment is not None else position
        if item is None or not item.has_position or not np.allclose(item.position, target, atol=1e-5):
            raise AssertionError(f"Applied landmark differs from candidate position: {point_id}")
        if segment is not None and (not item.has_line_segment or
                                    not np.allclose(item.position_b, segment[1], atol=1e-5)):
            raise AssertionError("Applied line segment differs from candidate")
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    if abs(rmse - candidate.fitted_rmse_px) > .02:
        raise AssertionError("Evaluated picked-point RMSE differs from candidate")
    if "calibration not validated" not in space.sync_status:
        raise AssertionError("Applied candidate lost its provisional status")
    return dict(passed=True, numerical_solves=0, saved_blend=False,
                applied_through_operator=through_operator,
                cameras=len(by_camera), picked_points=len(errors),
                landmarks=len(points), candidate_point_rmse_px=candidate.fitted_rmse_px,
                evaluated_pick_rmse_px=rmse,
                evaluated_projection_max_delta_px=float(max(native_differences)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("blend", "inputs", "result", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--operator", action="store_true",
                        help="Publish the saved candidate through Refine Lenses, then use its operator; zero solves")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    blend = args.blend.expanduser().resolve()
    if args.out.resolve() in {blend, args.inputs.resolve(), args.result.resolve()}:
        raise ValueError("Report path must differ from the blend and captured inputs")
    source_before = blend.stat()
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    inputs = json.loads(args.inputs.read_text())
    saved = json.loads(args.result.read_text())
    report = verify(inputs, saved, through_operator=args.operator)
    source_after = blend.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (
        source_after.st_size, source_after.st_mtime_ns
    ):
        raise AssertionError("Source blend changed during in-memory candidate verification")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("Candidate apply PASS:", report)


if __name__ == "__main__":
    main()
