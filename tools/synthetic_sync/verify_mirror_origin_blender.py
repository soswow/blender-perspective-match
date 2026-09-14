"""Check automatic free mirror-origin placement without numerical solves.

Run in factory-startup Blender with ``-- --out /tmp/pm-mirror-origin``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.blender_case import blender_pixels, register_extension
from tools.synthetic_sync.verify_landmark_mirror_blender import make_case
from tools.synthetic_sync.verify_focal_constraints_blender import build


def verify(out: Path) -> None:
    from match_perspective import properties, scene
    from match_perspective.core import sync

    case = make_case("sync")
    build(case, out)
    workspace = properties.workspace(bpy.context)
    if workspace.mirror_object is not None:
        bpy.data.objects.remove(workspace.mirror_object, do_unlink=True)
    workspace.mirror_object = None
    workspace.mirror_origin = "LANDMARK"
    workspace.mirror_plane = "YZ"
    workspace.mirror_landmark_id = case["request"]["mirror_landmark_id"]
    roots = {root.name: root for root in properties.iter_match_roots()}
    anchor = roots[case["request"]["anchor_id"]]
    anchor.pm_session.origin_is_set = False

    positions = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in case["truth"]["points"].items()
    }
    # Add one finite line below the point cloud, so its endpoint also sets the floor.
    positions["test_edge"] = np.array((0.2, -0.1, -3.0))
    segments = {
        "test_edge": (
            np.array((0.2, -0.1, -3.5)),
            np.array((0.3, 0.2, -2.5)),
        ),
    }

    def result():
        return sync.SyncSolveResult(
            similarities={key: sync.SimilarityTransform() for key in roots},
            landmarks={key: value.copy() for key, value in positions.items()},
            line_segments={key: (a.copy(), b.copy()) for key, (a, b) in segments.items()},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="Synthetic accepted result",
        )

    trial = result()
    trial.calibrations = {anchor.name: scene.calibration_from_settings(anchor.pm_session)}
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is not None

    anchor.pm_session.origin_is_set = True
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    anchor.pm_session.origin_is_set = False
    workspace.mirror_plane = "XY"
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    workspace.mirror_plane = "XZ"
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is not None
    workspace.mirror_plane = "YZ"
    other = next(root for root in roots.values() if root != anchor)
    other.pm_session.sync_role = "LOCK_POSE"
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    other.pm_session.sync_role = "SOLVE"
    point = next(item for item in workspace.landmarks if item.kind == "POINT")
    point.on_ground = True
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    point.on_ground = False
    known = bpy.data.objects.new("Known world control", None)
    bpy.context.scene.collection.objects.link(known)
    point.known_object = known
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    point.known_object = None
    bpy.data.objects.remove(known, do_unlink=True)
    orientation = bpy.data.objects.new("Orientation control", None)
    bpy.context.scene.collection.objects.link(orientation)
    orientation.rotation_euler.z = 0.3
    workspace.mirror_object = orientation
    assert scene._free_mirror_origin_offset(bpy.context, trial, anchor) is None
    workspace.mirror_object = None
    bpy.data.objects.remove(orientation, do_unlink=True)

    def projected(current):
        samples = {}
        for key, similarity in current.similarities.items():
            calibration = scene.calibration_from_settings(roots[key].pm_session)
            samples[key] = {
                landmark_id: sync.project_private_point(
                    similarity.inverse_point(value), calibration,
                )
                for landmark_id, value in current.landmarks.items()
            }
        return samples

    before = projected(trial)
    camera_specs = {item["id"]: item for item in case["request"]["cameras"]}
    reference_id = workspace.mirror_landmark_id
    native_before = {
        match_id: blender_pixels(
            root.pm_session.camera_object, camera_specs[match_id],
            [trial.landmarks[reference_id]],
        )
        for match_id, root in roots.items()
    }
    matches = [
        sync.SyncMatchInput(match_id=key, calibration=scene.calibration_from_settings(root.pm_session))
        for key, root in roots.items()
    ]
    applied = scene._apply_sync_solve_result(bpy.context, trial, matches)
    np.testing.assert_allclose(
        applied.calibrations[anchor.name].camera_center,
        scene.calibration_from_settings(anchor.pm_session).camera_center,
        atol=1e-7,
    )
    after = projected(applied)
    for match_id in before:
        for landmark_id in before[match_id]:
            np.testing.assert_allclose(
                after[match_id][landmark_id], before[match_id][landmark_id],
                atol=1e-6,
            )
    np.testing.assert_allclose(anchor.matrix_world.translation, (0, 0, 0), atol=1e-7)
    assert not anchor.pm_session.origin_is_set
    saved_anchor = next(item.calibration for item in matches if item.match_id == anchor.name)
    np.testing.assert_allclose(
        saved_anchor.camera_center,
        scene.calibration_from_settings(anchor.pm_session).camera_center,
        atol=1e-7,
    )
    next_request = scene.collect_sync_request(bpy.context)
    assert not next_request.fixed_similarities
    next_anchor = next(item.calibration for item in next_request.matches if item.match_id == anchor.name)
    np.testing.assert_allclose(next_anchor.camera_center, saved_anchor.camera_center, atol=1e-7)
    reference = applied.landmarks[workspace.mirror_landmark_id]
    np.testing.assert_allclose(reference[:2], (0, 0), atol=1e-7)
    for match_id, root in roots.items():
        native_after = blender_pixels(
            root.pm_session.camera_object, camera_specs[match_id], [reference],
        )
        np.testing.assert_allclose(native_after, native_before[match_id], atol=0.003)
    assert min(value[2] for value in applied.landmarks.values()) > 0
    assert min(point[2] for segment in applied.line_segments.values() for point in segment) > 0
    np.testing.assert_allclose(trial.landmarks[reference_id], positions[reference_id], atol=1e-7)

    # The next solve starts in the translated anchor frame and must not drift.
    repeat = sync.SyncSolveResult(
        similarities=applied.similarities.copy(),
        landmarks={key: value.copy() for key, value in applied.landmarks.items()},
        line_segments={key: (a.copy(), b.copy()) for key, (a, b) in applied.line_segments.items()},
        mean_reprojection_px=0.0,
        per_match_rmse_px={}, per_landmark_rmse_px={}, message="Repeat",
    )
    center = scene.calibration_from_settings(anchor.pm_session).camera_center.copy()
    repeat_applied = scene._apply_sync_solve_result(bpy.context, repeat, ())
    np.testing.assert_allclose(
        scene.calibration_from_settings(anchor.pm_session).camera_center,
        center, atol=1e-6,
    )
    workspace.mirror_plane = "XZ"
    moved = np.array((0.2, 0.4, 0.0))
    xz = sync.SyncSolveResult(
        similarities=repeat_applied.similarities.copy(),
        landmarks={key: value + moved for key, value in repeat_applied.landmarks.items()},
        line_segments={key: (a + moved, b + moved) for key, (a, b) in repeat_applied.line_segments.items()},
        mean_reprojection_px=0.0,
        per_match_rmse_px={}, per_landmark_rmse_px={}, message="XZ world plane",
    )
    xz_before = projected(xz)
    xz_applied = scene._apply_sync_solve_result(bpy.context, xz, ())
    xz_after = projected(xz_applied)
    for match_id in xz_before:
        for landmark_id in xz_before[match_id]:
            np.testing.assert_allclose(
                xz_after[match_id][landmark_id], xz_before[match_id][landmark_id],
                atol=1e-6,
            )
    np.testing.assert_allclose(
        xz_applied.landmarks[reference_id][:2], (0, 0), atol=1e-7,
    )
    assert min(value[2] for value in xz_applied.landmarks.values()) > 0
    saved_center = scene.calibration_from_settings(anchor.pm_session).camera_center.copy()
    path = out / "mirror-origin-reopen.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(path))
    workspace = properties.workspace(bpy.context)
    anchor = properties.anchor_root(bpy.context)
    assert anchor is not None and not anchor.pm_session.origin_is_set
    np.testing.assert_allclose(anchor.matrix_world.translation, (0, 0, 0), atol=1e-7)
    np.testing.assert_allclose(
        scene.calibration_from_settings(anchor.pm_session).camera_center,
        saved_center, atol=1e-6,
    )
    restored = next(item for item in workspace.landmarks if item.item_id == reference_id)
    np.testing.assert_allclose(restored.position[:2], (0, 0), atol=1e-6)
    scene.clear_sync_transforms(bpy.context)
    np.testing.assert_allclose(anchor.matrix_world.translation, (0, 0, 0), atol=1e-7)
    np.testing.assert_allclose(
        scene.calibration_from_settings(anchor.pm_session).camera_center,
        saved_center, atol=1e-6,
    )
    print("mirror origin placement: PASS (zero numerical solves)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    register_extension()
    verify(args.out)


if __name__ == "__main__":
    main()
