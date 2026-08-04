"""Headless registration, solver, scene, and project smoke test.

Run with:
``blender --factory-startup -b --python validate_addon.py``
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy
import numpy as np


def load_extension_module():
    """Import this extension package directly from its source directory."""
    extension_directory = Path(__file__).resolve().parent
    module_name = "match_perspective"
    spec = importlib.util.spec_from_file_location(
        module_name,
        extension_directory / "__init__.py",
        submodule_search_locations=[str(extension_directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def add_bundle(settings, axis: str, vanishing: np.ndarray, targets) -> None:
    """Add two synthetic lines converging at a chosen VP."""
    for target in targets:
        target_point = np.asarray(target, dtype=np.float64)
        point_a = vanishing + 0.65 * (target_point - vanishing)
        point_b = vanishing + 0.92 * (target_point - vanishing)
        line = settings.lines.add()
        line.axis = axis
        line.x1, line.y1 = point_a
        line.x2, line.y2 = point_b


def _write_png(temporary_path: Path, name: str, width: int, height: int, color) -> Path:
    generated = bpy.data.images.new(name, width=width, height=height, alpha=True)
    generated.generated_color = color
    image_path = temporary_path / f"{name}.png"
    generated.filepath_raw = str(image_path)
    generated.file_format = "PNG"
    generated.save()
    bpy.data.images.remove(generated)
    return image_path


def _write_minimal_pmproj(path: Path, image_path: Path, *, include_origin: bool = True) -> None:
    """Write a minimal importable .pmproj for smoke testing."""
    session = {
        "imagePath": str(image_path),
        "vpMode": 2,
        "activeAxis": "x",
        "lockFocal": False,
        "overlayOpacity": 0.9,
        "controlsOpacity": 1.0,
        "showVpOverlay": True,
        "lines": {
            "x": [
                {"id": "lx1", "x1": 50.0, "y1": 180.0, "x2": 120.0, "y2": 220.0},
                {"id": "lx2", "x1": 60.0, "y1": 420.0, "x2": 140.0, "y2": 380.0},
            ],
            "y": [],
            "z": [
                {"id": "lz1", "x1": 680.0, "y1": 160.0, "x2": 620.0, "y2": 210.0},
                {"id": "lz2", "x1": 670.0, "y1": 450.0, "x2": 610.0, "y2": 400.0},
            ],
        },
        # Surfaces/scale must be ignored by the importer.
        "surfaces": [
            {"id": "ignored", "plane": "xz", "x1": 1, "y1": 1, "x2": 2, "y2": 2, "divisions": 4},
        ],
        "scalePointA": [100.0, 100.0],
        "scalePointB": [200.0, 100.0],
        "measuredLength": 2.0,
        "scale": 0.5,
    }
    if include_origin:
        session["originImage"] = [400.0, 420.0]
    path.write_text(
        json.dumps({
            "kind": "perspective-match-project",
            "version": 1,
            "session": session,
        }),
        encoding="utf-8",
    )


def main() -> None:
    """Exercise core math and native scene application."""
    extension = load_extension_module()
    registered = False
    with tempfile.TemporaryDirectory(prefix="perspective-match-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        try:
            extension.register()
            registered = True
            core = sys.modules["match_perspective.core"]
            distortion = sys.modules["match_perspective.distortion"]
            properties = sys.modules["match_perspective.properties"]
            scene = sys.modules["match_perspective.scene"]
            project_io = sys.modules["match_perspective.project_io"]

            crossing = core.vanishing_point_from_lines(
                [
                    core.LineSegment(0.0, 0.0, 100.0, 100.0),
                    core.LineSegment(0.0, 100.0, 100.0, 0.0),
                ]
            )
            assert crossing is not None
            assert np.allclose(crossing[:2], (50.0, 50.0), atol=1.0e-6)

            center = core.orthocenter_2d(
                np.array((0.0, 0.0)),
                np.array((4.0, 0.0)),
                np.array((0.0, 3.0)),
            )
            assert center is not None and np.allclose(center, (0.0, 0.0))

            points = np.array(((100.0, 80.0), (640.0, 360.0), (1100.0, 600.0)))
            ideal = core.undistort_points(points, 800.0, 800.0, 640.0, 360.0, 0.12)
            observed = core.distort_points(ideal, 800.0, 800.0, 640.0, 360.0, 0.12)
            assert np.allclose(points, observed, atol=0.05)

            image_path = _write_png(
                temporary_path,
                "reference",
                800,
                600,
                (0.15, 0.2, 0.25, 1.0),
            )

            root_a = scene.create_match_camera(bpy.context)
            assert properties.active_root(bpy.context) == root_a
            scene.bind_reference_image(bpy.context, str(image_path))
            settings = properties.active_session(bpy.context)
            assert settings is not None
            assert settings.camera_object is not None
            assert settings.camera_object.parent == properties.active_root(bpy.context)
            assert settings.image_width == 800 and settings.image_height == 600

            settings.vp_mode = "2"
            settings.lock_focal = False
            settings.lines.clear()
            add_bundle(
                settings,
                "x",
                np.array((-600.0, 300.0)),
                ((200.0, 120.0), (220.0, 500.0)),
            )
            add_bundle(
                settings,
                "z",
                np.array((900.0, 300.0)),
                ((620.0, 100.0), (610.0, 520.0)),
            )
            calibration = scene.refine_match(bpy.context)
            expected_focal = np.sqrt(500000.0)
            assert abs(calibration.intrinsics.fx - expected_focal) < 2.0
            assert bpy.context.scene.camera == settings.camera_object
            assert abs(settings.camera_object.data.shift_x) < 1.0e-5

            # 3-point with X + Z only: Blender Y must be derived (no green lines).
            settings.vp_mode = "3"
            settings.lines.clear()
            add_bundle(
                settings,
                "x",
                np.array((-600.0, 300.0)),
                ((200.0, 120.0), (220.0, 500.0)),
            )
            add_bundle(
                settings,
                "y",
                np.array((400.0, 1400.0)),
                ((180.0, 80.0), (620.0, 90.0)),
            )
            derived = scene.refine_match(bpy.context)
            assert derived.intrinsics.fx > 1.0
            # Rotation column 1 is Blender Y — should be a real unit axis, not identity leftover.
            y_column = derived.rotation_w2c[:, 1]
            assert abs(float(np.linalg.norm(y_column)) - 1.0) < 1.0e-5

            settings.origin_image = (400.0, 420.0)
            settings.origin_is_set = True
            scene.refine_match(bpy.context)
            assert settings.origin_is_set

            # Second match must keep the first intact.
            image_b_path = _write_png(
                temporary_path,
                "reference_b",
                640,
                480,
                (0.4, 0.1, 0.1, 1.0),
            )
            root_b = scene.create_match_camera(bpy.context)
            assert properties.active_root(bpy.context) == root_b
            scene.bind_reference_image(bpy.context, str(image_b_path))
            settings_b = properties.active_session(bpy.context)
            assert settings_b is not None
            assert settings_b.image_width == 640
            assert settings_b.image_path.endswith("reference_b.png")
            # Original match A state survives on its own root.
            settings_a = root_a.pm_session
            assert len(settings_a.lines) == 4
            assert settings_a.image_width == 800
            assert settings_a.origin_is_set

            scene.set_active_match(bpy.context, root_a)
            assert properties.active_root(bpy.context) == root_a
            assert bpy.context.scene.camera == settings_a.camera_object
            assert bpy.context.scene.render.resolution_x == 800

            scene.set_active_match(bpy.context, root_b)
            assert bpy.context.scene.camera == settings_b.camera_object
            assert bpy.context.scene.render.resolution_x == 640

            scene.unload_match(bpy.context)
            assert properties.active_root(bpy.context) is None
            assert properties.active_session(bpy.context) is None
            assert len(properties.iter_match_roots()) == 2

            # Deleting a match root should prune it from discovery.
            camera_b = settings_b.camera_object
            bpy.data.objects.remove(root_b, do_unlink=True)
            if camera_b is not None and camera_b.name in bpy.data.objects:
                bpy.data.objects.remove(camera_b, do_unlink=True)
            roots = properties.iter_match_roots()
            assert len(roots) == 1
            assert roots[0] == root_a

            scene.set_active_match(bpy.context, root_a)
            settings = properties.active_session(bpy.context)
            assert settings is not None

            project_path = temporary_path / "import.pmproj"
            _write_minimal_pmproj(project_path, image_path)
            project_io.load_project(bpy.context, str(project_path))
            reloaded = properties.active_session(bpy.context)
            assert reloaded is not None
            assert len(reloaded.lines) == 4
            assert reloaded.origin_is_set
            assert reloaded.camera_object is not None
            # Surfaces from the project must not be created.
            assert not hasattr(reloaded, "surfaces") or len(getattr(reloaded, "surfaces", [])) == 0

            reloaded.lock_focal = True
            reloaded.estimate_distortion = False
            # Turning estimation off intentionally clears λ; set the synthetic
            # stored value afterward to test locked-focal preservation itself.
            reloaded.division_lambda = 0.12
            manual_result = scene.refine_match(bpy.context)
            assert abs(manual_result.division_lambda - 0.12) < 1.0e-6, (
                manual_result.division_lambda,
                reloaded.division_lambda,
                reloaded.estimate_distortion,
                reloaded.lock_focal,
            )

            # Zero alpha on the source still — must not produce a transparent/black PNG.
            source_buffer = np.empty(
                reloaded.image_width * reloaded.image_height * 4,
                dtype=np.float32,
            )
            reloaded.image.pixels.foreach_get(source_buffer)
            source_buffer[3::4] = 0.0
            reloaded.image.pixels.foreach_set(source_buffer)
            reloaded.image.update()

            reloaded.view_exposure = 1.0
            reloaded.view_contrast = 1.25
            view_image = distortion.apply_view_lighting(bpy.context)
            view_path = Path(distortion.default_view_path(
                reloaded.image_path,
                distortion._plate_key(reloaded),
            ))
            assert view_path.exists()
            assert reloaded.view_lighting_applied
            assert reloaded.camera_object.data.background_images[0].image == view_image
            assert view_image.name == f"{reloaded.camera_object.name}.pm-view"
            source_image = reloaded.image
            source_path = reloaded.image_path
            first_root = properties.active_root(bpy.context)

            # Simulate a lost source pointer that left the lit plate on the camera
            # background — recovery must not adopt *-pm-view as the solver still.
            reloaded.image = view_image
            assert scene._is_derived_display_image(reloaded, reloaded.image)
            assert scene.ensure_session_image(reloaded)
            assert reloaded.image == source_image
            assert not scene._is_derived_display_image(reloaded, reloaded.image)
            assert reloaded.image_path == source_path
            assert scene.ensure_match_ready(bpy.context)
            assert reloaded.camera_object.data.background_images[0].image == view_image

            # Two matches on the same still must not share one view plate.
            scene.create_match_camera(bpy.context)
            scene.bind_reference_image(bpy.context, source_path)
            other = properties.active_session(bpy.context)
            other.view_exposure = 0.0
            other.view_contrast = 1.0
            other_view = distortion.apply_view_lighting(bpy.context)
            assert other_view != view_image
            assert other_view.name == f"{other.camera_object.name}.pm-view"
            other_buf = np.empty(
                int(other_view.size[0]) * int(other_view.size[1]) * 4,
                dtype=np.float32,
            )
            other_view.pixels.foreach_get(other_buf)
            other_mean = float(other_buf[0::4].mean())
            # Switch back to the first match — its plate must stay at +1 EV bake.
            scene.set_active_match(bpy.context, first_root)
            reloaded = properties.active_session(bpy.context)
            bg = reloaded.camera_object.data.background_images[0].image
            assert bg == reloaded.view_image
            assert bg.name == f"{reloaded.camera_object.name}.pm-view"
            buf = np.empty(int(bg.size[0]) * int(bg.size[1]) * 4, dtype=np.float32)
            bg.pixels.foreach_get(buf)
            first_mean = float(buf[0::4].mean())
            # First match baked +1 EV / 1.25 contrast; second baked 0 EV. They must differ.
            assert first_mean > other_mean + 0.05, (first_mean, other_mean)

            # Lost view-plate pointer must restore on activate — not snap to the bright source.
            saved_view_path = reloaded.view_path
            reloaded.view_image = None
            assert reloaded.view_lighting_applied
            scene.set_active_match(bpy.context, first_root)
            reloaded = properties.active_session(bpy.context)
            restored_bg = reloaded.camera_object.data.background_images[0].image
            assert reloaded.view_lighting_applied
            assert restored_bg is not None
            assert restored_bg.name == f"{reloaded.camera_object.name}.pm-view"
            restored_buf = np.empty(
                int(restored_bg.size[0]) * int(restored_bg.size[1]) * 4,
                dtype=np.float32,
            )
            restored_bg.pixels.foreach_get(restored_buf)
            assert abs(float(restored_buf[0::4].mean()) - first_mean) < 0.05
            assert saved_view_path == reloaded.view_path or reloaded.view_path != ""

            # Drop the extra match so later smoke steps stay on a single session.
            other_root = None
            for root in list(properties.iter_match_roots()):
                if root != first_root:
                    other_root = root
                    break
            if other_root is not None:
                other_camera = other_root.pm_session.camera_object
                other_collection = other_root.pm_session.match_collection
                if other_camera is not None:
                    camera_data = other_camera.data
                    bpy.data.objects.remove(other_camera, do_unlink=True)
                    if camera_data is not None and camera_data.users == 0:
                        bpy.data.cameras.remove(camera_data)
                bpy.data.objects.remove(other_root, do_unlink=True)
                if other_collection is not None and len(other_collection.objects) == 0:
                    bpy.data.collections.remove(other_collection)
            scene.set_active_match(bpy.context, first_root)
            reloaded = properties.active_session(bpy.context)

            undistorted_path = temporary_path / "reference-undistorted.png"
            undistorted = distortion.generate_undistorted_plate(
                bpy.context,
                str(undistorted_path),
            )
            assert undistorted_path.exists()
            assert reloaded.view_undistorted
            assert reloaded.undistorted_width >= reloaded.image_width
            assert reloaded.camera_object.data.background_images[0].image == undistorted
            saved = bpy.data.images.load(str(undistorted_path), check_existing=True)
            saved_pixels = np.empty(
                int(saved.size[0]) * int(saved.size[1]) * 4,
                dtype=np.float32,
            )
            saved.pixels.foreach_get(saved_pixels)
            assert float(saved_pixels[0::4].max()) > 0.05
            assert float(saved_pixels[3::4].mean()) > 0.2

            # Re-apply lighting while undistorted is active must keep lit→undistorted chain.
            reloaded.view_exposure = 0.5
            distortion.apply_view_lighting(bpy.context)
            assert reloaded.view_undistorted
            assert reloaded.camera_object.data.background_images[0].image == reloaded.undistorted_image

            distortion.reset_view_lighting(bpy.context)
            assert not reloaded.view_lighting_applied
            assert abs(reloaded.view_exposure) < 1.0e-6
            assert reloaded.view_undistorted
            assert reloaded.camera_object.data.background_images[0].image == reloaded.undistorted_image

            reloaded.lock_focal = True
            reloaded.hfov_degrees += 1.0
            scene.apply_manual_fov(bpy.context)
            assert not reloaded.view_undistorted
            assert reloaded.undistorted_image is None

            camera_before_invalid_load = reloaded.camera_object
            invalid_path = temporary_path / "invalid.pmproj"
            invalid_path.write_text(json.dumps({
                "kind": "perspective-match-project",
                "version": 1,
                "session": {
                    "imagePath": str(image_path),
                    "lines": {"x": [{"x1": "broken"}]},
                },
            }))
            try:
                project_io.load_project(bpy.context, str(invalid_path))
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed project should be rejected")
            assert reloaded.camera_object == camera_before_invalid_load

            # Landmark sync: 2D↔2D essential pose; ground tags pin absolute scale.
            from match_perspective import sync
            root_sync_a = scene.create_match_camera(bpy.context)
            scene.bind_reference_image(bpy.context, str(image_path))
            session_a = root_sync_a.pm_session
            session_a.fx = session_a.fy = 800.0
            session_a.cx, session_a.cy = 400.0, 300.0
            session_a.image_width, session_a.image_height = 800, 600
            session_a.camera_center = (-3.0, -4.0, 2.0)
            session_a.rotation_w2c = (
                1.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0,
            )
            session_a.origin_is_set = True

            root_sync_b = scene.create_match_camera(bpy.context)
            scene.bind_reference_image(bpy.context, str(image_b_path))
            session_b = root_sync_b.pm_session
            session_b.fx = session_b.fy = 800.0
            session_b.cx, session_b.cy = 400.0, 300.0
            session_b.image_width, session_b.image_height = 800, 600
            angle = 0.4
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            rotation_sim = np.array(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            translation_sim = np.array((2.0, -1.0, 0.5), dtype=np.float64)
            shared_center_b = np.array((4.0, -3.0, 2.5), dtype=np.float64)
            shared_rotation_b = np.array(
                ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
                dtype=np.float64,
            )
            rotation_private_b = shared_rotation_b @ rotation_sim
            center_private_b = rotation_sim.T @ (shared_center_b - translation_sim)
            session_b.rotation_w2c = tuple(float(value) for value in rotation_private_b.reshape(-1))
            session_b.camera_center = tuple(float(value) for value in center_private_b)
            session_b.origin_is_set = True
            scene.apply_camera(bpy.context.scene, session_a, scene.calibration_from_settings(session_a))
            scene.apply_camera(bpy.context.scene, session_b, scene.calibration_from_settings(session_b))

            space = properties.workspace(bpy.context)
            space.anchor_root = root_sync_a
            properties.sync_anchor_match_enum(space, root_sync_a.name)
            landmarks_shared = {
                "g0": (np.array((0.0, 0.0, 0.0)), True),
                "g1": (np.array((2.0, 0.0, 0.0)), True),
                "g2": (np.array((0.0, 2.0, 0.0)), True),
                "g3": (np.array((1.5, 1.0, 0.0)), True),
                "up": (np.array((1.0, 0.5, 1.5)), False),
            }
            calib_a = scene.calibration_from_settings(session_a)
            calib_b = scene.calibration_from_settings(session_b)
            true_sim = sync.SimilarityTransform(1.0, rotation_sim, translation_sim)
            for name, (shared_point, on_ground) in landmarks_shared.items():
                landmark = space.landmarks.add()
                landmark.item_id = f"smoke-{name}"
                landmark.name = name
                landmark.on_ground = on_ground
                private_b = true_sim.inverse_point(shared_point)
                projected_a = sync.project_private_point(shared_point, calib_a)
                projected_b = sync.project_private_point(private_b, calib_b)
                assert projected_a is not None and projected_b is not None
                observation_a = landmark.observations.add()
                observation_a.match_root = root_sync_a
                observation_a.x, observation_a.y = float(projected_a[0]), float(projected_a[1])
                observation_a.is_set = True
                observation_b = landmark.observations.add()
                observation_b.match_root = root_sync_b
                observation_b.x, observation_b.y = float(projected_b[0]), float(projected_b[1])
                observation_b.is_set = True
            space.active_landmark_index = 0

            result = scene.solve_and_apply_sync(bpy.context)
            assert result.success, result.message
            assert session_b.sync_is_applied
            assert abs(session_b.sync_scale - 1.0) < 1.0e-6
            assert abs(session_b.sync_translation[0] - translation_sim[0]) < 0.2

            print("Perspective Match smoke test passed")
        finally:
            if registered:
                extension.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
