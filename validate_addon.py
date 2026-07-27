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

            surface = settings.surfaces.add()
            surface.plane = "xz"
            surface.x1, surface.y1 = (260.0, 340.0)
            surface.x2, surface.y2 = (560.0, 510.0)
            surface.divisions = 4
            scene.rebuild_surface_meshes(bpy.context, calibration)
            assert surface.mesh_object is not None
            assert len(surface.mesh_object.data.vertices) == 4

            settings.origin_image = (400.0, 420.0)
            settings.origin_is_set = True
            settings.scale_point_a = (330.0, 430.0)
            settings.scale_point_b = (520.0, 430.0)
            settings.scale_point_count = 2
            settings.measured_length = 2.0
            scene.refine_match(bpy.context)
            assert settings.solved_scale > 0.0

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

            project_path = temporary_path / "roundtrip.pmproj"
            project_io.save_project(settings, str(project_path))
            payload = json.loads(project_path.read_text())
            assert payload["kind"] == "perspective-match-project"
            assert len(payload["session"]["lines"]["x"]) == 2
            assert len(payload["session"]["surfaces"]) == 1

            project_io.load_project(bpy.context, str(project_path))
            reloaded = properties.active_session(bpy.context)
            assert reloaded is not None
            assert len(reloaded.lines) == 4
            assert len(reloaded.surfaces) == 1
            assert reloaded.camera_object is not None

            reloaded.division_lambda = 0.12
            reloaded.lock_focal = True
            reloaded.estimate_distortion = False
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

            plate_project = temporary_path / "with-plate.pmproj"
            project_io.save_project(reloaded, str(plate_project))
            project_io.load_project(bpy.context, str(plate_project))
            reloaded = properties.active_session(bpy.context)
            assert reloaded is not None
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is not None

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

            camera_json_path = temporary_path / "camera.json"
            project_io.save_camera_json(reloaded, str(camera_json_path))
            camera_payload = json.loads(camera_json_path.read_text())
            rotation = np.asarray(camera_payload["R"])
            center = np.asarray(camera_payload["cameraCenter"])
            assert np.allclose(camera_payload["t"], -rotation @ center)

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
