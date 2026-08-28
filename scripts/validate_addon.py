"""Headless registration, solver, and scene smoke test.

Run with:
``blender --factory-startup -b --python scripts/validate_addon.py``
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import traceback

import bpy
import numpy as np


def load_extension_module():
    """Import this extension package directly from its source directory."""
    extension_directory = Path(__file__).resolve().parents[1]
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
            pick_keys = [
                item
                for _keymap, item in extension._addon_keymaps
                if item.idname == "perspective_match.pick_in_active_match"
            ]
            assert len(pick_keys) == 1, pick_keys
            pick_key = pick_keys[0]
            assert pick_key.type == "A" and pick_key.value == "PRESS"
            assert pick_key.ctrl and pick_key.oskey
            assert not pick_key.shift and not pick_key.alt
            # Disable/re-enable and wheel extract can call register() twice.
            extension.register()
            extension.unregister()
            extension.register()
            core = sys.modules["match_perspective.core"]
            distortion = sys.modules["match_perspective.scene.distortion"]
            properties = sys.modules["match_perspective.properties"]
            scene = sys.modules["match_perspective.scene"]
            ui_panel = sys.modules["match_perspective.ui.panel"]

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
            counted_axis_items = properties._counted_axis_items(settings, bpy.context)
            counted_axes = {
                identifier: label
                for identifier, label, _description, _number in counted_axis_items
            }
            assert counted_axes == {
                "x": "X (Red) · 2",
                "z": "Y (Green) · 2",
                "y": "Z (Blue) · 0",
            }
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
            normal_polarity = scene.refine_match(bpy.context)
            assert settings.origin_is_set

            # Horizontal VP polarity flips X/Y together, immediately and after refine.
            settings.flip_xy_vp_polarity = True
            flipped = scene.calibration_from_settings(settings)
            assert np.allclose(
                flipped.rotation_w2c[:, :2],
                -normal_polarity.rotation_w2c[:, :2],
                atol=1.0e-7,
            )
            assert np.allclose(
                flipped.rotation_w2c[:, 2],
                normal_polarity.rotation_w2c[:, 2],
                atol=1.0e-7,
            )
            assert np.allclose(
                flipped.camera_center,
                (
                    -normal_polarity.camera_center[0],
                    -normal_polarity.camera_center[1],
                    normal_polarity.camera_center[2],
                ),
                atol=1.0e-7,
            )
            refined_flipped = scene.refine_match(bpy.context)
            assert np.allclose(
                refined_flipped.rotation_w2c,
                flipped.rotation_w2c,
                atol=1.0e-7,
            )
            settings.flip_xy_vp_polarity = False
            restored = scene.calibration_from_settings(settings)
            assert np.allclose(
                restored.rotation_w2c,
                normal_polarity.rotation_w2c,
                atol=1.0e-7,
            )

            # Replace Image keeps lines/origin when dimensions match.
            replacement_path = _write_png(
                temporary_path,
                "reference_replaced",
                800,
                600,
                (0.05, 0.35, 0.15, 1.0),
            )
            line_count_before = len(settings.lines)
            scene.replace_reference_image(bpy.context, str(replacement_path))
            assert settings.image_path.endswith("reference_replaced.png")
            assert len(settings.lines) == line_count_before
            assert settings.origin_is_set
            assert settings.origin_image[0] == 400.0
            mismatched = _write_png(
                temporary_path,
                "reference_wrong_size",
                640,
                480,
                (0.2, 0.2, 0.2, 1.0),
            )
            try:
                scene.replace_reference_image(bpy.context, str(mismatched))
                raise AssertionError("replace should reject size mismatch")
            except ValueError as error:
                assert "same size" in str(error).lower() or "×" in str(error)

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

            # Adjusted Camera is a live source, not a one-time snapshot. Changes
            # made after enabling it must survive View Match rehydration and a
            # switch away/back, including a later second pose/lens adjustment.
            camera_a = settings_a.camera_object
            settings_a.camera_control = "ADJUSTED"
            adjusted_matrix = camera_a.matrix_world.copy()
            adjusted_matrix.translation.x += 0.75
            adjusted_matrix.translation.z -= 0.25
            camera_a.matrix_world = adjusted_matrix
            camera_a.data.lens = 61.0
            first_live = scene.calibration_from_settings(settings_a)
            assert abs(first_live.hfov_degrees - np.degrees(camera_a.data.angle_x)) < 1.0e-5
            rebuilt_private = scene.private_camera_matrix(first_live)
            expected_private = root_a.matrix_world.inverted_safe() @ adjusted_matrix
            assert np.allclose(
                np.asarray(rebuilt_private),
                np.asarray(expected_private),
                atol=1.0e-6,
            )
            scene.set_active_match(bpy.context, root_b)
            scene.set_active_match(bpy.context, root_a)
            assert np.allclose(
                np.asarray(camera_a.matrix_world),
                np.asarray(adjusted_matrix),
                atol=1.0e-6,
            ), (camera_a.matrix_world, adjusted_matrix, camera_a.matrix_local)
            assert abs(camera_a.data.lens - 61.0) < 1.0e-6

            later_matrix = camera_a.matrix_world.copy()
            later_matrix.translation.y -= 0.5
            camera_a.matrix_world = later_matrix
            camera_a.data.lens = 73.0
            private_before_root_move = (
                root_a.matrix_world.inverted_safe() @ camera_a.matrix_world
            )
            root_matrix_before = root_a.matrix_world.copy()
            root_a.location.x += 2.0
            root_a.location.z += 1.0
            root_a.scale = (1.25, 1.25, 1.25)
            bpy.context.view_layer.update()
            live_under_moved_root = scene.calibration_from_settings(settings_a)
            assert np.allclose(
                np.asarray(scene.private_camera_matrix(live_under_moved_root)),
                np.asarray(private_before_root_move),
                atol=1.0e-6,
            )
            root_a.matrix_world = root_matrix_before
            bpy.context.view_layer.update()
            assert scene.ensure_match_ready(bpy.context)
            second_live = scene.calibration_from_settings(settings_a)
            assert not np.allclose(first_live.camera_center, second_live.camera_center)
            assert abs(second_live.hfov_degrees - np.degrees(camera_a.data.angle_x)) < 1.0e-5
            scene.set_active_match(bpy.context, root_b)
            scene.set_active_match(bpy.context, root_a)
            assert np.allclose(
                np.asarray(camera_a.matrix_world),
                np.asarray(later_matrix),
                atol=1.0e-6,
            )
            assert abs(camera_a.data.lens - 73.0) < 1.0e-6

            settings_a.camera_control = "MATCHED"
            assert scene.ensure_match_ready(bpy.context)
            assert abs(camera_a.data.lens - 73.0) > 1.0

            space = properties.workspace(bpy.context)
            settings_a.hide_origin_empty = True
            assert root_a.hide_get()
            assert not settings_a.camera_object.hide_viewport
            assert not settings_a.match_collection.hide_viewport
            assert not root_b.hide_get()
            scene.set_active_match(bpy.context, root_b)
            assert root_a.hide_get()
            assert not root_b.hide_get()
            assert not settings_b.hide_origin_empty
            root_b.hide_set(True)
            scene.set_active_match(bpy.context, root_a)
            assert settings_a.hide_origin_empty
            assert root_a.hide_get()
            scene.set_active_match(bpy.context, root_b)
            assert settings_b.hide_origin_empty
            assert root_b.hide_get()
            settings_b.hide_origin_empty = False
            assert not root_b.hide_get()
            assert root_a.hide_get()
            scene.set_active_match(bpy.context, root_a)

            # Dynamic Anchor enum can disagree with the pointer (saved index).
            # Getters and operator poll must not write Scene RNA to repair it.
            space.anchor_root = root_a
            properties.sync_anchor_match_enum(space, root_b.name)
            assert space.anchor_match == root_b.name
            assert properties.anchor_root(bpy.context) == root_a
            assert space.anchor_match == root_b.name
            bpy.ops.perspective_match.solve_sync.poll()
            bpy.ops.perspective_match.refine_lenses.poll()
            bpy.ops.perspective_match.diagnose_sync.poll()
            properties.reconcile_workspace_refs(space)
            assert space.anchor_match == root_a.name

            scene.set_active_match(bpy.context, root_b)
            assert bpy.context.scene.camera == settings_b.camera_object
            assert bpy.context.scene.render.resolution_x == 640

            scene.unload_match(bpy.context)
            assert properties.active_root(bpy.context) is None
            assert properties.active_session(bpy.context) is None
            assert len(properties.iter_match_roots()) == 2

            # Deleting a match root should prune it from discovery.
            scene.delete_match(bpy.context, root_b)
            roots = properties.iter_match_roots()
            assert len(roots) == 1
            assert roots[0] == root_a

            scene.set_active_match(bpy.context, root_a)
            # Rebind the original plate for downstream lighting/distortion checks.
            # Replace Image earlier left a different still on this match.
            scene.bind_reference_image(bpy.context, str(image_path))
            reloaded = properties.active_session(bpy.context)
            assert reloaded is not None
            reloaded.vp_mode = "2"
            reloaded.lines.clear()
            add_bundle(
                reloaded,
                "x",
                np.array((-600.0, 300.0)),
                ((200.0, 120.0), (220.0, 500.0)),
            )
            add_bundle(
                reloaded,
                "z",
                np.array((900.0, 300.0)),
                ((620.0, 100.0), (610.0, 520.0)),
            )
            reloaded.origin_image = (400.0, 420.0)
            reloaded.origin_is_set = True
            scene.refine_match(bpy.context)
            assert len(reloaded.lines) == 4
            assert reloaded.origin_is_set
            assert reloaded.camera_object is not None

            reloaded.lock_focal = True
            # Ordinary refine must keep a stored λ without re-fitting it.
            reloaded.division_lambda = 0.12
            manual_result = scene.refine_match(bpy.context)
            assert abs(manual_result.division_lambda - 0.12) < 1.0e-6, (
                manual_result.division_lambda,
                reloaded.division_lambda,
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
            assert view_path.parent.name == "post-processed"
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

            # PP drag previews on the active undistorted plate, then applies and
            # rebuilds it once on release. Mouse moves must not expose the source.
            ui_operators = sys.modules["match_perspective.ui.operators"]
            ui_overlay = sys.modules["match_perspective.ui.overlay"]
            original_cx = float(reloaded.cx)
            original_cy = float(reloaded.cy)
            pp_drag = SimpleNamespace(
                _drag_kind="PP",
                _start=None,
                _original=(original_cx, original_cy),
                _edit_index=-1,
                report=lambda *_args: None,
                _status_prompt=lambda: "Drag the principal point",
            )
            preview_point = (original_cx + 12.0, original_cy + 8.0)
            ui_operators.PM_OT_interact._update_drag(
                pp_drag,
                bpy.context,
                preview_point,
            )
            assert abs(float(reloaded.cx) - original_cx) < 1.0e-6
            assert abs(float(reloaded.cy) - original_cy) < 1.0e-6
            assert reloaded.camera_object.data.background_images[0].image == undistorted
            ui_operators.PM_OT_interact._complete_drag(
                pp_drag,
                bpy.context,
                preview_point,
            )
            assert abs(float(reloaded.cx) - preview_point[0]) < 1.0e-6
            assert abs(float(reloaded.cy) - preview_point[1]) < 1.0e-6
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is not None
            assert (
                reloaded.camera_object.data.background_images[0].image
                == reloaded.undistorted_image
            )
            ui_overlay.clear_preview(bpy.context)

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
            # FOV change regenerates the undistorted plate when one is being viewed.
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is not None

            scene.invalidate_undistorted_cache(reloaded)
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is None
            distortion.sync_undistorted_plate_after_refine(bpy.context)
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is not None

            reloaded.brown_conrady = (0.12, -0.25, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0)
            reloaded.division_lambda = 0.0
            distortion.generate_undistorted_plate(bpy.context)
            distortion.use_original_plate(bpy.context)
            assert not reloaded.view_undistorted
            assert abs(float(reloaded.brown_conrady[0]) - 0.12) < 1.0e-6
            distortion.use_undistorted_plate(bpy.context)
            assert reloaded.view_undistorted
            assert reloaded.undistorted_image is not None
            reloaded.brown_conrady = (0.0,) * 8
            reloaded.division_lambda = 0.12

            # Landmark sync: 2D↔2D essential pose; ground tags pin absolute scale.
            from match_perspective.core import sync
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
            helper = bpy.data.objects.get("PM_LM_g0")
            assert helper is not None, "Solve Sync should create landmark Empties"
            assert helper.get(scene.LANDMARK_HELPER_ID_KEY) == "smoke-g0"
            assert scene.landmark_index_for_helper(space, helper) == 0
            assert session_b.sync_is_applied
            assert abs(session_b.sync_scale - 1.0) < 1.0e-6, (
                session_b.sync_scale,
                result.message,
            )
            # This broad scene smoke accepts the solver's robust-BA variation;
            # exact synthetic recovery is covered in tests/test_sync.py.
            assert abs(session_b.sync_translation[0] - translation_sim[0]) < 0.35, (
                tuple(session_b.sync_translation),
                result.message,
            )

            # Diagnose copies bpy data on main, solves pure data, then applies on main.
            diagnose_progress = []
            diagnose_prep = scene.prepare_diagnose_sync(bpy.context)
            diagnose_result = scene.run_diagnose_sync(
                diagnose_prep,
                progress_callback=lambda step, total, label: diagnose_progress.append(
                    (step, total, label)
                ),
            )
            assert diagnose_result.success, diagnose_result.message
            scene.apply_diagnose_sync_result(
                bpy.context,
                diagnose_prep,
                diagnose_result,
            )
            assert diagnose_progress[-1][:2] == (6, 6), diagnose_progress[-1]

            # Landmark list filter keeps only picks defined in the active match.
            local_landmark = space.landmarks.add()
            local_landmark.item_id = "smoke-local-a"
            local_landmark.name = "Only in A"
            local_observation = local_landmark.observations.add()
            local_observation.match_root = root_sync_a
            local_observation.is_set = True
            space.landmarks_filter_current_match = True
            scene.set_active_match(bpy.context, root_sync_b)
            filter_bit = 1 << 30
            filter_owner = SimpleNamespace(bitflag_filter_item=filter_bit)
            filter_flags, _filter_order = ui_panel.PM_UL_landmarks.filter_items(
                filter_owner,
                bpy.context,
                space,
                "landmarks",
            )
            assert filter_flags[0] == filter_bit
            assert filter_flags[len(space.landmarks) - 1] == 0
            scene.set_active_match(bpy.context, root_sync_a)
            filter_flags, _filter_order = ui_panel.PM_UL_landmarks.filter_items(
                filter_owner,
                bpy.context,
                space,
                "landmarks",
            )
            assert filter_flags[len(space.landmarks) - 1] == filter_bit
            space.landmarks_filter_current_match = False

            # Full K + shared ground picks can initialize orientation without VPs.
            root_sync_c = scene.create_match_camera(bpy.context)
            scene.bind_reference_image(bpy.context, str(image_b_path))
            session_c = root_sync_c.pm_session
            session_c.lock_focal = True
            session_c.fx = session_c.fy = 800.0
            session_c.cx, session_c.cy = 400.0, 300.0
            session_c.image_width, session_c.image_height = 800, 600
            session_c.rotation_w2c = tuple(
                float(value) for value in calib_a.rotation_w2c.reshape(-1)
            )
            session_c.camera_center = (1.5, -3.5, 2.8)
            calib_c = scene.calibration_from_settings(session_c)
            for landmark in space.landmarks:
                if not landmark.item_id.startswith("smoke-g"):
                    continue
                shared_point = landmarks_shared[landmark.name][0]
                projected_c = sync.project_private_point(shared_point, calib_c)
                assert projected_c is not None
                observation_c = landmark.observations.add()
                observation_c.match_root = root_sync_c
                observation_c.x, observation_c.y = (
                    float(projected_c[0]),
                    float(projected_c[1]),
                )
                observation_c.is_set = True
            identity_rotation = (
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            for session in (session_a, session_b, session_c):
                session.lock_focal = True
                session.rotation_w2c = identity_rotation
            ground_note = scene.ensure_ground_frame_from_landmarks(bpy.context)
            assert ground_note.startswith("Ground frame inferred"), ground_note
            inferred_a = np.asarray(session_a.rotation_w2c).reshape(3, 3)
            assert float(np.dot(inferred_a[:, 2], calib_a.rotation_w2c[:, 2])) > 0.99
            session_c.sync_enabled = False

            # Bulk Create: skip stills that already have a match; copy locked K.
            bulk_dir = temporary_path / "bulk_stills"
            bulk_dir.mkdir()
            bulk_a = _write_png(bulk_dir, "bulk_a", 800, 600, (0.4, 0.1, 0.1, 1.0))
            _write_png(bulk_dir, "bulk_b", 800, 600, (0.1, 0.4, 0.1, 1.0))
            _write_png(bulk_dir, "bulk_c", 400, 300, (0.1, 0.1, 0.4, 1.0))
            scene.set_active_match(bpy.context, root_sync_a)
            scene.bind_reference_image(bpy.context, str(bulk_a))
            session_a.lock_focal = True
            session_a.fx = 910.0
            session_a.fy = 930.0
            session_a.cx = 390.0
            session_a.cy = 305.0
            session_a.brown_conrady = (
                -0.08,
                0.015,
                0.001,
                -0.002,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            source_k = (
                float(session_a.fx),
                float(session_a.fy),
                float(session_a.cx),
                float(session_a.cy),
            )
            created, skipped = scene.bulk_create_match_cameras(
                bpy.context, str(bulk_dir)
            )
            assert created == 2, (created, skipped)
            assert skipped == 1
            created_again, skipped_again = scene.bulk_create_match_cameras(
                bpy.context, str(bulk_dir)
            )
            assert created_again == 0
            assert skipped_again == 3
            last = properties.active_session(bpy.context)
            assert last is not None and last.lock_focal
            assert last.image_path.endswith("bulk_c.png")
            expected_k = tuple(value * 0.5 for value in source_k)
            actual_k = (
                float(last.fx),
                float(last.fy),
                float(last.cx),
                float(last.cy),
            )
            assert np.allclose(actual_k, expected_k, atol=1.0e-3), (
                actual_k,
                expected_k,
            )
            assert np.allclose(
                tuple(last.brown_conrady), tuple(session_a.brown_conrady)
            )
            assert last.view_undistorted
            assert last.undistorted_image is not None

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
