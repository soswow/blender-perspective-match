"""Dump Known 3D pin-refine inputs and trial metrics from a .blend.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-sync/probe_pin_refine.py -- \\
      --blend scene.blend [--match PM_construction-front_Origin]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import bpy
import numpy as np


def _load_extension():
    extension_directory = Path(__file__).resolve().parents[2]
    module_name = "match_perspective"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            extension_directory / "__init__.py",
            submodule_search_locations=[str(extension_directory)],
        )
        module = sys.modules[module_name] = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]
    if hasattr(module, "register"):
        try:
            module.register()
        except Exception:
            pass
    return module


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Known 3D camera pin polish")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--match", default="", help="Match root Empty name")
    return parser.parse_args(argv)


def _find_root(match_name: str):
    from match_perspective import properties

    roots = list(properties.iter_match_roots())
    if match_name:
        exact = next((root for root in roots if root.name == match_name), None)
        if exact is not None:
            return exact
        for root in roots:
            if match_name in root.name:
                return root
        raise SystemExit(
            f"Match '{match_name}' not found. Available: {[item.name for item in roots]}"
        )
    if len(roots) == 1:
        return roots[0]
    raise SystemExit(f"Pass --match. Available: {[item.name for item in roots]}")


def _collect_pins(root):
    from match_perspective import properties, scene
    from match_perspective.core import pin_refine
    from match_perspective.core import sync as sync_module

    space = properties.workspace(bpy.context)
    pins = []
    for landmark in space.landmarks:
        if getattr(landmark, "kind", "POINT") != "POINT":
            continue
        known_object = landmark.known_object
        if known_object is None or known_object.name not in bpy.data.objects:
            continue
        observation = scene.observation_for_match(landmark, root)
        if observation is None or not observation.is_set:
            continue
        world = known_object.matrix_world.to_translation()
        pins.append(
            pin_refine.KnownPin(
                landmark_id=str(landmark.item_id or landmark.name),
                point_private=scene._private_point_from_world(root, world),
                u=float(observation.x),
                v=float(observation.y),
                weight=sync_module.observation_weight(
                    observation.confidence,
                    float(getattr(landmark, "sync_weight", 1.0)),
                ),
                landmark_name=str(landmark.name or landmark.item_id),
            )
        )
    return pins


def _describe_candidate(label: str, calibration, pins, bundles) -> list[str]:
    from match_perspective import core
    from match_perspective.core import pin_refine

    metrics = pin_refine.pin_metrics(pins, calibration)
    vp_rms = core.vp_line_residual_rms(calibration, bundles)
    vp_ang = core.vp_angular_residual_degrees(calibration, bundles)
    lines = [
        f"{label}: fx={calibration.intrinsics.fx:.2f} "
        f"cx,cy=({calibration.intrinsics.cx:.1f},{calibration.intrinsics.cy:.1f}) "
        f"λ={calibration.division_lambda:.5f} "
        f"HFOV={calibration.hfov_degrees:.2f}°",
        f"  pin RMS={metrics.rms_px:.3f} max={metrics.max_px:.3f} "
        f"behind={list(metrics.behind_ids)}",
        f"  VP line RMSE={vp_rms:.3f} px  angle={vp_ang:.3f}°",
    ]
    axis_map = core.vp_line_axis_residuals(calibration, bundles)
    axis_text = " ".join(
        f"{axis}={axis_map[axis]:.2f}" for axis in ("x", "y", "z") if axis in axis_map
    )
    if axis_text:
        lines.append(f"  VP axis RMS (y=blue upright) {axis_text}")
    for pin, error in zip(pins, metrics.per_pin_px):
        lines.append(
            f"  {pin.landmark_name or pin.landmark_id}: "
            f"uv=({pin.u:.1f},{pin.v:.1f}) "
            f"xyz=({pin.point_private[0]:.3f},{pin.point_private[1]:.3f},"
            f"{pin.point_private[2]:.3f}) err={error:.2f}"
        )
    return lines


def main() -> int:
    args = _parse_args(sys.argv[sys.argv.index("--") + 1 :])
    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        print(f"Blend not found: {blend_path}", file=sys.stderr)
        return 2
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    from match_perspective import core, scene
    from match_perspective.core import pin_refine

    root = _find_root(args.match)
    settings = root.pm_session
    calibration = scene.calibration_from_settings(settings)
    bundles = scene.line_bundles_from_settings(settings)
    pins = _collect_pins(root)
    print(f"match={root.name}")
    print(
        f"image={settings.image_width}x{settings.image_height} "
        f"view_undistorted={bool(settings.view_undistorted)} "
        f"camera_control={settings.camera_control} "
        f"vp_mode={settings.vp_mode} lock_focal={settings.lock_focal}"
    )
    print(
        f"lines x={len(bundles.get('x', []))} "
        f"y={len(bundles.get('y', []))} z={len(bundles.get('z', []))}"
    )
    print(f"pins={len(pins)}")
    loc, rot, sca = root.matrix_world.decompose()
    print(
        f"root loc={tuple(round(v, 4) for v in loc)} "
        f"scale={tuple(round(v, 4) for v in sca)} "
        f"origin_set={settings.origin_is_set} "
        f"origin_uv={tuple(settings.origin_image) if settings.origin_is_set else None}"
    )
    print(
        f"cam_center={np.round(calibration.camera_center, 4).tolist()} "
        f"pp=({calibration.intrinsics.cx:.1f},{calibration.intrinsics.cy:.1f})"
    )
    from match_perspective.core.sync.projection import project_private_point

    for pin in pins[:2]:
        projected = project_private_point(pin.point_private, calibration)
        print(
            f"  project {pin.landmark_name}: "
            f"xyz={np.round(pin.point_private, 3).tolist()} "
            f"pick=({pin.u:.1f},{pin.v:.1f}) "
            f"proj={None if projected is None else (round(float(projected[0]), 1), round(float(projected[1]), 1))}"
        )
    print("\n".join(_describe_candidate("stored", calibration, pins, bundles)))

    pinhole_seed = pin_refine.copy_calibration(calibration)
    pinhole_seed.division_lambda = 0.0
    pinhole_seed.lambda_saturated = False
    pinhole_seed.brown_conrady = ()
    pinhole_seed.intrinsics.cx = float(calibration.intrinsics.image_width) * 0.5
    pinhole_seed.intrinsics.cy = float(calibration.intrinsics.image_height) * 0.5
    print("\n".join(_describe_candidate("pinhole seed (center PP)", pinhole_seed, pins, bundles)))

    reorient = core.refine_camera(
        bundles,
        pinhole_seed.intrinsics,
        lock_focal=True,
        estimate_principal_point=False,
        estimate_distortion=False,
        initial_division_lambda=0.0,
        initial_rotation=calibration.rotation_w2c,
    )
    reorient.camera_center = np.array(calibration.camera_center, copy=True)
    print("\n".join(_describe_candidate("pinhole + VP reorient", reorient, pins, bundles)))

    result = pin_refine.refine_from_known_pins(
        calibration, pins, bundles, orient_from_vp=True
    )
    print(
        f"\nresult success={result.success} hypothesis={result.hypothesis} "
        f"pin={result.pin_rms_px:.3f} vp={result.vp_line_rms_px:.3f} "
        f"axis={tuple(round(v, 2) for v in result.axis_rms_px)} "
        f"initial_pin={result.initial_pin_rms_px:.3f} "
        f"initial_vp={result.initial_vp_line_rms_px:.3f}"
    )
    print(f"message={result.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
