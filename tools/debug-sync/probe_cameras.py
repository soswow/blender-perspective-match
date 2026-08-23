"""Dump per-match K, distortion, and Blender camera FOV from a .blend.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-sync/probe_cameras.py -- \\
      --blend scene.blend [--out /tmp/pm-cameras.txt]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import bpy


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
    parser = argparse.ArgumentParser(description="Dump PM camera K / Blender lens")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def _report() -> str:
    from match_perspective import properties

    lines: list[str] = []
    scene_camera = bpy.context.scene.camera
    lines.append(
        f"scene.camera={scene_camera.name if scene_camera else None} "
        f"res={bpy.context.scene.render.resolution_x}x"
        f"{bpy.context.scene.render.resolution_y}"
    )
    roots = list(properties.iter_match_roots())
    lines.append(f"matches={len(roots)}")
    for root in roots:
        settings = getattr(root, "pm_session", None)
        if settings is None:
            lines.append(f"\n{root.name}  <no session>")
            continue
        camera = settings.camera_object
        data = camera.data if camera is not None else None
        bg = ""
        if data is not None and data.background_images:
            image = data.background_images[0].image
            bg = image.name if image is not None else "<none>"
        lens = float(data.lens) if data is not None else float("nan")
        sensor = float(data.sensor_width) if data is not None else float("nan")
        angle = float(data.angle) if data is not None else float("nan")
        shift_x = float(data.shift_x) if data is not None else float("nan")
        shift_y = float(data.shift_y) if data is not None else float("nan")
        cam_type = data.type if data is not None else "?"
        brown = tuple(float(value) for value in settings.brown_conrady[:5])
        hfov = 0.0
        if settings.fx > 1.0e-9 and settings.image_width > 0:
            hfov = math.degrees(
                2.0 * math.atan(settings.image_width / (2.0 * settings.fx))
            )
        scale = tuple(round(float(value), 6) for value in root.scale)
        lines.append(
            "\n"
            f"{root.name}  lock_focal={bool(settings.lock_focal)} "
            f"vp_mode={settings.vp_mode} "
            f"sync={bool(getattr(settings, 'sync_enabled', True))}"
        )
        lines.append(
            f"  image {settings.image_width}x{settings.image_height} "
            f"source_w={settings.source_image_width} "
            f"path={Path(settings.image_path).name if settings.image_path else ''}"
        )
        lines.append(
            f"  K fx={settings.fx:.3f} fy={settings.fy:.3f} "
            f"cx={settings.cx:.3f} cy={settings.cy:.3f} "
            f"hfov={hfov:.2f}°"
        )
        lines.append(
            f"  D λ={settings.division_lambda:.6g} brown={brown} "
            f"view_undistorted={bool(settings.view_undistorted)} "
            f"undistorted={settings.undistorted_width}x{settings.undistorted_height} "
            f"off=({settings.undistorted_offset_x:.1f},{settings.undistorted_offset_y:.1f})"
        )
        lines.append(
            f"  blender type={cam_type} lens={lens:.4f}mm sensor={sensor:.4f}mm "
            f"angle={math.degrees(angle):.2f}° shift=({shift_x:.4f},{shift_y:.4f}) "
            f"clip=({float(data.clip_start) if data else 0:.4g},"
            f"{float(data.clip_end) if data else 0:.4g}) bg={bg}"
        )
        lines.append(
            f"  empty_s={scale} cam_s="
            f"{tuple(round(float(v), 6) for v in camera.scale) if camera else ()}"
        )
        if camera is not None:
            loc = camera.matrix_world.translation
            lines.append(
                f"  cam_world=({loc.x:.4f},{loc.y:.4f},{loc.z:.4f}) "
                f"active={camera == scene_camera}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args(sys.argv[sys.argv.index("--") + 1 :])
    blend = Path(args.blend).expanduser().resolve()
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    text = _report()
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
