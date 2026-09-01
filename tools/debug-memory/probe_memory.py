"""Dump Blender datablock / Python memory from a .blend (read-only).

Does not write the blend. GPU / Metal heaps are invisible here — use
`probe_process.sh` against a live GUI process for IOAccelerator.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-memory/probe_memory.py -- \\
      --blend scene.blend [--out /tmp/pm-memory.txt] [--addon]
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
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
    parser = argparse.ArgumentParser(description="Dump PM-related memory facts")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--addon",
        action="store_true",
        help="Register this checkout's Perspective Match before dumping",
    )
    return parser.parse_args(argv)


def _rss_now_kb(pid: int) -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip() or 0)


def _numpy_live_bytes() -> int:
    try:
        import numpy as np
    except ImportError:
        return 0
    total = 0
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            try:
                total += int(obj.nbytes)
            except Exception:
                pass
    return total


def _report() -> str:
    gc.collect()
    pid = os.getpid()
    lines: list[str] = []
    filepath = bpy.data.filepath or "<unsaved>"
    lines.append(f"file={filepath}")
    lines.append(f"pid={pid} rss_now_mb={_rss_now_kb(pid) / 1024:.1f}")
    lines.append(f"addon_match_perspective={'match_perspective' in sys.modules}")
    undo_steps = bpy.context.preferences.edit.undo_steps
    lines.append(f"undo_steps={undo_steps}")

    images = list(bpy.data.images)
    pixel_bytes = 0
    packed = 0
    fake_user = 0
    pm_named = 0
    lines.append(f"images={len(images)}")
    for image in images:
        width, height = image.size
        size_bytes = int(width) * int(height) * (16 if image.is_float else 4)
        pixel_bytes += size_bytes
        is_packed = image.packed_file is not None
        packed += int(is_packed)
        fake_user += int(bool(image.use_fake_user))
        name = image.name
        if ".pm-" in name or name.endswith(".pm-view"):
            pm_named += 1
        lines.append(
            f"  {name!r} {width}x{height} "
            f"float={bool(image.is_float)} packed={is_packed} "
            f"fake_user={image.use_fake_user} users={image.users} "
            f"~{size_bytes / (1024 ** 2):.1f}MiB"
        )
    lines.append(
        f"image_pixels_approx_mb={pixel_bytes / (1024 ** 2):.1f} "
        f"packed={packed} fake_user={fake_user} pm_named={pm_named}"
    )
    npy = _numpy_live_bytes()
    lines.append(f"numpy_live_mb={npy / (1024 ** 2):.1f}")
    for kind in ("meshes", "objects", "cameras", "materials", "textures", "screens"):
        lines.append(f"bpy.data.{kind}={len(getattr(bpy.data, kind))}")

    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            bpy.ops.wm.memory_statistics()
        stats = captured.getvalue().strip()
        if stats:
            lines.append("=== memory_statistics ===")
            lines.append(stats)
    except Exception as exc:
        lines.append(f"memory_statistics_failed={exc!r}")
    return "\n".join(lines) + "\n"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)
    if args.addon:
        _load_extension()
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    text = _report()
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
