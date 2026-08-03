"""Blender-side dump of one Perspective Match session to JSON.

Invoked by ``explore.py``; not meant to be run by hand unless debugging:

  blender --factory-startup -b scene.blend --python extract_match.py -- \\
      --match "PM Root" --out /tmp/match.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import bpy


def _load_extension():
    """Register Perspective Match from the sibling repo checkout."""
    extension_directory = Path(__file__).resolve().parents[2]
    module_name = "match_perspective"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            extension_directory / "__init__.py",
            submodule_search_locations=[str(extension_directory)],
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec so dataclasses / relative imports resolve.
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    if hasattr(module, "register"):
        try:
            module.register()
        except Exception:
            # Already registered from a previous run in the same Blender process.
            pass
    return module


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump one PM match to JSON")
    parser.add_argument("--match", required=True, help="Match root Empty name (dropdown id)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--blend",
        default="",
        help="Optional .blend to open (when not already loaded via CLI)",
    )
    return parser.parse_args(argv)


def _session_payload(root, session) -> dict:
    lines = {
        "x": [],
        "y": [],
        "z": [],
    }
    for item in session.lines:
        lines[item.axis].append(
            {
                "x1": float(item.x1),
                "y1": float(item.y1),
                "x2": float(item.x2),
                "y2": float(item.y2),
            }
        )
    width = max(int(session.image_width), 1)
    height = max(int(session.image_height), 1)
    cx = float(session.cx)
    cy = float(session.cy)
    return {
        "match_name": root.name,
        "vp_mode": str(session.vp_mode),
        "lock_focal": bool(session.lock_focal),
        "image_width": width,
        "image_height": height,
        "hfov_degrees": float(session.hfov_degrees),
        "fx": float(session.fx),
        "fy": float(session.fy),
        "cx": cx,
        "cy": cy,
        "pp_offset_x": cx - 0.5 * width,
        "pp_offset_y": cy - 0.5 * height,
        "division_lambda": float(session.division_lambda),
        "residual_degrees": float(session.residual_degrees),
        "lines": lines,
        "line_counts": {axis: len(segments) for axis, segments in lines.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :])
    if args.blend:
        blend_path = Path(args.blend).expanduser().resolve()
        if not blend_path.is_file():
            print(f"Blend not found: {blend_path}", file=sys.stderr)
            return 2
        # PropertyGroups must be registered before opening the file.
        _load_extension()
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    else:
        _load_extension()

    from match_perspective import properties

    root = bpy.data.objects.get(args.match)
    if root is None or not properties.is_match_root(root):
        available = [
            obj.name
            for obj in bpy.data.objects
            if properties.is_match_root(obj)
        ]
        print(
            f"Match '{args.match}' not found. Available: {available}",
            file=sys.stderr,
        )
        return 3

    payload = _session_payload(root, root.pm_session)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    # Blender appends its own args; everything after "--" is ours.
    try:
        separator = sys.argv.index("--")
        cli_args = sys.argv[separator + 1 :]
    except ValueError:
        cli_args = []
    raise SystemExit(main(cli_args))
