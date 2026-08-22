"""Probe a recovered still: ground vs off-plane RMSE after Solve Sync.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-sync/probe_resected.py -- \\
      --blend scene.blend [--match PM_top_face_Origin]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np


def _load_extension():
    """Register Perspective Match from this repo (factory-startup has no add-on)."""
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
            traceback.print_exc()
    return module


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe recovered-still landmark RMSE")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--match", default="", help="Match id; default = recovered still")
    return parser.parse_args(argv)


def _rmse(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def _point_error(sync_module, similarity, calibration, point, observation):
    projected = sync_module.project_private_point(
        similarity.inverse_point(point), calibration
    )
    if projected is None:
        return 1.0e3
    return float(np.hypot(projected[0] - observation.u, projected[1] - observation.v))


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    blend_path = Path(args.blend).expanduser().resolve()
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    from match_perspective import properties, scene
    from match_perspective.core import sync as sync_module

    matches, observations, known_world, line_observations, known_lines, parallel = (
        scene.build_sync_problem(bpy.context)
    )
    match_map = {item.match_id: item for item in matches}
    space = bpy.context.scene.match_perspective
    names = {landmark.item_id: landmark.name for landmark in space.landmarks}
    on_ground = {
        landmark.item_id: bool(landmark.on_ground) for landmark in space.landmarks
    }
    anchor = properties.anchor_root(bpy.context)
    result = sync_module.solve_landmark_sync(
        matches,
        observations,
        anchor_id=anchor.name,
        known_world=known_world,
        line_observations=line_observations,
        known_lines=known_lines,
        parallel_pairs=parallel,
    )
    print(result.message)
    print(f"per_match={ {k: round(v, 1) for k, v in result.per_match_rmse_px.items()} }")

    match_id = args.match
    if not match_id and "recovered '" in result.message:
        start = result.message.find("recovered '") + len("recovered '")
        match_id = result.message[start : result.message.find("'", start)]
    if not match_id or match_id not in result.similarities:
        print("No recovered match in the solve; pass --match")
        return 2

    similarity = result.similarities[match_id]
    calibration = match_map[match_id].calibration
    print(
        f"\n=== {match_id}  fx={calibration.intrinsics.fx:.1f} "
        f"fy={calibration.intrinsics.fy:.1f} ==="
    )
    print(f"{'name':28s} g  err_px      X      Y      Z")
    ground_err: list[float] = []
    off_err: list[float] = []
    rows = []
    for landmark_id, point in result.landmarks.items():
        observation = next(
            (
                item
                for item in observations
                if item.landmark_id == landmark_id and item.match_id == match_id
            ),
            None,
        )
        if observation is None:
            continue
        error = _point_error(
            sync_module, similarity, calibration, point, observation
        )
        flag = int(on_ground.get(landmark_id, False) or observation.on_ground)
        label = names.get(landmark_id, landmark_id)
        rows.append((error, label, flag, point))
        if flag:
            ground_err.append(error)
        else:
            off_err.append(error)
    for error, label, flag, point in sorted(rows, reverse=True):
        print(
            f"{label:28s} {flag}  {error:6.1f}  "
            f"{point[0]:6.2f} {point[1]:6.2f} {point[2]:6.2f}"
        )
    print(
        f"ground n={len(ground_err)} rmse={_rmse(ground_err):.1f}  "
        f"off-plane n={len(off_err)} rmse={_rmse(off_err):.1f}"
    )
    return 0


if __name__ == "__main__":
    try:
        separator = sys.argv.index("--")
        cli_args = sys.argv[separator + 1 :]
    except ValueError:
        cli_args = []
    raise SystemExit(main(cli_args))
