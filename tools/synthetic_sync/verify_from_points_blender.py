"""Generated zero-solve RNA, stale-stroke and save/reopen checks for From Points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import create_scene, register_extension
from tools.synthetic_sync.scenarios import generate


def _by_id(space, item_id):
    return next(item for item in space.landmarks if item.item_id == item_id)


def check(out: Path) -> dict:
    from match_perspective import properties, scene
    from match_perspective.ui import operators, overlay

    space = properties.workspace(bpy.context)
    points = [item for item in space.landmarks if item.kind == "POINT"]
    assert len(points) >= 3
    endpoint_a, endpoint_b, other = points[:3]
    line = space.landmarks.add()
    line.item_id = "generated-from-points-line"
    line.name = "Generated From Points"
    line.kind = "LINE"
    line.line_source = "DRAWN"
    stale = line.observations.add()
    stale.match_root = properties.active_root(bpy.context)
    stale.x, stale.y, stale.x2, stale.y2 = 30, 40, 180, 210
    stale.is_set = True
    line.line_source = "FROM_POINTS"
    line.line_point_a = endpoint_a.item_id
    line.line_point_b = endpoint_b.item_id
    space.active_landmark_index = len(space.landmarks) - 1

    stale_known = bpy.data.objects.new("Generated Stale Known Line A", None)
    stale_known_b = bpy.data.objects.new("Generated Stale Known Line B", None)
    bpy.context.scene.collection.objects.link(stale_known)
    bpy.context.scene.collection.objects.link(stale_known_b)
    line.known_object = stale_known
    line.known_object_b = stale_known_b
    line.has_position = True
    line.has_line_segment = True
    line.position = (0.0, 0.0, 2.0)
    line.position_b = (1.0, 0.0, 2.0)

    request = scene.collect_sync_request(bpy.context)
    assert request.derived_lines == [
        (line.item_id, endpoint_a.item_id, endpoint_b.item_id)]
    assert all(item.landmark_id != line.item_id for item in request.observations)
    assert all(item.landmark_id != line.item_id for item in request.line_observations)
    assert line.item_id not in request.known_lines
    assert scene.landmark_rmse_px_in_match(line, properties.active_root(bpy.context)) is None
    assert not scene.project_known_object_into_match(
        line, properties.active_root(bpy.context))
    assert scene.landmark_viewport_object(line) is None

    # Exercise the real dashed subdivision while intercepting only the lowest
    # GPU primitives. Point landmarks keep their normal handles; the derived
    # line contributes dashed spans and no additional crosshairs.
    root = properties.active_root(bpy.context)
    expected_point_handles = sum(
        1 for item in space.landmarks if item.kind == "POINT"
        for observation in item.observations
        if observation.match_root == root and observation.is_set
    )
    spans, crosshairs = [], []
    with (patch.object(scene, "image_to_region",
                       side_effect=lambda _context, x, y: Vector((x, y))),
          patch.object(overlay, "_draw_line",
                       side_effect=lambda *args, **kwargs: spans.append((args, kwargs))),
          patch.object(overlay, "_draw_crosshair",
                       side_effect=lambda *args, **kwargs: crosshairs.append((args, kwargs)))):
        overlay._draw_landmarks(
            bpy.context, object(), properties.active_session(bpy.context))
    assert len(spans) >= 2
    assert len(crosshairs) == expected_point_handles
    initial_hash = request.evidence_sha256()
    line.line_point_b = other.item_id
    assert scene.collect_sync_request(bpy.context).evidence_sha256() != initial_hash
    line.line_point_b = endpoint_b.item_id

    # Stale Drawn data cannot be edited, cleared, or hit after source conversion.
    try:
        scene.set_landmark_line_observation(bpy.context, (1, 2), (3, 4))
    except ValueError as exc:
        assert "edited through" in str(exc)
    else:
        raise AssertionError("From Points accepted a drawn stroke")
    assert not scene.clear_landmark_observation_for_active(bpy.context)
    endpoint_states = []
    for item in space.landmarks:
        if item == line:
            continue
        for observation in item.observations:
            endpoint_states.append((observation, bool(observation.is_set)))
            observation.is_set = False
    with patch.object(scene, "image_to_region", return_value=Vector((50, 50))):
        assert operators._overlay_landmark_hit_index(
            bpy.context, Vector((50, 50))) == -1
    for observation, is_set in endpoint_states:
        observation.is_set = is_set

    # Stable enum numbers retain string IDs across reorder, unrelated deletion,
    # source-kind edits and a native save/reopen roundtrip.
    endpoint_ids = (endpoint_a.item_id, endpoint_b.item_id)
    index = next(i for i, item in enumerate(space.landmarks) if item.item_id == other.item_id)
    space.landmarks.move(index, 0)
    line = _by_id(space, "generated-from-points-line")
    assert (line.line_point_a, line.line_point_b) == endpoint_ids
    space.landmarks.remove(0)
    line = _by_id(space, "generated-from-points-line")
    assert (line.line_point_a, line.line_point_b) == endpoint_ids
    saved = out / "from-points.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(saved), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(saved), load_ui=False)
    space = properties.workspace(bpy.context)
    line = _by_id(space, "generated-from-points-line")
    assert (line.line_point_a, line.line_point_b) == endpoint_ids
    assert scene.collect_sync_request(bpy.context).derived_lines == [
        (line.item_id, *endpoint_ids)]

    endpoint_a = _by_id(space, endpoint_ids[0])
    endpoint_a.kind = "LINE"
    assert line.line_point_a == endpoint_ids[0]
    try:
        scene.collect_sync_request(bpy.context)
    except ValueError as exc:
        assert "missing or disabled point" in str(exc)
    else:
        raise AssertionError("Kind-changed endpoint was silently retargeted")
    endpoint_a.kind = "POINT"
    assert scene.collect_sync_request(bpy.context).derived_lines

    return {
        "passed": True,
        "numerical_solves": 0,
        "saved_reopened": True,
        "stable_endpoint_ids": list(endpoint_ids),
        "stale_stroke_ignored": True,
        "stale_known_ignored": True,
        "dashed_read_only_overlay": True,
        "missing_reference_diagnosed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=False)
    register_extension()
    create_scene(generate("free_scale", seed=0, noise_px=0.0), args.out, False)
    report = check(args.out)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("From Points Blender PASS:", report)


if __name__ == "__main__":
    main()
