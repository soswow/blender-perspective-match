"""Check no-VP/no-reference Blender preparation without running a numerical solve."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import create_scene, register_extension
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import environment


def check(case):
    from match_perspective import properties, scene
    from match_perspective.core import sync
    roots = list(properties.iter_match_roots())
    space = properties.workspace(bpy.context)
    assert all(not root.pm_session.lines and not root.pm_session.origin_is_set for root in roots)
    assert all(not point.on_ground and point.known_object is None for point in space.landmarks)
    before = scene.collect_sync_request(bpy.context).to_record()
    with patch.object(sync, "solve_landmark_sync", side_effect=AssertionError("Preparation must not solve")):
        prepared = scene.prepare_diagnose_sync(bpy.context)
        after = scene.collect_sync_request(bpy.context).to_record()
        assert before == after, "Preparation changed no-ground/no-VP inputs"
        assert not prepared.ground_frame_note
        assert not prepared.known_world and not prepared.fixed_similarities
        assert not scene.known_3d_iterate_roots(bpy.context)
        flags = {}
        for shared in (True, False):
            space.share_lens = shared
            prep = scene.prepare_lens_refine(bpy.context)
            flags[str(shared)] = {item.match_id: bool(item.freeze_focal) for item in prep.lens_inputs}
            assert all(value == (not shared) for value in flags[str(shared)].values())
    expected = {camera["id"]: camera for camera in case["request"]["cameras"]}
    for match in prepared.matches:
        wanted = expected[match.match_id]
        k = match.calibration.intrinsics
        assert np.allclose([k.fx, k.fy], [wanted["fx"], wanted["fy"]], rtol=2e-6, atol=1e-5)
        assert np.allclose(match.calibration.camera_center, wanted["center"], atol=2e-6)
        assert np.allclose(match.calibration.rotation_w2c, wanted["rotation"], atol=2e-6)
    return dict(passed=True, numerical_solves=0, request=before,
                freeze_focal=flags, environment=environment(), blender=bpy.app.version_string,
                limitations="Preparation and Manual FOV only; no numerical solve/apply, native modal or image decoding test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reopen", action="store_true", help="Read only the input generated in --out")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out.mkdir(parents=True, exist_ok=True)
    case = read_case(args.case)
    assert all(not p["ground"] and p["known"] is None for p in case["request"]["points"])
    assert not case["request"]["fixed_similarities"]
    register_extension()
    from match_perspective import core, properties, scene
    generated = args.out / "input.blend"
    if args.reopen:
        bpy.ops.wm.open_mainfile(filepath=str(generated.resolve()))
    else:
        create_scene(case, args.out, False)
        for root in properties.iter_match_roots():
            session = root.pm_session
            session.lines.clear()
            session.origin_is_set = False
            scene.set_active_match(bpy.context, root, record_history=False)
            session.hfov_degrees = core.hfov_from_focal(session.fx, session.image_width)
            scene.apply_manual_fov(bpy.context)
        bpy.ops.wm.save_as_mainfile(filepath=str(generated.resolve()), check_existing=False)
    report = check(case)
    (args.out / ("reopened.json" if args.reopen else "prepared.json")).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.reopen:
        previous = json.loads((args.out / "prepared.json").read_text())
        assert report["request"] == previous["request"], "Reopening changed captured inputs"
    else:
        subprocess.run([bpy.app.binary_path, "--factory-startup", "--disable-autoexec", "-b",
                        "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
                        "--", "--case", str(args.case.resolve()), "--out", str(args.out.resolve()),
                        "--reopen"], check=True, timeout=120)
    print("No-VP preparation verified; zero numerical solves")


if __name__ == "__main__":
    main()
