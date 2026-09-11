"""Check withheld object alignment across automatic origin preparation in Blender."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.blender_case import assert_equivalent, create_scene, register_extension, run_case
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import fingerprint, solver_arguments


def original_case(family, state):
    case = generate(family)
    case["name"] = f"preparation-{family}-{state}"
    case["preparation_state"] = state
    if state == "locked_origin":
        actual, stored = case["truth"]["cameras"][2], case["request"]["cameras"][2]
        rotation = np.asarray(actual["rotation"]).T @ np.asarray(stored["rotation"])
        case["request"]["fixed_similarities"]["view_2"] = dict(scale=1.0, rotation=rotation.tolist(),
            translation=(np.asarray(actual["center"])-rotation@stored["center"]).tolist())
    return case


def prepare_case(case):
    """Allow only the non-anchor private center to change; retain independent truth."""
    from match_perspective import properties, scene
    prep = scene.prepare_diagnose_sync(bpy.context)
    expected = solver_arguments(case["request"])
    for key, value in expected.items():
        if key in {"matches", "observations", "line_observations"}:
            continue
        assert_equivalent(value, getattr(prep, key), key)
    for key in ("observations", "line_observations"):
        actual = deepcopy(getattr(prep, key))
        for observation in actual:
            observation.landmark_name = ""
        assert_equivalent(sorted(expected[key], key=lambda o:(o.landmark_id,o.match_id)),
                          sorted(actual, key=lambda o:(o.landmark_id,o.match_id)), key)
    prepared = deepcopy(case)
    matches = {match.match_id: match for match in prep.matches}
    state = case["preparation_state"]
    for camera in prepared["request"]["cameras"]:
        wanted = next(match.calibration for match in expected["matches"] if match.match_id == camera["id"])
        actual = matches[camera["id"]].calibration
        assert_equivalent(wanted.intrinsics, actual.intrinsics, camera["id"]+".intrinsics")
        assert_equivalent(wanted.rotation_w2c, actual.rotation_w2c, camera["id"]+".rotation")
        assert_equivalent(wanted.brown_conrady, actual.brown_conrady, camera["id"]+".distortion")
        if state == "missing_origin" and camera["id"] == "view_2":
            assert np.linalg.norm(actual.camera_center-wanted.camera_center) > 0.1, "Origin was not actually initialized"
        else:
            assert_equivalent(wanted.camera_center, actual.camera_center, camera["id"]+".center")
        camera.update(center=actual.camera_center.tolist(), rotation=actual.rotation_w2c.tolist())
        camera.update(fx=actual.intrinsics.fx, fy=actual.intrinsics.fy, cx=actual.intrinsics.cx, cy=actual.intrinsics.cy)
    target = next(root for root in properties.iter_match_roots() if root.get("synthetic_match_id") == "view_2")
    assert bool(target.pm_session.origin_is_set) == (state != "locked_origin")
    assert bool(prep.auto_origin_notes) == (state == "missing_origin")
    prepared["preparation_notes"] = prep.auto_origin_notes
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("ground", "overhead"), default="ground")
    parser.add_argument("--state", choices=("preset", "missing_origin", "locked_origin"), default="missing_origin")
    parser.add_argument("--case", type=Path, help="Replay an exact original-case.json")
    parser.add_argument("--load", type=Path, help="Reopen its generated raw input.blend")
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    case = read_case(args.case) if args.case else original_case(args.family, args.state)
    write_case(case, args.out / "original-case.json")
    register_extension()
    from match_perspective import properties
    if args.load:
        bpy.ops.wm.open_mainfile(filepath=str(args.load.resolve()))
        if bpy.context.scene.get("synthetic_sync_request") != fingerprint(case["request"]):
            raise ValueError("Expected the generated raw input for this case")
    else:
        create_scene(case, args.out, False)
        if case["preparation_state"] != "preset":
            target = next(root for root in properties.iter_match_roots() if root.get("synthetic_match_id") == "view_2")
            target.pm_session.origin_is_set = False
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "input.blend"), check_existing=False)
    prepared = prepare_case(case)
    write_case(prepared, args.out / "prepared-case.json")
    bpy.context.scene["synthetic_sync_request"] = fingerprint(prepared["request"])
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "prepared.blend"), check_existing=False)
    passed = run_case(prepared, args.out)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "solved.blend"), check_existing=False)
    if args.roundtrip:
        process = subprocess.run([bpy.app.binary_path, "--factory-startup", "--disable-autoexec", "-b",
            "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
            "--case", str(args.out / "original-case.json"), "--load", str(args.out / "input.blend"),
            "--out", str(args.out / "reopened")])
        passed = passed and process.returncode == 0
    if not passed:
        raise RuntimeError("Origin preparation failed withheld-object verification; inspect report.html")


if __name__ == "__main__":
    main()
