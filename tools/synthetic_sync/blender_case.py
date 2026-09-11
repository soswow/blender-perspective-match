"""Build and verify synthetic .blend files; run with factory-startup Blender."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.synthetic_sync.evaluation import alignment, evaluate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import calibration, environment, fingerprint, result_record, solver_arguments
from tools.synthetic_sync.run import write_report


def register_extension():
    spec = importlib.util.spec_from_file_location("match_perspective", ROOT / "__init__.py",
                                                submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.register()
    return module


def native_camera(camera, collection):
    """Truth camera constructed directly, without PM's camera application code."""
    data = bpy.data.cameras.new("Truth_" + camera["id"])
    obj = bpy.data.objects.new(data.name, data)
    collection.objects.link(obj)
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = camera["fx"] * 36.0 / camera["width"]
    data.shift_x = 0.5 - camera["cx"] / camera["width"]
    data.shift_y = (camera["cy"] - camera["height"] / 2) / camera["width"]
    basis = np.asarray(camera["rotation"])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([basis[0], -basis[1], -basis[2]])
    matrix[:3, 3] = camera["center"]
    obj.matrix_world = Matrix(matrix)
    return obj


def blender_pixels(camera_obj, camera, points):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = camera["width"], camera["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    bpy.context.view_layer.update()
    evaluated = camera_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    coordinates = [world_to_camera_view(scene, evaluated, Vector(point)) for point in points]
    if any(p.z <= 0 for p in coordinates):
        raise AssertionError("Withheld geometry is behind an evaluated Blender camera")
    return np.array([(p.x * camera["width"], (1 - p.y) * camera["height"]) for p in coordinates])


def create_scene(case, out: Path, render: bool):
    from match_perspective import properties, scene
    # Factory scene only: this command never modifies a user's scene in place.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    collection = bpy.data.collections.new("Synthetic truth")
    bpy.context.scene.collection.children.link(collection)
    mesh = bpy.data.meshes.new("Withheld object geometry")
    mesh.from_pydata(case["truth"]["mesh"]["vertices"], [], case["truth"]["mesh"]["faces"])
    mesh.update()
    obj = bpy.data.objects.new("Reference object", mesh)
    collection.objects.link(obj)
    request = case["request"]
    images = {}
    for camera in case["truth"]["cameras"]:
        truth = native_camera(camera, collection)
        checks = [p["position"] for p in case["truth"]["checks"] if camera["id"] in p["views"]]
        expected = project(checks, camera)[0]
        actual = blender_pixels(truth, camera, checks)
        if not np.allclose(expected, actual, atol=0.002, rtol=0):
            raise AssertionError(f"Independent oracle disagrees with native Blender: {camera['id']}")
        if render:
            bpy.context.scene.camera = truth
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
            bpy.context.scene.display.shading.light = "STUDIO"
            bpy.context.scene.display.shading.color_type = "SINGLE"
            bpy.context.scene.display.shading.single_color = (0.32, 0.55, 0.7)
            bpy.context.scene.render.image_settings.file_format = "PNG"
            path = out / (camera["id"] + ".png")
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(path), check_existing=False)
        else:
            image = bpy.data.images.new("Blank_" + camera["id"], camera["width"], camera["height"])
            image.generated_color = (0.12, 0.16, 0.21, 1)
        image.pack()
        images[camera["id"]] = image
        truth.hide_set(True)
    roots = {}
    for camera in request["cameras"]:
        root = scene.create_match_camera(bpy.context)
        root.name = camera["id"]
        root["synthetic_match_id"] = camera["id"]
        roots[camera["id"]] = root
        session = root.pm_session
        session.image = images[camera["id"]]
        session.image_path = ""
        session.image_width, session.image_height = camera["width"], camera["height"]
        session.source_image_width = camera["width"]
        session.origin_is_set = True
        # Exercise Sync on stored VP-style calibration without invoking VP detection
        # or ground-only calibration initialization as an unrelated extra stage.
        session.lock_focal = False
        session.camera_object.data.background_images.new().image = session.image
        scene.apply_camera(bpy.context.scene, session, calibration(camera))
        fixed = request["fixed_similarities"].get(camera["id"])
        if fixed:
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(fixed["rotation"]) * fixed["scale"]
            matrix[:3, 3] = fixed["translation"]
            root.matrix_world = Matrix(matrix)
            session.sync_role = "LOCK_POSE"
            session.sync_lock_pose = True
        else:
            # Existing cases omit these keys; default Solve so Blender matches
            # the numerical solver's "every camera may move 3D" contract.
            location_ids = request.get("location_match_ids")
            location_ids = set(
                [item["id"] for item in request["cameras"]] if location_ids is None else location_ids
            )
            readonly_ids = set(request.get("readonly_match_ids") or [])
            if camera["id"] in readonly_ids or camera["id"] not in location_ids:
                session.sync_role = "FIT_ONLY"
                session.sync_lock_pose = False
            else:
                session.sync_role = "SOLVE"
                session.sync_lock_pose = False
    workspace = properties.workspace(bpy.context)
    workspace.show_landmark_empties = True
    workspace.anchor_root = roots[request["anchor_id"]]
    for key in ("lock_rotation", "lock_translation", "ground_slack", "known_3d_slack", "mirror_slack", "plane_slack"):
        if key in request:
            setattr(workspace, key, request[key])
    if request["mirror_plane"]:
        mirror = bpy.data.objects.new("Mirror evidence", None)
        bpy.context.scene.collection.objects.link(mirror)
        mirror.location = request["mirror_plane"][0]
        # Pilot scenarios use X-normal reflection; reject unsupported data.
        if request["mirror_plane"][1] != [1, 0, 0]:
            raise ValueError("Blender pilot currently supports X-normal mirror planes")
        workspace.mirror_object = mirror
        workspace.mirror_plane = "YZ"
    landmarks = {}

    def known_empty(item_id, position):
        pin = bpy.data.objects.new("Known_" + item_id, None)
        bpy.context.scene.collection.objects.link(pin)
        pin.location = position
        return pin

    for item in request["points"] + request["lines"]:
        landmark = workspace.landmarks.add()
        landmark.item_id = landmark.name = item["id"]
        landmark.kind = "POINT" if "ground" in item else "LINE"
        landmark.on_ground = item.get("ground", False)
        if item["known"] is not None:
            if landmark.kind == "POINT":
                landmark.known_object = known_empty(item["id"], item["known"])
            else:
                landmark.known_object = known_empty(item["id"] + "_a", item["known"][0])
                landmark.known_object_b = known_empty(item["id"] + "_b", item["known"][1])
    # Collection growth can invalidate previously returned RNA element wrappers.
    landmarks = {landmark.item_id: landmark for landmark in workspace.landmarks}
    for a, b in request["mirror_pairs"]:
        landmarks[a].mirror_of_id = b
    for a, b in request["parallel_pairs"]:
        landmarks[a].parallel_to = b
    for landmark_id, axis, group in request.get("plane_groups") or []:
        landmark = landmarks.get(landmark_id)
        if landmark is None:
            continue
        landmark.plane_axis = str(axis)
        landmark.plane_group = str(group)
    for observation in request["observations"] + request["line_observations"]:
        landmark = landmarks[observation["landmark_id"]]
        pick = landmark.observations.add()
        pick.match_root = roots[observation["match_id"]]
        pick.is_set = True
        pick.confidence = "NORMAL"
        if observation["weight"] != 1.0:
            raise ValueError("Blender pilot requires unit pick weights")
        if landmark.kind == "POINT":
            pick.x, pick.y = observation["u"], observation["v"]
        else:
            pick.x, pick.y, pick.x2, pick.y2 = (observation[k] for k in ("u1", "v1", "u2", "v2"))
    bpy.context.scene["synthetic_sync_request"] = fingerprint(request)
    properties.tag_sync_ui_redraw(bpy.context)
    scene.set_active_match(bpy.context, workspace.anchor_root)
    bpy.context.view_layer.update()


def assert_equivalent(expected, actual, path="request"):
    """Tolerate RNA float32 storage, not lost settings, observations or cameras."""
    if is_dataclass(expected):
        expected, actual = asdict(expected), asdict(actual)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"{path}: keys differ")
        for key in expected:
            assert_equivalent(expected[key], actual[key], path + "." + key)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: lengths differ ({len(expected)} expected, {len(actual)} actual)")
        for index, (a, b) in enumerate(zip(expected, actual)):
            assert_equivalent(a, b, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if not np.isclose(expected, actual, rtol=1e-6, atol=1e-4):
            raise AssertionError(f"{path}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def verify_inputs(case):
    from match_perspective import scene
    prep = scene.prepare_diagnose_sync(bpy.context)
    expected = solver_arguments(case["request"])
    for key, value in expected.items():
        actual = deepcopy(getattr(prep, key))
        if key == "matches":
            value, actual = (sorted(items, key=lambda m:m.match_id) for items in (value, actual))
        elif key in {"observations", "line_observations"}:
            value, actual = (sorted(items, key=lambda o:(o.landmark_id,o.match_id)) for items in (value,actual))
            wanted = {(o.landmark_id, o.match_id) for o in value}
            found = {(o.landmark_id, o.match_id) for o in actual}
            if wanted != found:
                raise AssertionError(f"{key}: missing {sorted(wanted-found)}, extra {sorted(found-wanted)}")
            # Display names are not evidence.
            for item in actual:
                item.landmark_name = ""
        elif key in {"parallel_pairs", "mirror_pairs"}:
            value, actual = (sorted([sorted(pair) for pair in items]) for items in (value, actual))
        assert_equivalent(value, actual, key)
    return prep


def run_case(case, out):
    from match_perspective import properties, scene
    from tools.sync_snapshot import write_snapshot
    prep = verify_inputs(case)
    write_snapshot(prep, out / "request.json", source_name="Generated scene")
    started = time.perf_counter()
    try:
        result = scene.solve_and_apply_sync(bpy.context)
    except scene.SyncSolveRejected as error:
        result = error.result
    record = result_record(result, case["request"]["cameras"])
    from match_perspective.ui import sync_report
    report = sync_report.build_sync_report(
        operation="Synthetic Solve Sync", source_name="Generated scene",
        matches=prep.matches, observations=prep.observations, line_observations=prep.line_observations,
        result=result, anchor_id=prep.anchor_id, known_world=prep.known_world, known_lines=prep.known_lines,
        parallel_pairs=prep.parallel_pairs, mirror_pairs=prep.mirror_pairs,
        fixed_match_ids=prep.fixed_similarities,
    )
    (out / "product-report.html").write_text(sync_report.render_sync_report_html(report))
    record.update(elapsed_s=time.perf_counter()-started, environment=environment(),
                  blender=bpy.app.version_string, request_sha256=fingerprint(case["request"]))
    assessment = evaluate(case, record)
    if record["success"]:
        workspace = properties.workspace(bpy.context)
        bpy.context.view_layer.update()
        for landmark in workspace.landmarks:
            key = landmark.item_id
            kind = "points" if landmark.kind == "POINT" else "lines"
            if key in case["expectation"].get("excluded_" + kind, []):
                if landmark.has_position or landmark.has_line_segment or scene.landmark_viewport_object(landmark):
                    assessment["violations"].append(f"{key}: unconstrained geometry remains in Blender")
            if key not in case["expectation"].get("required_" + kind, []):
                continue
            helper = scene.landmark_viewport_object(landmark)
            if not landmark.has_position or helper is None:
                assessment["violations"].append(f"{key}: reconstructed geometry was not applied to Blender")
                continue
            if kind == "points":
                expected = record["landmarks"].get(key)
                values = [list(landmark.position), list(helper.matrix_world.translation)]
            else:
                expected = record["line_segments"].get(key)
                if not landmark.has_line_segment or helper.type != "MESH":
                    assessment["violations"].append(f"{key}: reconstructed line helper is missing")
                    continue
                values = [[list(landmark.position), list(landmark.position_b)],
                          [list(helper.matrix_world @ v.co) for v in helper.data.vertices]]
            if expected is None or any(np.shape(value) != np.shape(expected)
                or not np.allclose(value, expected, atol=1e-5, rtol=1e-6) for value in values):
                assessment["violations"].append(f"{key}: Blender geometry differs from the solver result")
        scale, rotation, translation = alignment(case, record)
        for root in properties.iter_match_roots():
            key = root.get("synthetic_match_id")
            if key not in assessment["cameras"]:
                continue
            if assessment["cameras"][key]["actual_uv"] is None:
                continue  # Geometry evaluation already records this failure.
            camera = next(c for c in case["truth"]["cameras"] if c["id"] == key)
            points = np.array([p["position"] for p in case["truth"]["checks"] if key in p["views"]])
            # Express truth checks in the one shared gauge used by this solve.
            points = ((points - translation) @ rotation) / scale
            scene.set_active_match(bpy.context, root)
            actual = blender_pixels(root.pm_session.camera_object, camera, points)
            predicted = np.array(assessment["cameras"][key]["actual_uv"])
            error = float(np.max(np.linalg.norm(actual-predicted, axis=1)))
            assessment["cameras"][key]["blender_agreement_max_px"] = error
            if error > 0.01:
                assessment["violations"].append(f"{key}: evaluated Blender projection differs by {error:.4f}px")
        assessment["passed"] = not assessment["violations"]
    (out / "result.json").write_text(json.dumps(dict(result=record, assessment=assessment), indent=2, allow_nan=False)+"\n")
    images = {root.get("synthetic_match_id"): Path(bpy.path.abspath(root.pm_session.image.filepath))
              for root in properties.iter_match_roots() if root.pm_session.image}
    write_report([(case, record, assessment)], out / "report.html", images)
    print(f"Synthetic Blender {'PASS' if assessment['passed'] else 'FAIL'}: {case['name']}", flush=True)
    for issue in assessment["violations"]:
        print("  " + issue, flush=True)
    return assessment["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load", type=Path, help="Reopen a generated input file and verify its request fingerprint")
    parser.add_argument("--roundtrip", action="store_true", help="Also replay input.blend in a fresh Blender process")
    parser.add_argument("--render", action="store_true", help="Render and pack truth reference images")
    parser.add_argument("--drop-constraint", action="store_true", help="After a constraint case solves, remove its constraint and verify the live state")
    parser.add_argument("--role-case", type=Path, help="After solving, change only camera roles to this case and verify the live scene")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.case, args.out = args.case.resolve(), args.out.resolve()
    if args.load and args.load.resolve() in {args.out / "input.blend", args.out / "solved.blend"}:
        parser.error("Use a different output directory when reopening a generated file")
    args.out.mkdir(parents=True, exist_ok=True)
    case = read_case(args.case)
    role_case = read_case(args.role_case.resolve()) if args.role_case else None
    if role_case is not None:
        role_fields = {"location_match_ids", "readonly_match_ids"}
        before = {key: value for key, value in case["request"].items() if key not in role_fields}
        after = {key: value for key, value in role_case["request"].items() if key not in role_fields}
        if before != after or case["truth"] != role_case["truth"]:
            parser.error("--role-case may change camera participation and expectations only")
        if args.drop_constraint:
            parser.error("Choose one live transition: --role-case or --drop-constraint")
    if args.drop_constraint and case["family"] not in {"known_lines", "mirror_points", "mirror_lines"}:
        parser.error("--drop-constraint requires a constraint contribution case")
    register_extension()
    if args.load:
        bpy.ops.wm.open_mainfile(filepath=str(args.load.resolve()))
        if bpy.context.scene.get("synthetic_sync_request") != fingerprint(case["request"]):
            raise ValueError("File is not a generated input for this exact request")
    else:
        create_scene(case, args.out, args.render)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "input.blend"), check_existing=False)
    passed = run_case(case, args.out)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "solved.blend"), check_existing=False)
    if role_case is not None:
        from match_perspective import properties
        from tools.synthetic_sync.scenarios import write_case
        request = role_case["request"]
        location_ids = request.get("location_match_ids")
        location_ids = set(camera["id"] for camera in request["cameras"]) if location_ids is None else set(location_ids)
        readonly_ids = set(request.get("readonly_match_ids") or [])
        for root in properties.iter_match_roots():
            key = root.get("synthetic_match_id")
            if key in request["fixed_similarities"]:
                root.pm_session.sync_role = "LOCK_POSE"
            elif key in readonly_ids or key not in location_ids:
                root.pm_session.sync_role = "FIT_ONLY"
            else:
                root.pm_session.sync_role = "SOLVE"
        bpy.context.scene["synthetic_sync_request"] = fingerprint(request)
        folder = args.out / "after-role-change"
        folder.mkdir(exist_ok=True)
        write_case(role_case, folder / "case.json")
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "input.blend"), check_existing=False)
        passed = run_case(role_case, folder) and passed
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "solved.blend"), check_existing=False)
    if args.drop_constraint:
        from match_perspective import properties
        from tools.synthetic_sync.constraints import remove_constraint
        from tools.synthetic_sync.scenarios import write_case
        removed = remove_constraint(case)
        workspace = properties.workspace(bpy.context)
        if case["family"] == "known_lines":
            for landmark in workspace.landmarks:
                landmark.known_object = landmark.known_object_b = None
        else:
            for landmark in workspace.landmarks:
                landmark.mirror_of_id = "NONE"
            workspace.mirror_object = None
        bpy.context.scene["synthetic_sync_request"] = fingerprint(removed["request"])
        folder = args.out / "without-constraint"
        folder.mkdir(exist_ok=True)
        write_case(removed, folder / "case.json")
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "input.blend"), check_existing=False)
        before = {root.name: root.matrix_world.copy() for root in properties.iter_match_roots()}
        passed = run_case(removed, folder) and passed
        if removed["expectation"]["outcome"] == "reject":
            for root in properties.iter_match_roots():
                if root.matrix_world != before[root.name]:
                    raise AssertionError("Refused solve changed a previously valid camera")
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "solved.blend"), check_existing=False)
    if args.roundtrip:
        process = subprocess.run([bpy.app.binary_path, "--factory-startup", "--disable-autoexec", "-b",
            "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--case", str(args.case),
            "--out", str(args.out / "reopened"), "--load", str(args.load.resolve() if args.load else args.out / "input.blend")]
            + (["--drop-constraint"] if args.drop_constraint else [])
            + (["--role-case", str(args.role_case.resolve())] if args.role_case else []))
        passed = passed and process.returncode == 0
    if not passed:
        raise RuntimeError("Synthetic Blender verification failed; see report.html")


if __name__ == "__main__":
    main()
