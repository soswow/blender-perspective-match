"""Dump landmark overlap, stored vs solved residuals, and timed Solve Sync.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-sync/probe_graph.py -- \\
      --blend scene.blend [--out /tmp/pm-graph.txt] [--no-solve]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
from collections import defaultdict
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
    parser = argparse.ArgumentParser(description="Dump PM landmark graph + timed solve")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument(
        "--focus",
        default="",
        help="Comma-separated landmark name substrings to detail (default: bottom,id325)",
    )
    return parser.parse_args(argv)


def _angle_from_vertical_deg(direction: np.ndarray) -> float:
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    return float(math.degrees(math.acos(min(1.0, abs(float(unit[2]))))))


def _optical_axis_world(calibration) -> np.ndarray:
    axis = calibration.rotation_w2c.T @ np.array((0.0, 0.0, 1.0), dtype=np.float64)
    return axis / max(float(np.linalg.norm(axis)), 1.0e-12)


def _shared_optical_axis(calibration, similarity) -> np.ndarray:
    axis_private = _optical_axis_world(calibration)
    axis_shared = similarity.rotation @ axis_private
    return axis_shared / max(float(np.linalg.norm(axis_shared)), 1.0e-12)


def _short_match(match_id: str) -> str:
    name = match_id
    if name.startswith("PM_"):
        name = name[3:]
    if name.endswith("_Origin"):
        name = name[: -len("_Origin")]
    return name


def _radius_norm(sync_module, observation, calibration) -> float:
    return float(
        sync_module._image_radius_norm(observation.u, observation.v, calibration)
    )


def _project_error(sync_module, similarity, calibration, point, u, v) -> float:
    projected = sync_module.project_private_point(
        similarity.inverse_point(point), calibration
    )
    if projected is None:
        return 1.0e3
    return float(np.hypot(projected[0] - u, projected[1] - v))


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        print(f"Blend not found: {blend_path}", file=sys.stderr)
        return 2

    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)

    load_t0 = time.perf_counter()
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    load_s = time.perf_counter() - load_t0

    from match_perspective import properties, scene
    from match_perspective.core import sync as sync_module
    from match_perspective.core.sync import ba as ba_module
    from match_perspective.core.sync import pose as pose_module
    from match_perspective.core.sync import solve as solve_module

    space = bpy.context.scene.match_perspective
    anchor = properties.anchor_root(bpy.context)
    roots = sorted(
        [obj for obj in bpy.data.objects if properties.is_match_root(obj)],
        key=lambda item: item.name,
    )

    log(f"blend={blend_path}")
    log(f"load_s={load_s:.1f}")
    log(f"anchor={None if anchor is None else anchor.name}")
    log(f"landmarks={len(space.landmarks)}")
    log(f"sync_status={space.sync_status!r}")
    log()

    log("=== STORED CAMERAS ===")
    match_ids: list[str] = []
    for root in roots:
        session = root.pm_session
        match_ids.append(root.name)
        if session.image is None or float(session.fx) <= 0.0:
            log(f"{_short_match(root.name):24s} unsovled fx={session.fx:.1f}")
            continue
        calibration = scene.calibration_from_settings(session)
        optical = _optical_axis_world(calibration)
        center = calibration.camera_center
        empty_s = float(root.matrix_world.to_scale().x)
        log(
            f"{_short_match(root.name):24s} sync={int(session.sync_enabled)} "
            f"applied={int(session.sync_is_applied)} "
            f"fx={calibration.intrinsics.fx:7.1f} fy={calibration.intrinsics.fy:7.1f} "
            f"origin={int(session.origin_is_set)} "
            f"nadir={_angle_from_vertical_deg(optical):5.1f} "
            f"C=({center[0]:+6.2f},{center[1]:+6.2f},{center[2]:+6.2f}) "
            f"empty_s={empty_s:.3g} "
            f"session_s={float(session.sync_scale):.3g} "
            f"rmse={float(session.sync_rmse_px):.1f}px"
        )

    build_t0 = time.perf_counter()
    matches, observations, known_world, line_observations, known_lines, parallel = (
        scene.build_sync_problem(bpy.context)
    )
    build_s = time.perf_counter() - build_t0
    log()
    log(f"build_sync_problem_s={build_s:.2f}")
    log(
        f"matches={len(matches)} observations={len(observations)} "
        f"known3d={len(known_world)} lines={len(line_observations)} "
        f"known_lines={len(known_lines)} parallel={len(parallel)}"
    )

    name_by_id = {landmark.item_id: landmark.name for landmark in space.landmarks}
    ground_ids = {
        landmark.item_id for landmark in space.landmarks if landmark.on_ground
    }
    stored_pos = {
        landmark.item_id: np.array(landmark.position, dtype=np.float64)
        for landmark in space.landmarks
        if landmark.has_position
    }
    stored_rmse = {landmark.item_id: float(landmark.rmse_px) for landmark in space.landmarks}

    obs_by_match: dict[str, list] = defaultdict(list)
    obs_by_lm: dict[str, list] = defaultdict(list)
    for observation in observations:
        obs_by_match[observation.match_id].append(observation)
        obs_by_lm[observation.landmark_id].append(observation)

    log()
    log("=== PER-MATCH PICKS (g=On Ground landmarks visible here) ===")
    for match in matches:
        items = obs_by_match.get(match.match_id, [])
        n_ground = sum(1 for item in items if item.landmark_id in ground_ids)
        log(
            f"{_short_match(match.match_id):24s} picks={len(items):3d} "
            f"ground={n_ground:3d}"
        )

    log()
    log("=== ON GROUND LANDMARKS (which matches see them) ===")
    for landmark in space.landmarks:
        if not landmark.on_ground:
            continue
        seen = sorted(
            _short_match(item.match_id) for item in obs_by_lm.get(landmark.item_id, [])
        )
        z_bit = ""
        if landmark.has_position:
            z_bit = f" stored_xyz=({landmark.position[0]:.2f},{landmark.position[1]:.2f},{landmark.position[2]:.2f})"
        log(
            f"  {landmark.name:28s} n={len(seen):2d} {seen}{z_bit} "
            f"stored_rmse={stored_rmse.get(landmark.item_id, 0.0):.1f}px"
        )

    log()
    log("=== LANDMARK OVERLAP (shared pick count) ===")
    short_ids = [_short_match(match.match_id) for match in matches]
    picks_set = {
        match.match_id: {item.landmark_id for item in obs_by_match.get(match.match_id, [])}
        for match in matches
    }
    header = f"{'':16s}" + "".join(f"{name[:8]:>8s}" for name in short_ids)
    log(header)
    for match_a in matches:
        row = f"{_short_match(match_a.match_id)[:16]:16s}"
        set_a = picks_set[match_a.match_id]
        for match_b in matches:
            if match_a.match_id == match_b.match_id:
                row += f"{len(set_a):8d}"
            else:
                row += f"{len(set_a & picks_set[match_b.match_id]):8d}"
        log(row)

    focus_needles = [
        part.strip().lower()
        for part in (args.focus or "bottom,id325").split(",")
        if part.strip()
    ]

    def _is_focus(name: str) -> bool:
        lowered = name.lower()
        return any(needle in lowered for needle in focus_needles)

    log()
    log("=== STORED EMPTY vs CURRENT CAMERA (applied pose, not Diagnose) ===")
    log("(large err here with small stored_rmse ⇒ list px is stale / averaged)")
    match_map = {item.match_id: item for item in matches}
    for root in roots:
        session = root.pm_session
        if not session.sync_is_applied or root.name not in match_map:
            continue
        similarity = scene._similarity_from_session(session)
        calibration = match_map[root.name].calibration
        rows = []
        for observation in obs_by_match.get(root.name, []):
            point = stored_pos.get(observation.landmark_id)
            if point is None:
                continue
            error = _project_error(
                sync_module,
                similarity,
                calibration,
                point,
                observation.u,
                observation.v,
            )
            rows.append((error, name_by_id.get(observation.landmark_id, observation.landmark_id)))
        if not rows:
            log(f"{_short_match(root.name):24s} (no stored landmark positions)")
            continue
        rows.sort(reverse=True)
        rmse = float(np.sqrt(np.mean([error * error for error, _name in rows])))
        worst = ", ".join(f"{name} {error:.0f}px" for error, name in rows[:5])
        log(
            f"{_short_match(root.name):24s} n={len(rows):3d} stored-empty RMSE={rmse:6.1f}px "
            f"session={float(session.sync_rmse_px):.1f}px  worst: {worst}"
        )

    log()
    log("=== FOCUS LANDMARKS (stored) ===")
    for landmark in space.landmarks:
        if not _is_focus(landmark.name):
            continue
        seen = [
            _short_match(item.match_id) for item in obs_by_lm.get(landmark.item_id, [])
        ]
        pos_bit = "no stored xyz"
        if landmark.has_position:
            pos_bit = (
                f"xyz=({landmark.position[0]:.3f},{landmark.position[1]:.3f},"
                f"{landmark.position[2]:.3f})"
            )
        log(
            f"{landmark.name:28s} g={int(landmark.on_ground)} "
            f"use={int(landmark.use_in_sync)} stored_rmse={landmark.rmse_px:.2f}px "
            f"{pos_bit} seen={seen}"
        )
        if not landmark.has_position:
            continue
        point = np.array(landmark.position, dtype=np.float64)
        for observation in obs_by_lm.get(landmark.item_id, []):
            root = next((item for item in roots if item.name == observation.match_id), None)
            if root is None:
                continue
            session = root.pm_session
            if not session.sync_is_applied:
                log(f"    {_short_match(observation.match_id):24s} (sync not applied)")
                continue
            similarity = scene._similarity_from_session(session)
            calibration = match_map[observation.match_id].calibration
            error = _project_error(
                sync_module,
                similarity,
                calibration,
                point,
                observation.u,
                observation.v,
            )
            log(
                f"    {_short_match(observation.match_id):24s} "
                f"uv=({observation.u:.0f},{observation.v:.0f}) "
                f"empty_err={error:.1f}px"
            )

    if args.no_solve or not matches or anchor is None:
        report = "\n".join(lines) + "\n"
        if args.out:
            Path(args.out).expanduser().write_text(report, encoding="utf-8")
        print(report, end="")
        return 0

    timings: dict[str, float] = {}
    orig_register = solve_module._register_from_relative_pose
    orig_ba = ba_module._bundle_adjust_registration
    orig_resect = solve_module._resect_skipped_matches
    orig_relative = pose_module._relative_pose_from_correspondences
    relative_calls = {"n": 0, "s": 0.0}

    def timed_register(*args, **kwargs):
        started = time.perf_counter()
        result = orig_register(*args, **kwargs)
        timings["register"] = time.perf_counter() - started
        return result

    def timed_ba(*args, **kwargs):
        started = time.perf_counter()
        result = orig_ba(*args, **kwargs)
        timings["ba"] = timings.get("ba", 0.0) + (time.perf_counter() - started)
        return result

    def timed_resect(*args, **kwargs):
        started = time.perf_counter()
        result = orig_resect(*args, **kwargs)
        timings["resect"] = time.perf_counter() - started
        return result

    def timed_relative(*args, **kwargs):
        started = time.perf_counter()
        result = orig_relative(*args, **kwargs)
        relative_calls["n"] += 1
        relative_calls["s"] += time.perf_counter() - started
        return result

    solve_module._register_from_relative_pose = timed_register
    ba_module._bundle_adjust_registration = timed_ba
    solve_module._bundle_adjust_registration = timed_ba
    solve_module._resect_skipped_matches = timed_resect
    pose_module._relative_pose_from_correspondences = timed_relative
    solve_module._relative_pose_from_correspondences = timed_relative

    log()
    log("=== solve_landmark_sync ===")
    solve_t0 = time.perf_counter()
    try:
        result = sync_module.solve_landmark_sync(
            matches,
            observations,
            anchor_id=anchor.name,
            known_world=known_world,
            line_observations=line_observations,
            known_lines=known_lines,
            parallel_pairs=parallel,
        )
        solve_s = time.perf_counter() - solve_t0
        log(f"success={result.success}")
        log(f"message={result.message}")
        log(f"mean_rmse={result.mean_reprojection_px:.3f}")
        log(f"solve_s={solve_s:.2f}")
        log(
            f"stage_s register={timings.get('register', float('nan')):.2f} "
            f"ba={timings.get('ba', float('nan')):.2f} "
            f"resect={timings.get('resect', float('nan')):.2f} "
            f"relative_pose n={relative_calls['n']} {relative_calls['s']:.2f}s"
        )
        log("per_match=" + ", ".join(
            f"{_short_match(match_id)}={rmse:.1f}"
            for match_id, rmse in sorted(result.per_match_rmse_px.items())
        ))
        recovered = set()
        if "recovered '" in result.message:
            # Parse names already printed in message; also use similarities vs peel.
            pass
        log(f"registered={sorted(_short_match(match_id) for match_id in result.similarities)}")
        log()
        log("=== RECOVERED / SOLVED CAMERA POSES ===")
        for match_id, similarity in sorted(result.similarities.items()):
            item = match_map[match_id]
            axis = _shared_optical_axis(item.calibration, similarity)
            center = similarity.transform_point(item.calibration.camera_center)
            collapsed = abs(math.log(max(float(similarity.scale), 1.0e-12))) >= 17.5
            log(
                f"{_short_match(match_id):24s} s={similarity.scale:.4g} "
                f"nadir={_angle_from_vertical_deg(axis):5.1f} "
                f"C=({center[0]:+6.2f},{center[1]:+6.2f},{center[2]:+6.2f}) "
                f"axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}) "
                f"look_{'up' if axis[2] < -0.5 else 'down' if axis[2] > 0.5 else 'side'} "
                f"collapsed={int(collapsed)} "
                f"rmse={result.per_match_rmse_px.get(match_id, float('nan')):.1f}px"
            )

        log()
        log("=== RESIDUAL vs RADIUS (inner r<0.35, outer r>=0.55) ===")
        for match_id, similarity in sorted(result.similarities.items()):
            item = match_map[match_id]
            calibration = item.calibration
            inner: list[float] = []
            outer: list[float] = []
            all_err: list[tuple[float, float, str]] = []
            for observation in obs_by_match.get(match_id, []):
                point = result.landmarks.get(observation.landmark_id)
                if point is None:
                    continue
                error = _project_error(
                    sync_module,
                    similarity,
                    calibration,
                    point,
                    observation.u,
                    observation.v,
                )
                radius = _radius_norm(sync_module, observation, calibration)
                all_err.append(
                    (error, radius, name_by_id.get(observation.landmark_id, observation.landmark_id))
                )
                if radius < 0.35:
                    inner.append(error)
                elif radius >= 0.55:
                    outer.append(error)
            if not all_err:
                continue
            rmse = float(np.sqrt(np.mean([error * error for error, _r, _n in all_err])))
            inner_rmse = (
                float(np.sqrt(np.mean([error * error for error in inner])))
                if inner
                else float("nan")
            )
            outer_rmse = (
                float(np.sqrt(np.mean([error * error for error in outer])))
                if outer
                else float("nan")
            )
            worst_err, worst_r, worst_name = max(all_err)
            log(
                f"{_short_match(match_id):24s} n={len(all_err):3d} rmse={rmse:5.1f} "
                f"inner n={len(inner):2d} {inner_rmse:5.1f}px "
                f"outer n={len(outer):2d} {outer_rmse:5.1f}px "
                f"worst={worst_name} {worst_err:.0f}px r={worst_r:.2f}"
            )

        log()
        log("=== PER-OBSERVATION (solved 3D vs recovered pose) ===")
        missing_3d = []
        for landmark in space.landmarks:
            if not landmark.use_in_sync:
                continue
            items = obs_by_lm.get(landmark.item_id, [])
            if len(items) < 2 and landmark.item_id not in known_world:
                continue
            point = result.landmarks.get(landmark.item_id)
            if point is None:
                missing_3d.append(
                    (
                        landmark.name,
                        [_short_match(item.match_id) for item in items],
                        stored_rmse.get(landmark.item_id, 0.0),
                    )
                )
                continue
            errors = []
            for observation in items:
                similarity = result.similarities.get(observation.match_id)
                if similarity is None:
                    continue
                error = _project_error(
                    sync_module,
                    similarity,
                    match_map[observation.match_id].calibration,
                    point,
                    observation.u,
                    observation.v,
                )
                errors.append((error, observation.match_id))
            if not errors:
                continue
            rmse = float(np.sqrt(np.mean([error * error for error, _id in errors])))
            listed = stored_rmse.get(landmark.item_id, 0.0)
            worst_err, worst_id = max(errors)
            flag = ""
            if worst_err > 15.0 and listed < 2.0:
                flag = "  ** LIST-PX MISMATCH"
            if _is_focus(landmark.name) or worst_err > 20.0 or flag:
                bits = ", ".join(
                    f"{_short_match(match_id)} {error:.1f}px"
                    for error, match_id in sorted(errors, reverse=True)
                )
                z_val = float(point[2])
                log(
                    f"{landmark.name:28s} g={int(landmark.on_ground)} "
                    f"Z={z_val:+.2f} listed={listed:.1f}px solved_rmse={rmse:.1f}px "
                    f"worst={_short_match(worst_id)} {worst_err:.1f}px{flag}"
                )
                if _is_focus(landmark.name) or flag:
                    log(f"    {bits}")

        if missing_3d:
            log()
            log("=== LANDMARKS WITH PICKS BUT NO SOLVED 3D ===")
            log("(empties / listed px stay stale from the previous apply)")
            for name, seen, listed in missing_3d:
                log(
                    f"  {name:28s} listed={listed:.1f}px seen={seen}"
                )

        log()
        log("=== GROUND Z AFTER SOLVE ===")
        ground_z = []
        for landmark_id, point in result.landmarks.items():
            if landmark_id not in ground_ids:
                continue
            ground_z.append((float(point[2]), name_by_id.get(landmark_id, landmark_id)))
        ground_z.sort()
        for z_val, name in ground_z:
            log(f"  Z={z_val:+7.3f}  {name}")

    except Exception:
        log(traceback.format_exc())

    report = "\n".join(lines) + "\n"
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    print(report, end="")
    return 0


if __name__ == "__main__":
    try:
        separator = sys.argv.index("--")
        cli_args = sys.argv[separator + 1 :]
    except ValueError:
        cli_args = []
    raise SystemExit(main(cli_args))
