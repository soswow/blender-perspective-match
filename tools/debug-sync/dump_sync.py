"""Dump Perspective Match sync graph, camera tilt vs Z, and pairwise RMSE.

Usage (from repo root):

  blender --factory-startup -b --python tools/debug-sync/dump_sync.py -- \\
      --blend scene.blend [--out /tmp/pm-sync-dump.txt]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
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
            pass
    return module


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump PM sync state, optical-axis tilt, and pairwise RMSE",
    )
    parser.add_argument("--blend", required=True, help="Path to a .blend with PM matches")
    parser.add_argument("--out", default="", help="Optional text report path")
    return parser.parse_args(argv)


def _angle_from_vertical_deg(direction: np.ndarray) -> float:
    """0° = parallel to world ±Z, 90° = horizontal."""
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    return float(math.degrees(math.acos(min(1.0, abs(float(unit[2]))))))


def _optical_axis_world(calibration) -> np.ndarray:
    """Private-world direction of the camera +Z axis (principal ray)."""
    axis = calibration.rotation_w2c.T @ np.array((0.0, 0.0, 1.0), dtype=np.float64)
    return axis / max(float(np.linalg.norm(axis)), 1.0e-12)


def _image_center_ray_world(calibration) -> np.ndarray:
    from match_perspective.core import sync as sync_module

    width = calibration.intrinsics.image_width
    height = calibration.intrinsics.image_height
    _origin, direction = sync_module.camera_ray_private(
        0.5 * width,
        0.5 * height,
        calibration,
    )
    return direction / max(float(np.linalg.norm(direction)), 1.0e-12)


def _clone_obs_without_ground(observations):
    from match_perspective.core import sync as sync_module

    return [
        sync_module.SyncObservation(
            match_id=observation.match_id,
            landmark_id=observation.landmark_id,
            u=observation.u,
            v=observation.v,
            on_ground=False,
            landmark_name=observation.landmark_name,
            weight=observation.weight,
        )
        for observation in observations
    ]


def _shared_optical_axis(calibration, similarity) -> np.ndarray:
    """Optical axis of `calibration` after mapping private → shared world."""
    axis_private = _optical_axis_world(calibration)
    axis_shared = similarity.rotation @ axis_private
    return axis_shared / max(float(np.linalg.norm(axis_shared)), 1.0e-12)


def _homography_rmse(pairs) -> float | None:
    """Pixel RMSE of a DLT homography other ~ H @ anchor (planar degeneracy probe)."""
    if len(pairs) < 4:
        return None
    design = []
    for anchor_obs, other_obs in pairs:
        x_coord, y_coord = float(anchor_obs.u), float(anchor_obs.v)
        u_coord, v_coord = float(other_obs.u), float(other_obs.v)
        design.append(
            (-x_coord, -y_coord, -1.0, 0.0, 0.0, 0.0, u_coord * x_coord, u_coord * y_coord, u_coord)
        )
        design.append(
            (0.0, 0.0, 0.0, -x_coord, -y_coord, -1.0, v_coord * x_coord, v_coord * y_coord, v_coord)
        )
    _u_matrix, _singular, vt_matrix = np.linalg.svd(np.asarray(design, dtype=np.float64))
    homography = vt_matrix[-1].reshape(3, 3)
    errors: list[float] = []
    for anchor_obs, other_obs in pairs:
        mapped = homography @ np.array((anchor_obs.u, anchor_obs.v, 1.0), dtype=np.float64)
        if abs(float(mapped[2])) < 1.0e-12:
            continue
        mapped /= mapped[2]
        errors.append(
            math.hypot(float(mapped[0] - other_obs.u), float(mapped[1] - other_obs.v))
        )
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def _try_relative(sync_module, anchor_id, other_id, obs_by_lm, matches, known_world):
    solved, detail = sync_module._relative_pose_from_correspondences(
        anchor_id,
        other_id,
        obs_by_lm,
        matches,
        known_world,
    )
    rmse = None
    pairs = sync_module._point_pairs_between_matches(
        anchor_id,
        other_id,
        obs_by_lm,
        excluded_landmark_ids=set(known_world),
    )
    metric = sync_module._metric_landmarks(
        obs_by_lm,
        anchor_id,
        matches[anchor_id].calibration,
        known_world,
    )
    points_shared, points_image, _ids, weights = (
        sync_module._metric_pnp_correspondences(other_id, metric, obs_by_lm)
    )
    if solved is not None:
        errors = sync_module._reprojection_errors_for_similarity(
            solved,
            pairs,
            matches[anchor_id].calibration,
            matches[other_id].calibration,
            points_shared,
            points_image,
            point_weights=weights,
            weighted=True,
        )
        if errors:
            rmse = float(np.sqrt(np.mean(np.square(errors))))
    return solved, rmse, detail, pairs


def _group_obs(observations):
    by_lm: dict[str, list] = defaultdict(list)
    for observation in observations:
        by_lm[observation.landmark_id].append(observation)
    return by_lm


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        print(f"Blend not found: {blend_path}", file=sys.stderr)
        return 2

    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    from match_perspective import properties, scene
    from match_perspective.core import geometry as core
    from match_perspective.core import sync as sync_module

    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)

    space = bpy.context.scene.match_perspective
    anchor = properties.anchor_root(bpy.context)
    log(f"blend={blend_path}")
    log(f"anchor={None if anchor is None else anchor.name}")
    log(f"landmarks={len(space.landmarks)}")
    log(f"sync_status={space.sync_status!r}")
    log()
    log(
        "=== STORED CAMERA POSES (current Blender; leftover if sync never locked) ==="
    )
    log("(nadir_deg=0 looks along ±Z; 90 is horizontal)")

    roots = [obj for obj in bpy.data.objects if properties.is_match_root(obj)]
    roots.sort(key=lambda item: item.name)
    for root in roots:
        session = root.pm_session
        if session.image is None or float(session.fx) <= 0.0:
            log(
                f"{root.name:32s} sync={int(session.sync_enabled)} "
                f"last_ok={int(getattr(session, 'sync_last_ok', 0))} "
                f"fx={session.fx:.1f} (no plate / unsolved)"
            )
            continue
        calibration = scene.calibration_from_settings(session)
        optical = _optical_axis_world(calibration)
        center_ray = _image_center_ray_world(calibration)
        bundles = scene.line_bundles_from_settings(session)
        can_orient = core.can_solve_orientation(
            bundles,
            lock_focal=bool(session.lock_focal or session.vp_mode == "1"),
            vp_mode=session.vp_mode,
        )
        n_lines = sum(len(bundles.get(axis, [])) for axis in ("X", "Y", "Z"))
        log(
            f"{root.name:32s} sync={int(session.sync_enabled)} "
            f"last_ok={int(getattr(session, 'sync_last_ok', 0))} "
            f"fx={calibration.intrinsics.fx:8.1f} origin={int(session.origin_is_set)} "
            f"vp_lines={n_lines} can_orient={int(can_orient)} "
            f"nadir_deg={_angle_from_vertical_deg(optical):5.1f} "
            f"center_vs_z_deg={_angle_from_vertical_deg(center_ray):5.1f} "
            f"C_z={float(calibration.camera_center[2]):6.2f} "
            f"empty_s={float(root.matrix_world.to_scale().x):.3g} "
            f"axis=({optical[0]:+.2f},{optical[1]:+.2f},{optical[2]:+.2f})"
        )

    matches, observations, known_world, line_observations, known_lines, parallel = (
        scene.build_sync_problem(bpy.context)
    )
    if not matches or anchor is None:
        log()
        log("No sync-enabled solved matches / no anchor — stop.")
        report = "\n".join(lines) + "\n"
        if args.out:
            Path(args.out).expanduser().write_text(report, encoding="utf-8")
        print(report, end="")
        return 0

    match_map = {item.match_id: item for item in matches}
    obs_by_lm = _group_obs(observations)
    obs_free_by_lm = _group_obs(_clone_obs_without_ground(observations))
    known_ids = set(known_world)
    anchor_id = anchor.name

    log()
    log("=== PAIRWISE vs ANCHOR ===")
    log(
        "(mixed includes On Ground 3D; 2d is correspondences only; "
        "rec_* is the recovered pose in shared/anchor world, not the stored camera)"
    )
    outlier_ids: list[str] = []
    for item in matches:
        if item.match_id == anchor_id:
            continue
        pairs = sync_module._point_pairs_between_matches(
            anchor_id,
            item.match_id,
            obs_by_lm,
            excluded_landmark_ids=known_ids,
        )
        n_ground = sum(1 for a, _b in pairs if a.on_ground)
        n_off = len(pairs) - n_ground
        solved_m, rmse_m, detail_m, _pairs_m = _try_relative(
            sync_module, anchor_id, item.match_id, obs_by_lm, match_map, known_world
        )
        solved_2d, rmse_2d, detail_2d, _pairs_2d = _try_relative(
            sync_module, anchor_id, item.match_id, obs_free_by_lm, match_map, {}
        )
        mixed_bit = f"{rmse_m:.1f}" if rmse_m is not None else f"FAIL:{detail_m}"
        two_d_bit = f"{rmse_2d:.1f}" if rmse_2d is not None else f"FAIL:{detail_2d}"
        homography = _homography_rmse(pairs)
        homography_bit = f"{homography:.1f}" if homography is not None else "n/a"
        recovered = solved_2d if solved_2d is not None else solved_m
        rec_bit = ""
        if recovered is not None:
            rec_axis = _shared_optical_axis(item.calibration, recovered)
            rec_center = recovered.transform_point(item.calibration.camera_center)
            rec_bit = (
                f"  rec_nadir_deg={_angle_from_vertical_deg(rec_axis):5.1f} "
                f"rec_C_z={float(rec_center[2]):6.2f} "
                f"rec_axis=({rec_axis[0]:+.2f},{rec_axis[1]:+.2f},{rec_axis[2]:+.2f})"
            )
        log(
            f"{item.match_id}: pairs={len(pairs)} ground={n_ground} off-plane={n_off} "
            f"mixed_rmse={mixed_bit}  2d_rmse={two_d_bit}  "
            f"H_rmse={homography_bit}  "
            f"ok_mixed={int(solved_m is not None)} ok_2d={int(solved_2d is not None)}"
            f"{rec_bit}"
        )
        mixed_bad = (
            rmse_m is not None
            and rmse_2d is not None
            and rmse_m > 40.0
            and rmse_2d <= 40.0
        )
        if mixed_bad:
            outlier_ids.append(item.match_id)

    if outlier_ids:
        log()
        log("=== SHARED LANDMARKS (mixed On Ground disagrees with 2D pose) ===")
        log("(uv as fraction of image; g=On Ground)")
        for match_id in outlier_ids:
            other = match_map[match_id].calibration
            anchor_cal = match_map[anchor_id].calibration
            pairs = sync_module._point_pairs_between_matches(
                anchor_id,
                match_id,
                obs_by_lm,
                excluded_landmark_ids=known_ids,
            )
            log(f"-- {match_id} --")
            for anchor_obs, other_obs in sorted(
                pairs, key=lambda item: item[1].landmark_name
            ):
                log(
                    f"  {other_obs.landmark_name:28s} g={int(anchor_obs.on_ground)} "
                    f"anchor=({anchor_obs.u / anchor_cal.intrinsics.image_width:.2f},"
                    f"{anchor_obs.v / anchor_cal.intrinsics.image_height:.2f}) "
                    f"other=({other_obs.u / other.intrinsics.image_width:.2f},"
                    f"{other_obs.v / other.intrinsics.image_height:.2f})"
                )

            solved_2d, _rmse_2d, _detail_2d, _pairs_2d = _try_relative(
                sync_module, anchor_id, match_id, obs_free_by_lm, match_map, {}
            )
            metric = sync_module._metric_landmarks(
                obs_by_lm,
                anchor_id,
                match_map[anchor_id].calibration,
                known_world,
            )
            points_shared, points_image, _ids, weights = (
                sync_module._metric_pnp_correspondences(
                    match_id, metric, obs_by_lm
                )
            )
            if solved_2d is not None:
                pair_errors = sync_module._reprojection_errors_for_similarity(
                    solved_2d,
                    pairs,
                    match_map[anchor_id].calibration,
                    other,
                    [],
                    [],
                    weighted=True,
                )
                metric_errors = sync_module._reprojection_errors_for_similarity(
                    solved_2d,
                    [],
                    match_map[anchor_id].calibration,
                    other,
                    points_shared,
                    points_image,
                    point_weights=weights,
                    weighted=True,
                )
                pair_rmse = (
                    float(np.sqrt(np.mean(np.square(pair_errors))))
                    if pair_errors
                    else None
                )
                metric_rmse = (
                    float(np.sqrt(np.mean(np.square(metric_errors))))
                    if metric_errors
                    else None
                )
                pair_bit = f"{pair_rmse:.1f}" if pair_rmse is not None else "n/a"
                metric_bit = f"{metric_rmse:.1f}" if metric_rmse is not None else "n/a"
                log(
                    f"  2d-pose split: pairs_rmse={pair_bit} ground3d_rmse={metric_bit}"
                )

            planar_seeds = sync_module._planar_homography_similarities(
                np.stack(points_shared) if points_shared else np.zeros((0, 3)),
                np.stack(points_image) if points_image else np.zeros((0, 2)),
                other,
                weights=np.asarray(weights, dtype=np.float64) if weights else None,
            )
            if not planar_seeds:
                log("  planar homography: (no pose)")
            else:
                plane_inliers = sync_module._coplanar_inlier_indices(
                    np.stack(points_shared)
                )
                if plane_inliers is not None:
                    plane_pts = np.stack(points_shared)[plane_inliers]
                    plane_pix = np.stack(points_image)[plane_inliers]
                    centroid = plane_pts.mean(axis=0)
                    centered = plane_pts - centroid
                    _u_m, _s, vt_m = np.linalg.svd(centered, full_matrices=False)
                    normal = vt_m[-1]
                    if float(normal[2]) < 0.0:
                        normal = -normal
                    tangent_a, tangent_b = sync_module._plane_tangent_basis(normal)
                    plane_xy = np.column_stack(
                        (centered @ tangent_a, centered @ tangent_b)
                    )
                    normalized = []
                    for u_coord, v_coord in plane_pix:
                        ray = sync_module._normalized_camera_ray(
                            float(u_coord), float(v_coord), other
                        )
                        depth = float(ray[2])
                        if abs(depth) < 1.0e-12:
                            continue
                        normalized.append((ray[0] / depth, ray[1] / depth))
                    if len(normalized) >= 4:
                        homography = sync_module._fit_homography_dlt(
                            plane_xy[: len(normalized)],
                            np.asarray(normalized, dtype=np.float64),
                        )
                        if homography is not None:
                            fx_mean = 0.5 * (
                                float(other.intrinsics.fx) + float(other.intrinsics.fy)
                            )
                            errors = []
                            normalized_arr = np.asarray(normalized, dtype=np.float64)
                            plane_used = plane_xy[: len(normalized)]
                            for xy_coord, uv_coord in zip(plane_used, normalized):
                                mapped = homography @ np.array(
                                    (xy_coord[0], xy_coord[1], 1.0), dtype=np.float64
                                )
                                if abs(float(mapped[2])) < 1.0e-12:
                                    continue
                                mapped = mapped[:2] / mapped[2]
                                errors.append(
                                    math.hypot(
                                        float(mapped[0] - uv_coord[0]),
                                        float(mapped[1] - uv_coord[1]),
                                    )
                                    * fx_mean
                                )
                            if errors:
                                col_a = homography[:, 0]
                                col_b = homography[:, 1]
                                norm_a = float(np.linalg.norm(col_a))
                                norm_b = float(np.linalg.norm(col_b))
                                cosine = float(np.dot(col_a, col_b)) / max(
                                    norm_a * norm_b, 1.0e-12
                                )
                                log(
                                    f"  plane→image H_rmse="
                                    f"{float(np.sqrt(np.mean(np.square(errors)))):.1f}px "
                                    f"n={len(errors)} "
                                    f"fx={other.intrinsics.fx:.1f} fy={other.intrinsics.fy:.1f} "
                                    f"|h1|/|h2|={norm_a / max(norm_b, 1.0e-12):.3f} "
                                    f"h1·h2={cosine:.3f}"
                                )
                            square = abs(
                                float(other.intrinsics.fx) - float(other.intrinsics.fy)
                            ) > 0.05 * max(float(other.intrinsics.fx), 1.0)
                            if square:
                                cloned = core.Calibration(
                                    core.CameraIntrinsics(
                                        fx=other.intrinsics.fx,
                                        fy=other.intrinsics.fx,
                                        cx=other.intrinsics.cx,
                                        cy=other.intrinsics.cy,
                                        image_width=other.intrinsics.image_width,
                                        image_height=other.intrinsics.image_height,
                                    ),
                                    other.rotation_w2c.copy(),
                                    other.camera_center.copy(),
                                )
                                cloned_seeds = sync_module._planar_homography_similarities(
                                    np.stack(points_shared),
                                    np.stack(points_image),
                                    cloned,
                                )
                                if not cloned_seeds:
                                    log("  planar fy=fx: (no pose)")
                                else:
                                    best = None
                                    for seed in cloned_seeds:
                                        errors_sq = sync_module._reprojection_errors_for_similarity(
                                            seed,
                                            [],
                                            match_map[anchor_id].calibration,
                                            cloned,
                                            points_shared,
                                            points_image,
                                            point_weights=weights,
                                            weighted=True,
                                        )
                                        if not errors_sq:
                                            continue
                                        rmse_sq = float(
                                            np.sqrt(np.mean(np.square(errors_sq)))
                                        )
                                        if best is None or rmse_sq < best:
                                            best = rmse_sq
                                    log(
                                        f"  planar fy=fx: rmse="
                                        f"{best:.1f}" if best is not None else "n/a"
                                    )
                for index, seed in enumerate(planar_seeds):
                    errors = sync_module._reprojection_errors_for_similarity(
                        seed,
                        [],
                        match_map[anchor_id].calibration,
                        other,
                        points_shared,
                        points_image,
                        point_weights=weights,
                        weighted=True,
                    )
                    rmse = (
                        float(np.sqrt(np.mean(np.square(errors))))
                        if errors
                        else None
                    )
                    rec_axis = _shared_optical_axis(other, seed)
                    rec_center = seed.transform_point(other.camera_center)
                    rmse_bit = f"{rmse:.1f}" if rmse is not None else "n/a"
                    log(
                        f"  planar[{index}]: rmse={rmse_bit} "
                        f"nadir={_angle_from_vertical_deg(rec_axis):.1f} "
                        f"C_z={float(rec_center[2]):.2f}"
                    )

            ground_obs = [
                observation
                for observation in observations
                if observation.on_ground
                and observation.match_id in {anchor_id, match_id}
            ]
            ground_by_lm = _group_obs(ground_obs)
            solved_g, rmse_g, detail_g, _pairs_g = _try_relative(
                sync_module,
                anchor_id,
                match_id,
                ground_by_lm,
                match_map,
                known_world,
            )
            if solved_g is None:
                log(f"  ground-only PnP: FAIL:{detail_g}")
            else:
                rec_axis = _shared_optical_axis(other, solved_g)
                rec_center = solved_g.transform_point(other.camera_center)
                log(
                    f"  ground-only PnP: rmse={rmse_g:.1f} "
                    f"n={len(points_shared)} "
                    f"nadir={_angle_from_vertical_deg(rec_axis):.1f} "
                    f"C_z={float(rec_center[2]):.2f} "
                    f"axis=({rec_axis[0]:+.2f},{rec_axis[1]:+.2f},{rec_axis[2]:+.2f})"
                )

    log()
    log("=== solve_landmark_sync ===")
    try:
        result = sync_module.solve_landmark_sync(
            matches,
            observations,
            anchor_id=anchor_id,
            known_world=known_world,
            line_observations=line_observations,
            known_lines=known_lines,
            parallel_pairs=parallel,
        )
        log(f"success={result.success}")
        log(f"message={result.message}")
        log(f"mean_rmse={result.mean_reprojection_px:.3f}")
        log(f"per_match={result.per_match_rmse_px}")
        log(f"registered={sorted(result.similarities)}")
        for match_id in sorted(result.similarities):
            similarity = result.similarities[match_id]
            log(f"  {match_id} scale={similarity.scale:.6g}")
    except Exception:
        log(traceback.format_exc())

    report = "\n".join(lines) + "\n"
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        log_note = f"Wrote {out_path}"
        print(log_note)
    print(report, end="")
    return 0


if __name__ == "__main__":
    try:
        separator = sys.argv.index("--")
        cli_args = sys.argv[separator + 1 :]
    except ValueError:
        cli_args = []
    raise SystemExit(main(cli_args))
