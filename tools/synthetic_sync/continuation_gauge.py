"""Read-only orientation-gauge audit of a saved generated continuation endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.solver import load_core


def assess(archive: Path) -> dict:
    """Use only training geometry to identify and align an allowed frame gauge."""
    load_core()
    from match_perspective.core.focal_constraints import PointFocalConstraints
    from match_perspective.core.focal_line_constraints import LineFocalConstraints
    from match_perspective.core.focal_orientation import orientation_basis
    from match_perspective.core.sync.request import SyncSolveRequest
    from match_perspective.core.sync.solve import solution_result_from_seed

    case_file = archive / "joint-case.json"
    report_file = archive / "refine-result.json"
    case = json.loads(case_file.read_text())
    report = json.loads(report_file.read_text())
    request = SyncSolveRequest.from_record(report["post_request"])
    seed = request.initial_solution
    if seed is None or seed.evidence_sha256 != request.evidence_sha256():
        raise ValueError("Refine endpoint is not certified")
    result = solution_result_from_seed(seed)
    if result is None:
        raise ValueError("Refine endpoint lacks applied diagnostics")

    point_ids = sorted({item.landmark_id for item in request.observations})
    line_ids = sorted({item.landmark_id for item in request.line_observations or ()})
    anchor = seed.calibrations[request.anchor_id]
    rotation, center = anchor.rotation_w2c, anchor.camera_center
    centers = [similarity.transform_point(seed.calibrations[key].camera_center)
               for key, similarity in seed.similarities.items()
               if key != request.anchor_id]
    baseline = max(float(np.linalg.norm(value - center)) for value in centers)
    point_pairs = [pair for pair in request.mirror_pairs or ()
                   if pair[0] in point_ids and pair[1] in point_ids]
    line_pairs = [pair for pair in request.mirror_pairs or ()
                  if pair[0] in line_ids and pair[1] in line_ids]
    points = PointFocalConstraints.from_inputs(
        point_ids, anchor_rotation=rotation, anchor_center=center,
        baseline_world=baseline,
        plane_groups=[entry for entry in request.plane_groups or ()
                      if entry[0] in point_ids],
        plane_slack=request.plane_slack or 0.0,
        mirror_pairs=point_pairs, mirror_plane=request.mirror_plane,
        mirror_slack=request.mirror_slack or 0.0,
        mirror_landmark_id=request.mirror_landmark_id,
        extra_mirror_pairs=bool(line_pairs),
    )
    lines = LineFocalConstraints.from_inputs(
        point_ids, line_ids, plane_groups=request.plane_groups or (),
        parallel_pairs=request.parallel_pairs or (),
        anchor_rotation=rotation, plane_spring=points.plane_spring,
        baseline_world=baseline, hard_plane=points.hard_plane,
    )
    chart_points = np.asarray([
        rotation @ (result.landmarks[key] - center) / baseline
        for key in point_ids
    ])
    chart_lines = []
    for key in line_ids:
        first, second = result.line_segments[key]
        direction = second - first
        direction /= np.linalg.norm(direction)
        chart_lines.append((rotation @ (0.5 * (first + second) - center) / baseline,
                            rotation @ direction))
    basis = orientation_basis(points, lines, chart_points, chart_lines)
    output = dict(
        archive=str(archive.resolve()),
        case_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest(),
        observable_rotation_rank=len(basis),
        raw_withheld_rmse_px=report["assessment"]["withheld_rmse_px"],
        raw_accuracy_flags=report["assessment"]["accuracy_flags"],
    )
    axes = {axis for _key, axis, _bucket in request.plane_groups or ()
            if axis in {"X", "Y", "Z"}}
    if request.mirror_plane is not None and len(axes) == 1:
        axis = np.eye(3)["XYZ".index(next(iter(axes)))]
        normal = np.asarray(request.mirror_plane[1], float)
        normal /= np.linalg.norm(normal)
        output["world_axis_mirror_normal_angle_deg"] = float(np.degrees(
            np.arccos(np.clip(abs(float(axis @ normal)), 0.0, 1.0))))
    # The generated free case has one fixed X normal; a global X rotation
    # and positive scale preserve its hard relations. Fit both from training
    # 3D only, then check disjoint withheld image points without refitting.
    if (len(basis) == 2 and axes == {"X"} and
            not request.mirror_pairs and not request.known_world and
            not request.known_lines):
        truth = case["truth"]
        anchor_id = request.anchor_id
        true_cameras = {item["id"]: item for item in truth["cameras"]}
        true_center = np.asarray(true_cameras[anchor_id]["center"], float)
        fitted_center = np.asarray(report["recovered"]["cameras"][anchor_id]["center"], float)
        ids = sorted(truth["points"])
        source_points = np.asarray([truth["points"][key] for key in ids]) - true_center
        fitted_points = np.asarray([report["recovered"]["landmarks"][key]
                                    for key in ids]) - fitted_center
        cos_sum = float(np.sum(source_points[:, 1] * fitted_points[:, 1] +
                               source_points[:, 2] * fitted_points[:, 2]))
        sin_sum = float(np.sum(source_points[:, 1] * fitted_points[:, 2] -
                               source_points[:, 2] * fitted_points[:, 1]))
        theta = float(np.arctan2(sin_sum, cos_sum))
        cosine, sine = np.cos(theta), np.sin(theta)
        allowed_rotation = np.array(((1.0, 0.0, 0.0),
                                     (0.0, cosine, -sine),
                                     (0.0, sine, cosine)))
        rotated = source_points @ allowed_rotation.T
        scale = float(np.sum(fitted_points * rotated) / np.sum(source_points**2))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("Training points have no positive gauge alignment")
        errors = []
        for point_id, world in truth["holdouts"].items():
            aligned = fitted_center + scale * allowed_rotation @ (
                np.asarray(world, float) - true_center)
            for camera_id, camera in report["recovered"]["cameras"].items():
                pixel, depth = project([aligned], camera)
                if depth[0] <= 0:
                    raise ValueError("An aligned withheld point is behind a camera")
                oracle = truth["holdout_pixels"][point_id][camera_id]
                errors.append(float(np.linalg.norm(pixel[0] - oracle)))
        output["training_only_allowed_alignment"] = dict(
            rotation_about_x_deg=float(np.degrees(theta)), scale=scale,
            withheld_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
            withheld_max_px=max(errors),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(ROOT.resolve()):
        parser.error("Audit output must stay outside the repository")
    report = assess(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
