"""Frozen point-FOV constraint cases with a solver-independent pinhole oracle.

The request contains only stored camera poses, picks and user constraints.
Truth and unused 3D checks live outside the request. The scene is an ideal
unobstructed scaffold of visible markers: visibility means positive depth and
an interior image pixel, with no opaque surfaces or occlusion claims. No
production solver or projection module is imported here.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import look_at, project


ROOT = Path(__file__).resolve().parent / "cases" / "focal-constraints"
SCHEMA_VERSION = 1
CASE_NAMES = (
    "free-hard", "free-hard-removed", "free-soft", "free-removed",
    "axis-hard", "axis-hard-removed", "axis-soft", "axis-removed",
    "mirror-hard-offcenter", "mirror-biased-hard", "mirror-biased-soft",
    "mirror-tilted-hard",
    "mirror-removed", "mirror-through-anchor",
    "free-hard-exact", "free-hard-removed-exact",
    "axis-hard-exact", "axis-hard-removed-exact",
    "mirror-hard-offcenter-exact", "mirror-removed-exact",
    "mirror-through-anchor-exact",
)
ANCHOR = "view_0"
PICK_NOISE_PX = 0.35


def _unit(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("Invalid fixture direction")
    return vector / length


def _cameras() -> list[dict]:
    centers = ((-4.2, -5.0, 3.1), (4.4, -4.1, 2.8),
               (4.3, 4.4, 3.5), (-4.0, 4.1, 2.5))
    focals = (870.0, 1040.0, 760.0, 950.0)
    return [dict(id=f"view_{index}", width=1280, height=960,
                 fx=focal, fy=focal, cx=640.0, cy=480.0,
                 center=list(center), rotation=look_at(center, (0.0, 0.0, 1.0)))
            for index, (center, focal) in enumerate(zip(centers, focals))]


def _stored_cameras(truth: list[dict]) -> list[dict]:
    stored = deepcopy(truth)
    factors = (1.10, 0.91, 1.13, 0.88)
    for index, camera in enumerate(stored):
        camera["fx"] *= factors[index]
        camera["fy"] = camera["fx"]
        if index:
            # These private frames are unrelated to the common world. Their
            # input order must not affect truth or the supplied constraints.
            center = np.array((-2.3, -3.2, 2.2)) + index * np.array((0.2, -0.1, 0.15))
            camera["center"] = center.tolist()
            camera["rotation"] = look_at(center, (0.3, -0.2, 0.6))
    return stored


def _base_points() -> dict[str, list[float]]:
    xyz = (
        (-0.90, -0.65, 0.28), (-0.45, 0.62, 1.82), (0.78, -0.58, 1.35),
        (0.95, 0.65, 0.45), (-0.68, 0.30, 1.05), (0.10, -0.82, 1.95),
        (0.35, 0.78, 1.65), (-0.12, -0.12, 0.25), (0.72, 0.10, 2.10),
        (-0.82, -0.12, 1.54), (0.22, 0.45, 0.72), (-0.30, -0.72, 1.18),
    )
    return {f"base_{index:02d}": list(point) for index, point in enumerate(xyz)}


def _free_points(*, warped: bool) -> dict[str, list[float]]:
    xy = ((-0.72, -0.46), (-0.28, 0.67), (0.22, -0.76),
          (0.80, 0.42), (0.56, -0.20))
    departures = (0.0, 0.028, -0.025, 0.036, -0.019) if warped else (0.0,) * 5
    return {f"free_{index}": [x, y, 0.94 + 0.24 * x - 0.17 * y + departures[index]]
            for index, (x, y) in enumerate(xy)}


def _axis_points(*, warped: bool) -> dict[str, list[float]]:
    yz = ((-0.62, 0.45), (0.52, 0.88), (-0.24, 1.64), (0.34, 1.92))
    departures = (0.0, 0.035, -0.029, 0.018) if warped else (0.0,) * 4
    return {f"axis_{index}": [0.38 + departures[index], y, z]
            for index, (y, z) in enumerate(yz)}


def _mirror_geometry(*, through_anchor: bool, anchor_center) -> tuple[
    dict[str, list[float]], list[list[str]], list[list[float]]]:
    if through_anchor:
        # This normal is perpendicular to anchor_center minus (0,0,1).
        # Compute it exactly from that vector so the scale-gauge control is exact.
        displacement = np.asarray(anchor_center, float) - (0.0, 0.0, 1.0)
        normal = _unit((displacement[1], -displacement[0], 0.0))
        origin = np.asarray(anchor_center, float)
    else:
        normal = _unit((1.0, 0.22, 0.11))
        origin = np.asarray((0.16, 0.0, 0.96), float)
    tangent_a = _unit(np.cross(normal, (0.0, 0.0, 1.0)))
    tangent_b = _unit(np.cross(normal, tangent_a))
    points = {}
    pairs = []
    for index, (along, height, distance) in enumerate(
        ((-0.54, -0.42, 0.47), (0.33, 0.38, 0.60), (0.58, -0.15, 0.37))
    ):
        middle = np.asarray((0.0, 0.0, 1.0)) + along * tangent_a + height * tangent_b
        # The through-anchor plane passes through this chosen scene centre.
        if not through_anchor:
            middle += normal * float(normal @ (origin - middle))
        left = middle + distance * normal
        right = middle - distance * normal
        pair = [f"mirror_{index}_a", f"mirror_{index}_b"]
        points[pair[0]], points[pair[1]] = left.tolist(), right.tolist()
        pairs.append(pair)
    return points, pairs, [origin.tolist(), normal.tolist()]


def _holdouts() -> dict[str, list[float]]:
    xyz = ((-0.72, 0.75, 0.35), (0.83, -0.74, 0.83), (-0.33, -0.22, 2.20),
           (0.43, 0.18, 1.23), (-0.90, 0.15, 0.62), (0.13, -0.56, 1.67),
           (0.90, 0.54, 1.90), (-0.18, 0.84, 1.35))
    return {f"check_{index:02d}": list(point) for index, point in enumerate(xyz)}


def _inside(point, camera, *, margin: float = 20.0) -> bool:
    uv, depth = project([point], camera)
    return bool(np.isfinite(uv).all() and depth[0] > 0 and
                margin < uv[0, 0] < camera["width"] - margin and
                margin < uv[0, 1] < camera["height"] - margin)


def generate(name: str) -> dict:
    """Generate one deterministic request and its disjoint reference truth."""
    if name not in CASE_NAMES:
        raise ValueError(f"Unknown focal constraint case: {name}")
    exact_picks = name.endswith("-exact")
    base_name = name[:-6] if exact_picks else name
    truth_cameras = _cameras()
    anchor_center = truth_cameras[0]["center"]
    points = _base_points()
    family = base_name.split("-")[0]
    plane_groups: list[list] = []
    mirror_pairs: list[list[str]] = []
    mirror_plane = None
    plane_slack = mirror_slack = 0.0
    true_mirror_plane = None
    reference_truth_satisfies_request: bool | None = True
    if family == "free":
        points.update(_free_points(warped=base_name not in {"free-hard", "free-hard-removed"}))
        if base_name not in {"free-removed", "free-hard-removed"}:
            plane_groups = [[key, "FREE", 1] for key in points if key.startswith("free_")]
            plane_slack = 0.08 if base_name == "free-soft" else 0.0
        else:
            reference_truth_satisfies_request = None
    elif family == "axis":
        points.update(_axis_points(warped=base_name not in {"axis-hard", "axis-hard-removed"}))
        if base_name not in {"axis-removed", "axis-hard-removed"}:
            plane_groups = [[key, "X", 2] for key in points if key.startswith("axis_")]
            plane_slack = 0.08 if base_name == "axis-soft" else 0.0
        else:
            reference_truth_satisfies_request = None
    else:
        through_anchor = base_name == "mirror-through-anchor"
        mirror_points, pairs, true_mirror_plane = _mirror_geometry(
            through_anchor=through_anchor, anchor_center=anchor_center)
        points.update(mirror_points)
        if base_name != "mirror-removed":
            mirror_pairs = pairs
            mirror_plane = deepcopy(true_mirror_plane)
            if base_name.startswith("mirror-biased"):
                origin, normal = (np.asarray(value, float) for value in mirror_plane)
                mirror_plane[0] = (origin + 0.08 * normal).tolist()
                mirror_slack = 0.15 if base_name.endswith("soft") else 0.0
                reference_truth_satisfies_request = base_name.endswith("soft")
            elif base_name == "mirror-tilted-hard":
                normal = np.asarray(mirror_plane[1], float)
                tangent = _unit(np.cross(normal, (0.0, 0.0, 1.0)))
                mirror_plane[1] = _unit(normal + 0.18 * tangent).tolist()
                reference_truth_satisfies_request = False
        else:
            reference_truth_satisfies_request = None

    # Identical noisy picks across each hard/soft/removal comparison when
    # geometry matches. The seed does not depend on the case label.
    rng = np.random.default_rng(20260912)
    applied_noise = 0.0 if exact_picks else PICK_NOISE_PX
    observations = []
    oracle_pixels = {}
    for point_id, position in sorted(points.items()):
        oracle_pixels[point_id] = {}
        for camera in truth_cameras:
            if not _inside(position, camera):
                raise ValueError(f"Training point {point_id} is not visible in {camera['id']}")
            uv = project([position], camera)[0][0]
            oracle_pixels[point_id][camera["id"]] = uv.tolist()
            noisy = uv + rng.normal(0.0, applied_noise, 2)
            observations.append(dict(match_id=camera["id"], landmark_id=point_id,
                                     u=float(noisy[0]), v=float(noisy[1]), weight=1.0))
    holdouts = _holdouts()
    holdout_pixels = {}
    for point_id, position in holdouts.items():
        holdout_pixels[point_id] = {}
        for camera in truth_cameras:
            if not _inside(position, camera):
                raise ValueError(f"Holdout {point_id} is not visible in {camera['id']}")
            holdout_pixels[point_id][camera["id"]] = project([position], camera)[0][0].tolist()
    request = dict(
        cameras=_stored_cameras(truth_cameras),
        points=[dict(id=point_id, ground=False, known=None) for point_id in sorted(points)],
        observations=observations, lines=[], line_observations=[], anchor_id=ANCHOR,
        fixed_similarities={}, lock_rotation=False, lock_translation=False,
        ground_slack=0.0, known_3d_slack=0.0, mirror_pairs=mirror_pairs,
        mirror_plane=mirror_plane, mirror_slack=mirror_slack, parallel_pairs=[],
        plane_groups=plane_groups, plane_slack=plane_slack,
    )
    behavior = ("conditional_metric_bias" if base_name.startswith("mirror-biased")
                else "orientation_conflict_probe" if base_name == "mirror-tilted-hard"
                else "relation_removed" if base_name.endswith("removed")
                else "reference_geometry")
    reference_exact = (None if behavior == "relation_removed" else
                       base_name not in {"free-soft", "axis-soft", "mirror-biased-hard",
                                    "mirror-biased-soft", "mirror-tilted-hard"})
    case = dict(schema_version=SCHEMA_VERSION, name=name, family=family,
                pick_sigma_px=PICK_NOISE_PX, request=request,
                expectation=dict(reference_truth_satisfies_request=reference_truth_satisfies_request,
                                 reference_truth_exact=reference_exact,
                                 behavior=behavior,
                                 metric_truth_independently_known_to_solver=False,
                                 scale_gauge="free" if family != "mirror" or
                                 base_name in {"mirror-removed", "mirror-through-anchor"}
                                 else "anchor-plane-offset",
                                 one_view_members=False),
                truth=dict(cameras=truth_cameras, points=points,
                           scene="ideal_unobstructed_landmark_scaffold",
                           mirror_plane=true_mirror_plane, oracle_pixels=oracle_pixels,
                           holdouts=holdouts, holdout_pixels=holdout_pixels))
    validate(case)
    return case


def _reflect(point, plane) -> np.ndarray:
    origin, normal = (np.asarray(value, float) for value in plane)
    return np.asarray(point, float) - 2 * normal * float(normal @ (np.asarray(point, float) - origin))


def biased_mirror_scale_witness(case: dict) -> dict:
    """Exact pixel/reflection alternative when only mirror-plane offset is biased.

    The fixed anchor centre lies off the plane. Scaling all other centres and
    points about it can satisfy the shifted *hard* plane with unchanged image
    projections and focals. This is a metric-bias witness, not a solver result.
    The soft case also has this alternative, alongside the true-scale geometry
    with a normal-only plane shift within its slack.
    """
    if case["name"] not in {"mirror-biased-hard", "mirror-biased-soft"}:
        raise ValueError("Scale witness is defined only for biased mirror offsets")
    anchor = case["request"]["anchor_id"]
    true_cameras = case["truth"]["cameras"]
    center = np.asarray(next(item["center"] for item in true_cameras if item["id"] == anchor), float)
    true_origin, normal = (np.asarray(value, float) for value in case["truth"]["mirror_plane"])
    request_origin = np.asarray(case["request"]["mirror_plane"][0], float)
    true_distance = float(normal @ (true_origin - center))
    request_distance = float(normal @ (request_origin - center))
    if abs(true_distance) < 1e-9:
        raise ValueError("Offset bias has no scale witness through the anchor")
    scale = request_distance / true_distance
    if scale <= 0:
        raise ValueError("Biased plane requests a reversed scale")
    cameras = deepcopy(true_cameras)
    for camera in cameras:
        camera["center"] = (center + scale * (np.asarray(camera["center"]) - center)).tolist()
    points = {key: (center + scale * (np.asarray(point) - center)).tolist()
              for key, point in case["truth"]["points"].items()}
    return dict(scale=float(scale), cameras={item["id"]: item for item in cameras},
                landmarks=points)


def _plane_distance(points: dict, group: list[list]) -> np.ndarray:
    ids = [item[0] for item in group]
    xyz = np.asarray([points[key] for key in ids], float)
    axis = group[0][1]
    if axis != "FREE":
        coord = xyz[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        return coord - coord.mean()
    centered = xyz - xyz.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[-1]


def validate(case: dict) -> None:
    """Check fixture consistency without consulting the production solver."""
    request, truth = case["request"], case["truth"]
    if truth.get("scene") != "ideal_unobstructed_landmark_scaffold":
        raise ValueError("Fixture visibility must be declared as frustum-only")
    cameras = truth["cameras"]
    if len(cameras) not in (3, 4) or len({camera["id"] for camera in cameras}) != len(cameras):
        raise ValueError("Fixture needs three or four distinct cameras")
    anchor_truth = next(item for item in cameras if item["id"] == request["anchor_id"])
    anchor_stored = next(item for item in request["cameras"] if item["id"] == request["anchor_id"])
    for field in ("center", "rotation"):
        if not np.array_equal(anchor_truth[field], anchor_stored[field]):
            raise ValueError("External constraints require the true fixed anchor frame")
    if np.linalg.norm(np.asarray(anchor_truth["rotation"]) - np.eye(3)) < 0.1:
        raise ValueError("The anchor rotation should exercise transformed normals")
    if not any(np.linalg.norm(np.asarray(stored["center"]) - np.asarray(true["center"])) > 0.5
               for stored, true in zip(request["cameras"][1:], cameras[1:])):
        raise ValueError("Non-anchor private poses were not perturbed")
    point_ids = set(truth["points"])
    if point_ids & set(truth["holdouts"]):
        raise ValueError("Training and withheld points overlap")
    counts = {camera["id"]: 0 for camera in cameras}
    seen = set()
    for pick in request["observations"]:
        key = (pick["match_id"], pick["landmark_id"])
        if key in seen or key[1] not in point_ids or key[0] not in counts:
            raise ValueError("Duplicate or unknown pick")
        seen.add(key)
        counts[key[0]] += 1
        camera = next(item for item in cameras if item["id"] == key[0])
        exact = project([truth["points"][key[1]]], camera)[0][0]
        uv = np.asarray(truth["oracle_pixels"][key[1]][key[0]])
        if not np.allclose(exact, uv, rtol=0.0, atol=1e-10):
            raise ValueError("Stored training oracle differs from independent projection")
        if not np.isfinite(uv).all() or np.linalg.norm(uv - (pick["u"], pick["v"])) > 5 * PICK_NOISE_PX:
            raise ValueError("Pick disagrees with the independent oracle")
    if min(counts.values()) < 8 or any(sum(key[1] == point_id for key in seen) < 2
                                     for point_id in point_ids):
        raise ValueError("Fixture lacks multiview point support")
    for point_id, position in {**truth["points"], **truth["holdouts"]}.items():
        for camera in cameras:
            if not _inside(position, camera):
                raise ValueError(f"{point_id} is outside the camera frustum")
            if point_id in truth["holdouts"]:
                exact = project([position], camera)[0][0]
                stored = truth["holdout_pixels"][point_id][camera["id"]]
                if not np.allclose(exact, stored, rtol=0.0, atol=1e-10):
                    raise ValueError("Stored holdout oracle differs from independent projection")
    if request["plane_groups"]:
        distances = _plane_distance(truth["points"], request["plane_groups"])
        limit = 1e-10 if request["plane_slack"] == 0 else request["plane_slack"]
        if np.max(abs(distances)) > limit:
            raise ValueError("Fixture points violate their declared plane budget")
    if truth["mirror_plane"] is not None:
        for left, right in ([ [f"mirror_{index}_a", f"mirror_{index}_b"] for index in range(3) ]):
            if np.linalg.norm(_reflect(truth["points"][left], truth["mirror_plane"]) -
                              truth["points"][right]) > 1e-10:
                raise ValueError("True mirror pair does not reflect")
        if request["mirror_plane"] is not None:
            normal = np.asarray(request["mirror_plane"][1], float)
            anchor_distance = abs(normal @ (np.asarray(anchor_truth["center"]) -
                                            np.asarray(request["mirror_plane"][0])))
            base_name = case["name"].removesuffix("-exact")
            if base_name == "mirror-through-anchor" and anchor_distance > 1e-10:
                raise ValueError("Through-anchor mirror did not preserve scale gauge")
            if base_name != "mirror-through-anchor" and anchor_distance < 0.5:
                raise ValueError("Off-anchor mirror cannot exercise metric scale")
            if base_name in {"mirror-biased-hard", "mirror-tilted-hard"}:
                if not any(np.linalg.norm(_reflect(truth["points"][left], request["mirror_plane"]) -
                                          truth["points"][right]) > 0.08
                           for left, right in request["mirror_pairs"]):
                    raise ValueError("Reference truth does not depart from the supplied hard mirror")
            if base_name in {"mirror-biased-hard", "mirror-biased-soft"}:
                witness = biased_mirror_scale_witness(case)
                for left, right in request["mirror_pairs"]:
                    if np.linalg.norm(_reflect(witness["landmarks"][left], request["mirror_plane"]) -
                                      witness["landmarks"][right]) > 1e-9:
                        raise ValueError("Biased mirror scale witness does not reflect")
                for point_id, view_pixels in truth["oracle_pixels"].items():
                    for camera_id, expected in view_pixels.items():
                        pixel = project([witness["landmarks"][point_id]], witness["cameras"][camera_id])[0][0]
                        if not np.allclose(pixel, expected, rtol=0.0, atol=1e-9):
                            raise ValueError("Biased mirror scale witness changes image evidence")


def assess(case: dict, fitted: dict, *, validated: bool = False) -> dict:
    """Independently score fitted world cameras/points against withheld truth.

    ``fitted`` uses ``{cameras: {id: {fx, fy, center, rotation}}, landmarks: {id: xyz}}``.
    Scale-free cases align only their global scale about the fixed anchor. The
    reported scale ratio itself is never hidden by that alignment.
    """
    if not validated:
        validate(case)
    truth = case["truth"]
    true_cameras = {item["id"]: item for item in truth["cameras"]}
    fitted_cameras = fitted["cameras"]
    if set(fitted_cameras) != set(true_cameras):
        raise ValueError("Fitted cameras differ from the fixture graph")
    anchor = case["request"]["anchor_id"]
    true_center = np.asarray(true_cameras[anchor]["center"], float)
    fitted_center = np.asarray(fitted_cameras[anchor]["center"], float)
    true_baseline = max(np.linalg.norm(np.asarray(item["center"]) - true_center)
                        for key, item in true_cameras.items() if key != anchor)
    fitted_baseline = max(np.linalg.norm(np.asarray(item["center"]) - fitted_center)
                          for key, item in fitted_cameras.items() if key != anchor)
    scale_ratio = fitted_baseline / true_baseline
    if not np.isfinite(scale_ratio) or scale_ratio <= 0:
        raise ValueError("Fitted baseline is invalid")
    free_scale = case["expectation"]["scale_gauge"] == "free"
    alignment_scale = 1.0
    if free_scale:
        # The baseline ratio is a diagnostic, not a reliable alignment for
        # noisy free-scale reconstructions. Fit exactly one positive global
        # scale from training 3D truth, never from withheld points or cameras.
        point_ids = sorted(truth["points"])
        reference = np.asarray([truth["points"][key] for key in point_ids], float) - true_center
        reconstructed = np.asarray([fitted["landmarks"][key] for key in point_ids], float) - fitted_center
        denominator = float(np.sum(reference * reference))
        alignment_scale = float(np.sum(reference * reconstructed) / denominator)
        if not np.isfinite(alignment_scale) or alignment_scale <= 0:
            raise ValueError("Fitted training points have no positive scale alignment")
    errors = []
    for point_id, world in truth["holdouts"].items():
        point = np.asarray(world, float)
        if free_scale:
            point = fitted_center + alignment_scale * (point - true_center)
        for camera_id, camera in fitted_cameras.items():
            # The joint fitter reports pose and focal; principal point and
            # image dimensions remain fixed at the input values.
            predicted, depth = project([point], {**true_cameras[camera_id], **camera})
            if depth[0] <= 0 or not np.isfinite(predicted).all():
                errors.append(float("inf"))
                continue
            target = truth["holdout_pixels"][point_id][camera_id]
            errors.append(float(np.linalg.norm(predicted[0] - target)))
    focal_relative = {key: float(camera["fx"] / true_cameras[key]["fx"] - 1)
                      for key, camera in fitted_cameras.items()}
    center_errors = []
    rotation_errors = []
    for camera_id, camera in fitted_cameras.items():
        center = np.asarray(camera["center"], float)
        if free_scale:
            center = true_center + (center - fitted_center) / alignment_scale
        center_errors.append(float(np.linalg.norm(center - true_cameras[camera_id]["center"])))
        relative = np.asarray(camera["rotation"], float) @ np.asarray(
            true_cameras[camera_id]["rotation"], float).T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    point_errors = []
    for point_id, truth_point in truth["points"].items():
        point = np.asarray(fitted["landmarks"][point_id], float)
        if free_scale:
            point = true_center + (point - fitted_center) / alignment_scale
        point_errors.append(float(np.linalg.norm(point - truth_point)))
    plane_rms = None
    if case["request"]["plane_groups"]:
        distances = _plane_distance(fitted["landmarks"], case["request"]["plane_groups"])
        plane_rms = float(np.sqrt(np.mean(distances**2)))
    mirror_gap = midpoint_shift = adjusted_gap = midpoint_spread = None
    if case["request"]["mirror_pairs"]:
        plane = case["request"]["mirror_plane"]
        origin, normal = (np.asarray(value, float) for value in plane)
        gaps, offsets = [], []
        for left, right in case["request"]["mirror_pairs"]:
            a, b = (np.asarray(fitted["landmarks"][key], float) for key in (left, right))
            gaps.append(float(np.linalg.norm(_reflect(a, plane) - b)))
            offsets.append(float(normal @ (0.5 * (a + b) - origin)))
        mirror_gap = max(gaps)
        midpoint_shift = float(np.mean(offsets))
        midpoint_spread = float(np.sqrt(np.mean((np.asarray(offsets) - midpoint_shift)**2)))
        effective_plane = [(origin + midpoint_shift * normal).tolist(), normal.tolist()]
        adjusted_gap = max(float(np.linalg.norm(
            _reflect(fitted["landmarks"][left], effective_plane) - fitted["landmarks"][right]))
            for left, right in case["request"]["mirror_pairs"])
    return dict(scale_ratio=float(scale_ratio), alignment_scale=float(alignment_scale),
                focal_relative=focal_relative,
                withheld_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
                withheld_max_px=float(max(errors)),
                center_rmse_world=float(np.sqrt(np.mean(np.square(center_errors)))),
                landmark_rmse_world=float(np.sqrt(np.mean(np.square(point_errors)))),
                max_rotation_error_deg=max(rotation_errors), plane_rms_world=plane_rms,
                mirror_max_gap_world=mirror_gap,
                mirror_midpoint_offset_world=midpoint_shift,
                mirror_midpoint_spread_world=midpoint_spread,
                mirror_max_gap_shifted_world=adjusted_gap)


def write_cases(directory: Path = ROOT) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in CASE_NAMES:
        path = directory / f"{name}.json"
        path.write_text(json.dumps(generate(name), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Freeze generated cases")
    options = parser.parse_args()
    if options.write:
        write_cases()
    else:
        for case_name in CASE_NAMES:
            validate(json.loads((ROOT / f"{case_name}.json").read_text()))
