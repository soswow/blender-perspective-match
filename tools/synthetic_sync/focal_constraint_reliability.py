"""Frozen weak-evidence plane/mirror controls with independent projection truth.

The scaffold checks frustum visibility only. Its cameras have a shorter,
mostly lateral baseline than the earlier constraint cases; each point is picked
in two or three views. Stored non-anchor poses remain deliberately private.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .focal_constraints import assess as assess_fixed_frame
from .focal_constraints import generate as generate_strong
from .geometry import look_at, project


ROOT = Path(__file__).resolve().parent / "cases" / "focal-constraint-reliability"
CASE_BASE = {
    "weak-axis-hard": "axis-hard",
    "weak-axis-removed": "axis-hard",
    "weak-axis-wrong-member": "axis-hard",
    "weak-free-hard": "free-hard",
    "weak-free-removed": "free-hard",
    "weak-mirror-hard": "mirror-hard-offcenter",
    "weak-mirror-removed": "mirror-hard-offcenter",
    "weak-mirror-tilted": "mirror-hard-offcenter",
    "weak-mirror-anchor-rotated": "mirror-hard-offcenter",
    "weak-mirror-anchor-rotated-removed": "mirror-hard-offcenter",
}
CASE_NAMES = tuple(CASE_BASE)
CENTERS = ((-2.0, -6.8, 2.70), (-0.7, -7.2, 2.80),
           (0.7, -6.9, 2.75), (2.0, -7.1, 2.65))
TARGET = (0.0, 0.0, 1.0)
PICK_SIGMA_PX = 0.35
MIRROR_NORMAL_TILT = 0.04
ANCHOR_YAW_RAD = 0.025


def _unit(vector):
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def _inside(point, camera):
    uv, depth = project([point], camera)
    return bool(depth[0] > 0 and np.isfinite(uv).all() and
                20 < uv[0, 0] < camera["width"] - 20 and
                20 < uv[0, 1] < camera["height"] - 20)


def generate(name: str) -> dict:
    if name not in CASE_BASE:
        raise ValueError(name)
    base = generate_strong(CASE_BASE[name])
    case = deepcopy(base)
    case["schema_version"] = 2
    case["name"] = name
    case["scaffold"] = dict(centers=CENTERS, target=TARGET,
                            observation_rule="(point_index + camera_index) % 3 != 0",
                            mirror_normal_tilt=MIRROR_NORMAL_TILT,
                            anchor_yaw_rad=ANCHOR_YAW_RAD)
    truth, request = case["truth"], case["request"]
    for i, camera in enumerate(truth["cameras"]):
        camera["center"] = list(CENTERS[i])
        camera["rotation"] = look_at(CENTERS[i], TARGET)
    anchor = request["cameras"][0]
    anchor["center"] = truth["cameras"][0]["center"]
    anchor["rotation"] = truth["cameras"][0]["rotation"]
    if name.startswith("weak-axis"):
        # The decoy is present in all three matched-pick requests. Only the
        # wrong-reference variant adds it to the axis bucket (4 cm off-plane).
        truth["points"]["axis_decoy"] = [0.42, 0.11, 1.31]
        request["points"].append(dict(id="axis_decoy", ground=False, known=None))
        request["points"].sort(key=lambda item: item["id"])
    if name in {"weak-axis-removed", "weak-free-removed"}:
        request["plane_groups"] = []
    if name == "weak-axis-wrong-member":
        request["plane_groups"].append(["axis_decoy", "X", 2])
    if name in {"weak-mirror-removed", "weak-mirror-anchor-rotated-removed"}:
        request["mirror_pairs"] = []
        request["mirror_plane"] = None
    if name == "weak-mirror-tilted":
        normal = np.asarray(request["mirror_plane"][1], float)
        tangent = _unit(np.cross(normal, (0.0, 0.0, 1.0)))
        request["mirror_plane"][1] = _unit(normal + MIRROR_NORMAL_TILT * tangent).tolist()
    if "anchor-rotated" in name:
        angle = ANCHOR_YAW_RAD
        turn = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        anchor["rotation"] = (np.asarray(anchor["rotation"]) @ turn).tolist()

    rng = np.random.default_rng(20260912)
    request["observations"] = []
    truth["oracle_pixels"] = {}
    for point_index, (point_id, point) in enumerate(sorted(truth["points"].items())):
        truth["oracle_pixels"][point_id] = {}
        for camera_index, camera in enumerate(truth["cameras"]):
            if not _inside(point, camera):
                raise ValueError(f"{point_id} outside {camera['id']}")
            uv = project([point], camera)[0][0]
            truth["oracle_pixels"][point_id][camera["id"]] = uv.tolist()
            noise = rng.normal(0.0, PICK_SIGMA_PX, 2)
            if (point_index + camera_index) % 3:
                request["observations"].append(dict(
                    match_id=camera["id"], landmark_id=point_id,
                    u=float(uv[0] + noise[0]), v=float(uv[1] + noise[1]), weight=1.0))
    truth["holdout_pixels"] = {}
    for point_id, point in truth["holdouts"].items():
        truth["holdout_pixels"][point_id] = {}
        for camera in truth["cameras"]:
            if not _inside(point, camera):
                raise ValueError(f"Holdout {point_id} outside {camera['id']}")
            truth["holdout_pixels"][point_id][camera["id"]] = project([point], camera)[0][0].tolist()
    case["expectation"].update(
        behavior=("wrong_anchor_frame" if "anchor-rotated" in name else
                  "wrong_relation" if name.endswith(("wrong-member", "tilted")) else
                  "relation_removed" if name.endswith("removed") else "reference_geometry"),
        reference_truth_exact=name in {"weak-axis-hard", "weak-free-hard", "weak-mirror-hard"},
        reference_truth_satisfies_request=name in {"weak-axis-hard", "weak-free-hard", "weak-mirror-hard"},
        partial_overlap=True,
    )
    validate(case)
    return case


def validate(case: dict) -> None:
    name = case["name"]
    if name not in CASE_NAMES or case["schema_version"] != 2:
        raise ValueError("Unknown reliability fixture")
    truth, request = case["truth"], case["request"]
    if truth["scene"] != "ideal_unobstructed_landmark_scaffold":
        raise ValueError("Missing visibility limitation")
    if set(truth["points"]) & set(truth["holdouts"]):
        raise ValueError("Training and holdout overlap")
    counts = {camera["id"]: 0 for camera in truth["cameras"]}
    picked = {point_id: set() for point_id in truth["points"]}
    for observation in request["observations"]:
        camera_id, point_id = observation["match_id"], observation["landmark_id"]
        if point_id not in picked or camera_id in picked[point_id]:
            raise ValueError("Unknown or repeated training pick")
        camera = next(item for item in truth["cameras"] if item["id"] == camera_id)
        uv = project([truth["points"][point_id]], camera)[0][0]
        if not _inside(truth["points"][point_id], camera) or not np.allclose(
                uv, truth["oracle_pixels"][point_id][camera_id], rtol=0, atol=1e-10):
            raise ValueError("Training oracle mismatch")
        if np.linalg.norm(uv - (observation["u"], observation["v"])) > 5 * PICK_SIGMA_PX:
            raise ValueError("Training pick beyond declared noise")
        picked[point_id].add(camera_id)
        counts[camera_id] += 1
    if min(counts.values()) < 8 or any(len(views) < 2 or len(views) > 3
                                     for views in picked.values()):
        raise ValueError("Weak graph has insufficient or full-view support")
    for point_id, point in truth["holdouts"].items():
        for camera in truth["cameras"]:
            uv = project([point], camera)[0][0]
            if not _inside(point, camera) or not np.allclose(
                    uv, truth["holdout_pixels"][point_id][camera["id"]], rtol=0, atol=1e-10):
                raise ValueError("Holdout oracle mismatch")
    truth_anchor = truth["cameras"][0]
    stored_anchor = request["cameras"][0]
    rotation_gap = np.linalg.norm(np.asarray(truth_anchor["rotation"]) - stored_anchor["rotation"])
    if ("anchor-rotated" in name) != (rotation_gap > 0.01):
        raise ValueError("Anchor mismatch is not the declared intervention")
    if not np.array_equal(truth_anchor["center"], stored_anchor["center"]):
        raise ValueError("Anchor center changed")
    if name == "weak-axis-wrong-member":
        if request["plane_groups"][-1] != ["axis_decoy", "X", 2]:
            raise ValueError("Wrong plane member missing")
        if abs(truth["points"]["axis_decoy"][0] - 0.38) < 0.03:
            raise ValueError("Wrong member is effectively on the plane")
    if name == "weak-mirror-tilted":
        base = generate_strong("mirror-hard-offcenter")
        original = np.asarray(base["request"]["mirror_plane"][1])
        requested = np.asarray(request["mirror_plane"][1])
        angle = np.degrees(np.arccos(np.clip(original @ requested, -1, 1)))
        if not 2.0 < angle < 2.5:
            raise ValueError("Mirror tilt differs from frozen intervention")


def _withheld_rmse(case: dict, fitted: dict, world_transform) -> float:
    errors = []
    cameras = fitted["cameras"]
    for point_id, point in case["truth"]["holdouts"].items():
        world = world_transform(np.asarray(point, float))
        for camera_id, camera in cameras.items():
            reference = next(item for item in case["truth"]["cameras"] if item["id"] == camera_id)
            pixels, depth = project([world], {**reference, **camera})
            target = case["truth"]["holdout_pixels"][point_id][camera_id]
            errors.append(float(np.linalg.norm(pixels[0] - target)) if depth[0] > 0 else float("inf"))
    return float(np.sqrt(np.mean(np.square(errors))))


def assess(case: dict, fitted: dict) -> dict:
    """Report raw frame and one training-only global-similarity shape check."""
    validate(case)
    usual = assess_fixed_frame(case, fitted, validated=True)
    truth = case["truth"]
    ids = sorted(truth["points"])
    source = np.asarray([truth["points"][key] for key in ids], float)
    target = np.asarray([fitted["landmarks"][key] for key in ids], float)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    u, singular, vh = np.linalg.svd(source_zero.T @ target_zero)
    sign = np.diag([1.0, 1.0, np.linalg.det(vh.T @ u.T)])
    rotation = vh.T @ sign @ u.T
    scale = float(np.trace(np.diag(singular) @ sign) / np.sum(source_zero ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training points do not determine a positive similarity")
    transformed = lambda world: target_mean + scale * rotation @ (world - source_mean)
    translation = target_mean - scale * rotation @ source_mean
    usual["raw_frame_withheld_rmse_px"] = _withheld_rmse(case, fitted, lambda world: world)
    usual["shape_withheld_rmse_px"] = _withheld_rmse(case, fitted, transformed)
    usual["shape_training_scale"] = scale
    usual["shape_training_rotation_w2w"] = rotation.tolist()
    usual["shape_training_translation_world"] = translation.tolist()
    usual["shape_training_point_rmse_world"] = float(np.sqrt(np.mean(np.sum(
        (source @ (scale * rotation).T + translation - target) ** 2, axis=1))))
    usual["shape_training_rotation_deg"] = float(np.degrees(np.arccos(np.clip(
        (np.trace(rotation) - 1) / 2, -1, 1))))
    return usual


def write_cases(directory: Path = ROOT) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in CASE_NAMES:
        (directory / f"{name}.json").write_text(
            json.dumps(generate(name), indent=2, allow_nan=False) + "\n")


def reassess_bundle(case_name: str, ledger: Path, output: Path) -> None:
    """Reassess archived numerical output; never invoke a solver."""
    case = json.loads((ROOT / f"{case_name}.json").read_text())
    validate(case)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    start = next(row for row in rows if row.get("kind") == "started" and
                 row.get("label") == f"{case_name}:joint-bundle")
    completed = next(row for row in rows if row.get("kind") == "completed" and
                     row.get("key") == start["key"])
    record = completed["result"]
    if not record["accepted"]:
        raise ValueError("Archived bundle was refused")
    result = dict(case_name=case_name, ledger=str(ledger),
                  numeric_record_sha256=hashlib.sha256(json.dumps(
                      record, sort_keys=True).encode()).hexdigest(),
                  assessment=assess(case, record["fitted"]))
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--reassess-bundle", nargs=3,
                        metavar=("CASE", "LEDGER", "OUTPUT"))
    options = parser.parse_args()
    if options.reassess_bundle:
        case_name, ledger, output = options.reassess_bundle
        reassess_bundle(case_name, Path(ledger), Path(output))
    elif options.write:
        write_cases()
    else:
        for filename in sorted(ROOT.glob("weak-*.json")):
            validate(json.loads(filename.read_text()))
