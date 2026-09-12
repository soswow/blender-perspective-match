"""Exact-pick orientation-gauge fixture with independent world constraints."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import focal_constraints
from .geometry import project


def rotation() -> np.ndarray:
    axis = np.asarray((0.71, -0.46, 0.53), float)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(12.0)
    cross = np.array(((0., -axis[2], axis[1]),
                      (axis[2], 0., -axis[0]),
                      (-axis[1], axis[0], 0.)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def generate(*, constrained: bool = True) -> dict:
    """Tilt only the stored anchor; image evidence retains the original truth."""
    case = focal_constraints.generate("axis-hard-exact")
    mirror = focal_constraints.generate("mirror-hard-offcenter-exact")
    request, truth = case["request"], case["truth"]
    mirror_ids = set(mirror["truth"]["points"]) - set(truth["points"])
    for point_id in sorted(mirror_ids):
        truth["points"][point_id] = mirror["truth"]["points"][point_id]
        truth["oracle_pixels"][point_id] = mirror["truth"]["oracle_pixels"][point_id]
    request["points"].extend(p for p in mirror["request"]["points"] if p["id"] in mirror_ids)
    request["observations"].extend(o for o in mirror["request"]["observations"]
                                   if o["landmark_id"] in mirror_ids)
    request["mirror_pairs"] = deepcopy(mirror["request"]["mirror_pairs"])
    request["mirror_plane"] = deepcopy(mirror["request"]["mirror_plane"])
    truth["mirror_plane"] = deepcopy(mirror["truth"]["mirror_plane"])
    anchor = request["cameras"][0]
    anchor["rotation"] = (np.asarray(anchor["rotation"]) @ rotation().T).tolist()
    if not constrained:
        request["plane_groups"] = []
        request["mirror_pairs"] = []
        request["mirror_plane"] = None
    case["name"] = "tilted-anchor-world-priors" if constrained else "tilted-anchor-free-gauge"
    case["expectation"]["scale_gauge"] = "anchor-plane-offset" if constrained else "free"
    validate(case)
    return case


def validate(case: dict) -> None:
    """Check exact pixels and the deliberately misoriented stored anchor."""
    request, truth = case["request"], case["truth"]
    anchor = request["anchor_id"]
    true_camera = next(c for c in truth["cameras"] if c["id"] == anchor)
    stored = next(c for c in request["cameras"] if c["id"] == anchor)
    assert np.allclose(stored["center"], true_camera["center"])
    assert np.allclose(stored["rotation"], np.asarray(true_camera["rotation"]) @ rotation().T)
    assert np.linalg.norm(np.asarray(stored["rotation"]) - true_camera["rotation"]) > .2
    assert set(truth["points"]).isdisjoint(truth["holdouts"])
    for item in request["observations"]:
        camera = next(c for c in truth["cameras"] if c["id"] == item["match_id"])
        pixel, depth = project([truth["points"][item["landmark_id"]]], camera)
        assert depth[0] > 0
        assert np.allclose(pixel[0], (item["u"], item["v"]), atol=1e-10)
    if request["plane_groups"]:
        x = [truth["points"][p][0] for p, axis, _ in request["plane_groups"] if axis == "X"]
        assert np.ptp(x) < 1e-12
    if request["mirror_pairs"]:
        for a, b in request["mirror_pairs"]:
            assert np.linalg.norm(focal_constraints._reflect(
                truth["points"][a], request["mirror_plane"]) - truth["points"][b]) < 1e-10


def anchor_gauge_start(case: dict) -> tuple[dict, dict]:
    """Return exact points and cameras in the tilted anchor's private chart."""
    center = np.asarray(case["truth"]["cameras"][0]["center"])
    turn = rotation()
    points = {key: (center + turn @ (np.asarray(value) - center)).tolist()
              for key, value in case["truth"]["points"].items()}
    cameras = {}
    for camera in case["truth"]["cameras"]:
        copy = deepcopy(camera)
        copy["center"] = (center + turn @ (np.asarray(camera["center"]) - center)).tolist()
        copy["rotation"] = (np.asarray(camera["rotation"]) @ turn.T).tolist()
        cameras[camera["id"]] = copy
        for key, point in points.items():
            expected = case["truth"]["oracle_pixels"][key][camera["id"]]
            assert np.allclose(project([point], copy)[0][0], expected, atol=1e-9)
    return points, cameras
