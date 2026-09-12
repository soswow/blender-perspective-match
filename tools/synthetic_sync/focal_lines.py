"""Independent point and line focal scaffold with withheld pixel truth."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import focal_live_reference
from .geometry import project


def generate(*, inconsistent: bool = False, weak: bool = False) -> dict:
    """Supply varied line strokes without supplying their 3D endpoints to the fit."""
    case = deepcopy(focal_live_reference.generate())
    request, truth = case["request"], case["truth"]
    reference = np.asarray(truth["points"][request["mirror_landmark_id"]], float)
    normal = np.asarray(request["mirror_plane"][1], float)
    tangent = np.cross(normal, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    other = np.cross(normal, tangent)
    center = reference + 0.20 * tangent + 0.16 * other
    start = center + 0.45 * normal - 0.51 * other
    end = center + 0.45 * normal + 0.51 * other
    reflected = lambda point: point - 2 * normal * float(normal @ (point - reference))
    lines = {"free_edge": [
        (reference - 0.53 * tangent - 0.43 * other).tolist(),
        (reference - 0.53 * tangent + 0.43 * other).tolist()],
        "mirror_edge_a": [start.tolist(), end.tolist()],
        "mirror_edge_b": [reflected(start).tolist(), reflected(end).tolist()]}
    # Each camera marks a different physical interval; odd cameras reverse the
    # stroke direction. These pixels constrain infinite lines, not endpoints.
    intervals = ((0.10, 0.83), (0.76, 0.18), (0.25, 0.94), (0.88, 0.05))
    observations = []
    oracle = {}
    for line_id, endpoints in lines.items():
        oracle[line_id] = {}
        a, b = np.asarray(endpoints, float)
        for camera_index, camera in enumerate(truth["cameras"]):
            if weak and camera_index > 0 and line_id != "free_edge":
                continue
            t1, t2 = intervals[camera_index]
            xyz = [a + t1 * (b - a), a + t2 * (b - a)]
            uv, depth = project(xyz, camera)
            if np.min(depth) <= 0 or np.any(uv < 10) or np.any(uv[:, 0] > camera["width"] - 10) or np.any(uv[:, 1] > camera["height"] - 10):
                raise ValueError(f"Line {line_id} leaves {camera['id']} image")
            oracle[line_id][camera["id"]] = uv.tolist()
            if inconsistent and line_id == "mirror_edge_b" and camera_index == 2:
                direction = uv[1] - uv[0]
                perpendicular = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                uv = uv + 75.0 * perpendicular
            observations.append(dict(match_id=camera["id"], landmark_id=line_id,
                                     u1=float(uv[0, 0]), v1=float(uv[0, 1]),
                                     u2=float(uv[1, 0]), v2=float(uv[1, 1]), weight=1.0))
    request["lines"] = [dict(id=key, known=None) for key in lines]
    request["line_observations"] = observations
    request["mirror_pairs"].append(["mirror_edge_a", "mirror_edge_b"])
    truth["lines"] = lines
    truth["line_oracle_pixels"] = oracle
    case["name"] = "focal-lines-inconsistent" if inconsistent else "focal-lines-weak" if weak else "focal-lines"
    return case
