"""Independent camera oracle for line plane and parallel FOV relations."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import focal_constraints
from .geometry import project


def generate(*, inconsistent: str | None = None, relations: bool = True,
             axis_parallel: bool = False) -> dict:
    """Build point-supported planes and parallel free lines from withheld 3D truth."""
    case = deepcopy(focal_constraints.generate("free-hard-exact"))
    request, truth = case["request"], case["truth"]
    axis_points = focal_constraints._axis_points(warped=False)
    for key, position in axis_points.items():
        truth["points"][key] = position
        request["points"].append(dict(id=key, ground=False, known=None))
        truth["oracle_pixels"][key] = {}
        for camera in truth["cameras"]:
            uv, depth = project([position], camera)
            assert depth[0] > 0
            truth["oracle_pixels"][key][camera["id"]] = uv[0].tolist()
            request["observations"].append(dict(match_id=camera["id"], landmark_id=key,
                                                 u=float(uv[0, 0]), v=float(uv[0, 1]), weight=1.0))
        request["plane_groups"].append([key, "X", 2])
    # Both free-plane members have direction (0.8, 0.35, 0.1325), exactly
    # tangent to z = 0.94 + 0.24*x - 0.17*y. A separate axis line lies at X=.38.
    lines = {
        "axis_edge": ([[.38, -.20, .45], [.38, -.20, 1.90]] if axis_parallel else
                      [[.38, -.56, .56], [.38, .49, 1.71]]),
        "free_edge_a": [[-.65, -.42, .8554], [.15, -.07, .9879]],
        "free_edge_b": [[-.28, .40, .8048], [.52, .75, .9373]],
    }
    intervals = ((.10, .87), (.78, .12), (.22, .94), (.89, .06))
    truth["lines"] = lines
    truth["line_oracle_pixels"] = {}
    request["lines"] = [dict(id=key, known=None) for key in lines]
    for key, ends in lines.items():
        a, b = np.asarray(ends, float)
        truth["line_oracle_pixels"][key] = {}
        for index, camera in enumerate(truth["cameras"]):
            t1, t2 = intervals[index]
            uv, depth = project([a + t1 * (b-a), a + t2 * (b-a)], camera)
            assert min(depth) > 0 and np.all(uv > 10)
            assert np.all(uv[:, 0] < camera["width"] - 10)
            assert np.all(uv[:, 1] < camera["height"] - 10)
            truth["line_oracle_pixels"][key][camera["id"]] = uv.tolist()
            if inconsistent == "parallel" and key == "free_edge_b" and index == 2:
                direction = uv[1] - uv[0]
                uv += 65 * np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            if inconsistent == "plane" and key == "axis_edge" and index == 2:
                direction = uv[1] - uv[0]
                uv += 65 * np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            request["line_observations"].append(dict(
                match_id=camera["id"], landmark_id=key,
                u1=float(uv[0, 0]), v1=float(uv[0, 1]),
                u2=float(uv[1, 0]), v2=float(uv[1, 1]), weight=1.0))
    if relations:
        request["plane_groups"] += [["axis_edge", "X", 2],
                                     ["free_edge_a", "FREE", 1],
                                     ["free_edge_b", "FREE", 1]]
        request["parallel_pairs"] = [["free_edge_a", "free_edge_b"]]
        if axis_parallel:
            request["parallel_pairs"].append(["axis_edge", "WORLD_AXIS_Z"])
    case["name"] = "focal-line-relations"
    case["family"] = "focal_line_relations"
    return case
