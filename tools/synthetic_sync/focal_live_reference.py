"""Generic live-landmark mirror case using an independent pixel oracle."""

from __future__ import annotations

import numpy as np

from . import focal_constraints
from .geometry import project


def generate() -> dict:
    """Add a fully picked on-plane reference to the exact point-mirror scaffold."""
    case = focal_constraints.generate("mirror-hard-offcenter-exact")
    request = case["request"]
    first_pair = request["mirror_pairs"][0]
    reference_id = "reference"
    reference = (np.asarray(case["truth"]["points"][first_pair[0]]) +
                 np.asarray(case["truth"]["points"][first_pair[1]])) / 2
    case["truth"]["points"][reference_id] = reference.tolist()
    case["truth"]["oracle_pixels"][reference_id] = {}
    for camera in case["truth"]["cameras"]:
        uv = project([reference], camera)[0][0]
        case["truth"]["oracle_pixels"][reference_id][camera["id"]] = uv.tolist()
        request["observations"].append(dict(
            match_id=camera["id"], landmark_id=reference_id,
            u=float(uv[0]), v=float(uv[1]), weight=1.0))
    request["points"].append(dict(id=reference_id, ground=False, known=None))
    request["mirror_landmark_id"] = reference_id
    # The location is deliberately unrelated to the plane; the live point
    # supplies that part of the constraint, while this pair supplies its normal.
    request["mirror_plane"][0] = [12.0, -8.0, 5.0]
    case["name"] = "mirror-live-reference"
    case["expectation"]["scale_gauge"] = "free"
    return case
