"""Known off-center image crop of the independent live-reference focal oracle.

The crop changes pixels by (u, v) -> (u-left, v-top), image dimensions by
(left+right, top+bottom), and the principal point by the same translation.
All cameras retain the exact same physical poses, focal lengths and 3D scene.
No user imagery or private scene data is used.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import focal_live_reference
from .geometry import project

CROP = (400, 80, 200, 80)  # left, right, top, bottom, source pixels


def generate(*, centered_principal_point: bool = False) -> dict:
    """Return a cropped oracle, optionally with deliberately wrong centered K."""
    case = deepcopy(focal_live_reference.generate())
    left, right, top, bottom = CROP
    request, truth = case["request"], case["truth"]
    for camera in truth["cameras"]:
        camera["width"] -= left + right
        camera["height"] -= top + bottom
        camera["cx"] -= left
        camera["cy"] -= top
    for camera in request["cameras"]:
        camera["width"] -= left + right
        camera["height"] -= top + bottom
        camera["cx"] -= left
        camera["cy"] -= top
        if centered_principal_point:
            camera["cx"] = camera["width"] / 2
            camera["cy"] = camera["height"] / 2
    for observation in request["observations"]:
        observation["u"] -= left
        observation["v"] -= top
    for field in ("oracle_pixels", "holdout_pixels"):
        for views in truth[field].values():
            for uv in views.values():
                uv[0] -= left
                uv[1] -= top
    for camera in truth["cameras"]:
        image_id = camera["id"]
        for point_id, point in {**truth["points"], **truth["holdouts"]}.items():
            uv, depth = project([point], camera)
            assert depth[0] > 0
            assert 0 < uv[0, 0] < camera["width"] and 0 < uv[0, 1] < camera["height"], (image_id, point_id)
            expected = truth["oracle_pixels" if point_id in truth["points"] else "holdout_pixels"][point_id][image_id]
            np.testing.assert_allclose(uv[0], expected, atol=1e-9)
    case["name"] = "known-crop-centered-input" if centered_principal_point else "known-crop-shifted-input"
    case["crop"] = dict(left=left, right=right, top=top, bottom=bottom,
                        input_principal_point="centered" if centered_principal_point else "shifted")
    return case
