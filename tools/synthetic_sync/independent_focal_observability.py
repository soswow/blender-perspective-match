"""Read-only image-homography and fitted-depth checks of noisy focal candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.synthetic_sync.independent_focal import CASES
from tools.synthetic_sync.independent_focal_noise import CORPUS


def normalized(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    distance = np.linalg.norm(points-center, axis=1).mean()
    scale = np.sqrt(2.) / distance
    transform = np.array(((scale, 0., -scale*center[0]),
                          (0., scale, -scale*center[1]), (0., 0., 1.)))
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :2], transform


def homography_transfer_rmse(first: np.ndarray, second: np.ndarray) -> float:
    """Fit a whole-pair homography, then score both transfer directions."""
    a, ta = normalized(first)
    b, tb = normalized(second)
    design = []
    for (x, y), (u, v) in zip(a, b):
        design.append((-x, -y, -1., 0., 0., 0., u*x, u*y, u))
        design.append((0., 0., 0., -x, -y, -1., v*x, v*y, v))
    _, _, vh = np.linalg.svd(np.asarray(design), full_matrices=False)
    h = np.linalg.inv(tb) @ vh[-1].reshape(3, 3) @ ta
    def transfer(matrix, points):
        mapped = (matrix @ np.column_stack((points, np.ones(len(points)))).T).T
        return mapped[:, :2] / mapped[:, 2, None]
    residuals = np.concatenate((transfer(h, first)-second,
                                transfer(np.linalg.inv(h), second)-first))
    return float(np.sqrt(np.mean(np.sum(residuals*residuals, axis=1))))


def inspect(request: dict, record: dict) -> dict:
    picks = {}
    for observation in request["observations"]:
        picks.setdefault(observation["landmark_id"], {})[observation["match_id"]] = (
            observation["u"], observation["v"])
    ids = [camera["id"] for camera in request["cameras"]]
    pairs = {}
    for i, first in enumerate(ids):
        for second in ids[i+1:]:
            common = [items for items in picks.values() if first in items and second in items]
            pairs[f"{first}/{second}"] = dict(count=len(common),
                homography_symmetric_transfer_rmse_px=homography_transfer_rmse(
                    np.asarray([items[first] for items in common]),
                    np.asarray([items[second] for items in common])))
    cameras = [record["cameras"][key] for key in ids]
    depths = []
    angles = []
    for point in record["landmarks"].values():
        point = np.asarray(point)
        rays = []
        for camera in cameras:
            offset = point-np.asarray(camera["center"])
            depths.append(float((np.asarray(camera["rotation"]) @ offset)[2]))
            rays.append(offset / np.linalg.norm(offset))
        angles.append(max(float(np.degrees(np.arccos(np.clip(a @ b, -1., 1.))))
                          for i, a in enumerate(rays) for b in rays[i+1:]))
    return dict(pairs=pairs, all_fitted_points_in_front=all(d > 0 for d in depths),
                min_fitted_depth_baselines=float(min(depths)),
                median_max_pair_parallax_deg=float(np.median(angles)))


def main() -> None:
    output = CORPUS / "observability.json"
    if output.exists():
        raise FileExistsError("Read-only observability report cannot be silently rerun")
    rows = []
    for name in CASES:
        request = json.loads((CORPUS / f"{name}.json").read_text())["request"]
        dense = json.loads((CORPUS / "dense" /
            f"{name}-dense-exact-start-1.json").read_text())
        rows.append(dict(name=name, request_sha256=dense["request_sha256"],
                         **inspect(request, dense["record"])))
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(dict(rows=rows, source_sha256=source),
        indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
