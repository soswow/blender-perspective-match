"""Seeded, JSON-serializable Sync evidence and separately retained truth."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .geometry import look_at, object_mesh, project, surface_points, visible


SCHEMA_VERSION = 1
FAMILIES = (
    "ground", "known_3d", "free_scale", "mixed_lines", "partial_symmetry",
    "locked_bridge", "overhead", "disconnected", "collinear",
)


def generate(family: str = "ground", seed: int = 0, noise_px: float = 0.0) -> dict:
    """Create evidence without calling the solver or its projection helpers."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; choose from {FAMILIES}")
    if not np.isfinite(noise_px) or noise_px < 0:
        raise ValueError("noise_px must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    mesh = object_mesh()
    centers = [[-4., -6., 3.8], [4., -6., 3.1], [5., 1., 4.2]]
    if family == "overhead":
        centers[2] = [0.2, -0.1, 6.0]
    if family == "collinear":
        centers = centers[:2]
    if family == "disconnected":
        centers.append([-5., 2., 3.])
    truth_cameras, stored_cameras, fixed = [], [], {}
    for index, center in enumerate(centers):
        center = np.asarray(center) + rng.uniform(-0.08, 0.08, 3)
        focal = float(740 + index * 105 + rng.uniform(-20, 20))
        camera = dict(id=f"view_{index}", width=960, height=720, fx=focal, fy=focal,
                      cx=473.0 + index * 3, cy=353.0 - index * 2,
                      center=center.tolist(), rotation=look_at(center, [0, 0, 0.7]))
        truth_cameras.append(camera)
        stored = deepcopy(camera)
        if index:
            # Deliberately unrelated private pose; truth is not a warm start.
            private_center = np.array([-2.0, -4.0, 2.2])
            stored.update(center=private_center.tolist(), rotation=look_at(private_center, [0.3, 0.1, 0.4]))
        stored_cameras.append(stored)
        if family == "locked_bridge" and index == 1:
            rotation = np.asarray(camera["rotation"]).T @ np.asarray(stored["rotation"])
            fixed[camera["id"]] = dict(scale=1.0, rotation=rotation.tolist(),
                translation=(center - rotation @ stored["center"]).tolist())

    training = surface_points(mesh, [(0.22,0.25), (0.73,0.28), (0.28,0.71), (0.69,0.74)])
    points = [{"id": f"point_{i:02d}", "position": p, "ground": False, "known": False}
              for i, p in enumerate(training)]
    has_ground = family not in {"known_3d", "free_scale", "collinear"}
    if has_ground:
        points += [dict(id=f"ground_{i}", position=p, ground=True, known=False)
                   for i, p in enumerate([[-2,-2,0], [0,-2.2,0], [2,-2,0], [2.3,0,0], [2,2,0], [-2,2,0]])]
    if family == "collinear":
        points = [dict(id=f"point_{i}", position=[-1+i/3, -1.2, 0.6], ground=False, known=False) for i in range(7)]
    mirrors = []
    if family == "partial_symmetry":
        for i, (y, z) in enumerate([(-0.9, 0.3), (-0.9, 1.1), (0.1, 1.4)]):
            ids = [f"mirror_{i}_{side}" for side in ("left", "right")]
            for item_id, x in zip(ids, [-1.0, 1.0]):
                points.append(dict(id=item_id, position=[x,y,z], ground=False, known=False))
            mirrors.append(ids)

    observations = []
    for point in points:
        for index, camera in enumerate(truth_cameras):
            if family == "disconnected" and index == 3:
                continue
            if not visible(point["position"], camera, mesh):
                continue
            # A controlled missing-pick pattern, independently of occlusion.
            if family == "locked_bridge" and index == 0 and int(point["id"].split("_")[-1]) % 2:
                continue
            uv = project([point["position"]], camera)[0][0]
            uv = uv + rng.normal(0, noise_px, 2)
            observations.append(dict(match_id=camera["id"], landmark_id=point["id"], u=float(uv[0]), v=float(uv[1]), weight=1.0))
    observed = {item["landmark_id"] for item in observations}
    points = [point for point in points if point["id"] in observed]
    if family == "known_3d":
        for point in points:
            point["known"] = True

    lines, line_observations, parallel = [], [], []
    if family == "mixed_lines":
        for i, x in enumerate([-1.2, 1.2]):
            line = dict(id=f"edge_{i}", ends=[[x,-0.9,0.1], [x,-0.9,1.3]], known=(i == 0))
            lines.append(line)
            parallel.append([line["id"], "WORLD_AXIS_Z"])
            for index, camera in enumerate(truth_cameras):
                a, b = np.asarray(line["ends"])
                # Different extents in each view, on the same infinite edge.
                ends = [a + (0.05 * index) * (b-a), b - (0.08 * index) * (b-a)]
                if not all(visible(p, camera, mesh) for p in ends):
                    continue
                uv = project(ends, camera)[0] + rng.normal(0, noise_px, (2,2))
                line_observations.append(dict(match_id=camera["id"], landmark_id=line["id"],
                    u1=float(uv[0,0]), v1=float(uv[0,1]), u2=float(uv[1,0]), v2=float(uv[1,1]), weight=1.0))

    checks = []
    candidates = mesh["vertices"] + surface_points(mesh, [(0.13,0.51), (0.52,0.87), (0.86,0.46)])
    for i, position in enumerate(candidates):
        views = [c["id"] for c in truth_cameras if visible(position, c, mesh)]
        if views:
            checks.append(dict(id=f"check_{i:03d}", position=position, views=views))
    request = dict(cameras=stored_cameras,
        points=[dict(id=p["id"], ground=p["ground"], known=p["position"] if p["known"] else None) for p in points],
        observations=observations, lines=[dict(id=line["id"], known=line["ends"] if line["known"] else None) for line in lines],
        line_observations=line_observations, anchor_id="view_0", fixed_similarities=fixed,
        lock_rotation=False, lock_translation=False, ground_slack=0.0, known_3d_slack=0.0,
        mirror_pairs=mirrors, mirror_plane=[[0,0,0],[1,0,0]] if mirrors else None,
        mirror_slack=0.0, parallel_pairs=parallel, plane_groups=[], plane_slack=0.0)
    return dict(schema_version=SCHEMA_VERSION, name=f"{family}-{seed}", family=family,
        seed=seed, noise_px=noise_px, request=request,
        expectation=dict(outcome="reject" if family == "collinear" else "solve",
            cameras=[c["id"] for c in truth_cameras if not (family == "disconnected" and c["id"] == "view_3")],
            excluded_cameras=["view_3"] if family == "disconnected" else [],
            gauge="similarity" if family == "free_scale" else "anchor",
            holdout_rmse_px=max(1.0, noise_px * 6), rotation_deg=1.0, center_fraction=0.02),
        truth=dict(cameras=truth_cameras, points={p["id"]:p["position"] for p in points},
                   lines={line["id"]:line["ends"] for line in lines}, mesh=mesh, checks=checks))


def write_case(case: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_case(path: Path) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema: {case.get('schema_version')}")
    return case
