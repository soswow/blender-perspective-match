"""Independent pinhole/mesh reference. Never import Perspective Match here."""

from __future__ import annotations

import numpy as np


def look_at(center, target) -> list:
    """World-to-camera basis, right/down/forward; world Z is up."""
    forward = np.asarray(target, dtype=float) - center
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(forward, right), forward]).tolist()


def project(points, camera) -> tuple[np.ndarray, np.ndarray]:
    """Project to continuous top-left source pixels; return pixels and signed depth."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    local = (points - camera["center"]) @ np.asarray(camera["rotation"]).T
    depth = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = local[:, :2] / depth[:, None]
        uv = uv * [camera["fx"], camera["fy"]] + [camera["cx"], camera["cy"]]
    return uv, depth


def visible(point, camera, mesh) -> bool:
    """Frustum plus ray/triangle occlusion, with the target surface excluded."""
    uv, depth = project([point], camera)
    if depth[0] <= 0 or not (0 < uv[0, 0] < camera["width"] and 0 < uv[0, 1] < camera["height"]):
        return False
    origin = np.asarray(camera["center"], dtype=float)
    ray = np.asarray(point, dtype=float) - origin
    vertices = np.asarray(mesh["vertices"])
    for face in mesh["faces"]:
        for i in range(1, len(face) - 1):
            a, b, c = vertices[[face[0], face[i], face[i + 1]]]
            edge1, edge2 = b - a, c - a
            cross = np.cross(ray, edge2)
            determinant = edge1 @ cross
            if abs(determinant) < 1e-10:
                continue
            offset = origin - a
            u = (offset @ cross) / determinant
            q = np.cross(offset, edge1)
            v = (ray @ q) / determinant
            t = (edge2 @ q) / determinant
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9 and 1e-7 < t < 1 - 1e-7:
                return False
    return True


def object_mesh() -> dict:
    """A symmetric box with an asymmetric raised detail; units are arbitrary."""
    vertices, faces = [], []
    for low, high in [((-1.2, -0.9, 0.0), (1.2, 0.9, 1.4)), ((0.15, -0.25, 1.4), (0.75, 0.4, 2.0))]:
        x, y, z = low
        X, Y, Z = high
        offset = len(vertices)
        vertices.extend([(x,y,z), (X,y,z), (X,Y,z), (x,Y,z), (x,y,Z), (X,y,Z), (X,Y,Z), (x,Y,Z)])
        faces.extend([[offset + i for i in face] for face in [(0,3,2,1), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7), (4,5,6,7)]])
    return {"vertices": vertices, "faces": faces}


def surface_points(mesh, fractions) -> list[list]:
    """Sample quad interiors; the train/check callers supply disjoint fractions."""
    vertices = np.asarray(mesh["vertices"])
    return [
        ((1-u)*(1-v)*vertices[a] + u*(1-v)*vertices[b] + u*v*vertices[c] + (1-u)*v*vertices[d]).tolist()
        for a, b, c, d in mesh["faces"]
        for u, v in fractions
    ]
