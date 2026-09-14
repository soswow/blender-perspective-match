"""Point and stroke projection shared by joint focal fitting."""

from __future__ import annotations

import numpy as np

from . import geometry
from .sync.projection import _project_world_line_to_image
from .sync.types import SimilarityTransform


MIN_DEPTH = 1.0e-6
DEPTH_PENALTY_PX = 1.0e3
INVALID_LINE_RESIDUAL_PX = 1.0e6
DISTORTED_LINE_ANGULAR_HALF_SPAN = 0.05


def stroke_line_witness(point: np.ndarray, direction: np.ndarray,
                        rotation_w2c: np.ndarray, camera_center: np.ndarray,
                        intrinsics: geometry.CameraIntrinsics, endpoints: np.ndarray,
                        *, division_lambda: float = 0.0,
                        brown_conrady: tuple[float, ...] = ()) -> np.ndarray:
    """Choose a 3D line witness by the stroke's midpoint ray, independent of endpoints in 3D."""
    point = np.asarray(point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    center = np.asarray(camera_center, dtype=np.float64)
    image_midpoint = np.mean(np.asarray(endpoints, dtype=np.float64).reshape(2, 2), axis=0)
    ideal = geometry.undistort_points(
        image_midpoint.reshape(1, 2), intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy, division_lambda, brown_conrady)[0]
    ray_camera = np.array(((ideal[0] - intrinsics.cx) / intrinsics.fx,
                           (ideal[1] - intrinsics.cy) / intrinsics.fy, 1.0))
    ray = np.asarray(rotation_w2c, dtype=np.float64).T @ ray_camera
    ray /= np.linalg.norm(ray)
    from_camera = point - center
    cosine = float(unit @ ray)
    denominator = 1.0 - cosine * cosine
    if denominator < 1.0e-8:
        along = -float(unit @ from_camera)
    else:
        along = (cosine * float(ray @ from_camera) -
                 float(unit @ from_camera)) / denominator
    return point + along * unit


def project_stroke_line(point: np.ndarray, direction: np.ndarray,
                        rotation_w2c: np.ndarray, camera_center: np.ndarray,
                        intrinsics: geometry.CameraIntrinsics, endpoints: np.ndarray,
                        *, division_lambda: float = 0.0,
                        brown_conrady: tuple[float, ...] = ()) -> np.ndarray | None:
    """Project a stroke-local line with sample rays invariant to world scale and line slide."""
    calibration = geometry.Calibration(
        intrinsics, np.asarray(rotation_w2c, dtype=np.float64),
        np.asarray(camera_center, dtype=np.float64),
        division_lambda=division_lambda, brown_conrady=brown_conrady)
    if not calibration.has_distortion:
        return _project_world_line_to_image(
            np.asarray(point, dtype=np.float64), np.asarray(direction, dtype=np.float64),
            calibration, SimilarityTransform())
    witness = stroke_line_witness(
        point, direction, rotation_w2c, camera_center, intrinsics, endpoints,
        division_lambda=division_lambda, brown_conrady=brown_conrady)
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm <= 1.0e-12:
        return None
    unit /= norm
    center = np.asarray(camera_center, dtype=np.float64)
    perpendicular = witness - center
    perpendicular -= float(perpendicular @ unit) * unit
    radius = float(np.linalg.norm(perpendicular))
    if radius <= 1.0e-10:
        return None
    span = DISTORTED_LINE_ANGULAR_HALF_SPAN * radius
    samples = np.stack((witness - span * unit, witness + span * unit))
    q = (np.asarray(rotation_w2c, dtype=np.float64) @ (samples - center).T).T
    if np.any(q[:, 2] <= MIN_DEPTH):
        return None
    pixels, _, _ = project_camera_points(
        q, intrinsics, division_lambda=division_lambda,
        brown_conrady=brown_conrady, jacobian=False)
    line = np.cross(np.append(pixels[0], 1.0), np.append(pixels[1], 1.0))
    magnitude = float(np.linalg.norm(line[:2]))
    if magnitude <= 1.0e-12:
        return None
    return line / magnitude


def project_camera_points(
    camera_points: np.ndarray,
    intrinsics: geometry.CameraIntrinsics,
    *,
    division_lambda: float = 0.0,
    brown_conrady: tuple[float, ...] = (),
    jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Project camera-space points; return pixels, d(pixel)/d(point), d(pixel)/d(log focal scale).

    The focal derivative scales fx and fy together. The returned pixel values
    include the joint fitter's clipped-depth penalty; depth screening remains
    the caller's responsibility.
    """
    q = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
    depth = q[:, 2]
    safe_depth = np.maximum(depth, MIN_DEPTH)
    focal = np.array((intrinsics.fx, intrinsics.fy), dtype=np.float64)
    principal = np.array((intrinsics.cx, intrinsics.cy), dtype=np.float64)
    ideal = q[:, :2] / safe_depth[:, None] * focal + principal
    distorted = geometry.distort_points(
        ideal, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
        division_lambda, brown_conrady,
    )
    pixels = distorted + np.maximum(MIN_DEPTH - depth, 0.0)[:, None] * DEPTH_PENALTY_PX
    if not jacobian:
        return pixels, None, None

    # Distortion is a function of normalized camera coordinates. Scaling both
    # focals multiplies the offset from principal even for division/Brown D.
    log_focal_jacobian = distorted - principal
    map_jacobian = np.broadcast_to(np.eye(2), (len(q), 2, 2)).copy()
    if geometry.has_lens_distortion(division_lambda, brown_conrady, threshold=1.0e-15):
        # Four vectorized distortion calls avoid a Python loop over picks.
        # Centered differences also handle Brown's tangential/rational terms.
        for axis in range(2):
            step = 1.0e-5 * np.maximum(1.0, np.abs(ideal[:, axis]))
            delta = np.zeros_like(ideal)
            delta[:, axis] = step
            above = geometry.distort_points(
                ideal + delta, intrinsics.fx, intrinsics.fy, intrinsics.cx,
                intrinsics.cy, division_lambda, brown_conrady,
            )
            below = geometry.distort_points(
                ideal - delta, intrinsics.fx, intrinsics.fy, intrinsics.cx,
                intrinsics.cy, division_lambda, brown_conrady,
            )
            map_jacobian[:, :, axis] = (above - below) / (2.0 * step[:, None])

    ideal_jacobian = np.zeros((len(q), 2, 3), dtype=np.float64)
    ideal_jacobian[:, 0, 0] = intrinsics.fx / safe_depth
    ideal_jacobian[:, 1, 1] = intrinsics.fy / safe_depth
    in_front = depth > MIN_DEPTH
    ideal_jacobian[in_front, 0, 2] = (
        -intrinsics.fx * q[in_front, 0] / depth[in_front] ** 2
    )
    ideal_jacobian[in_front, 1, 2] = (
        -intrinsics.fy * q[in_front, 1] / depth[in_front] ** 2
    )
    point_jacobian = np.einsum("nij,njk->nik", map_jacobian, ideal_jacobian)
    point_jacobian[~in_front, :, 2] = -DEPTH_PENALTY_PX
    return pixels, point_jacobian, log_focal_jacobian


def line_endpoint_distances(
    point: np.ndarray,
    direction: np.ndarray,
    rotation_w2c: np.ndarray,
    camera_center: np.ndarray,
    intrinsics: geometry.CameraIntrinsics,
    endpoints: np.ndarray,
    *,
    division_lambda: float = 0.0,
    brown_conrady: tuple[float, ...] = (),
) -> np.ndarray:
    """Signed endpoint distances to Sync's projected infinite 3D line."""
    line = project_stroke_line(
        point, direction, rotation_w2c, camera_center, intrinsics, endpoints,
        division_lambda=division_lambda, brown_conrady=brown_conrady)
    if line is None:
        return np.full(2, INVALID_LINE_RESIDUAL_PX)
    uv = np.asarray(endpoints, dtype=np.float64).reshape(2, 2)
    return uv @ line[:2] + line[2]
