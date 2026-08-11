"""Perspective-matching geometry shared by Blender operators and overlays.

The module intentionally depends only on NumPy, which Blender ships. Image
coordinates use a top-left origin. Axis ids retain the desktop application's
mapping: red ``x`` -> Blender X, yellow ``z`` -> Blender Y, blue ``y`` ->
Blender Z (up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AxisId = Literal["x", "y", "z"]
SurfacePlane = Literal["xz", "yz", "yx"]


@dataclass
class LineSegment:
    """One user-drawn segment in source-image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass
class Calibration:
    """Solved camera state in OpenCV world-to-camera convention."""

    intrinsics: CameraIntrinsics
    rotation_w2c: np.ndarray
    camera_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    division_lambda: float = 0.0
    lambda_saturated: bool = False

    @property
    def hfov_degrees(self) -> float:
        return hfov_from_focal(self.intrinsics.fx, self.intrinsics.image_width)

    @property
    def vfov_degrees(self) -> float:
        return vfov_from_focal(self.intrinsics.fy, self.intrinsics.image_height)


def focal_from_hfov(hfov_degrees: float, image_width: int) -> float:
    """Convert horizontal field of view to focal length in pixels."""
    angle = np.radians(max(1.0e-4, min(179.0, hfov_degrees)))
    return float(image_width / (2.0 * np.tan(angle * 0.5)))


def hfov_from_focal(focal_pixels: float, image_width: int) -> float:
    """Convert focal length in pixels to horizontal field of view."""
    return float(np.degrees(2.0 * np.arctan(image_width / (2.0 * max(focal_pixels, 1.0e-9)))))


def vfov_from_focal(focal_pixels: float, image_height: int) -> float:
    """Convert focal length in pixels to vertical field of view."""
    return float(np.degrees(2.0 * np.arctan(image_height / (2.0 * max(focal_pixels, 1.0e-9)))))


def segment_length(segment: LineSegment) -> float:
    """Return the segment length in image pixels."""
    return float(np.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1))


def _line_homogeneous(segment: LineSegment) -> np.ndarray:
    point_a = np.array([segment.x1, segment.y1, 1.0], dtype=np.float64)
    point_b = np.array([segment.x2, segment.y2, 1.0], dtype=np.float64)
    line = np.cross(point_a, point_b)
    normal_length = float(np.linalg.norm(line[:2]))
    if normal_length > 1.0e-12:
        line /= normal_length
    return line


def vanishing_point_from_lines(segments: list[LineSegment]) -> np.ndarray | None:
    """Fit a robust, length-weighted vanishing point with Huber IRLS."""
    if len(segments) < 2:
        return None

    lines = np.stack([_line_homogeneous(segment) for segment in segments], axis=0)
    lengths = np.array([max(segment_length(segment), 1.0) for segment in segments])
    base_weights = lengths / float(np.mean(lengths))
    weights = base_weights.copy()

    def solve(current_weights: np.ndarray) -> np.ndarray | None:
        weighted = lines * np.sqrt(current_weights)[:, None]
        if float(np.linalg.norm(weighted)) < 1.0e-12:
            return None
        _u_matrix, _singular_values, v_transpose = np.linalg.svd(weighted)
        return v_transpose[-1]

    vanishing_h = solve(weights)
    if vanishing_h is None:
        return None

    for _iteration in range(8):
        if abs(float(vanishing_h[2])) < 1.0e-10:
            break
        point = np.array(
            [vanishing_h[0] / vanishing_h[2], vanishing_h[1] / vanishing_h[2], 1.0],
            dtype=np.float64,
        )
        residuals = np.abs(lines @ point)
        median = float(np.median(residuals)) if len(residuals) else 1.0
        sigma = max(median * 1.4826, 1.5)
        normalized = residuals / (1.345 * sigma)
        huber_weights = np.where(
            normalized <= 1.0,
            1.0,
            1.0 / np.maximum(normalized, 1.0e-6),
        )
        next_vanishing = solve(base_weights * huber_weights)
        if next_vanishing is None:
            break
        if abs(float(next_vanishing[2])) > 1.0e-10:
            previous_xy = vanishing_h[:2] / vanishing_h[2]
            next_xy = next_vanishing[:2] / next_vanishing[2]
            vanishing_h = next_vanishing
            if float(np.linalg.norm(previous_xy - next_xy)) < 0.05:
                break
        else:
            vanishing_h = next_vanishing

    if abs(float(vanishing_h[2])) < 1.0e-10:
        direction = vanishing_h[:2]
        direction_length = float(np.linalg.norm(direction))
        if direction_length < 1.0e-12:
            return None
        return np.array([direction[0] / direction_length, direction[1] / direction_length, 0.0])
    return vanishing_h / vanishing_h[2]


def undistort_points(
    points_xy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    division_lambda: float,
) -> np.ndarray:
    """Map observed pixels to ideal pixels using the Fitzgibbon division model."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if abs(division_lambda) < 1.0e-15:
        return points.copy()
    normalized_x = (points[:, 0] - cx) / fx
    normalized_y = (points[:, 1] - cy) / fy
    radius_squared = normalized_x * normalized_x + normalized_y * normalized_y
    denominator = 1.0 + division_lambda * radius_squared
    denominator = np.where(
        np.abs(denominator) < 1.0e-12,
        np.where(denominator < 0.0, -1.0e-12, 1.0e-12),
        denominator,
    )
    return np.column_stack(
        [normalized_x / denominator * fx + cx, normalized_y / denominator * fy + cy]
    )


def distort_points(
    points_xy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    division_lambda: float,
) -> np.ndarray:
    """Map ideal pixels back to observed pixels."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if abs(division_lambda) < 1.0e-15:
        return points.copy()
    ideal_x = (points[:, 0] - cx) / fx
    ideal_y = (points[:, 1] - cy) / fy
    ideal_radius_squared = ideal_x * ideal_x + ideal_y * ideal_y
    ideal_radius = np.sqrt(ideal_radius_squared)
    observed_radius = ideal_radius.copy()
    valid = ideal_radius > 1.0e-12
    if np.any(valid):
        discriminant = np.clip(
            1.0 - 4.0 * division_lambda * ideal_radius_squared[valid],
            0.0,
            None,
        )
        denominator = 2.0 * division_lambda * ideal_radius[valid]
        quadratic = np.abs(denominator) > 1.0e-12
        values = ideal_radius[valid].copy()
        values[quadratic] = (
            1.0 - np.sqrt(discriminant[quadratic])
        ) / denominator[quadratic]
        observed_radius[valid] = np.abs(values)
    scale = np.ones_like(ideal_radius)
    scale[valid] = observed_radius[valid] / ideal_radius[valid]
    return np.column_stack([ideal_x * scale * fx + cx, ideal_y * scale * fy + cy])


def undistort_line_bundles(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    division_lambda: float,
) -> dict[AxisId, list[LineSegment]]:
    output: dict[AxisId, list[LineSegment]] = {"x": [], "y": [], "z": []}
    for axis, segments in line_bundles.items():
        for segment in segments:
            mapped = undistort_points(
                np.array([[segment.x1, segment.y1], [segment.x2, segment.y2]]),
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                division_lambda,
            )
            output[axis].append(
                LineSegment(mapped[0, 0], mapped[0, 1], mapped[1, 0], mapped[1, 1])
            )
    return output


def _concurrency_cost(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    division_lambda: float,
) -> float:
    mapped = undistort_line_bundles(line_bundles, intrinsics, division_lambda)
    total = 5.0 * division_lambda * division_lambda
    usable = False
    for segments in mapped.values():
        vanishing = vanishing_point_from_lines(segments)
        if vanishing is None:
            continue
        usable = True
        point = (
            np.array([vanishing[0], vanishing[1], 0.0])
            if abs(float(vanishing[2])) < 1.0e-10
            else np.array([vanishing[0] / vanishing[2], vanishing[1] / vanishing[2], 1.0])
        )
        for segment in segments:
            residual = float(_line_homogeneous(segment) @ point)
            total += max(segment_length(segment), 1.0) * residual * residual
    return float(total) if usable else float("inf")


def estimate_division_lambda(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    *,
    bound: float = 0.6,
) -> float:
    """Estimate radial division λ when one axis has at least three segments."""
    if not any(len(segments) >= 3 for segments in line_bundles.values()):
        return 0.0

    def objective(value: float) -> float:
        return _concurrency_cost(line_bundles, intrinsics, value)

    samples = np.linspace(-bound, bound, 49)
    costs = np.array([objective(float(value)) for value in samples])
    best_index = int(np.argmin(costs))
    low = float(samples[max(0, best_index - 1)])
    high = float(samples[min(len(samples) - 1, best_index + 1)])
    for _iteration in range(24):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if objective(first) < objective(second):
            high = second
        else:
            low = first
    best = 0.5 * (low + high)
    return 0.0 if objective(0.0) - objective(best) < 0.5 else float(best)


def collect_vanishing_points(
    line_bundles: dict[AxisId, list[LineSegment]],
) -> dict[AxisId, np.ndarray]:
    """Fit every usable axis bundle (≥2 concurrent segments per axis)."""
    output: dict[AxisId, np.ndarray] = {}
    for axis, segments in line_bundles.items():
        vanishing = vanishing_point_from_lines(segments)
        if vanishing is not None:
            output[axis] = vanishing
    return output


def axis_line_counts(
    line_bundles: dict[AxisId, list[LineSegment]],
) -> dict[AxisId, int]:
    """Return how many segments are drawn on each colored axis."""
    return {
        "x": len(line_bundles.get("x", [])),
        "y": len(line_bundles.get("y", [])),
        "z": len(line_bundles.get("z", [])),
    }


def can_solve_orientation(
    line_bundles: dict[AxisId, list[LineSegment]],
    *,
    lock_focal: bool,
    vp_mode: str,
) -> bool:
    """Whether drawn VP lines are sufficient to recover camera orientation.

    Classic path: ≥2 concurrent segments on each of two axes (mode-dependent).
    Known-K path (3-point + Manual FOV / imported intrinsics): ≥1 segment on
    every axis — orthogonality + full K locate the three vanishing points.
    """
    counts = axis_line_counts(line_bundles)
    if vp_mode == "1":
        return counts["y"] >= 2 and counts["z"] >= 2
    if vp_mode == "2":
        return counts["x"] >= 2 and counts["z"] >= 2
    # 3-point: two bundles of ≥2 lines, or known K with one line per axis.
    if sum(1 for axis in ("x", "y", "z") if counts[axis] >= 2) >= 2:
        return True
    if lock_focal and all(counts[axis] >= 1 for axis in ("x", "y", "z")):
        return True
    return False


def orientation_solve_hint(
    line_bundles: dict[AxisId, list[LineSegment]],
    *,
    lock_focal: bool,
    vp_mode: str,
) -> str:
    """Human-readable tip when orientation cannot be solved yet."""
    counts = axis_line_counts(line_bundles)
    summary = f"X {counts['x']} · Y {counts['z']} · Z {counts['y']}"
    if vp_mode == "1":
        return f"1-point needs 2+ Y and 2+ Z lines ({summary})"
    if vp_mode == "2":
        return f"2-point needs 2+ X and 2+ Y lines ({summary})"
    if lock_focal:
        return (
            "3-point with known K needs 1+ line on each axis, "
            f"or 2+ lines on any two axes ({summary})"
        )
    return (
        "3-point needs 2+ lines on any two axes "
        f"(or Manual FOV / Import YAML + 1 line per axis) ({summary})"
    )


def _line_plane_normal(
    segment: LineSegment,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, float] | None:
    """Back-project an image line to a camera-space plane normal + length weight."""
    line = _line_homogeneous(segment)
    # n = K^T ℓ — rays in that plane are orthogonal to n.
    normal = np.array(
        [
            intrinsics.fx * line[0],
            intrinsics.fy * line[1],
            intrinsics.cx * line[0] + intrinsics.cy * line[1] + line[2],
        ],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(normal))
    if length < 1.0e-12:
        return None
    return normal / length, max(segment_length(segment), 1.0)


def _upright_rotation(rotation: np.ndarray) -> np.ndarray:
    """Match orthonormalize_axes handedness / upright conventions."""
    result = np.asarray(rotation, dtype=np.float64).copy()
    if result[1, 2] > 0.0:
        result[:, 1] *= -1.0
        result[:, 2] *= -1.0
    if float(np.linalg.det(result)) < 0.0:
        result[:, 1] *= -1.0
    return result


def _orthogonal_line_constraint_cost(
    rotation: np.ndarray,
    column_normals: dict[int, list[tuple[np.ndarray, float]]],
) -> float:
    """Length-weighted sum of (plane-normal · axis-direction)²."""
    cost = 0.0
    for column, weighted_normals in column_normals.items():
        direction = rotation[:, column]
        for normal, weight in weighted_normals:
            residual = float(np.dot(normal, direction))
            cost += weight * residual * residual
    return cost


def _polish_rotation_from_line_normals(
    rotation: np.ndarray,
    column_normals: dict[int, list[tuple[np.ndarray, float]]],
    *,
    iterations: int = 24,
) -> np.ndarray:
    """Alternating LS polish: each axis → best direction for its lines, then Procrustes.

    One line leaves a free plane (project the current column). Two or more pin a
    direction via the smallest eigenvector of Σ w nnᵀ — so adding segments on an
    axis continuously nudges orientation even before every axis has a pair.
    """
    polished = np.asarray(rotation, dtype=np.float64).copy()
    if float(np.linalg.det(polished)) < 0.0:
        polished[:, 2] *= -1.0

    for _iteration in range(iterations):
        measured = np.zeros((3, 3), dtype=np.float64)
        for column in (0, 1, 2):
            weighted = column_normals[column]
            accumulator = np.zeros((3, 3), dtype=np.float64)
            for normal, weight in weighted:
                accumulator += weight * np.outer(normal, normal)
            eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
            # Rank-1 (single plane): whole tangent plane is null — project current.
            if float(eigenvalues[1]) <= 1.0e-10 * max(float(eigenvalues[2]), 1.0e-12):
                normal = weighted[0][0]
                direction = polished[:, column] - normal * float(
                    np.dot(normal, polished[:, column])
                )
                if float(np.linalg.norm(direction)) < 1.0e-12:
                    direction = eigenvectors[:, 0]
            else:
                direction = eigenvectors[:, 0]
            direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
            if float(np.dot(direction, polished[:, column])) < 0.0:
                direction = -direction
            measured[:, column] = direction

        u_matrix, _singular, v_transpose = np.linalg.svd(measured)
        candidate = u_matrix @ v_transpose
        if float(np.linalg.det(candidate)) < 0.0:
            u_matrix = u_matrix.copy()
            u_matrix[:, -1] *= -1.0
            candidate = u_matrix @ v_transpose
        if float(np.max(np.abs(candidate - polished))) < 1.0e-12:
            polished = candidate
            break
        polished = candidate
    return polished


def rotation_from_orthogonal_lines(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    *,
    initial_rotation: np.ndarray | None = None,
) -> np.ndarray | None:
    """Recover world-to-camera rotation from ≥1 line on each axis with known K.

    Each image line back-projects to a plane through the camera; the matching
    world-axis direction must lie in that plane. With three mutually orthogonal
    axes this pins orientation (discrete sign flips resolved by convention).
    Extra lines on any axis are folded in by a joint LS polish after the
    closed-form seed, so 2+1+1 already averages the doubled axis.
    """
    if not all(len(line_bundles.get(axis, [])) >= 1 for axis in ("x", "y", "z")):
        return None

    # UI axis → rotation column (X, Blender-Y/UI-z, Blender-Z/UI-y).
    axis_columns: tuple[tuple[AxisId, int], ...] = (("x", 0), ("z", 1), ("y", 2))
    column_normals: dict[int, list[tuple[np.ndarray, float]]] = {0: [], 1: [], 2: []}
    for axis, column in axis_columns:
        for segment in line_bundles[axis]:
            mapped = _line_plane_normal(segment, intrinsics)
            if mapped is not None:
                column_normals[column].append(mapped)
    if any(not column_normals[column] for column in (0, 1, 2)):
        return None

    def fixed_direction(column: int) -> np.ndarray | None:
        """Nullspace direction when ≥2 planes pin the axis; else None."""
        weighted = column_normals[column]
        if len(weighted) < 2:
            return None
        matrix = np.stack(
            [normal * np.sqrt(weight) for normal, weight in weighted],
            axis=0,
        )
        _u_matrix, _singular, v_transpose = np.linalg.svd(matrix, full_matrices=True)
        direction = v_transpose[-1]
        return direction / max(float(np.linalg.norm(direction)), 1.0e-12)

    def primary_normal(column: int) -> np.ndarray:
        weighted = column_normals[column]
        if len(weighted) == 1:
            return weighted[0][0]
        # Length-weighted average normal for soft single-plane fallback.
        accumulator = np.zeros(3, dtype=np.float64)
        for normal, weight in weighted:
            accumulator += weight * normal
        return accumulator / max(float(np.linalg.norm(accumulator)), 1.0e-12)

    fixed = {column: fixed_direction(column) for column in (0, 1, 2)}
    fixed_count = sum(1 for direction in fixed.values() if direction is not None)

    candidates: list[np.ndarray] = []

    if fixed_count == 3:
        measured = np.column_stack([fixed[0], fixed[1], fixed[2]])
        u_matrix, _singular, v_transpose = np.linalg.svd(measured)
        rotation = u_matrix @ v_transpose
        if float(np.linalg.det(rotation)) < 0.0:
            u_matrix = u_matrix.copy()
            u_matrix[:, -1] *= -1.0
            rotation = u_matrix @ v_transpose
        candidates.append(rotation)
    elif fixed_count == 2:
        missing = next(column for column in (0, 1, 2) if fixed[column] is None)
        kept = [column for column in (0, 1, 2) if column != missing]
        direction_a = np.asarray(fixed[kept[0]], dtype=np.float64)
        direction_b = np.asarray(fixed[kept[1]], dtype=np.float64)
        direction_a = direction_a / max(float(np.linalg.norm(direction_a)), 1.0e-12)
        direction_b = direction_b - direction_a * float(np.dot(direction_a, direction_b))
        if float(np.linalg.norm(direction_b)) < 1.0e-12:
            return None
        direction_b = direction_b / float(np.linalg.norm(direction_b))
        for sign_a in (1.0, -1.0):
            for sign_b in (1.0, -1.0):
                trial = np.zeros((3, 3), dtype=np.float64)
                trial[:, kept[0]] = sign_a * direction_a
                trial[:, kept[1]] = sign_b * direction_b
                if missing == 2:
                    trial[:, 2] = np.cross(trial[:, 0], trial[:, 1])
                elif missing == 0:
                    trial[:, 0] = np.cross(trial[:, 1], trial[:, 2])
                else:
                    trial[:, 1] = np.cross(trial[:, 2], trial[:, 0])
                if float(np.linalg.det(trial)) < 0.0:
                    trial[:, missing] *= -1.0
                candidates.append(trial)
    else:
        # ≤1 axis fully pinned: solve the free plane angle analytically.
        # d0 = cosθ u + sinθ v in plane 0; d1 ∥ d0 × n1; d2 = d0 × d1.
        # Orthogonality n2·d2 = 0 ⇒ trig equation in 2θ.
        free_columns = [
            column for column in (0, 1, 2) if fixed[column] is None
        ] or [0, 1, 2]
        for free_column in free_columns:
            order = [free_column, (free_column + 1) % 3, (free_column + 2) % 3]
            normal_a = primary_normal(order[0])
            normal_b = primary_normal(order[1])
            normal_c = primary_normal(order[2])
            helper = (
                np.array([1.0, 0.0, 0.0])
                if abs(float(normal_a[0])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            basis_u = np.cross(normal_a, helper)
            if float(np.linalg.norm(basis_u)) < 1.0e-12:
                helper = np.array([0.0, 0.0, 1.0])
                basis_u = np.cross(normal_a, helper)
            basis_u = basis_u / max(float(np.linalg.norm(basis_u)), 1.0e-12)
            basis_v = np.cross(normal_a, basis_u)
            basis_v = basis_v / max(float(np.linalg.norm(basis_v)), 1.0e-12)

            # (d0·nb)(nc·d0) = nc·nb with d0 = c u + s v.
            coeff_a = float(np.dot(basis_u, normal_b))
            coeff_b = float(np.dot(basis_v, normal_b))
            coeff_d = float(np.dot(normal_c, basis_u))
            coeff_e = float(np.dot(normal_c, basis_v))
            coeff_f = float(np.dot(normal_c, normal_b))
            # C1 cos 2θ + C2 sin 2θ = -C0
            coeff_c0 = 0.5 * (coeff_a * coeff_d + coeff_b * coeff_e) - coeff_f
            coeff_c1 = 0.5 * (coeff_a * coeff_d - coeff_b * coeff_e)
            coeff_c2 = 0.5 * (coeff_a * coeff_e + coeff_b * coeff_d)
            radius = float(np.hypot(coeff_c1, coeff_c2))
            angles: list[float] = []
            if radius < 1.0e-14:
                if abs(coeff_c0) < 1.0e-10:
                    angles.extend(
                        float(index * np.pi / 8.0) for index in range(16)
                    )
            else:
                ratio = float(np.clip(-coeff_c0 / radius, -1.0, 1.0))
                offset = float(np.arccos(ratio))
                phase = float(np.arctan2(coeff_c2, coeff_c1))
                for double_angle in (phase + offset, phase - offset):
                    angles.append(0.5 * double_angle)
                    angles.append(0.5 * double_angle + np.pi)

            for angle in angles:
                direction_a = np.cos(angle) * basis_u + np.sin(angle) * basis_v
                cross_ab = np.cross(direction_a, normal_b)
                if float(np.linalg.norm(cross_ab)) < 1.0e-9:
                    continue
                direction_b = cross_ab / float(np.linalg.norm(cross_ab))
                for sign_b in (1.0, -1.0):
                    axis_b = sign_b * direction_b
                    axis_c = np.cross(direction_a, axis_b)
                    if float(np.linalg.norm(axis_c)) < 1.0e-9:
                        continue
                    axis_c = axis_c / float(np.linalg.norm(axis_c))
                    rotation = np.zeros((3, 3), dtype=np.float64)
                    rotation[:, order[0]] = direction_a
                    rotation[:, order[1]] = axis_b
                    rotation[:, order[2]] = axis_c
                    if float(np.linalg.det(rotation)) < 0.0:
                        rotation[:, order[2]] *= -1.0
                    candidates.append(rotation)

    if initial_rotation is not None:
        candidates.append(np.asarray(initial_rotation, dtype=np.float64))

    if not candidates:
        return None

    # Several SO(3) frames can place each axis in its plane; pick the min-cost
    # seed closest to the prior (or identity), polish with every line, upright.
    reference = (
        np.asarray(initial_rotation, dtype=np.float64)
        if initial_rotation is not None
        else np.eye(3, dtype=np.float64)
    )
    scored_candidates: list[tuple[float, float, np.ndarray]] = []
    for candidate in candidates:
        rotation = np.asarray(candidate, dtype=np.float64)
        if float(np.linalg.det(rotation)) < 0.0:
            rotation = rotation.copy()
            rotation[:, 2] *= -1.0
        cost = _orthogonal_line_constraint_cost(rotation, column_normals)
        alignment = float(np.trace(rotation.T @ reference))
        scored_candidates.append((cost, alignment, rotation))
    best_cost = min(cost for cost, _alignment, _rotation in scored_candidates)
    cost_tolerance = max(1.0e-8, 1.0e-4 * best_cost + 1.0e-8)

    best_rotation: np.ndarray | None = None
    best_alignment = -float("inf")
    for cost, alignment, rotation in scored_candidates:
        if cost > best_cost + cost_tolerance:
            continue
        if alignment > best_alignment:
            best_alignment = alignment
            best_rotation = rotation
    if best_rotation is None:
        return None
    polished = _polish_rotation_from_line_normals(best_rotation, column_normals)
    # Keep the seed if polish somehow worsened the joint residual.
    if _orthogonal_line_constraint_cost(
        polished, column_normals
    ) <= _orthogonal_line_constraint_cost(best_rotation, column_normals) + 1.0e-12:
        best_rotation = polished
    return _upright_rotation(best_rotation)


def vanishing_points_from_known_intrinsics(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    *,
    initial_rotation: np.ndarray | None = None,
) -> dict[AxisId, np.ndarray]:
    """Estimate all three VPs when K is known and each axis has ≥1 line.

    Prefer classic multi-line fits when an axis already has concurrent segments;
    fill any remaining axes from the joint orthogonal solve.
    """
    output = collect_vanishing_points(line_bundles)
    if all(axis in output for axis in ("x", "y", "z")):
        return output
    if not all(len(line_bundles.get(axis, [])) >= 1 for axis in ("x", "y", "z")):
        return output

    rotation = rotation_from_orthogonal_lines(
        line_bundles,
        intrinsics,
        initial_rotation=initial_rotation,
    )
    if rotation is None:
        return output
    # Emit VPs for every axis from the joint orientation (overrides partial classic
    # fits so single-line axes stay consistent with orthogonality + K).
    return {
        "x": vanishing_from_camera_direction(rotation[:, 0], intrinsics),
        "z": vanishing_from_camera_direction(rotation[:, 1], intrinsics),
        "y": vanishing_from_camera_direction(rotation[:, 2], intrinsics),
    }


def _finite_xy(vanishing: np.ndarray) -> np.ndarray | None:
    if abs(float(vanishing[2])) < 1.0e-10:
        return None
    return vanishing[:2] / vanishing[2]


def orthocenter_2d(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> np.ndarray | None:
    """Return the orthocenter of a non-degenerate 2D triangle."""
    vector_ab = point_b - point_a
    vector_ac = point_c - point_a
    area_twice = abs(float(vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0]))
    if area_twice < 1.0:
        return None
    vector_bc = point_c - point_b
    matrix = np.array(
        [[vector_bc[0], vector_bc[1]], [vector_ac[0], vector_ac[1]]],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        return None
    right_hand = np.array(
        [float(np.dot(point_a, vector_bc)), float(np.dot(point_b, vector_ac))]
    )
    try:
        return np.linalg.solve(matrix, right_hand)
    except np.linalg.LinAlgError:
        return None


def principal_point_from_three_vps(
    vanishing_points: dict[AxisId, np.ndarray],
    image_width: int,
    image_height: int,
) -> tuple[float, float] | None:
    """Solve principal point as the orthocenter of three finite orthogonal VPs."""
    if not all(axis in vanishing_points for axis in ("x", "y", "z")):
        return None
    finite = [_finite_xy(vanishing_points[axis]) for axis in ("x", "y", "z")]
    if any(point is None for point in finite):
        return None
    center = orthocenter_2d(finite[0], finite[1], finite[2])
    if center is None:
        return None
    if (
        abs(float(center[0]) - image_width * 0.5) > image_width * 0.45
        or abs(float(center[1]) - image_height * 0.5) > image_height * 0.45
    ):
        return None
    return float(center[0]), float(center[1])


def focal_from_vanishing_pair(
    first: np.ndarray,
    second: np.ndarray,
    cx: float,
    cy: float,
) -> float | None:
    """Solve focal pixels from two finite orthogonal vanishing points."""
    first_xy = _finite_xy(first)
    second_xy = _finite_xy(second)
    if first_xy is None or second_xy is None:
        return None
    focal_squared = -float(
        (first_xy[0] - cx) * (second_xy[0] - cx)
        + (first_xy[1] - cy) * (second_xy[1] - cy)
    )
    return float(np.sqrt(focal_squared)) if focal_squared > 1.0 else None


def focal_estimates_by_pair(
    vanishing_points: dict[AxisId, np.ndarray],
    cx: float,
    cy: float,
) -> dict[str, float]:
    """Return focal estimates for each available world plane."""
    output: dict[str, float] = {}
    for label, first, second in (("XY", "x", "z"), ("ZY", "y", "z"), ("ZX", "x", "y")):
        if first in vanishing_points and second in vanishing_points:
            focal = focal_from_vanishing_pair(
                vanishing_points[first],
                vanishing_points[second],
                cx,
                cy,
            )
            if focal is not None:
                output[label] = focal
    return output


def _normalized_direction(
    vanishing: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    if abs(float(vanishing[2])) < 1.0e-10:
        direction = np.array(
            [vanishing[0] / intrinsics.fx, vanishing[1] / intrinsics.fy, 0.0]
        )
    else:
        direction = np.array(
            [
                (vanishing[0] / vanishing[2] - intrinsics.cx) / intrinsics.fx,
                (vanishing[1] / vanishing[2] - intrinsics.cy) / intrinsics.fy,
                1.0,
            ]
        )
    return direction / max(float(np.linalg.norm(direction)), 1.0e-12)


def orthonormalize_axes(directions: dict[AxisId, np.ndarray]) -> np.ndarray:
    """Build nearest world-to-camera rotation from measured colored axes."""
    x_direction = directions.get("x")
    y_direction = directions.get("z")
    z_direction = directions.get("y")

    def unit(vector: np.ndarray) -> np.ndarray:
        return vector / max(float(np.linalg.norm(vector)), 1.0e-12)

    def finish(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        if rotation[1, 2] > 0.0:
            rotation[:, 1] *= -1.0
            rotation[:, 2] *= -1.0
        if float(np.linalg.det(rotation)) < 0.0:
            rotation[:, 1] *= -1.0
        return rotation

    if x_direction is not None and y_direction is not None and z_direction is not None:
        measured = np.column_stack([unit(x_direction), unit(y_direction), unit(z_direction)])
        if measured[1, 2] > 0.0:
            measured[:, 2] *= -1.0
        if float(np.dot(np.cross(measured[:, 0], measured[:, 1]), measured[:, 2])) < 0.0:
            measured[:, 1] *= -1.0
        u_matrix, _singular_values, v_transpose = np.linalg.svd(measured)
        rotation = u_matrix @ v_transpose
        if float(np.linalg.det(rotation)) < 0.0:
            u_matrix[:, -1] *= -1.0
            rotation = u_matrix @ v_transpose
        if rotation[1, 2] > 0.0:
            rotation[:, 1] *= -1.0
            rotation[:, 2] *= -1.0
        return rotation

    if z_direction is not None and (x_direction is not None or y_direction is not None):
        z_axis = unit(z_direction)
        if x_direction is not None:
            x_axis = unit(x_direction - z_axis * np.dot(x_direction, z_axis))
            y_axis = unit(np.cross(z_axis, x_axis))
        else:
            y_axis = unit(y_direction - z_axis * np.dot(y_direction, z_axis))
            x_axis = unit(np.cross(y_axis, z_axis))
        return finish(x_axis, y_axis, z_axis)

    if x_direction is not None and y_direction is not None:
        x_axis = unit(x_direction)
        y_axis = unit(y_direction - x_axis * np.dot(y_direction, x_axis))
        return finish(x_axis, y_axis, unit(np.cross(x_axis, y_axis)))
    return np.eye(3, dtype=np.float64)


def refine_camera(
    line_bundles: dict[AxisId, list[LineSegment]],
    intrinsics: CameraIntrinsics,
    *,
    lock_focal: bool,
    estimate_principal_point: bool = True,
    estimate_distortion: bool = False,
    initial_division_lambda: float = 0.0,
    initial_rotation: np.ndarray | None = None,
) -> Calibration:
    """Refine focal, principal point, radial λ, and orientation from line bundles.

    With ``lock_focal`` and ≥1 line on each axis, orientation comes from the
    known-K orthogonal-line solve (vanishing points need not be intersected
    from parallel pairs).
    """
    current = CameraIntrinsics(**intrinsics.__dict__)
    division_lambda = float(initial_division_lambda)
    lambda_saturated = False
    vanishing_points: dict[AxisId, np.ndarray] = {}
    known_k_rotation: np.ndarray | None = None

    for _pass_index in range(1 if lock_focal else 8):
        previous = (
            current.fx,
            current.fy,
            current.cx,
            current.cy,
            division_lambda,
        )
        if estimate_distortion:
            division_lambda = estimate_division_lambda(line_bundles, current)
            lambda_saturated = abs(division_lambda) >= 0.588
            if lambda_saturated:
                division_lambda = 0.0
        working = undistort_line_bundles(line_bundles, current, division_lambda)
        use_known_k = lock_focal and all(
            len(working.get(axis, [])) >= 1 for axis in ("x", "y", "z")
        )
        if use_known_k:
            known_k_rotation = rotation_from_orthogonal_lines(
                working,
                current,
                initial_rotation=initial_rotation if known_k_rotation is None else known_k_rotation,
            )
            if known_k_rotation is not None:
                vanishing_points = {
                    "x": vanishing_from_camera_direction(known_k_rotation[:, 0], current),
                    "z": vanishing_from_camera_direction(known_k_rotation[:, 1], current),
                    "y": vanishing_from_camera_direction(known_k_rotation[:, 2], current),
                }
            else:
                vanishing_points = collect_vanishing_points(working)
        else:
            known_k_rotation = None
            vanishing_points = collect_vanishing_points(working)
        if estimate_principal_point:
            principal = principal_point_from_three_vps(
                vanishing_points,
                current.image_width,
                current.image_height,
            )
            if principal is not None:
                current.cx, current.cy = principal
        if not lock_focal:
            estimates = focal_estimates_by_pair(
                vanishing_points,
                current.cx,
                current.cy,
            )
            if estimates:
                focal = float(np.sqrt(np.mean(np.square(list(estimates.values())))))
                current.fx = focal
                current.fy = focal
        now = (current.fx, current.fy, current.cx, current.cy, division_lambda)
        if max(abs(now[index] - previous[index]) for index in range(4)) < 0.25 and abs(
            now[4] - previous[4]
        ) < 1.0e-4:
            break

    if known_k_rotation is not None:
        rotation = known_k_rotation
    else:
        directions = {
            axis: _normalized_direction(vanishing, current)
            for axis, vanishing in vanishing_points.items()
        }
        rotation = orthonormalize_axes(directions)
    return Calibration(
        intrinsics=current,
        rotation_w2c=rotation,
        camera_center=default_camera_center(rotation),
        division_lambda=division_lambda,
        lambda_saturated=lambda_saturated,
    )


def vp_angular_residual_degrees(
    calibration: Calibration,
    line_bundles: dict[AxisId, list[LineSegment]],
) -> float:
    """Max angle (°) between measured VP directions and orthonormalized axes.

    Used as a soft prior when varying focal length: at a candidate fx the
    orientation is rebuilt from VPs, and this residual says how well those
    directions still fit three orthogonal axes.
    """
    working_lines = undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    vanishing_points = collect_vanishing_points(working_lines)
    if not vanishing_points:
        return 180.0
    directions = {
        axis: _normalized_direction(vanishing, calibration.intrinsics)
        for axis, vanishing in vanishing_points.items()
    }
    axis_columns = {"x": 0, "z": 1, "y": 2}
    errors = []
    for axis, direction in directions.items():
        column = axis_columns[axis]
        cosine = abs(float(np.dot(direction, calibration.rotation_w2c[:, column])))
        errors.append(float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))))
    return max(errors, default=0.0)


# World-axis id → column in rotation_w2c (X, Z/UI-Y, Y/UI-Z).
_AXIS_ROTATION_COLUMNS = {"x": 0, "z": 1, "y": 2}


def _homogeneous_image_point(vanishing: np.ndarray) -> np.ndarray:
    """Normalize a homogeneous VP to (x, y, 1) or a line-at-infinity (x, y, 0)."""
    if abs(float(vanishing[2])) < 1.0e-10:
        return np.array([vanishing[0], vanishing[1], 0.0], dtype=np.float64)
    return np.array(
        [
            vanishing[0] / vanishing[2],
            vanishing[1] / vanishing[2],
            1.0,
        ],
        dtype=np.float64,
    )


def ideal_axis_vanishing_point(
    calibration: Calibration,
    axis: AxisId,
) -> np.ndarray | None:
    """Project a solved camera axis to its ideal homogeneous vanishing point."""
    column = _AXIS_ROTATION_COLUMNS.get(axis)
    if column is None:
        return None
    return vanishing_from_camera_direction(
        calibration.rotation_w2c[:, column],
        calibration.intrinsics,
    )


def vp_ideal_segment_residual_px(
    calibration: Calibration,
    axis: AxisId,
    ideal_segment: LineSegment,
) -> float | None:
    """Perpendicular distance (px) of an already-undistorted segment to the ideal VP."""
    vanishing = ideal_axis_vanishing_point(calibration, axis)
    if vanishing is None:
        return None
    point = _homogeneous_image_point(vanishing)
    return abs(float(_line_homogeneous(ideal_segment) @ point))


def vp_line_residual_rms(
    calibration: Calibration,
    line_bundles: dict[AxisId, list[LineSegment]],
) -> float:
    """Length-weighted RMS of line-to-VP distances against the solved axes.

    Projects each rotation column back to an ideal vanishing point, then
    measures how far the undistorted VP segments miss those points. Prefer
    this over the max angular residual when scoring lens trials — it keeps
    every drawn segment honest, not just the worst axis direction.
    """
    working_lines = undistort_line_bundles(
        line_bundles,
        calibration.intrinsics,
        calibration.division_lambda,
    )
    weighted_sse = 0.0
    weight_sum = 0.0
    for axis, segments in working_lines.items():
        if not segments or axis not in _AXIS_ROTATION_COLUMNS:
            continue
        vanishing = ideal_axis_vanishing_point(calibration, axis)
        if vanishing is None:
            continue
        point = _homogeneous_image_point(vanishing)
        for segment in segments:
            weight = max(segment_length(segment), 1.0)
            residual = float(_line_homogeneous(segment) @ point)
            weighted_sse += weight * residual * residual
            weight_sum += weight
    if weight_sum < 1.0e-12:
        return 180.0
    return float(np.sqrt(weighted_sse / weight_sum))


def default_camera_center(rotation_w2c: np.ndarray, *, height: float = 1.7) -> np.ndarray:
    """Place the camera above Z=0 with its principal ray near the world origin."""
    forward = rotation_w2c.T @ np.array([0.0, 0.0, 1.0])
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    if abs(float(forward[2])) > 1.0e-5:
        distance = -height / float(forward[2])
        if distance > 0.2:
            return -distance * forward
    return -5.0 * forward + np.array([0.0, 0.0, height])


def pixel_ray(u_coordinate: float, v_coordinate: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Build a normalized OpenCV camera ray for an image pixel."""
    ray = np.array(
        [
            (u_coordinate - intrinsics.cx) / intrinsics.fx,
            (v_coordinate - intrinsics.cy) / intrinsics.fy,
            1.0,
        ]
    )
    return ray / max(float(np.linalg.norm(ray)), 1.0e-12)


def ground_hit(
    image_point: tuple[float, float],
    calibration: Calibration,
) -> np.ndarray | None:
    """Intersect an image ray with the private-world ground plane Z=0."""
    camera_center = calibration.camera_center
    camera_to_world = calibration.rotation_w2c.T
    ideal_point = undistort_points(
        np.array([[image_point[0], image_point[1]]], dtype=np.float64),
        calibration.intrinsics.fx,
        calibration.intrinsics.fy,
        calibration.intrinsics.cx,
        calibration.intrinsics.cy,
        calibration.division_lambda,
    )[0]
    direction = camera_to_world @ pixel_ray(
        float(ideal_point[0]),
        float(ideal_point[1]),
        calibration.intrinsics,
    )
    if abs(float(direction[2])) < 1.0e-8:
        return None
    distance = -float(camera_center[2]) / float(direction[2])
    return camera_center + distance * direction if distance > 0.0 else None


def apply_origin_and_scale(
    calibration: Calibration,
    origin_image: tuple[float, float],
    scale_point_a: tuple[float, float] | None = None,
    scale_point_b: tuple[float, float] | None = None,
    measured_length: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Return a camera center shifted to the picked ground origin and scaled."""
    camera_center = calibration.camera_center.copy()

    origin_world = ground_hit(origin_image, calibration)
    if origin_world is None:
        origin_world = np.zeros(3)
    scale = 1.0
    if scale_point_a is not None and scale_point_b is not None and measured_length > 0.0:
        world_a = ground_hit(scale_point_a, calibration)
        world_b = ground_hit(scale_point_b, calibration)
        if world_a is not None and world_b is not None:
            current_length = float(np.linalg.norm(world_a - world_b))
            if current_length > 1.0e-8:
                scale = measured_length / current_length
    return (camera_center - origin_world) * scale, scale


def _direction_to_vanishing(point: np.ndarray, vanishing: np.ndarray) -> np.ndarray:
    direction = (
        vanishing[:2]
        if abs(float(vanishing[2])) < 1.0e-10
        else vanishing[:2] / vanishing[2] - point
    )
    return direction / max(float(np.linalg.norm(direction)), 1.0e-12)


def _intersect_2d(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray | None:
    determinant = float(
        direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    )
    if abs(determinant) < 1.0e-10:
        return None
    distance = float(
        ((point_b[0] - point_a[0]) * direction_b[1]
        - (point_b[1] - point_a[1]) * direction_b[0])
        / determinant
    )
    return point_a + distance * direction_a


def perspective_rectangle_corners(
    corner_a: tuple[float, float],
    corner_c: tuple[float, float],
    vanishing_u: np.ndarray,
    vanishing_v: np.ndarray,
) -> list[tuple[float, float]] | None:
    """Construct a perspective rectangle from opposite corners and two VPs."""
    point_a = np.array(corner_a, dtype=np.float64)
    point_c = np.array(corner_c, dtype=np.float64)
    point_b = _intersect_2d(
        point_a,
        _direction_to_vanishing(point_a, vanishing_u),
        point_c,
        _direction_to_vanishing(point_c, vanishing_v),
    )
    point_d = _intersect_2d(
        point_a,
        _direction_to_vanishing(point_a, vanishing_v),
        point_c,
        _direction_to_vanishing(point_c, vanishing_u),
    )
    if point_b is None or point_d is None:
        return None
    corners = [point_a, point_b, point_c, point_d]
    edge_lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    if min(edge_lengths) < 2.0:
        return None
    return [(float(point[0]), float(point[1])) for point in corners]


def implied_vertical_vanishing(
    vanishing_x: np.ndarray | None,
    vanishing_z: np.ndarray | None,
) -> np.ndarray | None:
    """Return the infinite upright direction perpendicular to the 2-point horizon."""
    if vanishing_x is None or vanishing_z is None:
        return None
    x_point = _finite_xy(vanishing_x)
    z_point = _finite_xy(vanishing_z)
    if x_point is None or z_point is None:
        return None
    horizon = z_point - x_point
    length = float(np.linalg.norm(horizon))
    if length < 1.0e-8:
        return None
    direction = np.array([-horizon[1], horizon[0]]) / length
    if direction[1] > 0.0:
        direction *= -1.0
    return np.array([direction[0], direction[1], 0.0])


def vanishing_from_camera_direction(
    direction: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project a camera-space direction to a homogeneous ideal vanishing point."""
    vector = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vector = vector / length
    if abs(float(vector[2])) < 1.0e-10:
        return np.array(
            [vector[0] * intrinsics.fx, vector[1] * intrinsics.fy, 0.0],
            dtype=np.float64,
        )
    return np.array(
        [
            vector[0] / vector[2] * intrinsics.fx + intrinsics.cx,
            vector[1] / vector[2] * intrinsics.fy + intrinsics.cy,
            1.0,
        ],
        dtype=np.float64,
    )


def complete_vanishing_points(
    vanishing_points: dict[AxisId, np.ndarray],
    intrinsics: CameraIntrinsics,
) -> dict[AxisId, np.ndarray]:
    """Fill missing orthogonal VPs from measured ones (for horizon / overlay guides)."""
    resolved = dict(vanishing_points)
    if len(resolved) < 2:
        return resolved
    if all(axis in resolved for axis in ("x", "y", "z")):
        return resolved

    working = CameraIntrinsics(**intrinsics.__dict__)
    estimates = focal_estimates_by_pair(resolved, working.cx, working.cy)
    if estimates:
        focal = float(np.sqrt(np.mean(np.square(list(estimates.values())))))
        working.fx = focal
        working.fy = focal

    directions = {
        axis: _normalized_direction(vanishing, working)
        for axis, vanishing in resolved.items()
    }
    rotation = orthonormalize_axes(directions)
    # Columns are world X/Y/Z in camera space; colored axes map x→X, z→Y, y→Z.
    for axis, column in (("x", 0), ("z", 1), ("y", 2)):
        if axis not in resolved:
            resolved[axis] = vanishing_from_camera_direction(rotation[:, column], working)
    return resolved


def plane_axes(plane: SurfacePlane) -> tuple[AxisId, AxisId]:
    """Return the two colored VP axes defining a surface plane."""
    if plane == "xz":
        return "x", "z"
    if plane == "yz":
        return "y", "z"
    return "y", "x"


def surface_vanishing_points(
    plane: SurfacePlane,
    vanishing_points: dict[AxisId, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve surface VPs, including implied 2-point uprights."""
    resolved = dict(vanishing_points)
    if "y" not in resolved:
        implied = implied_vertical_vanishing(resolved.get("x"), resolved.get("z"))
        if implied is not None:
            resolved["y"] = implied
    first_axis, second_axis = plane_axes(plane)
    if first_axis not in resolved or second_axis not in resolved:
        return None
    return resolved[first_axis], resolved[second_axis]


def surface_grid(
    corners: list[tuple[float, float]],
    divisions: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Create perspective-correct interior grid segments via a quad homography."""
    if len(corners) != 4 or divisions <= 1:
        return []
    source = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    design_rows = []
    for (u_coordinate, v_coordinate), (x_coordinate, y_coordinate) in zip(source, corners):
        design_rows.extend(
            [
                [u_coordinate, v_coordinate, 1.0, 0.0, 0.0, 0.0,
                 -x_coordinate * u_coordinate, -x_coordinate * v_coordinate, -x_coordinate],
                [0.0, 0.0, 0.0, u_coordinate, v_coordinate, 1.0,
                 -y_coordinate * u_coordinate, -y_coordinate * v_coordinate, -y_coordinate],
            ]
        )
    _u_matrix, _singular_values, v_transpose = np.linalg.svd(
        np.asarray(design_rows, dtype=np.float64)
    )
    homography = v_transpose[-1].reshape(3, 3)

    def apply(u_coordinate: float, v_coordinate: float) -> tuple[float, float]:
        result = homography @ np.array([u_coordinate, v_coordinate, 1.0])
        denominator = result[2] if abs(float(result[2])) > 1.0e-12 else 1.0
        return float(result[0] / denominator), float(result[1] / denominator)

    grid = []
    for index in range(1, max(1, min(64, divisions))):
        ratio = index / divisions
        grid.append((apply(ratio, 0.0), apply(ratio, 1.0)))
        grid.append((apply(0.0, ratio), apply(1.0, ratio)))
    return grid


def reconstruct_surface_world(
    plane: SurfacePlane,
    image_corners: list[tuple[float, float]],
    calibration: Calibration,
) -> list[tuple[float, float, float]] | None:
    """Ray-cast a perspective quad onto a Blender coordinate plane."""
    if len(image_corners) != 4:
        return None
    normals = {
        "xz": np.array([0.0, 0.0, 1.0]),
        "yz": np.array([1.0, 0.0, 0.0]),
        "yx": np.array([0.0, 1.0, 0.0]),
    }
    normal = normals[plane]
    camera_to_world = calibration.rotation_w2c.T
    hits: list[np.ndarray] = []
    for u_coordinate, v_coordinate in image_corners:
        direction = camera_to_world @ pixel_ray(
            u_coordinate,
            v_coordinate,
            calibration.intrinsics,
        )
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1.0e-9:
            return None
        distance = -float(np.dot(normal, calibration.camera_center)) / denominator
        if distance <= 1.0e-6:
            return None
        hits.append(calibration.camera_center + distance * direction)

    point_a, point_b, point_c, point_d = hits

    def edge_constant(first: float, second: float, epsilon: float) -> float:
        # Pin only when the full perspective edge already lies on a world axis.
        return 0.0 if abs(first) < epsilon and abs(second) < epsilon else 0.5 * (first + second)

    def expanded(first: float, second: float) -> tuple[float, float]:
        if abs(second - first) >= 1.0e-3:
            return first, second
        midpoint = 0.5 * (first + second)
        return midpoint - 5.0e-4, midpoint + 5.0e-4

    span = max(
        float(np.linalg.norm(point_a - point_b)),
        float(np.linalg.norm(point_b - point_c)),
        1.0e-3,
    )
    axis_epsilon = max(1.0e-4, 0.01 * span)
    if plane == "xz":
        x_left = edge_constant(point_a[0], point_d[0], axis_epsilon)
        x_right = edge_constant(point_b[0], point_c[0], axis_epsilon)
        y_near = edge_constant(point_a[1], point_b[1], axis_epsilon)
        y_far = edge_constant(point_c[1], point_d[1], axis_epsilon)
        x_left, x_right = expanded(x_left, x_right)
        y_near, y_far = expanded(y_near, y_far)
        return [(x_left, y_near, 0.0), (x_right, y_near, 0.0),
                (x_right, y_far, 0.0), (x_left, y_far, 0.0)]
    if plane == "yz":
        y_near = edge_constant(point_a[1], point_b[1], axis_epsilon)
        y_far = edge_constant(point_c[1], point_d[1], axis_epsilon)
        z_low = edge_constant(point_a[2], point_d[2], axis_epsilon)
        z_high = edge_constant(point_b[2], point_c[2], axis_epsilon)
        y_near, y_far = expanded(y_near, y_far)
        z_low, z_high = expanded(z_low, z_high)
        return [(0.0, y_near, z_low), (0.0, y_near, z_high),
                (0.0, y_far, z_high), (0.0, y_far, z_low)]
    x_left = edge_constant(point_a[0], point_b[0], axis_epsilon)
    x_right = edge_constant(point_c[0], point_d[0], axis_epsilon)
    z_low = edge_constant(point_a[2], point_d[2], axis_epsilon)
    z_high = edge_constant(point_b[2], point_c[2], axis_epsilon)
    x_left, x_right = expanded(x_left, x_right)
    z_low, z_high = expanded(z_low, z_high)
    return [(x_left, 0.0, z_low), (x_left, 0.0, z_high),
            (x_right, 0.0, z_high), (x_right, 0.0, z_low)]
