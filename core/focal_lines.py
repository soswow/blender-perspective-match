"""Minimal line coordinates and image residuals for independent focal fitting."""

from __future__ import annotations

import numpy as np


def canonical_line_point(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Represent an infinite line by its closest point to the local origin."""
    unit = np.asarray(direction, float)
    unit = unit / np.linalg.norm(unit)
    point = np.asarray(point, float)
    return point - float(point @ unit) * unit


def line_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable orthonormal basis perpendicular to a unit direction."""
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


class LineChart:
    """Four coordinates for an infinite unoriented line, near a seed line."""

    def __init__(self, point: np.ndarray, direction: np.ndarray):
        self.seed_direction = np.asarray(direction, float) / np.linalg.norm(direction)
        self.seed_basis = np.asarray(line_frame(self.seed_direction))
        closest = np.asarray(point, float) - np.dot(point, self.seed_direction) * self.seed_direction
        self.seed_point = closest
        self.initial = np.asarray((0.0, 0.0, *[float(closest @ b) for b in self.seed_basis]))

    def decode(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = self.seed_direction + params[:2] @ self.seed_basis
        direction /= np.linalg.norm(direction)
        first = self.seed_basis[0] - direction * np.dot(direction, self.seed_basis[0])
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        point = params[2] * first + params[3] * second
        return point, direction


def endpoint_distances(point: np.ndarray, direction: np.ndarray,
                       rotation: np.ndarray, center: np.ndarray, focal: float,
                       principal: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    """Signed pixel distances from two stroke endpoints to the projected 3D line."""
    a = rotation @ (point - center)
    b = rotation @ direction
    # Homogeneous image line K^-T (a x b). This remains valid when the
    # visible portion of the infinite line lies far from the closest point.
    line = np.cross(a, b)
    normal = line[:2] / focal
    intercept = line[2] - np.dot(normal, principal)
    length = np.linalg.norm(normal)
    if length < 1.0e-12:
        return np.full(2, 1.0e6)
    return (endpoints @ normal + intercept) / length
