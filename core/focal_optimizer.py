"""Bound-aware least-squares steps for the independent focal fit."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def bounded_lm_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    parameters: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    damping: float,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Return a feasible damped step, re-solving after bound hits.

    This is an active-set step, not a general exact bounded-QP solver.
    """
    if damping <= 0:
        raise ValueError("damping must be positive")
    column_scale = np.maximum(np.linalg.norm(jacobian, axis=0), 1.0e-8)
    scaled = jacobian / column_scale
    gradient = scaled.T @ residual
    at_lower = parameters <= lower + 1.0e-10
    at_upper = parameters >= upper - 1.0e-10
    fixed = (at_lower & (gradient > 0)) | (at_upper & (gradient < 0))
    scaled_step = np.zeros(len(parameters))
    scaled_lower = (lower - parameters) * column_scale
    scaled_upper = (upper - parameters) * column_scale
    for _ in range(len(parameters) + 1):
        if cancel_check and cancel_check():
            raise InterruptedError("Bounded focal step cancelled")
        free = ~fixed
        if not np.any(free):
            break
        target = -residual - scaled[:, fixed] @ scaled_step[fixed]
        system = np.vstack((scaled[:, free], math.sqrt(damping) * np.eye(np.count_nonzero(free))))
        rhs = np.concatenate((target, np.zeros(np.count_nonzero(free))))
        candidate = np.linalg.lstsq(system, rhs, rcond=None)[0]
        indices = np.flatnonzero(free)
        violated_lower = candidate < scaled_lower[indices]
        violated_upper = candidate > scaled_upper[indices]
        if not np.any(violated_lower | violated_upper):
            scaled_step[indices] = candidate
            break
        violated = violated_lower | violated_upper
        hit = indices[violated]
        scaled_step[hit] = np.where(violated_lower[violated], scaled_lower[hit], scaled_upper[hit])
        fixed[hit] = True
    return scaled_step / column_scale
