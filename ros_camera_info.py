"""Parse ROS ``camera_info`` YAML (OpenCV / camera_calibration_parsers layout).

Blender does not ship PyYAML, so this module only understands the flat
scalar + one-level matrix block shape used by that format.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RosCameraInfo:
    """Intrinsics extracted from a ROS camera_info YAML file."""

    image_width: int
    image_height: int
    camera_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    # Project extension — None when the file omits it (stock ROS YAML).
    fitzgibbon_lambda: float | None = None


_KEY_VALUE = re.compile(r"^(\s*)([A-Za-z_][\w]*)\s*:\s*(.*?)\s*$")


def _parse_scalar(raw: str) -> Any:
    """Parse a YAML scalar used by ROS camera_info exports."""
    if raw == "" or raw is None:
        return None
    if raw in {"null", "Null", "NULL", "~"}:
        return None
    if raw in {"true", "True", "TRUE"}:
        return True
    if raw in {"false", "False", "FALSE"}:
        return False
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid YAML list: {raw}") from error
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected a list, got {raw}")
        return [float(item) for item in value]
    try:
        if any(char in raw for char in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_ros_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse top-level keys with optional one-level nested mappings."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = _KEY_VALUE.match(raw_line)
        if match is None:
            raise ValueError(f"Unsupported YAML on line {line_number}: {raw_line}")
        indent, key, value_text = match.groups()
        depth = len(indent.replace("\t", "  "))
        while len(stack) > 1 and depth <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((depth, child))
            continue
        parent[key] = _parse_scalar(value_text)

    return root


def _matrix_data(block: Any, label: str, expected_len: int) -> list[float]:
    if not isinstance(block, dict):
        raise ValueError(f"Missing or invalid {label}")
    data = block.get("data")
    if not isinstance(data, list) or len(data) < expected_len:
        raise ValueError(f"{label}.data must contain at least {expected_len} numbers")
    values = []
    for index, item in enumerate(data[:expected_len]):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.data[{index}] is not numeric") from error
        if not math.isfinite(number):
            raise ValueError(f"{label}.data[{index}] is not finite")
        values.append(number)
    return values


def parse_ros_camera_info_yaml(text: str) -> RosCameraInfo:
    """Parse ROS camera_info YAML text into pinhole intrinsics."""
    mapping = _parse_ros_yaml_mapping(text)
    try:
        width = int(mapping["image_width"])
        height = int(mapping["image_height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("YAML must include image_width and image_height") from error
    if width <= 0 or height <= 0:
        raise ValueError("YAML image_width / image_height must be positive")

    camera_matrix = _matrix_data(mapping.get("camera_matrix"), "camera_matrix", 9)
    # Row-major K: fx, 0, cx, 0, fy, cy, 0, 0, 1
    fx, _zero_a, cx, _zero_b, fy, cy = camera_matrix[:6]
    if fx <= 1.0e-6 or fy <= 1.0e-6:
        raise ValueError("camera_matrix fx/fy must be positive")

    distortion_model = str(mapping.get("distortion_model") or "")
    distortion_block = mapping.get("distortion_coefficients")
    coefficients: tuple[float, ...] = ()
    if isinstance(distortion_block, dict) and "data" in distortion_block:
        raw = distortion_block["data"]
        if isinstance(raw, list):
            coefficients = tuple(float(item) for item in raw)

    camera_name = mapping.get("camera_name")
    if camera_name is None:
        camera_name = ""
    else:
        camera_name = str(camera_name)

    # Optional project extension (ignored by stock ROS/OpenCV loaders).
    fitzgibbon_lambda: float | None = None
    if "fitzgibbon_lambda" in mapping:
        try:
            fitzgibbon_lambda = float(mapping["fitzgibbon_lambda"])
        except (TypeError, ValueError) as error:
            raise ValueError("fitzgibbon_lambda must be numeric") from error
        if not math.isfinite(fitzgibbon_lambda):
            raise ValueError("fitzgibbon_lambda must be finite")

    return RosCameraInfo(
        image_width=width,
        image_height=height,
        camera_name=camera_name,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
        fitzgibbon_lambda=fitzgibbon_lambda,
    )


def scale_intrinsics_to_image(
    info: RosCameraInfo,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float, bool]:
    """Return fx, fy, cx, cy for the target plate; True when scaled."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Target image size must be positive")
    if info.image_width == image_width and info.image_height == image_height:
        return info.fx, info.fy, info.cx, info.cy, False
    scale_x = float(image_width) / float(info.image_width)
    scale_y = float(image_height) / float(info.image_height)
    return (
        info.fx * scale_x,
        info.fy * scale_y,
        info.cx * scale_x,
        info.cy * scale_y,
        True,
    )
