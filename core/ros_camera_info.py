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


_BROWN_MODELS = {"", "plumb_bob", "rational_polynomial"}
_UNSUPPORTED_DISTORTION_MODELS = {
    "equidistant",
    "kannala_brandt",
    "fisheye",
}


@dataclass(frozen=True)
class ImportedDistortion:
    """How ROS ``distortion_coefficients`` should be applied on import."""

    brown_conrady: tuple[float, ...]
    skip_reason: str = ""


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


def remap_intrinsics_to_size(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, float, float, float, str]:
    """Map pinhole K from one plate size to another.

    Same size: unchanged (kind ``""``).
    Exact axis swap, e.g. 3000×4000 ↔ 4000×3000: swap fx/fy and cx/cy
    (kind ``"rotated"``) — same camera and pixel pitch, phone rotated 90°.
    Any other size change: scale fx and fy by width so pixels stay square;
    cx follows width, cy follows height (kind ``"scaled"``).
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target image size must be positive")
    source_width = max(int(source_width), 1)
    source_height = max(int(source_height), 1)
    if source_width == target_width and source_height == target_height:
        return float(fx), float(fy), float(cx), float(cy), ""
    if (
        source_width == target_height
        and source_height == target_width
        and source_width != source_height
    ):
        return float(fy), float(fx), float(cy), float(cx), "rotated"
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    return (
        float(fx) * scale_x,
        float(fy) * scale_x,
        float(cx) * scale_x,
        float(cy) * scale_y,
        "scaled",
    )


def scale_intrinsics_to_image(
    info: RosCameraInfo,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float, str]:
    """Return fx, fy, cx, cy for the target plate, and ``""`` / ``scaled`` / ``rotated``."""
    return remap_intrinsics_to_size(
        info.fx,
        info.fy,
        info.cx,
        info.cy,
        info.image_width,
        info.image_height,
        image_width,
        image_height,
    )


def interpret_distortion(info: RosCameraInfo) -> ImportedDistortion:
    """Pick OpenCV Brown–Conrady D from a ROS camera_info, or a skip reason.

    Coefficients live in the normalized plane, so they are not scaled with
    image size. Fitzgibbon λ is handled separately by the importer.
    """
    coefficients = info.distortion_coefficients
    if not coefficients or not any(abs(float(value)) > 1.0e-12 for value in coefficients):
        return ImportedDistortion((), "")

    model = str(info.distortion_model or "").strip().lower().replace("-", "_")
    if model in _UNSUPPORTED_DISTORTION_MODELS:
        return ImportedDistortion((), f"{info.distortion_model} D skipped")

    extra = coefficients[8:]
    extra_skip = ""
    if extra and any(abs(float(value)) > 1.0e-12 for value in extra):
        extra_skip = "thin-prism/tilt D skipped"

    if model not in _BROWN_MODELS and not (4 <= len(coefficients) <= 8):
        return ImportedDistortion((), f"{info.distortion_model} D skipped")
    if len(coefficients) < 4:
        return ImportedDistortion((), "D skipped (need ≥4 coefficients)")

    values = [float(item) for item in coefficients[:8]]
    while len(values) < 8:
        values.append(0.0)
    return ImportedDistortion(tuple(values), extra_skip)
