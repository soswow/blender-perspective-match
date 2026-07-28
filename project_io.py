"""Perspective Match project import (.pmproj)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from . import core, properties, scene

PROJECT_KIND = "perspective-match-project"
PROJECT_VERSION = 1


def _session_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") == PROJECT_KIND and isinstance(payload.get("session"), dict):
        if int(payload.get("version", 1)) > PROJECT_VERSION:
            raise ValueError("This project was created by a newer Perspective Match version")
        return payload["session"]
    if "imagePath" in payload or "lines" in payload:
        return payload
    raise ValueError("Not a Perspective Match project file")


def _validate_session(session: dict[str, Any]) -> None:
    """Validate fields needed for a safe import; surfaces/scale are ignored."""
    if int(session.get("vpMode", 2)) not in {1, 2, 3}:
        raise ValueError("Project has an invalid VP mode")
    if session.get("activeAxis", "x") not in {"x", "y", "z"}:
        raise ValueError("Project has an invalid active axis")

    def finite_number(value: Any, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Project {label} is not numeric") from error
        if not math.isfinite(number):
            raise ValueError(f"Project {label} is not finite")
        return number

    line_groups = session.get("lines") or {}
    if not isinstance(line_groups, dict):
        raise ValueError("Project lines must be grouped by axis")
    for axis in ("x", "y", "z"):
        group = line_groups.get(axis, []) or []
        if not isinstance(group, list):
            raise ValueError(f"Project {axis} lines must be a list")
        for line in group:
            if not isinstance(line, dict):
                raise ValueError("Project line record is invalid")
            for coordinate in ("x1", "y1", "x2", "y2"):
                finite_number(line.get(coordinate), f"line {coordinate}")

    calibration = session.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ValueError("Project calibration is invalid")
        width = int(calibration.get("imageWidth", 0))
        height = int(calibration.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Project calibration has invalid image dimensions")
        for key in ("fx", "fy", "cx", "cy"):
            finite_number(calibration.get(key), f"calibration {key}")
        finite_number(calibration.get("divisionLambda", 0.0), "calibration divisionLambda")
        rotation = np.asarray(calibration.get("rotation"), dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Project calibration rotation must be a finite 3×3 matrix")
        translation = np.asarray(calibration.get("translation", (0.0, 0.0, 0.0)))
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Project camera center must contain three finite values")

    for key in ("overlayOpacity", "controlsOpacity"):
        finite_number(session.get(key, 1.0), key)
    for key in ("undistortedOffsetX", "undistortedOffsetY"):
        finite_number(session.get(key, 0.0), key)
    origin = session.get("originImage")
    if origin is not None:
        if not isinstance(origin, (list, tuple)) or len(origin) < 2:
            raise ValueError("Project originImage must contain two coordinates")
        finite_number(origin[0], "originImage x")
        finite_number(origin[1], "originImage y")


def load_project(context: bpy.types.Context, path: str) -> None:
    """Load a version-1 desktop or Blender project into the current scene."""
    project_path = Path(path).expanduser().resolve()
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    session = _session_from_payload(payload)
    _validate_session(session)
    image_path_value = session.get("imagePath")
    if not image_path_value:
        raise ValueError("Project does not reference an image")
    image_path = Path(str(image_path_value)).expanduser()
    if not image_path.is_absolute():
        image_path = project_path.parent / image_path
    scene.bind_reference_image(context, str(image_path))
    settings = properties.active_session(context)
    if settings is None:
        raise ValueError("Failed to create or activate a match session")
    settings.project_path = str(project_path)
    settings.source_session_json = json.dumps(session)
    settings.vp_mode = str(int(session.get("vpMode", 2)))
    settings.active_axis = session.get("activeAxis", "x")
    settings.lock_focal = bool(session.get("lockFocal", False))
    settings.overlay_opacity = float(session.get("overlayOpacity", 0.9))
    settings.show_vp_overlay = bool(session.get("showVpOverlay", True))

    settings.lines.clear()
    line_groups = session.get("lines") or {}
    for axis in ("x", "y", "z"):
        for raw in line_groups.get(axis, []) or []:
            item = settings.lines.add()
            item.item_id = str(raw.get("id") or "")
            item.axis = axis
            item.x1 = float(raw["x1"])
            item.y1 = float(raw["y1"])
            item.x2 = float(raw["x2"])
            item.y2 = float(raw["y2"])

    # Surfaces and scale fields from desktop projects are intentionally ignored.

    calibration_raw = session.get("calibration") or {}
    if calibration_raw:
        width = int(calibration_raw.get("imageWidth", settings.image_width))
        height = int(calibration_raw.get("imageHeight", settings.image_height))
        intrinsics = core.CameraIntrinsics(
            fx=float(calibration_raw.get("fx", settings.fx)),
            fy=float(calibration_raw.get("fy", settings.fy)),
            cx=float(calibration_raw.get("cx", width * 0.5)),
            cy=float(calibration_raw.get("cy", height * 0.5)),
            image_width=width,
            image_height=height,
        )
        rotation = np.asarray(calibration_raw.get("rotation", np.eye(3)), dtype=np.float64)
        translation = np.asarray(
            calibration_raw.get("translation", core.default_camera_center(rotation)),
            dtype=np.float64,
        )
        calibration = core.Calibration(
            intrinsics=intrinsics,
            rotation_w2c=rotation,
            camera_center=translation,
            division_lambda=float(calibration_raw.get("divisionLambda", 0.0)),
        )
        scene.apply_camera(context.scene, settings, calibration)

    origin = session.get("originImage")
    settings.origin_is_set = bool(origin and len(origin) >= 2)
    if settings.origin_is_set:
        settings.origin_image = (float(origin[0]), float(origin[1]))
        scene.reapply_placement(context)

    settings.undistorted_path = str(session.get("undistortedImagePath") or "")
    settings.undistorted_offset_x = float(session.get("undistortedOffsetX", 0.0))
    settings.undistorted_offset_y = float(session.get("undistortedOffsetY", 0.0))
    requested_undistorted = bool(session.get("viewUndistorted", False))
    if settings.undistorted_path:
        cached_path = Path(settings.undistorted_path).expanduser()
        if not cached_path.is_absolute():
            cached_path = project_path.parent / cached_path
        if cached_path.exists():
            try:
                cached_image = bpy.data.images.load(str(cached_path.resolve()), check_existing=True)
            except (OSError, RuntimeError):
                # The cache is optional; the editable project remains valid without it.
                settings.undistorted_path = ""
            else:
                settings.undistorted_image = cached_image
                settings.undistorted_path = str(cached_path.resolve())
                settings.undistorted_width = int(cached_image.size[0])
                settings.undistorted_height = int(cached_image.size[1])
                settings.view_undistorted = requested_undistorted
    scene.apply_camera(
        context.scene,
        settings,
        scene.calibration_from_settings(settings),
    )
    settings.status = f"Project imported: {project_path.name}"
