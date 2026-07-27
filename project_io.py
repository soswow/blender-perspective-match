"""Perspective Match project and camera JSON serialization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from . import core, properties, scene

PROJECT_KIND = "perspective-match-project"
PROJECT_VERSION = 1


def _calibration_dict(settings) -> dict[str, Any]:
    calibration = scene.calibration_from_settings(settings)
    return {
        "fx": calibration.intrinsics.fx,
        "fy": calibration.intrinsics.fy,
        "cx": calibration.intrinsics.cx,
        "cy": calibration.intrinsics.cy,
        "imageWidth": calibration.intrinsics.image_width,
        "imageHeight": calibration.intrinsics.image_height,
        "hfovDeg": calibration.hfov_degrees,
        "vfovDeg": calibration.vfov_degrees,
        "gravity": [float(value) for value in -calibration.rotation_w2c[:, 2]],
        "rotation": calibration.rotation_w2c.tolist(),
        "translation": calibration.camera_center.tolist(),
        "uncertainty": {},
        "confidence": "medium",
        "divisionLambda": calibration.division_lambda,
    }


def build_project_dict(settings) -> dict[str, Any]:
    """Build a version-1 project compatible with the desktop app."""
    lines: dict[str, list[dict[str, float | str]]] = {"x": [], "y": [], "z": []}
    for index, item in enumerate(settings.lines):
        lines[item.axis].append(
            {
                "id": item.item_id or f"blender-line-{index}",
                "x1": item.x1,
                "y1": item.y1,
                "x2": item.x2,
                "y2": item.y2,
            }
        )
    surfaces = [
        {
            "id": item.item_id or f"blender-surface-{index}",
            "plane": item.plane,
            "x1": item.x1,
            "y1": item.y1,
            "x2": item.x2,
            "y2": item.y2,
            "divisions": item.divisions,
        }
        for index, item in enumerate(settings.surfaces)
    ]
    session = {
        "imagePath": settings.image_path or None,
        "imageUrl": None,
        "calibration": _calibration_dict(settings),
        "calibrationBaseline": None,
        "vpMode": int(settings.vp_mode),
        "activeAxis": settings.active_axis,
        "lines": lines,
        "autoLines": {"x": [], "y": [], "z": []},
        "suggestedVpMode": None,
        "vpClassificationReason": None,
        "surfaces": surfaces,
        "activeSurfacePlane": settings.active_surface_plane,
        "selectedSurfaceId": None,
        "originImage": list(settings.origin_image) if settings.origin_is_set else None,
        "scalePointA": list(settings.scale_point_a) if settings.scale_point_count >= 1 else None,
        "scalePointB": list(settings.scale_point_b) if settings.scale_point_count >= 2 else None,
        "measuredLength": settings.measured_length,
        "scale": settings.solved_scale,
        "workMode": "navigate",
        "lockFocal": settings.lock_focal,
        "fovSourcePlane": None,
        "guideStep": "export",
        "selectedLineId": None,
        "overlayOpacity": settings.overlay_opacity,
        "controlsOpacity": settings.controls_opacity,
        "showVpOverlay": settings.show_vp_overlay,
        "showSurfacesOverlay": settings.show_surface_overlay,
        "viewExposure": 0,
        "viewContrast": 1,
        "viewZoom": 1,
        "viewPanX": 0,
        "viewPanY": 0,
        "viewUndistorted": settings.view_undistorted,
        "undistortedImagePath": settings.undistorted_path or None,
        "undistortedOffsetX": settings.undistorted_offset_x,
        "undistortedOffsetY": settings.undistorted_offset_y,
        "projectPath": settings.project_path or None,
        "sidebarWidth": 400,
        "status": settings.status,
        "busy": False,
        "error": None,
    }
    if settings.source_session_json:
        try:
            preserved = json.loads(settings.source_session_json)
        except (TypeError, json.JSONDecodeError):
            preserved = {}
        # Preserve desktop-only fields that Blender does not own or edit.
        for key in (
            "calibrationBaseline",
            "autoLines",
            "suggestedVpMode",
            "vpClassificationReason",
            "fovSourcePlane",
            "viewExposure",
            "viewContrast",
            "viewZoom",
            "viewPanX",
            "viewPanY",
            "sidebarWidth",
        ):
            if key in preserved:
                session[key] = preserved[key]
    return {
        "kind": PROJECT_KIND,
        "version": PROJECT_VERSION,
        "savedAt": datetime.now(timezone.utc).isoformat(),
        "session": session,
    }


def save_project(settings, path: str) -> Path:
    """Write a desktop-compatible `.pmproj` file."""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".pmproj":
        output = output.with_suffix(".pmproj")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings.project_path = str(output)
    output.write_text(json.dumps(build_project_dict(settings), indent=2) + "\n", encoding="utf-8")
    settings.status = f"Project saved: {output.name}"
    return output


def _session_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") == PROJECT_KIND and isinstance(payload.get("session"), dict):
        if int(payload.get("version", 1)) > PROJECT_VERSION:
            raise ValueError("This project was created by a newer Perspective Match version")
        return payload["session"]
    if "imagePath" in payload or "lines" in payload:
        return payload
    raise ValueError("Not a Perspective Match project file")


def _validate_session(session: dict[str, Any]) -> None:
    """Validate all fields that could otherwise fail after replacing scene data."""
    if int(session.get("vpMode", 2)) not in {1, 2, 3}:
        raise ValueError("Project has an invalid VP mode")
    if session.get("activeAxis", "x") not in {"x", "y", "z"}:
        raise ValueError("Project has an invalid active axis")
    if session.get("activeSurfacePlane", "xz") not in {"xz", "yz", "yx"}:
        raise ValueError("Project has an invalid surface plane")

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

    surfaces = session.get("surfaces") or []
    if not isinstance(surfaces, list):
        raise ValueError("Project surfaces must be a list")
    for surface in surfaces:
        if not isinstance(surface, dict) or surface.get("plane") not in {"xz", "yz", "yx"}:
            raise ValueError("Project surface record is invalid")
        for coordinate in ("x1", "y1", "x2", "y2"):
            finite_number(surface.get(coordinate), f"surface {coordinate}")
        try:
            divisions = int(surface.get("divisions", 4))
        except (TypeError, ValueError) as error:
            raise ValueError("Project surface divisions must be an integer") from error
        if divisions < 1:
            raise ValueError("Project surface divisions must be positive")

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
    for key in ("scalePointA", "scalePointB"):
        point = session.get(key)
        if point is not None:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"Project {key} must contain two coordinates")
            finite_number(point[0], f"{key} x")
            finite_number(point[1], f"{key} y")
    measured_length = finite_number(session.get("measuredLength", 1.0), "measuredLength")
    if measured_length <= 0.0:
        raise ValueError("Project measuredLength must be positive")
    finite_number(session.get("scale", 1.0) or 1.0, "scale")


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
    settings.active_surface_plane = session.get("activeSurfacePlane", "xz")
    settings.lock_focal = bool(session.get("lockFocal", False))
    settings.overlay_opacity = float(session.get("overlayOpacity", 0.9))
    settings.controls_opacity = float(session.get("controlsOpacity", 1.0))
    settings.show_vp_overlay = bool(session.get("showVpOverlay", True))
    settings.show_surface_overlay = bool(session.get("showSurfacesOverlay", True))

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

    settings.surfaces.clear()
    for raw in session.get("surfaces") or []:
        if raw.get("plane") not in {"xz", "yz", "yx"}:
            continue
        item = settings.surfaces.add()
        item.item_id = str(raw.get("id") or "")
        item.plane = raw["plane"]
        item.x1 = float(raw["x1"])
        item.y1 = float(raw["y1"])
        item.x2 = float(raw["x2"])
        item.y2 = float(raw["y2"])
        item.divisions = max(1, min(64, int(raw.get("divisions", 4))))

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
    point_a = session.get("scalePointA")
    point_b = session.get("scalePointB")
    settings.scale_point_count = 0
    if point_a and len(point_a) >= 2:
        settings.scale_point_a = (float(point_a[0]), float(point_a[1]))
        settings.scale_point_count = 1
    if point_b and len(point_b) >= 2:
        settings.scale_point_b = (float(point_b[0]), float(point_b[1]))
        settings.scale_point_count = 2
    settings.measured_length = max(float(session.get("measuredLength", 1.0)), 1.0e-6)
    settings.solved_scale = float(session.get("scale", 1.0) or 1.0)
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
    scene.rebuild_surface_meshes(context)
    settings.status = f"Project loaded: {project_path.name}"


def save_camera_json(settings, path: str) -> Path:
    """Write generic intrinsics/pose JSON for external tools."""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    calibration = _calibration_dict(settings)
    rotation = np.asarray(settings.rotation_w2c, dtype=np.float64).reshape(3, 3)
    camera_center = np.asarray(settings.camera_center, dtype=np.float64)
    translation = -rotation @ camera_center
    calibration.update(
        {
            "imagePath": settings.image_path,
            "imageSize": [settings.image_width, settings.image_height],
            "K": [
                [settings.fx, 0.0, settings.cx],
                [0.0, settings.fy, settings.cy],
                [0.0, 0.0, 1.0],
            ],
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "cameraCenter": camera_center.tolist(),
            "units": "meters",
            "mmEquivFocal": settings.fx * 36.0 / max(settings.source_image_width, 1),
        }
    )
    output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    settings.status = f"Camera JSON saved: {output.name}"
    return output
