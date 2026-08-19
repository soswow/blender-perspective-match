"""Detect AprilTag fiducials in a match still and map them to sync landmarks.

Hardcoded family: AprilTag 25h9 (same as ``tools/print-apriltags``).
Landmark names use ``idNN-25h9`` (NN = 00–99). OpenCV with aruco is optional
(bundled wheel ``opencv-contrib-python-headless`` in ``./wheels/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

# Must match tools/print-apriltags default dictionary.
APRILTAG_DICTIONARY = "apriltag-25h9"
APRILTAG_FAMILY_SUFFIX = "25h9"
_OPENCV_DICT_NAME = "DICT_APRILTAG_25h9"


@dataclass(frozen=True)
class DetectedTag:
    """One AprilTag hit in source-image pixels (top-left origin)."""

    tag_id: int
    center_xy: tuple[float, float]
    corners_xy: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class AprilTagAssignResult:
    """Counts from applying detections onto the landmark list."""

    detected: int
    updated: int
    created: int
    skipped: int


class AprilTagDependencyError(RuntimeError):
    """Raised when OpenCV / aruco is unavailable for detection."""


def landmark_name_for_tag(tag_id: int) -> str:
    """Canonical landmark name for a 25h9 tag id (zero-padded 00–99)."""
    if tag_id < 0 or tag_id > 99:
        raise ValueError(f"AprilTag id must be 0–99, got {tag_id}")
    return f"id{tag_id:02d}-{APRILTAG_FAMILY_SUFFIX}"


def find_landmark_for_tag(landmarks, tag_id: int):
    """Return the first landmark whose name starts with ``idNN-25h9``."""
    prefix = landmark_name_for_tag(tag_id)
    for landmark in landmarks:
        name = getattr(landmark, "name", "") or ""
        if name.startswith(prefix):
            return landmark
    return None


def _import_cv2():
    """Import OpenCV with aruco; raise AprilTagDependencyError if missing."""
    from .opencv import capabilities

    caps = capabilities()
    if caps.module is None or not caps.apriltag_25h9:
        detail = caps.error or "OpenCV with aruco is not available"
        raise AprilTagDependencyError(
            "AprilTag detection needs the bundled OpenCV wheel. "
            "Run ./scripts/fetch-wheels.sh, then disable and re-enable Perspective Match "
            f"(or install a fresh platform zip from ./scripts/build-extension.sh). ({detail})"
        )
    return caps.module


def _aruco_dictionary(cv2_module):
    """Return the hardcoded AprilTag 25h9 dictionary."""
    if not hasattr(cv2_module.aruco, _OPENCV_DICT_NAME):
        raise AprilTagDependencyError(
            f"This OpenCV build has no {_OPENCV_DICT_NAME}"
        )
    return cv2_module.aruco.getPredefinedDictionary(
        getattr(cv2_module.aruco, _OPENCV_DICT_NAME)
    )


def _gray_from_path(cv2_module, image_path: str) -> np.ndarray | None:
    """Load a grayscale image from disk; None if the path cannot be read."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        return None
    bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2GRAY)


def _gray_from_blender_image(cv2_module, image) -> np.ndarray:
    """Convert a Blender Image to grayscale uint8 (top-left origin)."""
    from ..scene import distortion

    rgba = distortion._image_pixels_top_left(image)
    rgb_u8 = np.clip(np.round(rgba[:, :, :3] * 255.0), 0, 255).astype(np.uint8)
    return cv2_module.cvtColor(rgb_u8, cv2_module.COLOR_RGB2GRAY)


def _load_detection_gray(cv2_module, settings) -> np.ndarray:
    """Prefer the on-disk still; fall back to Blender's pixel buffer."""
    image_path = getattr(settings, "image_path", "") or ""
    if image_path:
        from_disk = _gray_from_path(cv2_module, image_path)
        if from_disk is not None:
            return from_disk
    image = getattr(settings, "image", None)
    if image is None:
        raise ValueError("Active match has no reference image")
    return _gray_from_blender_image(cv2_module, image)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _projective_quad_center(points_xy: np.ndarray) -> np.ndarray:
    """Return the intersection of a quadrilateral's corner diagonals.

    OpenCV orders marker corners around the perimeter. The diagonal
    intersection is the projection of the tag's physical center; averaging
    the four image-space corners is only correct for affine projections.
    """
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    first_direction = points[2] - points[0]
    second_direction = points[3] - points[1]
    denominator = _cross_2d(first_direction, second_direction)
    if abs(denominator) < 1.0e-12:
        # A decoded marker should never be degenerate, but retain a finite
        # fallback if a detector supplies pathological corners.
        return points.mean(axis=0)
    distance = _cross_2d(points[1] - points[0], second_direction) / denominator
    return points[0] + distance * first_direction


def detect_apriltags_25h9(gray: np.ndarray) -> list[DetectedTag]:
    """Run ArUco AprilTag 25h9 detection; return tag id + center pixels."""
    cv2 = _import_cv2()
    if gray.ndim != 2:
        raise ValueError("Detection expects a single-channel grayscale image")
    dictionary = _aruco_dictionary(cv2)
    parameters = cv2.aruco.DetectorParameters()
    # Stock defaults drop fairly small tags on high-res stills (perimeter ≥ 3%
    # of max(W,H), canonical side ≥ 32 px). Halve both so mid-distance printed
    # markers still decode; no UI knobs yet.
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minSideLengthCanonicalImg = 8
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners_list, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    detections: list[DetectedTag] = []
    for corners, tag_id_array in zip(corners_list, ids):
        tag_id = int(np.asarray(tag_id_array).reshape(-1)[0])
        # Skip ids outside the naming scheme (print sheets use 0–19 typically).
        if tag_id < 0 or tag_id > 99:
            continue
        # corners shape: (1, 4, 2), ordered around the marker perimeter.
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        center = _projective_quad_center(points)
        detections.append(
            DetectedTag(
                tag_id=tag_id,
                center_xy=(float(center[0]), float(center[1])),
                corners_xy=tuple(
                    (float(point[0]), float(point[1])) for point in points
                ),
            )
        )
    # Stable order for UI / status messages.
    detections.sort(key=lambda item: item.tag_id)
    return detections


def _correct_centers_for_lens(
    detections: list[DetectedTag], settings
) -> list[DetectedTag]:
    """Recompute tag centers in ideal space when the source has distortion."""
    # Lens distortion breaks projective straight lines. Find each center after
    # mapping its four corners into the ideal pinhole image, then map the center
    # back to source-image storage coordinates. Overlay and sync will apply the
    # normal source→ideal mapping exactly once later.
    from .. import core

    division_lambda = float(getattr(settings, "division_lambda", 0.0))
    brown_conrady = tuple(getattr(settings, "brown_conrady", ()))
    if not core.has_lens_distortion(
        division_lambda,
        brown_conrady,
        threshold=1.0e-15,
    ):
        return detections

    fx = max(float(getattr(settings, "fx", 0.0)), 1.0e-6)
    fy = max(float(getattr(settings, "fy", 0.0)), 1.0e-6)
    cx = float(getattr(settings, "cx", 0.0))
    cy = float(getattr(settings, "cy", 0.0))
    corrected: list[DetectedTag] = []
    for detection in detections:
        if len(detection.corners_xy) != 4:
            corrected.append(detection)
            continue
        ideal_corners = core.undistort_points(
            np.asarray(detection.corners_xy, dtype=np.float64),
            fx,
            fy,
            cx,
            cy,
            division_lambda,
            brown_conrady,
        )
        ideal_center = _projective_quad_center(ideal_corners)
        storage_center = core.distort_points(
            ideal_center.reshape(1, 2),
            fx,
            fy,
            cx,
            cy,
            division_lambda,
            brown_conrady,
        )[0]
        corrected.append(
            DetectedTag(
                tag_id=detection.tag_id,
                center_xy=(float(storage_center[0]), float(storage_center[1])),
                corners_xy=detection.corners_xy,
            )
        )
    return corrected


def detect_apriltags_in_session(settings) -> list[DetectedTag]:
    """Detect AprilTags in the active match's reference still."""
    cv2 = _import_cv2()
    gray = _load_detection_gray(cv2, settings)
    detections = detect_apriltags_25h9(gray)
    return _correct_centers_for_lens(detections, settings)


def _set_point_observation(
    landmark,
    root,
    image_point: tuple[float, float],
    confidence: str,
) -> None:
    """Write / overwrite a POINT observation for ``root`` without activating it."""
    from .. import scene

    observation = scene.observation_for_match(landmark, root)
    if observation is None:
        observation = landmark.observations.add()
        observation.match_root = root
    observation.x, observation.y = float(image_point[0]), float(image_point[1])
    observation.is_set = True
    observation.confidence = confidence


def apply_apriltag_detections(
    context,
    detections: list[DetectedTag],
) -> AprilTagAssignResult:
    """Assign detections to landmarks on the active match (create if missing)."""
    from .. import properties

    space = properties.workspace(context)
    root = properties.active_root(context)
    if root is None:
        raise ValueError("Activate a match camera first")

    confidence = space.landmark_pick_confidence
    updated = 0
    created = 0
    skipped = 0

    for detection in detections:
        landmark = find_landmark_for_tag(space.landmarks, detection.tag_id)
        if landmark is not None:
            if getattr(landmark, "kind", "POINT") == "LINE":
                # Tag centres are points; leave line landmarks alone.
                skipped += 1
                continue
            _set_point_observation(landmark, root, detection.center_xy, confidence)
            updated += 1
            continue

        landmark = space.landmarks.add()
        landmark.item_id = f"landmark-{uuid4().hex}"
        landmark.kind = "POINT"
        landmark.name = landmark_name_for_tag(detection.tag_id)
        _set_point_observation(landmark, root, detection.center_xy, confidence)
        created += 1

    if created:
        properties.ensure_landmark_creation_indices(space)
        space.active_landmark_index = len(space.landmarks) - 1

    properties.tag_viewport_redraw(context)
    return AprilTagAssignResult(
        detected=len(detections),
        updated=updated,
        created=created,
        skipped=skipped,
    )


def find_and_assign_apriltags(context) -> AprilTagAssignResult:
    """Detect 25h9 tags in the active still and map them onto landmarks."""
    from .. import properties

    settings = properties.active_session(context)
    if settings is None or settings.image is None:
        raise ValueError("Load a reference image on the active match first")
    detections = detect_apriltags_in_session(settings)
    return apply_apriltag_detections(context, detections)
