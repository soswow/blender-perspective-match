"""Optional OpenCV probe for AprilTag and auto VP-line detection.

Core matching does not need OpenCV. Detection extras use the bundled
``opencv-contrib-python-headless`` wheel when it is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenCvCapabilities:
    """What this Blender session can do with the loaded OpenCV (if any)."""

    module: Any
    apriltag_dictionaries: tuple[str, ...]
    line_segment_detector: bool
    error: str

    @property
    def available(self) -> bool:
        return self.module is not None

    @property
    def apriltags(self) -> bool:
        """Whether at least one configured AprilTag family is available."""
        return bool(self.apriltag_dictionaries)

    @property
    def apriltag_25h9(self) -> bool:
        """Compatibility flag for the original detector family."""
        return "DICT_APRILTAG_25h9" in self.apriltag_dictionaries

    @property
    def apriltag_36h10(self) -> bool:
        return "DICT_APRILTAG_36h10" in self.apriltag_dictionaries


_cached: OpenCvCapabilities | None = None


def cached_capabilities() -> OpenCvCapabilities | None:
    """Last probe result, or ``None`` if ``cv2`` has not been imported yet.

    UI poll/draw should use this so enabling the add-on does not ``import cv2``
    on the main thread (OpenCV's native load is seconds).
    """
    return _cached


def capabilities(refresh: bool = False) -> OpenCvCapabilities:
    """Probe OpenCV once; later calls reuse the result unless ``refresh``."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    _cached = _probe()
    return _cached


def load_warning() -> str:
    """Info-editor line when detection extras are unavailable; else empty."""
    caps = capabilities()
    hidden: list[str] = []
    if not caps.line_segment_detector:
        hidden.append("Detect VP Lines")
    if not caps.apriltags:
        hidden.append("Find AprilTags")
    if not hidden:
        return ""
    if caps.module is None:
        reason = "OpenCV is not available"
    else:
        reason = "OpenCV is missing aruco/LSD (need opencv-contrib-python-headless)"
    names = " and ".join(hidden)
    return f"Perspective Match: {reason} — {names} hidden"


def _probe() -> OpenCvCapabilities:
    try:
        import cv2
    except Exception as error:
        return OpenCvCapabilities(
            module=None,
            apriltag_dictionaries=(),
            line_segment_detector=False,
            error=str(error) or error.__class__.__name__,
        )

    available_apriltag_dictionaries: tuple[str, ...] = ()
    try:
        from .apriltags import APRILTAG_FAMILIES

        if hasattr(cv2, "aruco"):
            available_apriltag_dictionaries = tuple(
                family.opencv_name
                for family in APRILTAG_FAMILIES
                if hasattr(cv2.aruco, family.opencv_name)
            )
    except Exception:
        available_apriltag_dictionaries = ()

    has_lsd = False
    try:
        has_lsd = hasattr(cv2, "createLineSegmentDetector")
    except Exception:
        has_lsd = False

    missing: list[str] = []
    if not available_apriltag_dictionaries:
        missing.append("aruco AprilTag dictionaries")
    if not has_lsd:
        missing.append("Line Segment Detector")
    error = ""
    if missing:
        error = "OpenCV loaded without " + " and ".join(missing)

    return OpenCvCapabilities(
        module=cv2,
        apriltag_dictionaries=available_apriltag_dictionaries,
        line_segment_detector=has_lsd,
        error=error,
    )
