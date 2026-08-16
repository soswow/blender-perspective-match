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
    apriltag_25h9: bool
    line_segment_detector: bool
    error: str

    @property
    def available(self) -> bool:
        return self.module is not None


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
    if not caps.apriltag_25h9:
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
            apriltag_25h9=False,
            line_segment_detector=False,
            error=str(error) or error.__class__.__name__,
        )

    has_apriltag = False
    try:
        has_apriltag = hasattr(cv2, "aruco") and hasattr(
            cv2.aruco, "DICT_APRILTAG_25h9"
        )
    except Exception:
        has_apriltag = False

    has_lsd = False
    try:
        has_lsd = hasattr(cv2, "createLineSegmentDetector")
    except Exception:
        has_lsd = False

    missing: list[str] = []
    if not has_apriltag:
        missing.append("aruco AprilTag 25h9")
    if not has_lsd:
        missing.append("Line Segment Detector")
    error = ""
    if missing:
        error = "OpenCV loaded without " + " and ".join(missing)

    return OpenCvCapabilities(
        module=cv2,
        apriltag_25h9=has_apriltag,
        line_segment_detector=has_lsd,
        error=error,
    )
