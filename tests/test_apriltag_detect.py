"""Unit tests for AprilTag landmark naming and matching helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# Load apriltags without executing the add-on's bpy-dependent __init__.
_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package
if "match_perspective.detect" not in sys.modules:
    _detect = types.ModuleType("match_perspective.detect")
    _detect.__path__ = [str(_ROOT / "detect")]
    sys.modules["match_perspective.detect"] = _detect

_spec = importlib.util.spec_from_file_location(
    "match_perspective.detect.apriltags",
    _ROOT / "detect/apriltags.py",
    submodule_search_locations=[str(_ROOT / "detect")],
)
assert _spec is not None and _spec.loader is not None
apriltag_detect = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.detect.apriltags"] = apriltag_detect
_spec.loader.exec_module(apriltag_detect)


class AprilTagNamingTests(unittest.TestCase):
    def test_landmark_name_zero_padded(self) -> None:
        self.assertEqual(apriltag_detect.landmark_name_for_tag(0), "id00-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(5), "id05-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(19), "id19-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(99), "id99-25h9")

    def test_landmark_name_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            apriltag_detect.landmark_name_for_tag(-1)
        with self.assertRaises(ValueError):
            apriltag_detect.landmark_name_for_tag(100)

    def test_find_landmark_startswith(self) -> None:
        landmarks = [
            SimpleNamespace(name="Landmark 1"),
            SimpleNamespace(name="id05-25h9"),
            SimpleNamespace(name="id07-25h9-extra"),
        ]
        found = apriltag_detect.find_landmark_for_tag(landmarks, 5)
        self.assertIs(found, landmarks[1])
        found_extra = apriltag_detect.find_landmark_for_tag(landmarks, 7)
        self.assertIs(found_extra, landmarks[2])
        self.assertIsNone(apriltag_detect.find_landmark_for_tag(landmarks, 3))


class AprilTagDetectSmokeTests(unittest.TestCase):
    """Optional OpenCV smoke: generate a 25h9 marker and detect its centre."""

    def test_detect_generated_marker_centre(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(cv2, "aruco") or not hasattr(
            cv2.aruco, "DICT_APRILTAG_25h9"
        ):
            self.skipTest("OpenCV aruco AprilTag 25h9 unavailable")

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_25h9
        )
        tag_id = 5
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 200)
        # Pad so the tag is not flush with the image edge.
        canvas = cv2.copyMakeBorder(
            marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255
        )
        detections = apriltag_detect.detect_apriltags_25h9(canvas)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].tag_id, tag_id)
        center_x, center_y = detections[0].center_xy
        # Marker is 200×200 placed at (40, 40) → centre ≈ (140, 140).
        self.assertAlmostEqual(center_x, 140.0, delta=2.0)
        self.assertAlmostEqual(center_y, 140.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
