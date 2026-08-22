"""Unit tests for AprilTag landmark naming and matching helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

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
)
assert _spec is not None and _spec.loader is not None
apriltag_detect = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.detect.apriltags"] = apriltag_detect
_spec.loader.exec_module(apriltag_detect)

from match_perspective import core


class AprilTagNamingTests(unittest.TestCase):
    def test_landmark_name_zero_padded(self) -> None:
        self.assertEqual(apriltag_detect.landmark_name_for_tag(0), "id000-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(5), "id005-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(19), "id019-25h9")
        self.assertEqual(apriltag_detect.landmark_name_for_tag(99), "id099-25h9")
        self.assertEqual(apriltag_detect.format_tag_id(2319), "2319")

    def test_landmark_name_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            apriltag_detect.landmark_name_for_tag(-1)
        with self.assertRaises(ValueError):
            apriltag_detect.landmark_name_for_tag(100)

    def test_landmark_name_distinguishes_family_and_allows_large_36h10_id(self) -> None:
        self.assertEqual(
            apriltag_detect.landmark_name_for_tag(5, "36h10"),
            "id005-36h10",
        )
        self.assertEqual(
            apriltag_detect.landmark_name_for_tag(2319, "36h10"),
            "id2319-36h10",
        )
        with self.assertRaises(ValueError):
            apriltag_detect.landmark_name_for_tag(2320, "36h10")

    def test_find_landmark_startswith(self) -> None:
        landmarks = [
            SimpleNamespace(name="Landmark 1"),
            SimpleNamespace(name="id005-25h9"),
            SimpleNamespace(name="id007-25h9-extra"),
        ]
        found = apriltag_detect.find_landmark_for_tag(landmarks, 5)
        self.assertIs(found, landmarks[1])
        found_extra = apriltag_detect.find_landmark_for_tag(landmarks, 7)
        self.assertIs(found_extra, landmarks[2])
        self.assertIsNone(apriltag_detect.find_landmark_for_tag(landmarks, 3))

    def test_find_landmark_accepts_historical_two_digit_names(self) -> None:
        landmarks = [
            SimpleNamespace(name="id05-25h9"),
            SimpleNamespace(name="id07-25h9-extra"),
        ]
        found = apriltag_detect.find_landmark_for_tag(landmarks, 5)
        self.assertIs(found, landmarks[0])
        found_extra = apriltag_detect.find_landmark_for_tag(landmarks, 7)
        self.assertIs(found_extra, landmarks[1])

    def test_canonical_padding_rewrites_historical_names(self) -> None:
        self.assertEqual(
            apriltag_detect._with_canonical_tag_padding("id05-25h9", 5),
            "id005-25h9",
        )
        self.assertEqual(
            apriltag_detect._with_canonical_tag_padding("id07-25h9-extra", 7),
            "id007-25h9-extra",
        )
        self.assertEqual(
            apriltag_detect._with_canonical_tag_padding("id005-25h9", 5),
            "id005-25h9",
        )

    def test_find_landmark_does_not_cross_families(self) -> None:
        landmarks = [
            SimpleNamespace(name="id005-25h9"),
            SimpleNamespace(name="id005-36h10"),
        ]
        found = apriltag_detect.find_landmark_for_tag(landmarks, 5, "36h10")
        self.assertIs(found, landmarks[1])


class AprilTagCenterTests(unittest.TestCase):
    def test_projective_center_is_diagonal_intersection(self) -> None:
        # The mean is (2, 1), but perspective maps the square's physical center
        # to the diagonal intersection (2, 4/3).
        trapezoid = np.array(
            ((0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0)),
            dtype=np.float64,
        )
        center = apriltag_detect._projective_quad_center(trapezoid)
        self.assertTrue(np.allclose(center, (2.0, 4.0 / 3.0), atol=1.0e-9))

    def test_degenerate_corners_have_finite_fallback(self) -> None:
        collapsed = np.array(
            ((1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)),
            dtype=np.float64,
        )
        center = apriltag_detect._projective_quad_center(collapsed)
        self.assertTrue(np.allclose(center, (1.0, 2.0), atol=1.0e-9))

    def test_distorted_corners_are_centered_in_ideal_space(self) -> None:
        ideal_corners = np.array(
            (
                (240.0, 180.0),
                (760.0, 210.0),
                (690.0, 640.0),
                (300.0, 590.0),
            ),
            dtype=np.float64,
        )
        fx, fy, cx, cy = 780.0, 790.0, 500.0, 400.0
        coefficients = (-0.18, 0.035, 0.002, -0.001, 0.0, 0.0, 0.0, 0.0)
        observed_corners = core.distort_points(
            ideal_corners,
            fx,
            fy,
            cx,
            cy,
            0.0,
            coefficients,
        )
        detection = apriltag_detect.DetectedTag(
            tag_id=6,
            center_xy=tuple(observed_corners.mean(axis=0)),
            corners_xy=tuple(tuple(point) for point in observed_corners),
        )
        settings = SimpleNamespace(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            division_lambda=0.0,
            brown_conrady=coefficients,
        )

        corrected = apriltag_detect._correct_centers_for_lens(
            [detection], settings
        )[0]
        recovered_ideal = core.undistort_points(
            np.asarray([corrected.center_xy]),
            fx,
            fy,
            cx,
            cy,
            0.0,
            coefficients,
        )[0]
        expected_ideal = apriltag_detect._projective_quad_center(ideal_corners)
        self.assertTrue(np.allclose(recovered_ideal, expected_ideal, atol=1.0e-3))


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


class AprilTagMultiFamilyTests(unittest.TestCase):
    def test_detect_scans_every_configured_family(self) -> None:
        class FakeAruco:
            DICT_APRILTAG_25h9 = 25
            DICT_APRILTAG_36h10 = 36

            @staticmethod
            def getPredefinedDictionary(dictionary_id):
                return dictionary_id

            @staticmethod
            def DetectorParameters():
                return SimpleNamespace()

            @staticmethod
            def ArucoDetector(dictionary, _parameters):
                tag_id = 5 if dictionary == 25 else 120
                offset = 0.0 if dictionary == 25 else 20.0
                corners = np.array(
                    [[[offset, 0.0], [offset + 10.0, 0.0],
                      [offset + 10.0, 10.0], [offset, 10.0]]],
                    dtype=np.float32,
                )
                return SimpleNamespace(
                    detectMarkers=lambda _gray: (
                        [corners],
                        np.array([[tag_id]], dtype=np.int32),
                        [],
                    )
                )

        fake_cv2 = SimpleNamespace(aruco=FakeAruco())
        with mock.patch.object(
            apriltag_detect,
            "_import_cv2",
            return_value=fake_cv2,
        ):
            detections = apriltag_detect.detect_apriltags(
                np.zeros((40, 40), dtype=np.uint8)
            )

        self.assertEqual(
            [(item.tag_id, item.family_suffix) for item in detections],
            [(5, "25h9"), (120, "36h10")],
        )


if __name__ == "__main__":
    unittest.main()
