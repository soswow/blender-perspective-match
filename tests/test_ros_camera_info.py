"""Tests for ROS camera_info YAML parsing (no Blender runtime)."""

from __future__ import annotations

import unittest
from pathlib import Path

from match_perspective.core import ros_camera_info

_SAMPLE = """\
image_width: 2160
image_height: 3840
camera_name: pixel_1x
camera_matrix:
  rows: 3
  cols: 3
  data: [2780.598265886072, 0.0, 1060.0920399982783, 0.0, 2795.470643590964, 1900.7906308873787, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.016138166817855664, 0.0, 0.0, 0.0, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [2780.598265886072, 0.0, 1060.0920399982783, 0.0, 0.0, 2795.470643590964, 1900.7906308873787, 0.0, 0.0, 0.0, 1.0, 0.0]
fitzgibbon_lambda: -0.00058388
"""


class RosCameraInfoTests(unittest.TestCase):
    def test_parse_sample(self) -> None:
        info = ros_camera_info.parse_ros_camera_info_yaml(_SAMPLE)
        self.assertEqual(info.image_width, 2160)
        self.assertEqual(info.image_height, 3840)
        self.assertEqual(info.camera_name, "pixel_1x")
        self.assertAlmostEqual(info.fx, 2780.598265886072)
        self.assertAlmostEqual(info.fy, 2795.470643590964)
        self.assertAlmostEqual(info.cx, 1060.0920399982783)
        self.assertAlmostEqual(info.cy, 1900.7906308873787)
        self.assertEqual(info.distortion_model, "plumb_bob")
        self.assertAlmostEqual(info.distortion_coefficients[0], 0.016138166817855664)
        self.assertAlmostEqual(info.fitzgibbon_lambda, -0.00058388)

    def test_parse_without_lambda(self) -> None:
        text = "\n".join(
            line
            for line in _SAMPLE.splitlines()
            if not line.startswith("fitzgibbon_lambda")
        )
        info = ros_camera_info.parse_ros_camera_info_yaml(text)
        self.assertIsNone(info.fitzgibbon_lambda)

    def test_parse_real_file_when_present(self) -> None:
        path = Path(
            "/Users/sasha/hobby/camera-calibration-checkerboard/"
            "output/intrinsics-0.5x-my-camera-full.yaml"
        )
        if not path.is_file():
            self.skipTest("sample YAML not on this machine")
        info = ros_camera_info.parse_ros_camera_info_yaml(path.read_text(encoding="utf-8"))
        self.assertEqual(info.image_width, 2160)
        self.assertGreater(info.fx, 0.0)
        self.assertIsNotNone(info.fitzgibbon_lambda)
        self.assertAlmostEqual(info.fitzgibbon_lambda, 1.4618140765392445e-05)

    def test_scale_intrinsics(self) -> None:
        info = ros_camera_info.parse_ros_camera_info_yaml(_SAMPLE)
        fx, fy, cx, cy, scaled = ros_camera_info.scale_intrinsics_to_image(
            info, 1080, 1920
        )
        self.assertTrue(scaled)
        self.assertAlmostEqual(fx, info.fx * 0.5)
        self.assertAlmostEqual(fy, info.fy * 0.5)
        self.assertAlmostEqual(cx, info.cx * 0.5)
        self.assertAlmostEqual(cy, info.cy * 0.5)
        self.assertEqual(scaled, "scaled")

    def test_interpret_plumb_bob(self) -> None:
        info = ros_camera_info.parse_ros_camera_info_yaml(_SAMPLE)
        imported = ros_camera_info.interpret_distortion(info)
        self.assertEqual(imported.skip_reason, "")
        self.assertAlmostEqual(imported.brown_conrady[0], 0.016138166817855664)
        self.assertEqual(len(imported.brown_conrady), 8)

    def test_interpret_zero_d_is_identity(self) -> None:
        text = _SAMPLE.replace(
            "data: [0.016138166817855664, 0.0, 0.0, 0.0, 0.0]",
            "data: [0.0, 0.0, 0.0, 0.0, 0.0]",
        )
        info = ros_camera_info.parse_ros_camera_info_yaml(text)
        imported = ros_camera_info.interpret_distortion(info)
        self.assertEqual(imported.brown_conrady, ())
        self.assertEqual(imported.skip_reason, "")

    def test_interpret_equidistant_skipped(self) -> None:
        text = _SAMPLE.replace("distortion_model: plumb_bob", "distortion_model: equidistant")
        info = ros_camera_info.parse_ros_camera_info_yaml(text)
        imported = ros_camera_info.interpret_distortion(info)
        self.assertEqual(imported.brown_conrady, ())
        self.assertIn("equidistant", imported.skip_reason)

    def test_interpret_real_no_crop_yaml_when_present(self) -> None:
        path = Path(
            "/Users/sasha/hobby/camera-calibration-checkerboard/"
            "output/intrinsics-0.5x-no-crop-simple.yaml"
        )
        if not path.is_file():
            self.skipTest("sample YAML not on this machine")
        info = ros_camera_info.parse_ros_camera_info_yaml(path.read_text(encoding="utf-8"))
        imported = ros_camera_info.interpret_distortion(info)
        self.assertEqual(info.distortion_model, "plumb_bob")
        self.assertAlmostEqual(imported.brown_conrady[0], -0.0014045245854740534)
        self.assertAlmostEqual(imported.brown_conrady[1], 0.0004527363803767526)
        self.assertEqual(imported.skip_reason, "")

    def test_no_scale_when_sizes_match(self) -> None:
        info = ros_camera_info.parse_ros_camera_info_yaml(_SAMPLE)
        fx, fy, cx, cy, scaled = ros_camera_info.scale_intrinsics_to_image(
            info, 2160, 3840
        )
        self.assertEqual(scaled, "")
        self.assertEqual((fx, fy, cx, cy), (info.fx, info.fy, info.cx, info.cy))

    def test_portrait_yaml_on_landscape_swaps_axes(self) -> None:
        """Same pixel count, rotated 90°: keep fx/fy, do not scale by 4/3."""
        info = ros_camera_info.parse_ros_camera_info_yaml(_SAMPLE)
        self.assertEqual((info.image_width, info.image_height), (2160, 3840))
        fx, fy, cx, cy, kind = ros_camera_info.scale_intrinsics_to_image(
            info, 3840, 2160
        )
        self.assertEqual(kind, "rotated")
        self.assertAlmostEqual(fx, info.fy)
        self.assertAlmostEqual(fy, info.fx)
        self.assertAlmostEqual(cx, info.cy)
        self.assertAlmostEqual(cy, info.cx)
        self.assertNotAlmostEqual(fx, info.fx * (3840 / 2160))

    def test_pixel_charuco_yaml_landscape_keeps_focal(self) -> None:
        path = Path(
            "/Users/sasha/hobby/camera-calibration/output/"
            "pixel-1x-no-crop-charuco-full.yaml"
        )
        if not path.is_file():
            self.skipTest("pixel charuco YAML not on this machine")
        info = ros_camera_info.parse_ros_camera_info_yaml(
            path.read_text(encoding="utf-8")
        )
        self.assertEqual((info.image_width, info.image_height), (3000, 4000))
        fx, fy, cx, cy, kind = ros_camera_info.scale_intrinsics_to_image(
            info, 4000, 3000
        )
        self.assertEqual(kind, "rotated")
        self.assertAlmostEqual(fx, info.fy)
        self.assertAlmostEqual(fy, info.fx)
        self.assertAlmostEqual(cx, info.cy)
        self.assertAlmostEqual(cy, info.cx)
        self.assertAlmostEqual(fx, 2867.7920482469162, places=4)


if __name__ == "__main__":
    unittest.main()
