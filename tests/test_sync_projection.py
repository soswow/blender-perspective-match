"""Pinhole projection of infinite Sync lines."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package

from match_perspective import core
from match_perspective.core.sync.projection import _project_world_line_to_image
from match_perspective.core.sync.types import SimilarityTransform


def _camera(rotation: np.ndarray, center: np.ndarray) -> core.Calibration:
    return core.Calibration(
        core.CameraIntrinsics(810.0, 760.0, 480.0, 360.0, 960, 720),
        rotation,
        center,
    )


def _oracle_pixel(
    point_shared: np.ndarray,
    camera: core.Calibration,
    similarity: SimilarityTransform,
) -> np.ndarray:
    private = similarity.rotation.T @ (point_shared - similarity.translation) / similarity.scale
    xyz = camera.rotation_w2c @ (private - camera.camera_center)
    if xyz[2] <= 1e-8:
        raise ValueError("Oracle point must be in front")
    intrinsics = camera.intrinsics
    return np.array(
        (intrinsics.fx * xyz[0] / xyz[2] + intrinsics.cx,
         intrinsics.fy * xyz[1] / xyz[2] + intrinsics.cy),
        dtype=float,
    )


class InfiniteLineProjectionTests(unittest.TestCase):
    def _assert_same_infinite_line(
        self,
        point: np.ndarray,
        direction: np.ndarray,
        camera: core.Calibration,
        similarity: SimilarityTransform,
        *,
        sample_offsets: tuple[float, float],
    ) -> None:
        point_before = point.copy()
        direction_before = direction.copy()
        original = _project_world_line_to_image(point, direction, camera, similarity)
        self.assertIsNotNone(original)
        assert original is not None
        self.assertAlmostEqual(float(np.linalg.norm(original[:2])), 1.0, places=11)
        for offset in sample_offsets:
            uv = _oracle_pixel(point + offset * direction, camera, similarity)
            self.assertLess(abs(float(original @ np.r_[uv, 1.0])), 1e-8)
        unit = direction / np.linalg.norm(direction)
        for distance in (-9000.0, -100.0, 100.0, 9000.0):
            shifted = _project_world_line_to_image(
                point + distance * unit, direction, camera, similarity
            )
            self.assertIsNotNone(shifted, f"same infinite line shifted {distance}")
            assert shifted is not None
            self.assertLess(
                min(np.linalg.norm(shifted - original), np.linalg.norm(shifted + original)),
                1e-8,
            )
        np.testing.assert_array_equal(point, point_before)
        np.testing.assert_array_equal(direction, direction_before)

    def test_axis_line_remains_visible_after_midpoint_slides(self) -> None:
        # The camera looks down; a large positive Z slide hides every old
        # finite sample although the infinite line still crosses its view.
        camera = _camera(
            np.diag((1.0, -1.0, -1.0)),
            np.array((0.0, 0.0, 3.0)),
        )
        self._assert_same_infinite_line(
            np.array((1.0, 0.2, 0.5)),
            np.array((0.0, 0.0, 2.4)),
            camera,
            SimilarityTransform(),
            sample_offsets=(-0.2, 0.2),
        )

    def test_nonaxis_line_with_similarity_and_nonunit_direction(self) -> None:
        angle = 0.37
        rotation = np.array(
            ((np.cos(angle), -np.sin(angle), 0.0),
             (np.sin(angle), np.cos(angle), 0.0),
             (0.0, 0.0, 1.0))
        )
        similarity = SimilarityTransform(1.7, rotation, np.array((0.3, -0.2, 0.5)))
        private_point = np.array((1.0, 0.3, 4.0))
        private_direction = np.array((0.2, 0.4, 1.0))
        tilt = 0.23
        camera_rotation = np.array(
            ((1.0, 0.0, 0.0),
             (0.0, np.cos(tilt), -np.sin(tilt)),
             (0.0, np.sin(tilt), np.cos(tilt)))
        )
        self._assert_same_infinite_line(
            similarity.transform_point(private_point),
            1.7 * rotation @ private_direction,
            _camera(camera_rotation, np.array((0.2, -0.1, 0.3))),
            similarity,
            sample_offsets=(-0.5, 0.5),
        )

    def test_distorted_camera_keeps_sampled_projection(self) -> None:
        camera = _camera(np.eye(3), np.zeros(3))
        camera.division_lambda = 0.012
        line = _project_world_line_to_image(
            np.array((1.0, 0.3, 4.0)),
            np.array((0.2, 0.4, 1.0)),
            camera,
            SimilarityTransform(),
        )
        self.assertIsNotNone(line)
        np.testing.assert_allclose(
            line,
            np.array((-0.986859872369904, -0.16157844010280759, 741.0821388365509)),
            atol=1e-10,
            rtol=0.0,
        )

    def test_camera_crossing_and_behind_parallel_lines_are_refused(self) -> None:
        camera = _camera(np.eye(3), np.zeros(3))
        similarity = SimilarityTransform()
        self.assertIsNone(_project_world_line_to_image(
            camera.camera_center.copy(), np.array((1.0, 0.0, 1.0)), camera, similarity
        ))
        self.assertIsNone(_project_world_line_to_image(
            np.array((1.0, 0.2, -2.0)), np.array((1.0, 0.5, 0.0)), camera, similarity
        ))
        front = _project_world_line_to_image(
            np.array((1.0, 0.2, 2.0)), np.array((1.0, 0.5, 0.0)), camera, similarity
        )
        self.assertIsNotNone(front)
        # A nonparallel infinite line can cross from behind to in front.
        crossing = _project_world_line_to_image(
            np.array((1.0, 0.2, -100.0)), np.array((0.2, 0.4, 1.0)), camera, similarity
        )
        self.assertIsNotNone(crossing)


if __name__ == "__main__":
    unittest.main()
