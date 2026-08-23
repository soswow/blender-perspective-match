"""Unit tests for AprilTag-like landmark click snap."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
    "match_perspective.detect.tag_snap",
    _ROOT / "detect/tag_snap.py",
    submodule_search_locations=[str(_ROOT / "detect")],
)
assert _spec is not None and _spec.loader is not None
tag_snap = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.detect.tag_snap"] = tag_snap
_spec.loader.exec_module(tag_snap)


def _paint_axis_aligned_tag(
    *,
    width: int = 160,
    height: int = 120,
    left: int = 50,
    top: int = 35,
    inner: int = 28,
    border: int = 4,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Cardboard plate with a dark square, white bits, and a white quiet zone."""
    gray = np.full((height, width), 0.55, dtype=np.float64)
    outer = inner + 2 * border
    gray[top : top + outer, left : left + outer] = 0.95
    gray[top + border : top + border + inner, left + border : left + border + inner] = 0.12
    # Interior white data bits (holes inside the black body).
    bit = max(inner // 7, 2)
    gray[
        top + border + bit : top + border + 2 * bit,
        left + border + bit : left + border + 2 * bit,
    ] = 0.92
    gray[
        top + border + 3 * bit : top + border + 5 * bit,
        left + border + 4 * bit : left + border + 6 * bit,
    ] = 0.92
    center = (
        left + border + (inner - 1) / 2.0,
        top + border + (inner - 1) / 2.0,
    )
    return gray, center


def _paint_parallelogram_tag() -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """Oblique tag whose true centre is the diagonal intersection, not the mean."""
    gray = np.full((140, 180), 0.58, dtype=np.float64)
    # Inner black parallelogram (clockwise / CCW around the body).
    inner = np.array(
        (
            (40.0, 40.0),
            (100.0, 36.0),
            (118.0, 78.0),
            (58.0, 82.0),
        ),
        dtype=np.float64,
    )
    # Expand ~4 px along vertex normals for the white quiet zone.
    centroid = inner.mean(axis=0)
    outer = centroid + 1.35 * (inner - centroid)
    _fill_quad(gray, outer, 0.94)
    _fill_quad(gray, inner, 0.14)
    # A couple of white blotches inside.
    _fill_quad(
        gray,
        np.array(
            (
                (58.0, 50.0),
                (70.0, 49.0),
                (72.0, 58.0),
                (60.0, 59.0),
            ),
            dtype=np.float64,
        ),
        0.9,
    )
    center = tag_snap.projective_quad_center(inner)
    return gray, (float(center[0]), float(center[1])), inner


def _fill_quad(gray: np.ndarray, corners: np.ndarray, value: float) -> None:
    """Rasterize a convex quad by testing edge half-planes."""
    ordered = tag_snap._order_corners(np.asarray(corners, dtype=np.float64))
    height, width = gray.shape
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    x0 = int(max(0, np.floor(xs.min())))
    x1 = int(min(width, np.ceil(xs.max()) + 1))
    y0 = int(max(0, np.floor(ys.min())))
    y1 = int(min(height, np.ceil(ys.max()) + 1))
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).astype(np.float64)
    inside = np.ones(points.shape[0], dtype=bool)
    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        edge = end - start
        inside &= (
            edge[0] * (points[:, 1] - start[1]) - edge[1] * (points[:, 0] - start[0])
        ) >= 0.0
    region = inside.reshape(grid_y.shape)
    gray[y0:y1, x0:x1][region] = value


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


class TagSnapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _import_cv2() is None:
            raise unittest.SkipTest("OpenCV not installed")

    def test_snaps_offset_click_to_square_center(self) -> None:
        gray, center = _paint_axis_aligned_tag()
        click = (center[0] + 4.0, center[1] - 3.0)
        result = tag_snap.snap_click_to_tag_center(gray, click)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center_xy[0], center[0], delta=1.5)
        self.assertAlmostEqual(result.center_xy[1], center[1], delta=1.5)
        self.assertGreater(result.mean_shift_px, 2.0)

    def test_click_on_interior_white_bit_still_snaps(self) -> None:
        gray, center = _paint_axis_aligned_tag()
        # The first white bit is near top-left of the black body.
        click = (50 + 4 + 5, 35 + 4 + 5)
        result = tag_snap.snap_click_to_tag_center(gray, click)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center_xy[0], center[0], delta=2.0)
        self.assertAlmostEqual(result.center_xy[1], center[1], delta=2.0)

    def test_parallelogram_uses_diagonal_intersection(self) -> None:
        gray, center, inner = _paint_parallelogram_tag()
        mean = tuple(inner.mean(axis=0))
        click = (center[0] + 2.5, center[1] + 2.0)
        result = tag_snap.snap_click_to_tag_center(gray, click)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center_xy[0], center[0], delta=2.5)
        self.assertAlmostEqual(result.center_xy[1], center[1], delta=2.5)
        # Must not rest on the naive corner average when that differs.
        mean_error = np.linalg.norm(
            np.array(result.center_xy) - np.array(mean)
        )
        true_error = np.linalg.norm(
            np.array(result.center_xy) - np.array(center)
        )
        self.assertLess(true_error, mean_error + 0.5)

    def test_tiny_blurred_tag(self) -> None:
        gray, center = _paint_axis_aligned_tag(
            width=80, height=60, left=28, top=18, inner=10, border=2
        )
        blurred = gray.copy()
        # 3x3 box blur to mimic a far / compressed still.
        padded = np.pad(blurred, 1, mode="edge")
        acc = np.zeros_like(blurred)
        for dy in range(3):
            for dx in range(3):
                acc += padded[dy : dy + blurred.shape[0], dx : dx + blurred.shape[1]]
        blurred = acc / 9.0
        result = tag_snap.snap_click_to_tag_center(
            blurred, (center[0] + 1.5, center[1] - 1.0)
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center_xy[0], center[0], delta=2.5)
        self.assertAlmostEqual(result.center_xy[1], center[1], delta=2.5)

    def test_returns_none_on_empty_plate(self) -> None:
        gray = np.full((80, 80), 0.5, dtype=np.float64)
        self.assertIsNone(tag_snap.snap_click_to_tag_center(gray, (40.0, 40.0)))

    def test_ignores_distant_tag(self) -> None:
        gray, _center = _paint_axis_aligned_tag(
            width=400, height=300, left=50, top=35
        )
        result = tag_snap.snap_click_to_tag_center(gray, (380.0, 280.0))
        self.assertIsNone(result)

    def test_nearby_dark_patch_does_not_steal_the_snap(self) -> None:
        gray, center = _paint_axis_aligned_tag()
        gray[:28, 110:] = 0.12
        result = tag_snap.snap_click_to_tag_center(gray, center)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center_xy[0], center[0], delta=2.0)
        self.assertAlmostEqual(result.center_xy[1], center[1], delta=2.0)

    def test_clicks_inside_the_tag_agree(self) -> None:
        gray, center = _paint_axis_aligned_tag()
        clicks = (
            (center[0] - 3.0, center[1] - 2.0),
            (center[0] + 4.0, center[1] + 1.0),
            (center[0] - 1.0, center[1] + 3.5),
            (center[0] + 2.0, center[1] - 3.0),
        )
        centres = []
        for click in clicks:
            result = tag_snap.snap_click_to_tag_center(gray, click)
            self.assertIsNotNone(result, msg=f"no snap for {click}")
            assert result is not None
            centres.append(np.array(result.center_xy))
        spread = float(np.max(np.linalg.norm(np.stack(centres) - centres[0], axis=1)))
        self.assertLess(spread, 0.75)

    def test_blurry_oblique_tag_with_nearby_dark_furniture(self) -> None:
        """Clicks inside a small blurry tag must not jump toward nearby dark clutter."""
        gray = np.full((90, 110), 0.58, dtype=np.float64)
        gray[:28, 78:] = 0.18
        inner = np.array(
            (
                (38.0, 32.0),
                (62.0, 28.0),
                (70.0, 48.0),
                (46.0, 52.0),
            ),
            dtype=np.float64,
        )
        centroid = inner.mean(axis=0)
        _fill_quad(gray, centroid + 1.28 * (inner - centroid), 0.92)
        _fill_quad(gray, inner, 0.16)
        _fill_quad(
            gray,
            np.array(
                (
                    (48.0, 36.0),
                    (54.0, 35.0),
                    (55.0, 40.0),
                    (49.0, 41.0),
                ),
                dtype=np.float64,
            ),
            0.88,
        )
        padded = np.pad(gray, 2, mode="edge")
        acc = np.zeros_like(gray)
        for dy in range(5):
            for dx in range(5):
                acc += padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
        gray = acc / 25.0
        true_center = tag_snap.projective_quad_center(inner)
        clicks = (
            (true_center[0] - 2.0, true_center[1] + 1.5),
            (true_center[0] + 2.5, true_center[1] - 1.0),
            (true_center[0] + 0.5, true_center[1] + 2.0),
            (true_center[0] - 1.5, true_center[1] - 2.0),
        )
        centres = []
        for click in clicks:
            result = tag_snap.snap_click_to_tag_center(gray, click)
            self.assertIsNotNone(result, msg=f"no snap for {click}")
            assert result is not None
            centres.append(np.array(result.center_xy))
            self.assertLess(
                float(np.linalg.norm(np.array(result.center_xy) - true_center)),
                4.0,
            )
        spread = float(np.max(np.linalg.norm(np.stack(centres) - centres[0], axis=1)))
        self.assertLess(spread, 1.25)


class TagSnapWithoutOpenCvTests(unittest.TestCase):
    def test_returns_none_when_opencv_is_missing(self) -> None:
        gray = np.full((40, 40), 0.5, dtype=np.float64)
        with patch.object(tag_snap, "_cv2_module", return_value=None):
            self.assertIsNone(
                tag_snap.snap_click_to_tag_center(gray, (20.0, 20.0))
            )


def _ground_truth_tags(cv2, gray, red_bgr):
    """Red-mask connected components → (mask, diagonal-intersection centre)."""
    mask = (
        (red_bgr[:, :, 2] > 128)
        & (red_bgr[:, :, 1] < 80)
        & (red_bgr[:, :, 0] < 80)
    ).astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    tags = []
    for index in range(1, count):
        region = labels == index
        contours, _unused = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = None
        for fraction in np.linspace(0.01, 0.08, 15):
            candidate = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(candidate) == 4:
                approx = np.asarray(candidate, dtype=np.float64).reshape(4, 2)
                break
        if approx is None:
            approx = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
        centroid = approx.mean(axis=0)
        angles = np.arctan2(approx[:, 1] - centroid[1], approx[:, 0] - centroid[0])
        corners = approx[np.argsort(angles)]
        tags.append((region, tag_snap.projective_quad_center(corners)))
    return tags


_SAMPLE_FIXTURES = (
    ("sample-small-tags", 8),
    ("sample-small-tags2", 5),
)


class SampleSmallTagSnapTests(unittest.TestCase):
    """Regression: clicks inside hand-traced sample tags snap to centre."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cv2 = _import_cv2()
        cls.plates: dict[str, tuple[np.ndarray, list]] = {}
        if cls.cv2 is None:
            return
        for stem, _expected in _SAMPLE_FIXTURES:
            plate = _ROOT / "tests" / "fixtures" / f"{stem}.png"
            overlay = _ROOT / "tests" / "fixtures" / f"{stem}-just-red.png"
            bgr = cls.cv2.imread(str(plate), cls.cv2.IMREAD_COLOR)
            red = cls.cv2.imread(str(overlay), cls.cv2.IMREAD_COLOR)
            if bgr is None or red is None:
                continue
            gray = cls.cv2.cvtColor(bgr, cls.cv2.COLOR_BGR2GRAY)
            cls.plates[stem] = (gray, _ground_truth_tags(cls.cv2, gray, red))

    def test_clicks_inside_each_traced_tag_snap_to_its_centre(self) -> None:
        if self.cv2 is None:
            self.skipTest("OpenCV not installed")
        rng = np.random.default_rng(0)
        for stem, expected_count in _SAMPLE_FIXTURES:
            with self.subTest(stem=stem):
                self.assertIn(stem, self.plates)
                gray, tags = self.plates[stem]
                self.assertEqual(len(tags), expected_count)
                for tag_index, (region, centre) in enumerate(tags):
                    rows, cols = np.nonzero(region)
                    pixels = np.column_stack((cols, rows))
                    clicks = [(float(centre[0]), float(centre[1]))]
                    chosen = rng.choice(
                        len(pixels), size=min(5, len(pixels)), replace=False
                    )
                    for pixel in pixels[chosen]:
                        clicks.append((float(pixel[0]) + 0.5, float(pixel[1]) + 0.5))
                    for click in clicks:
                        result = tag_snap.snap_click_to_tag_center(gray, click)
                        self.assertIsNotNone(
                            result,
                            msg=f"{stem} tag {tag_index} no snap at {click}",
                        )
                        assert result is not None
                        error = float(
                            np.linalg.norm(np.array(result.center_xy) - centre)
                        )
                        self.assertLess(
                            error,
                            4.0,
                            msg=(
                                f"{stem} tag {tag_index} click {click} snapped to "
                                f"{result.center_xy} (want {tuple(centre.round(1))}, "
                                f"err {error:.2f}px)"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
