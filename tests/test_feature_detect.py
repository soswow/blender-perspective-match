"""Unit tests for automatic ORB/SIFT feature matching helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Load feature_detect without executing the add-on's bpy-dependent __init__.
_ROOT = Path(__file__).resolve().parents[1]
if "match_perspective" not in sys.modules:
    _package = types.ModuleType("match_perspective")
    _package.__path__ = [str(_ROOT)]
    _package.__file__ = str(_ROOT / "__init__.py")
    sys.modules["match_perspective"] = _package

_spec = importlib.util.spec_from_file_location(
    "match_perspective.feature_detect",
    _ROOT / "feature_detect.py",
    submodule_search_locations=[str(_ROOT)],
)
assert _spec is not None and _spec.loader is not None
feature_detect = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.feature_detect"] = feature_detect
_spec.loader.exec_module(feature_detect)


def _synthetic_pair(seed: int = 0):
    """Two textured stills related by a similarity (for OpenCV smoke tests)."""
    try:
        import cv2
    except ImportError:
        return None, None
    rng = np.random.RandomState(seed)
    img = np.full((480, 640), 40, dtype=np.uint8)
    for _ in range(60):
        x, y = rng.randint(30, 610), rng.randint(30, 450)
        w, h = rng.randint(15, 90), rng.randint(15, 90)
        img[y : y + h, x : x + w] = rng.randint(70, 220)
    for _ in range(30):
        cv2.circle(
            img,
            (rng.randint(30, 610), rng.randint(30, 450)),
            rng.randint(6, 28),
            int(rng.randint(90, 240)),
            -1,
        )
    noise = rng.randint(0, 35, img.shape, np.uint8)
    img = cv2.add(img, noise)
    height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 6.0, 0.94)
    matrix[0, 2] += 28.0
    matrix[1, 2] -= 18.0
    img2 = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return img, img2


class AutoTrackFilterTests(unittest.TestCase):
    def test_filter_keeps_best_percentile(self) -> None:
        tracks = [
            feature_detect.TrackData(
                track_id=f"t{index}",
                observations=[
                    feature_detect.TrackObservationData("a", 0.0, 0.0),
                    feature_detect.TrackObservationData("b", 1.0, 1.0),
                ],
                residual_px=float(index),
                multi_view=True,
            )
            for index in range(10)
        ]
        kept, dropped = feature_detect._filter_multi_view_tracks(
            tracks,
            keep_percentile=50.0,
            max_multi_view=100,
        )
        multi = [track for track in kept if track.multi_view]
        self.assertEqual(dropped, 5)
        self.assertEqual(len(multi), 5)
        self.assertTrue(all(track.residual_px <= 4.0 + 1.0e-9 for track in multi))

    def test_filter_caps_max_multi_view(self) -> None:
        tracks = [
            feature_detect.TrackData(
                track_id=f"t{index}",
                observations=[
                    feature_detect.TrackObservationData("a", 0.0, 0.0),
                    feature_detect.TrackObservationData("b", 1.0, 1.0),
                ],
                residual_px=float(index) * 0.1,
                multi_view=True,
            )
            for index in range(20)
        ]
        kept, _dropped = feature_detect._filter_multi_view_tracks(
            tracks,
            keep_percentile=100.0,
            max_multi_view=7,
        )
        self.assertEqual(sum(1 for track in kept if track.multi_view), 7)


class AutoFeatureDetectSmokeTests(unittest.TestCase):
    def test_orb_pair_builds_multi_view_tracks(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("ORB unavailable")

        img1, img2 = _synthetic_pair()
        self.assertIsNotNone(img1)
        result = feature_detect.build_tracks_from_images(
            [("match_a", img1), ("match_b", img2)],
            detector="ORB",
            max_features=800,
            keep_percentile=90.0,
            max_multi_view=200,
            max_orphans_per_match=40,
        )
        self.assertGreater(result.multi_view_count, 20)
        self.assertGreater(result.pair_inliers, 20)
        multi = [track for track in result.tracks if track.multi_view]
        self.assertTrue(all(len(track.observations) >= 2 for track in multi))
        # Residuals for kept multi-view tracks should be finite and modest.
        self.assertTrue(all(np.isfinite(track.residual_px) for track in multi))
        self.assertLess(np.median([track.residual_px for track in multi]), 5.0)

    def test_pure_rotation_uses_homography_fallback(self) -> None:
        """Stills with almost no translation make F degenerate — H must still match."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        if not hasattr(cv2, "ORB_create"):
            self.skipTest("ORB unavailable")

        img1, _img2 = _synthetic_pair(seed=3)
        height, width = img1.shape
        # Pure rotation around image centre (no translation) — F is unreliable.
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 9.0, 1.0)
        img2 = cv2.warpAffine(
            img1, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
        )
        result = feature_detect.build_tracks_from_images(
            [("a", img1), ("b", img2)],
            detector="ORB",
            max_features=400,
            max_orphans_per_match=20,
        )
        self.assertGreater(result.multi_view_count, 10)
        self.assertGreater(result.ratio_matches, 10)


if __name__ == "__main__":
    unittest.main()
