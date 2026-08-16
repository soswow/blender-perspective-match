"""Unit tests for the optional OpenCV capability probe."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

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
    "match_perspective.detect.opencv",
    _ROOT / "detect/opencv.py",
    submodule_search_locations=[str(_ROOT / "detect")],
)
assert _spec is not None and _spec.loader is not None
opencv_support = importlib.util.module_from_spec(_spec)
sys.modules["match_perspective.detect.opencv"] = opencv_support
_spec.loader.exec_module(opencv_support)


class OpenCvProbeTests(unittest.TestCase):
    def test_cached_capabilities_none_before_probe(self) -> None:
        previous = opencv_support._cached
        opencv_support._cached = None
        try:
            self.assertIsNone(opencv_support.cached_capabilities())
        finally:
            opencv_support._cached = previous

    def test_capabilities_are_booleans(self) -> None:
        caps = opencv_support.capabilities(refresh=True)
        self.assertIsInstance(caps.apriltag_25h9, bool)
        self.assertIsInstance(caps.line_segment_detector, bool)
        self.assertEqual(caps.available, caps.module is not None)

    def test_load_warning_matches_hidden_features(self) -> None:
        caps = opencv_support.capabilities(refresh=True)
        warning = opencv_support.load_warning()
        if caps.apriltag_25h9 and caps.line_segment_detector:
            self.assertEqual(warning, "")
            return
        self.assertTrue(warning.startswith("Perspective Match:"))
        if not caps.line_segment_detector:
            self.assertIn("Detect VP Lines", warning)
        if not caps.apriltag_25h9:
            self.assertIn("Find AprilTags", warning)
