"""Geometry controls for the read-only mirror-line diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import TestCase

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "debug-sync"))
from mirror_line_diagnostic import mirror_line_diagnostic


class MirrorLineDiagnosticTests(TestCase):
    def test_finite_extents_are_separate_from_infinite_line_relation(self):
        report = mirror_line_diagnostic(
            ((1.0, 1.0, 0.0), (1.0, 3.0, 0.0)),
            ((-1.0, 10.0, 0.0), (-1.0, 20.0, 0.0)),
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        )
        self.assertLess(report["scorer_position_gap"], 1.0e-12)
        self.assertLess(report["infinite_direction_sine"], 1.0e-12)
        self.assertLess(report["midpoint_perpendicular_norm"], 1.0e-12)
        self.assertGreater(abs(report["midpoint_along"]), 10.0)

    def test_position_term_for_nonparallel_lines_depends_on_reference_origin(self):
        left = ((-0.09, 9.0, 0.0), (-0.11, 11.0, 0.0))
        right = ((0.0, 9.0, 0.0), (0.0, 11.0, 0.0))
        near_crossing = mirror_line_diagnostic(
            left, right, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            coordinate_origin=(0.0, 0.0, 0.0),
        )
        near_helpers = mirror_line_diagnostic(
            left, right, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            coordinate_origin=(0.0, 10.0, 0.0),
        )
        self.assertAlmostEqual(
            near_crossing["infinite_direction_sine"],
            near_helpers["infinite_direction_sine"], places=12,
        )
        self.assertLess(near_crossing["scorer_position_gap"], 0.001)
        self.assertGreater(near_helpers["scorer_position_gap"], 0.099)
        self.assertAlmostEqual(
            near_helpers["midpoint_perpendicular_norm"], 0.1, places=3,
        )
