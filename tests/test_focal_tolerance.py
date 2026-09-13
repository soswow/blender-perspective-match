"""Bounded weak-axis convergence with independent withheld-shape evidence."""

import json
from unittest import TestCase
from unittest.mock import patch

from match_perspective.core import focal_bundle
from tools.synthetic_sync import focal_constraint_reliability as fixture
from tools.synthetic_sync import focal_constraint_trial as trial
from tools.synthetic_sync.solver import calibration, solver_arguments


class FocalToleranceTests(TestCase):
    def test_weak_axis_converges_before_cap_and_preserves_independent_shape(self):
        case = json.loads((fixture.ROOT / "weak-axis-hard.json").read_text())
        source = fixture.ROOT / "run-04"
        initial, _world = trial._replay_initial(case, source)
        request = case["request"]
        calibrations = {item["id"]: calibration(item) for item in request["cameras"]}
        arguments = solver_arguments(request)
        endpoints = []
        # The archived 1e-9 control needed 171 steps. This fixed cap checks
        # that the 1e-7 criterion stops sooner without relaxing fit gates.
        with patch.object(focal_bundle, "MAX_ITERATIONS", 165):
            result = focal_bundle.fit_independent_focals(
                calibrations, arguments["observations"], initial,
                anchor_id=request["anchor_id"], pick_sigma_px=case["pick_sigma_px"],
                fx_span=.4, plane_groups=arguments["plane_groups"],
                plane_slack=arguments["plane_slack"],
                mirror_pairs=arguments["mirror_pairs"],
                mirror_plane=arguments["mirror_plane"],
                mirror_slack=arguments["mirror_slack"],
                diagnostic_callback=endpoints.append)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(len(endpoints), 1)
        self.assertTrue(endpoints[0]["converged"])
        self.assertLess(endpoints[0]["iterations"], 165)
        fitted = trial._outcome_record(result, request["cameras"])["fitted"]
        assessment = fixture.assess(case, fitted)
        self.assertLess(assessment["shape_withheld_rmse_px"], 4.)
        self.assertLess(assessment["plane_rms_world"], 1e-4)
        for camera in case["truth"]["cameras"]:
            low, high = result.intervals_px[camera["id"]]
            self.assertLessEqual(low, camera["fx"])
            self.assertGreaterEqual(high, camera["fx"])
