"""Missing-camera starts remain tentative until the complete focal fit passes."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core import sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_lines
from tools.synthetic_sync.geometry import project
try:
    from .test_focal_lines_integration import _inputs
except ImportError:
    from test_focal_lines_integration import _inputs


class FocalStartupTests(TestCase):
    def test_search_bound_is_reported_even_before_convergence(self):
        case = focal_lines.generate()
        calibrations, picks, initial = _inputs(case)
        with patch('match_perspective.core.focal_bundle.MAX_ITERATIONS', 1):
            result = fit_independent_focals(calibrations, picks, initial,
                anchor_id=case['request']['anchor_id'], fx_span=.01)
        self.assertFalse(result.accepted)
        self.assertIn('search bound', result.reason)
        self.assertIn('view_', result.reason)

    def fit_missing(self, bad=False):
        case = focal_lines.generate(inconsistent=bad)
        calibrations, picks, initial = _inputs(case)
        missing = case['request']['cameras'][-1]['id']
        del initial.similarities[missing]
        request = case['request']
        result = fit_independent_focals(
            calibrations, picks, initial, anchor_id=request['anchor_id'],
            line_observations=[sync.SyncLineObservation(**p) for p in request['line_observations']],
            mirror_pairs=request['mirror_pairs'], mirror_plane=request['mirror_plane'],
            mirror_landmark_id=request['mirror_landmark_id'])
        self.assertNotIn(missing, initial.similarities)
        return case, result

    def test_missing_camera_can_enter_joint_fit_without_becoming_known_geometry(self):
        case, result = self.fit_missing()
        self.assertTrue(result.accepted, result.reason)
        truth = case['truth']
        anchor = np.asarray(next(c['center'] for c in truth['cameras'] if c['id'] == case['request']['anchor_id']))
        ids = list(truth['points'])
        relative = np.asarray([truth['points'][key] for key in ids]) - anchor
        fitted = np.asarray([result.sync_result.landmarks[key] for key in ids]) - anchor
        scale = np.sum(relative * fitted) / np.sum(relative**2)
        holdouts = np.asarray(list(truth['holdouts'].values()))
        for camera in truth['cameras']:
            key = camera['id']
            cal, sim = result.calibrations[key], result.sync_result.similarities[key]
            self.assertLess(abs(cal.intrinsics.fx / camera['fx'] - 1), 1e-4)
            fitted_camera = dict(camera, fx=cal.intrinsics.fx, fy=cal.intrinsics.fy,
                center=sim.transform_point(cal.camera_center), rotation=cal.rotation_w2c @ sim.rotation.T)
            uv = project(anchor + scale * (holdouts - anchor), fitted_camera)[0]
            np.testing.assert_allclose(uv, project(holdouts, camera)[0], atol=.01)

    def test_provisional_camera_does_not_bypass_inconsistent_evidence_refusal(self):
        _case, result = self.fit_missing(bad=True)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.sync_result)
