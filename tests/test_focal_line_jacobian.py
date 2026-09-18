"""Dependency and arithmetic checks for finite-difference line pixels."""

from unittest import TestCase, mock

import numpy as np

from match_perspective.core import focal_bundle
from match_perspective.core import sync
from match_perspective.core.focal_lines import LineChart, endpoint_distances
from match_perspective.core.sync.projection import _rodrigues
from tools.synthetic_sync import focal_lines

try:
    from .test_focal_lines_integration import _inputs
except ImportError:
    from test_focal_lines_integration import _inputs


class _CountingChart(LineChart):
    def __init__(self, point, direction):
        super().__init__(point, direction)
        self.calls = 0

    def decode(self, params):
        self.calls += 1
        return super().decode(params)


class FocalLineJacobianTests(TestCase):
    def test_fit_evaluates_only_dependent_strokes_before_first_step(self):
        case = focal_lines.generate()
        request = case["request"]
        calibrations, points, initial = _inputs(case)
        lines = [sync.SyncLineObservation(**item)
                 for item in request["line_observations"]]
        line_ids = {item.landmark_id for item in lines}
        independent_point_pairs = [pair for pair in request["mirror_pairs"]
                                   if not set(pair) & line_ids]

        class StopAfterJacobian(Exception):
            pass

        with mock.patch.object(focal_bundle, "endpoint_distances",
                               wraps=endpoint_distances) as measured, \
             mock.patch.object(focal_bundle, "bounded_lm_step",
                               side_effect=StopAfterJacobian):
            with self.assertRaises(StopAfterJacobian):
                focal_bundle.fit_independent_focals(
                    calibrations, points, initial, anchor_id=request["anchor_id"],
                    pick_sigma_px=case["pick_sigma_px"],
                    line_observations=lines, mirror_pairs=independent_point_pairs,
                    mirror_plane=request["mirror_plane"],
                    mirror_landmark_id=request["mirror_landmark_id"])
        # One full line residual at startup and one at the Jacobian state;
        # each camera and independent line chart then touches only its strokes.
        # Mirrored line charts share geometry and therefore have cross-line derivatives.
        camera_counts = [sum(item.match_id == camera for item in lines)
                         for camera in calibrations]
        camera_parameters = (1, 6, 7, 7)
        expected = 2 * len(lines) + sum(a * b for a, b in zip(
            camera_counts, camera_parameters)) + 4 * len(lines)
        self.assertEqual(measured.call_count, expected)

    def test_sparse_columns_equal_dense_differences_with_fewer_stroke_evaluations(self):
        charts = [
            _CountingChart(np.array((0.3, 0.6, 4.0)), np.array((0.2, 0.8, 0.1))),
            _CountingChart(np.array((-0.7, 0.2, 5.0)), np.array((0.7, -0.1, 0.3))),
        ]
        lci = np.array((0, 1, 1, 2, 0, 2))
        lli = np.array((0, 0, 1, 1, 1, 0))
        luv = np.array((
            ((280., 220.), (420., 290.)),
            ((300., 210.), (450., 270.)),
            ((200., 270.), (390., 260.)),
            ((230., 250.), (410., 280.)),
            ((210., 260.), (380., 240.)),
            ((310., 200.), (460., 300.)),
        ))
        weights = np.array((1., 0.7, 1.3, 0.9, 1.1, 0.8))
        pp = np.array(((320., 240.), (330., 235.), (310., 245.)))
        base_fx = np.array((700., 750., 680.))
        line_offset = 17  # Three non-image point coordinates separate poses from lines.
        params = np.zeros(line_offset + 4 * len(charts))
        params[:3] = (0.03, -0.02, 0.01)
        params[3:8] = (0.02, -0.01, 0.03, 0.04, -0.02)
        params[8:14] = (-0.01, 0.03, 0.02, -0.8, 0.3, 0.1)
        params[14:17] = (0.2, -0.4, 0.6)
        params[line_offset:] = (0.03, -0.04, 0.5, 0.7, -0.02, 0.01, -0.4, 0.6)
        camera_columns = [(0,), (1, 3, 4, 5, 6, 7), (2, 8, 9, 10, 11, 12, 13)]
        camera_strokes = [np.flatnonzero(lci == i) for i in range(3)]
        line_strokes = [np.flatnonzero(lli == i) for i in range(len(charts))]
        direction = np.array((1., 0.2, 0.1))
        direction /= np.linalg.norm(direction)
        tangent_a = np.array((-direction[1], direction[0], 0.))
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(direction, tangent_a)

        def camera_state(value, camera):
            focal = (base_fx * np.exp(value[:3]))[camera]
            if camera == 0:
                return focal, np.eye(3), np.zeros(3)
            if camera == 1:
                raw = direction + value[6] * tangent_a + value[7] * tangent_b
                return focal, _rodrigues(value[3:6]), raw / np.linalg.norm(raw)
            return focal, _rodrigues(value[8:11]), value[11:14]

        def changed_camera_state(value, camera, column):
            changed = camera_state(value, camera)
            if column == camera:
                return changed[0], None, None
            if (camera == 1 and column < 6) or (camera == 2 and column < 11):
                return None, changed[1], None
            return None, None, changed[2]

        def decode_geometry(value):
            return [chart.decode(value[line_offset + 4*i:line_offset + 4*i + 4])
                    for i, chart in enumerate(charts)]

        def all_strokes(value):
            states = [camera_state(value, camera) for camera in range(3)]
            geometry = decode_geometry(value)
            return np.asarray([
                weights[stroke] * endpoint_distances(
                    *geometry[lli[stroke]], states[lci[stroke]][1],
                    states[lci[stroke]][2], states[lci[stroke]][0],
                    pp[lci[stroke]], luv[stroke])
                for stroke in range(len(lci))]).ravel()

        baseline = all_strokes(params)
        dense = np.zeros((len(baseline), len(params)))
        columns = sorted({column for group in camera_columns for column in group} |
                         set(range(line_offset, len(params))))
        for column in columns:
            step = 1e-6 * max(1.0, abs(params[column]))
            shifted = params.copy()
            shifted[column] += step
            dense[:, column] = (all_strokes(shifted) - baseline) / step

        for chart in charts:
            chart.calls = 0
        states = [camera_state(params, camera) for camera in range(3)]
        geometry = decode_geometry(params)
        for chart in charts:
            chart.calls = 0
        with mock.patch.object(focal_bundle, "endpoint_distances",
                               wraps=endpoint_distances) as measured:
            sparse = focal_bundle._line_image_fd_jacobian(
                params, baseline, camera_columns=camera_columns,
                camera_strokes=camera_strokes, line_strokes=line_strokes,
                camera_state=changed_camera_state, line_charts=charts,
                line_offset=line_offset, lci=lci, lli=lli, luv=luv,
                line_weights=weights, pp=pp,
                focal=np.array([state[0] for state in states]),
                rotations=[state[1] for state in states],
                centers=np.array([state[2] for state in states]), geometry=geometry)
        np.testing.assert_array_equal(sparse, dense)
        expected_evaluations = sum(len(strokes) * len(columns)
                                   for strokes, columns in zip(camera_strokes, camera_columns))
        expected_evaluations += 4 * len(lci)
        self.assertEqual(measured.call_count, expected_evaluations)
        self.assertEqual(sum(chart.calls for chart in charts), 4 * len(charts))
        self.assertLess(expected_evaluations, len(columns) * len(lci))
