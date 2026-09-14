"""Hard-plane line pixels retain the coupled point and frame derivatives."""

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_bundle import (
    _active_pixel_jacobian,
    _coupled_hard_line_columns,
    _coupled_hard_line_image_fd_jacobian,
    _line_image_fd_jacobian,
    _redundant_hard_line_columns,
)
from match_perspective.core.focal_line_constraints import LineFocalConstraints
from match_perspective.core.focal_lines import LineChart, endpoint_distances
from match_perspective.core.focal_orientation import overall_rotation


class HardPlaneLineJacobianTests(TestCase):
    def _case(self, *, free):
        points = (np.array(((0., 0., 2.), (1., 0., 2.), (0., 1., 2.)))
                  if free else np.array(((0.25, 0., 0.),)))
        group = [(key, 'FREE' if free else 'X', 1)
                 for key in (('a', 'b', 'c', 'edge') if free else ('a', 'edge'))]
        model = LineFocalConstraints.from_inputs(
            ['a', 'b', 'c'] if free else ['a'], ['edge'], plane_groups=group,
            parallel_pairs=(), anchor_rotation=np.eye(3),
            plane_spring=100., baseline_world=4., hard_plane=True)
        chart = LineChart(
            np.array((0.3, 0.25, 2.)) if free else np.array((0.25, 0.2, 1.)),
            np.array((0.4, 0.9, 0.01)))
        line_offset = 3 * len(points)
        orientation_column = line_offset + 4
        params = np.concatenate((points.ravel(), chart.initial, np.zeros(1)))
        centers = [np.array((0., -2., -3.)), np.array((1., -1., -4.))]
        strokes = [np.array(((520., 450.), (760., 610.))),
                   np.array(((540., 470.), (730., 590.)))]

        def point_values_at(values):
            return values[:line_offset].reshape(-1, 3)

        def geometry_at(values, local_points):
            point, direction = chart.decode(values[line_offset:line_offset + 4])
            rotation = overall_rotation(values[orientation_column:], np.array(((0., 1., 0.),)))
            physical = model.constrain_hard_directions(
                local_points @ rotation.T, [(rotation @ point, rotation @ direction)])
            return [(rotation.T @ physical[0][0], rotation.T @ physical[0][1])]

        def line_residual_at(geometry):
            point, direction = geometry[0]
            return np.concatenate([
                endpoint_distances(point, direction, np.eye(3), center,
                                   900., np.array((640., 480.)), stroke)
                for center, stroke in zip(centers, strokes)])

        return (model, chart, points, params, line_offset, orientation_column,
                point_values_at, geometry_at, line_residual_at, centers, strokes)

    def test_free_plane_point_support_column_matches_independent_difference(self):
        (model, chart, points, params, line_offset, _orientation_column,
         point_values_at, geometry_at, line_residual_at, _centers, _strokes) = self._case(free=True)
        baseline = line_residual_at(geometry_at(params, points))
        column = 8  # Z of the third independent plane supporter.
        self.assertIn(column, _coupled_hard_line_columns(
            model, 0, line_offset + 4, line_offset + 5))
        jac = _coupled_hard_line_image_fd_jacobian(
            params, baseline, {column}, point_values_at=point_values_at,
            geometry_at=geometry_at, line_residual_at=line_residual_at)
        step = 2e-6
        minus, plus = params.copy(), params.copy()
        minus[column] -= step
        plus[column] += step
        expected = (line_residual_at(geometry_at(plus, point_values_at(plus))) -
                    line_residual_at(geometry_at(minus, point_values_at(minus)))) / (2 * step)
        self.assertGreater(float(np.linalg.norm(expected)), 1e-3)
        np.testing.assert_allclose(jac[:, column], expected, rtol=2e-4, atol=2e-5)
        disabled = _redundant_hard_line_columns(model, points, [chart], line_offset)
        self.assertEqual(len(disabled), 1)
        self.assertTrue(disabled <= {line_offset, line_offset + 1})

    def test_common_rotation_and_active_line_chart_columns(self):
        (model, chart, points, params, line_offset, rotation_column,
         point_values_at, geometry_at, line_residual_at, centers, strokes) = self._case(free=False)
        physical = geometry_at(params, points)
        baseline = line_residual_at(physical)
        self.assertEqual(_coupled_hard_line_columns(
            model, 0, rotation_column, rotation_column + 1), {rotation_column})
        coupled = _coupled_hard_line_image_fd_jacobian(
            params, baseline, {rotation_column}, point_values_at=point_values_at,
            geometry_at=geometry_at, line_residual_at=line_residual_at)
        step = 2e-6
        minus, plus = params.copy(), params.copy()
        minus[rotation_column] -= step
        plus[rotation_column] += step
        expected = (line_residual_at(geometry_at(plus, points)) -
                    line_residual_at(geometry_at(minus, points))) / (2 * step)
        self.assertGreater(float(np.linalg.norm(expected)), 1e-3)
        np.testing.assert_allclose(coupled[:, rotation_column], expected,
                                   rtol=2e-4, atol=2e-5)
        line_jac = _line_image_fd_jacobian(
            params, baseline, camera_columns=[(), ()],
            camera_strokes=[np.empty(0, int), np.empty(0, int)],
            line_strokes=[np.array((0, 1))],
            camera_state=lambda *_args: (None, None, None),
            line_charts=[chart], line_offset=line_offset,
            lci=np.array((0, 1)), lli=np.array((0, 0)), luv=np.asarray(strokes),
            line_weights=np.ones(2), pp=np.array(((640., 480.), (640., 480.))),
            focal=np.array((900., 900.)), rotations=[np.eye(3), np.eye(3)],
            centers=np.asarray(centers), geometry=physical,
            geometry_at=lambda shifted: geometry_at(shifted, points))
        disabled = _redundant_hard_line_columns(model, points, [chart], line_offset)
        active = np.array([column for column in range(line_offset, rotation_column + 1)
                           if column not in disabled], int)
        self.assertEqual(len(active), 4)  # Three line DOFs plus common rotation.
        reduced = _active_pixel_jacobian(
            line_jac + coupled, len(baseline), active, np.ones(len(baseline)))
        self.assertEqual(reduced.shape, (4, 4))
        np.testing.assert_allclose(reduced[:, -1], expected, rtol=2e-4, atol=2e-5)
