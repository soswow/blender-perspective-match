"""Fixed Known 3D line midpoints independently support shared line planes."""

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_line_constraints import (
    LineFocalConstraints,
    validate_line_relations,
)
from match_perspective.core.focal_lines import endpoint_distances


def compile_relations(point_ids, line_ids, groups, known_line_ids):
    return LineFocalConstraints.from_inputs(
        point_ids, line_ids, plane_groups=groups, parallel_pairs=(),
        known_line_ids=known_line_ids, anchor_rotation=np.eye(3),
        plane_spring=100., baseline_world=4., hard_plane=True)


class KnownLinePlaneSupportTests(TestCase):
    def test_hard_plane_direction_remains_in_plane_at_visible_segment(self):
        model = compile_relations(
            ['support'], ['edge'],
            [('support', 'X', 1), ('edge', 'X', 1)], [])
        points = np.array([[0.25, 0., 0.]])
        chart_point = np.array((0.25, 0.2, 1.))
        tilted = np.array((0.0004, 1., 0.2))
        tilted /= np.linalg.norm(tilted)
        physical = model.constrain_hard_directions(points, [(chart_point, tilted)])
        position, direction = physical[0]
        visible_midpoint = position + 5.0 * direction
        self.assertGreater(abs((chart_point + 5.0 * tilted)[0] - points[0, 0]), 1e-3)
        self.assertLess(abs(visible_midpoint[0] - points[0, 0]), 1e-12)
        self.assertLess(model.world_gaps(points, [(visible_midpoint, direction)])[0], 1e-12)
        self.assertLess(model.world_gaps(points, [(visible_midpoint, direction)])[1], 1e-12)
        camera_rotation = np.eye(3)
        camera_center = np.array((0., -2., -3.))
        strokes = np.array(((520., 450.), (760., 610.)))
        chart_residual = endpoint_distances(
            position, direction, camera_rotation, camera_center,
            900., np.array((640., 480.)), strokes)
        segment_residual = endpoint_distances(
            visible_midpoint, direction, camera_rotation, camera_center,
            900., np.array((640., 480.)), strokes)
        np.testing.assert_allclose(chart_residual, segment_residual, atol=1e-10)

    def test_free_hard_plane_direction_tracks_support_and_preserves_known_line(self):
        model = compile_relations(
            ['a', 'b', 'c'], ['known', 'edge'],
            [(key, 'FREE', 1) for key in ('a', 'b', 'c', 'known', 'edge')],
            ['known'])
        points = np.array(((0., 0., 2.), (1., 0., 2.), (0., 1., 2.)))
        known = (np.array((0.4, 0.5, 2.)), np.array((1., 0., 0.)))
        free = (np.array((0.3, 0.2, 2.)), np.array((0., 1., 0.001)))
        geometry = model.constrain_hard_directions(points, [known, free])
        np.testing.assert_allclose(geometry[0][1], known[1])
        self.assertAlmostEqual(float(geometry[1][1] @ np.array((0., 0., 1.))), 0., places=12)
        moved = points.copy()
        moved[2, 2] = 2.1
        changed = model.constrain_hard_directions(moved, [known, free])
        self.assertGreater(np.linalg.norm(changed[1][1] - geometry[1][1]), 1e-3)

    def test_axis_plane_uses_fixed_line_without_point_pick(self):
        model = compile_relations(
            [], ['known', 'free'], [('known', 'Z', 1), ('free', 'Z', 1)],
            ['known'])
        geometry = [
            (np.array([1., 2., 3.]), np.array([1., 0., 0.])),
            (np.array([4., 5., 3.]), np.array([0., 1., 0.])),
        ]
        np.testing.assert_allclose(model.residual(np.empty((0, 3)), geometry), 0)
        self.assertEqual(model.world_gaps(np.empty((0, 3)), geometry), (0., 0.))
        geometry[1] = (geometry[1][0] + [0., 0., .25], geometry[1][1])
        self.assertEqual(model.world_gaps(np.empty((0, 3)), geometry), (1., 0.))

    def test_free_plane_uses_mixed_fixed_points_and_line_midpoints(self):
        groups = [(key, 'FREE', 1) for key in ('point', 'known_a', 'known_b', 'free')]
        model = compile_relations(['point'], ['known_a', 'known_b', 'free'],
                                  groups, ['known_a', 'known_b'])
        points = np.array([[0., 0., 2.]])
        geometry = [
            (np.array([1., 0., 2.]), np.array([1., 0., 0.])),
            (np.array([0., 1., 2.]), np.array([0., 1., 0.])),
            (np.array([3., 4., 2.]), np.array([1., 0., 0.])),
        ]
        np.testing.assert_allclose(model.residual(points, geometry), 0, atol=1e-12)
        self.assertEqual(model.world_gaps(points, geometry), (0., 0.))
        geometry[2] = (geometry[2][0] + [0., 0., .25], geometry[2][1])
        self.assertAlmostEqual(model.world_gaps(points, geometry)[0], 1.)

    def test_free_plane_uses_only_three_known_line_midpoints(self):
        keys = ('known_a', 'known_b', 'known_c', 'free')
        model = compile_relations(
            [], list(keys), [(key, 'FREE', 1) for key in keys], keys[:3])
        geometry = [
            (np.array([0., 0., 2.]), np.array([1., 0., 0.])),
            (np.array([1., 0., 2.]), np.array([0., 1., 0.])),
            (np.array([0., 1., 2.]), np.array([1., 1., 0.]) / np.sqrt(2.)),
            (np.array([3., 4., 2.]), np.array([1., 0., 0.])),
        ]
        np.testing.assert_allclose(model.residual(np.empty((0, 3)), geometry), 0,
                                   atol=1e-12)

    def test_free_line_cannot_support_itself_or_replace_missing_reference(self):
        groups = [(key, 'FREE', 1) for key in ('point', 'known', 'free')]
        with self.assertRaisesRegex(ValueError, 'three non-collinear fixed members'):
            validate_line_relations(['point'], ['known', 'free'], groups, (),
                                    known_line_ids=['known'])
        with self.assertRaisesRegex(ValueError, 'fixed member'):
            compile_relations([], ['known', 'free'],
                              [('known', 'X', 1), ('free', 'X', 1)], [])

    def test_collinear_fixed_references_refuse_free_plane(self):
        model = compile_relations(
            ['point'], ['known_a', 'known_b', 'free'],
            [(key, 'FREE', 1) for key in ('point', 'known_a', 'known_b', 'free')],
            ['known_a', 'known_b'])
        geometry = [(np.array([1., 0., 0.]), np.array([1., 0., 0.])),
                    (np.array([2., 0., 0.]), np.array([1., 0., 0.])),
                    (np.array([0., 1., 0.]), np.array([0., 1., 0.]))]
        with self.assertRaisesRegex(ValueError, 'non-collinear fixed members'):
            model.residual(np.array([[0., 0., 0.]]), geometry)

    def test_known_segment_midpoint_is_stable_when_line_chart_slides(self):
        model = LineFocalConstraints.from_inputs(
            [], ['known', 'free'],
            plane_groups=[('known', 'X', 1), ('free', 'X', 1)],
            parallel_pairs=(), known_line_ids={'known'},
            known_line_positions={'known': np.array((0.25, 0.4, 0.5))},
            anchor_rotation=np.eye(3), plane_spring=100.,
            baseline_world=4., hard_plane=True)
        free = (np.array((0.25, -0.2, 1.0)), np.array((0., 1., 0.)))
        base = [(np.array((0.25, 0.4, 0.5)), np.array((0., 1., 0.))), free]
        slid = [(base[0][0] + np.array((0., 100., 0.)), base[0][1]), free]
        np.testing.assert_allclose(model.residual(np.empty((0, 3)), base),
                                   model.residual(np.empty((0, 3)), slid))
        self.assertEqual(model.world_gaps(np.empty((0, 3)), base),
                         model.world_gaps(np.empty((0, 3)), slid))

    def test_known_line_free_plane_is_rigid_frame_invariant(self):
        names = ['known_a', 'known_b', 'known_c', 'free']
        midpoints = {name: np.array(value) for name, value in zip(
            names[:3], ((0., 0., 2.), (1., 0., 2.), (0., 1., 2.)))}
        geometry = [(midpoints[name], np.array((1., 0., 0.)))
                    for name in names[:3]]
        geometry.append((np.array((0.4, 0.6, 2.1)),
                         np.array((1., 0., 0.))))
        groups = [(name, 'FREE', 1) for name in names]
        def model(positions):
            return LineFocalConstraints.from_inputs(
                [], names, plane_groups=groups, parallel_pairs=(),
                known_line_ids=names[:3], known_line_positions=positions,
                anchor_rotation=np.eye(3), plane_spring=100.,
                baseline_world=4., hard_plane=True)
        original = model(midpoints)
        angle = 0.31
        rotation = np.array(((np.cos(angle), 0., -np.sin(angle)),
                             (0., 1., 0.),
                             (np.sin(angle), 0., np.cos(angle))))
        translation = np.array((4., -2., 1.))
        transformed_geometry = [(rotation @ point + translation,
                                 rotation @ direction)
                                for point, direction in geometry]
        transformed = model({key: rotation @ point + translation
                             for key, point in midpoints.items()})
        rows_a = original.residual(np.empty((0, 3)), geometry)
        rows_b = transformed.residual(np.empty((0, 3)), transformed_geometry)
        self.assertAlmostEqual(float(rows_a @ rows_a), float(rows_b @ rows_b), places=9)
        np.testing.assert_allclose(original.world_gaps(np.empty((0, 3)), geometry),
                                   transformed.world_gaps(np.empty((0, 3)), transformed_geometry),
                                   atol=1e-12)
