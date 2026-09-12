"""Geometric contracts for focal line planes and parallel directions."""

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_line_constraints import LineFocalConstraints, validate_line_relations


def compile_relations(groups=(), pairs=(), *, rotation=None, hard=True):
    return LineFocalConstraints.from_inputs(
        ['a', 'b', 'c'], ['l', 'm'], plane_groups=groups, parallel_pairs=pairs,
        anchor_rotation=np.eye(3) if rotation is None else rotation,
        plane_spring=100., baseline_world=4., hard_plane=hard)


class FocalLineConstraintTests(TestCase):
    def test_axis_plane_checks_position_and_direction_separately(self):
        model = compile_relations([('a', 'Z', 1), ('l', 'Z', 1)])
        points = np.zeros((3, 3))
        on_plane = [(np.array([5., 2., 0.]), np.array([1., 0., 0.]))]
        np.testing.assert_allclose(model.residual(points, on_plane), 0)
        self.assertEqual(model.world_gaps(points, on_plane), (0., 0.))
        offset = [(np.array([5., 2., .25]), np.array([1., 0., 0.]))]
        self.assertEqual(model.world_gaps(points, offset), (1., 0.))
        tilted = [(on_plane[0][0], np.array([.8, 0., .6]))]
        self.assertEqual(model.world_gaps(points, tilted), (0., .6))
        self.assertGreater(np.linalg.norm(model.residual(points, offset)), 0)
        self.assertGreater(np.linalg.norm(model.residual(points, tilted)), 0)

    def test_soft_plane_changes_position_acceptance_not_direction(self):
        model = compile_relations([('a', 'Z', 1), ('l', 'Z', 1)], hard=False)
        geometry = [(np.array([0., 0., .25]), np.array([.8, 0., .6]))]
        self.assertEqual(model.world_gaps(np.zeros((3, 3)), geometry), (0., .6))

    def test_free_plane_follows_points_and_is_normal_sign_independent(self):
        groups = [(key, 'FREE', 1) for key in ('a', 'b', 'c', 'l')]
        model = compile_relations(groups)
        points = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        geometry = [(np.array([3., 4., 1.]), np.array([1., 0., 0.]))]
        np.testing.assert_allclose(model.residual(points, geometry), 0, atol=1e-12)
        shifted = points + [0., 0., .2]
        self.assertGreater(np.linalg.norm(model.residual(shifted, geometry)), 1)
        np.testing.assert_allclose(model.residual(shifted, geometry),
                                   model.residual(shifted[[2, 1, 0]], geometry), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'non-collinear'):
            model.residual(np.zeros((3, 3)), geometry)

    def test_parallel_ignores_position_and_drawing_direction(self):
        model = compile_relations(pairs=[('l', 'm')])
        geometry = [(np.array([1., 2., 3.]), np.array([1., 0., 0.])),
                    (np.array([30., -5., 6.]), np.array([-1., 0., 0.]))]
        np.testing.assert_allclose(model.residual(np.zeros((3, 3)), geometry), 0)
        geometry[1] = (geometry[1][0], np.array([0., 1., 0.]))
        self.assertEqual(model.world_gaps(np.zeros((3, 3)), geometry), (0., 1.))

    def test_world_axis_is_converted_to_anchor_frame(self):
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        model = compile_relations(pairs=[('WORLD_AXIS_Z', 'l')], rotation=rotation)
        geometry = [(np.zeros(3), np.array([-1., 0., 0.]))]
        np.testing.assert_allclose(model.residual(np.zeros((3, 3)), geometry), 0)

    def test_missing_support_and_invalid_links_refuse(self):
        for groups, pairs in [([('l', 'X', 1)], []),
                              ([('a', 'FREE', 1), ('l', 'FREE', 1)], []),
                              ([], [('l', 'a')]), ([], [('l', 'missing')]),
                              ([], [('l', 'l')]), ([], [('l', 'm'), ('m', 'l')]),
                              ([], [('WORLD_AXIS_X', 'WORLD_AXIS_Y')]),
                              ([('a', 'Z', 1), ('a', 'Z', 1)], [])]:
            with self.subTest(groups=groups, pairs=pairs), self.assertRaises(ValueError):
                validate_line_relations(['a', 'b', 'c'], ['l', 'm'], groups, pairs)
