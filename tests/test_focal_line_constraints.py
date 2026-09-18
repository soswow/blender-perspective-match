"""Geometric contracts for focal line planes and parallel directions."""

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_line_constraints import LineFocalConstraints, validate_line_relations
from match_perspective.core.derived_lines import derived_line_geometry


def compile_relations(groups=(), pairs=(), *, rotation=None, hard=True):
    return LineFocalConstraints.from_inputs(
        ['a', 'b', 'c'], ['l', 'm'], plane_groups=groups, parallel_pairs=pairs,
        anchor_rotation=np.eye(3) if rotation is None else rotation,
        plane_spring=100., baseline_world=4., hard_plane=hard)


class FocalLineConstraintTests(TestCase):
    def test_derived_line_relations_depend_on_endpoint_points(self):
        points = np.array((
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0),
        ))
        constraints = LineFocalConstraints.from_inputs_with_derived(
            ["a", "b", "support"], ["edge"],
            plane_groups=[("support", "X", 1), ("edge", "X", 1)],
            parallel_pairs=[("edge", "WORLD_AXIS_Z")],
            anchor_rotation=np.eye(3), plane_spring=100.0,
            baseline_world=1.0, hard_plane=True,
            derived_lines=[("edge", "a", "b")],
        )

        def geometry(values):
            first, second = values[:2]
            direction = second - first
            direction /= np.linalg.norm(direction)
            return [(0.5 * (first + second), direction)]

        self.assertLess(np.linalg.norm(constraints.residual(points, geometry(points))), 1e-12)
        moved = points.copy()
        moved[1, 0] = 0.2
        self.assertGreater(np.linalg.norm(constraints.residual(moved, geometry(moved))), 1.0)
        self.assertEqual(constraints.point_indices, {0, 1, 2})

        soft = LineFocalConstraints.from_inputs_with_derived(
            ["a", "b", "support"], ["edge"],
            plane_groups=[("a", "X", 1), ("b", "X", 1),
                          ("support", "X", 1), ("edge", "X", 1)],
            parallel_pairs=[], anchor_rotation=np.eye(3), plane_spring=10.0,
            baseline_world=1.0, hard_plane=False,
            derived_lines=[("edge", "a", "b")],
        )
        tilted = points.copy()
        tilted[1, 0] = 0.2
        self.assertGreater(np.linalg.norm(soft.residual(tilted, geometry(tilted))), 1.0)

    def test_parallel_to_derived_or_drawn_line_keeps_endpoint_dependencies(self):
        points = {"a": np.array((0., 0., 0.)), "b": np.array((0., 0., 2.)),
                  "c": np.array((1., 0., 2.)), "d": np.array((1., 0., 0.))}
        definitions = [("first", "a", "b"), ("second", "c", "d")]
        model = LineFocalConstraints.from_inputs_with_derived(
            list(points), ["first", "second", "drawn"], plane_groups=[],
            parallel_pairs=[("first", "second"), ("first", "drawn")],
            anchor_rotation=np.eye(3), plane_spring=100., baseline_world=1.,
            hard_plane=True, derived_lines=definitions)

        def residual(values):
            _, geometry = derived_line_geometry(values, definitions)
            return model.residual(np.asarray(list(values.values())), [
                geometry["first"], geometry["second"],
                (np.array((4., 5., 6.)), np.array((0., 0., 1.)))])

        np.testing.assert_allclose(residual(points), 0., atol=1e-12)
        self.assertEqual(model.point_indices, {0, 1, 2, 3})
        for endpoint in points:
            changed = {key: value.copy() for key, value in points.items()}
            changed[endpoint][0] += .2
            self.assertGreater(np.linalg.norm(residual(changed)), 1.)

    def test_derived_world_direction_can_enter_feasibility_continuation(self):
        points = {"a": np.array((0., 0., 0.)), "b": np.array((.1, 0., 2.))}
        model = LineFocalConstraints.from_inputs_with_derived(
            list(points), ["edge"], plane_groups=[],
            parallel_pairs=[("edge", "WORLD_AXIS_Z")], anchor_rotation=np.eye(3),
            plane_spring=100., baseline_world=1., hard_plane=True,
            derived_lines=[("edge", "a", "b")])
        _, geometry = derived_line_geometry(points, [("edge", "a", "b")])
        values = np.asarray(list(points.values()))
        ordinary = np.linalg.norm(model.residual(values, [geometry["edge"]]))
        self.assertTrue(model.has_derived_fixed_direction)
        model.direction_feasibility_scale = 10.
        self.assertAlmostEqual(np.linalg.norm(model.residual(values, [geometry["edge"]])),
                               ordinary * 10.)

    def test_derived_fixed_direction_gap_ignores_other_line_relations(self):
        model = LineFocalConstraints.from_inputs_with_derived(
            ["a", "b"], ["edge", "other"], plane_groups=[],
            parallel_pairs=[("edge", "WORLD_AXIS_Z"), ("other", "WORLD_AXIS_Z")],
            anchor_rotation=np.eye(3), plane_spring=100., baseline_world=1.,
            hard_plane=True, derived_lines=[("edge", "a", "b")])
        geometry = [(np.zeros(3), np.array((0., 0., 1.))),
                    (np.zeros(3), np.array((1., 0., 0.)))]
        self.assertEqual(model.derived_fixed_direction_gap(geometry), 0.)

    def test_derived_known_line_target_is_fixed_but_drawn_target_is_not(self):
        derived = LineFocalConstraints.from_inputs_with_derived(
            ["a", "b"], ["edge", "known"], plane_groups=[],
            parallel_pairs=[("edge", "known")], anchor_rotation=np.eye(3),
            plane_spring=100., baseline_world=1., hard_plane=True,
            known_line_ids={"known"}, known_line_positions={"known": np.zeros(3)},
            derived_lines=[("edge", "a", "b")])
        drawn = LineFocalConstraints.from_inputs_with_derived(
            ["a", "b"], ["edge", "drawn"], plane_groups=[],
            parallel_pairs=[("edge", "drawn")], anchor_rotation=np.eye(3),
            plane_spring=100., baseline_world=1., hard_plane=True,
            derived_lines=[("edge", "a", "b")])
        self.assertTrue(derived.has_derived_fixed_direction)
        self.assertFalse(drawn.has_derived_fixed_direction)
        geometry = [(np.zeros(3), np.array((.1, 0., 1.))),
                    (np.zeros(3), np.array((0., 0., 1.)))]
        values = np.asarray(((0., 0., 0.), (0., 0., 2.)))
        ordinary = np.linalg.norm(derived.residual(values, geometry))
        derived.direction_feasibility_scale = 10.
        self.assertAlmostEqual(np.linalg.norm(derived.residual(values, geometry)),
                               ordinary * 10.)
        drawn_ordinary = np.linalg.norm(drawn.residual(values, geometry))
        drawn.direction_feasibility_scale = 10.
        self.assertAlmostEqual(np.linalg.norm(drawn.residual(values, geometry)),
                               drawn_ordinary)

    def test_derived_free_plane_does_not_project_line_away_from_endpoints(self):
        points = {"a": np.array((.2, .3, 1.)), "b": np.array((.8, .3, 1.)),
                  "s0": np.array((0., 0., 1.)), "s1": np.array((1., 0., 1.)),
                  "s2": np.array((0., 1., 1.))}
        definitions = [("edge", "a", "b")]
        model = LineFocalConstraints.from_inputs_with_derived(
            list(points), ["edge"], parallel_pairs=[],
            plane_groups=[(key, "FREE", 1) for key in ("s0", "s1", "s2", "edge")],
            anchor_rotation=np.eye(3), plane_spring=100., baseline_world=1.,
            hard_plane=True, derived_lines=definitions)
        _, geometry = derived_line_geometry(points, definitions)
        np.testing.assert_allclose(model.residual(np.asarray(list(points.values())),
                                                 [geometry["edge"]]), 0., atol=1e-12)
        points["b"][2] += .2
        _, geometry = derived_line_geometry(points, definitions)
        values = np.asarray(list(points.values()))
        self.assertGreater(np.linalg.norm(model.residual(values, [geometry["edge"]])), 1.)
        self.assertEqual(model.hard_direction_normals(values, [geometry["edge"]]), {})
        self.assertGreater(model.world_gaps(values, [geometry["edge"]])[0], .09)

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
