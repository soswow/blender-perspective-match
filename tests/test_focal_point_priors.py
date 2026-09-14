"""Point reference constraints in the joint focal anchor chart."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from match_perspective.core.focal_point_priors import PointReferenceConstraints


class PointReferenceConstraintsTests(TestCase):
    def setUp(self) -> None:
        angle = np.pi / 4
        self.rotation = np.array((
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        ))
        self.center = np.array((1.0, -2.0, 3.0))
        self.baseline = 4.0

    def chart(self, world: np.ndarray) -> np.ndarray:
        return (self.rotation @ (world - self.center).T).T / self.baseline

    def model(self, *, known=None, known_slack=0.0, ground=(), ground_slack=0.0):
        return PointReferenceConstraints.from_inputs(
            ["known", "ground", "other"], anchor_rotation=self.rotation,
            anchor_center=self.center, baseline_world=self.baseline,
            known_world=known or {}, known_3d_slack=known_slack,
            ground_landmark_ids=list(ground), ground_slack=ground_slack,
        )

    def test_hard_known_xyz_and_oblique_ground_plane_are_exact(self) -> None:
        target = np.array((2.0, -1.0, 0.0))
        model = self.model(known={"known": target}, ground=("known", "ground"))
        self.assertEqual(set(model.hard_xyz), {0})
        self.assertEqual(set(model.hard_z), {1})
        normal, offset = model.hard_z[1]
        np.testing.assert_allclose(normal, self.rotation[:, 2])
        self.assertAlmostEqual(offset, -self.center[2] / self.baseline)
        points = self.chart(np.array((target + (0.2, 0.1, 0.4),
                                      (3.0, 1.0, 0.6), (0.0, 0.0, 1.0))))
        projected = model.project_hard(points)
        np.testing.assert_allclose(projected[0], self.chart(target), atol=1e-12)
        self.assertAlmostEqual(float(normal @ projected[1]), offset, places=12)
        self.assertAlmostEqual(model.world_gaps(projected)[0], 0.0, places=12)
        self.assertAlmostEqual(model.world_gaps(projected)[1], 0.0, places=12)
        residual, jac = model.residual_and_jacobian(
            projected, point_offset=7, parameter_count=16, jacobian=True)
        self.assertEqual(residual.shape, (0,))
        self.assertEqual(jac.shape, (0, 16))

    def test_conflicting_hard_known_height_and_ground_refuse(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hard Known 3D and On Ground"):
            self.model(known={"known": np.array((2.0, -1.0, 0.1))},
                       ground=("known",), known_slack=0.0,
                       ground_slack=0.0)

    def test_soft_known_and_ground_use_tighter_z_slack_without_known_z_row(self) -> None:
        target = np.array((2.0, -1.0, 0.3))
        world = np.array(((2.05, -1.04, 0.02),
                          (3.0, 1.0, -0.03), (0.0, 0.0, 1.0)))
        points = self.chart(world)
        model = self.model(known={"known": target}, known_slack=0.05,
                           ground=("known", "ground"), ground_slack=0.02)
        residual, jac = model.residual_and_jacobian(
            points, point_offset=5, parameter_count=14, jacobian=True)
        # Ground rows precede Known 3D XY, matching Sync's geometric row order.
        np.testing.assert_allclose(residual, (6.0, -9.0, 6.0, -4.8), atol=1e-10)
        self.assertEqual(jac.shape, (4, 14))
        no_jac, absent = model.residual_and_jacobian(
            points, point_offset=5, parameter_count=14, jacobian=False)
        np.testing.assert_array_equal(no_jac, residual)
        self.assertIsNone(absent)
        for point in range(3):
            for axis in range(3):
                shifted = points.copy()
                shifted[point, axis] += 1e-6
                trial, _ = model.residual_and_jacobian(
                    shifted, point_offset=5, parameter_count=14, jacobian=False)
                np.testing.assert_allclose(
                    (trial - residual) / 1e-6, jac[:, 5 + 3*point + axis],
                    atol=1e-6)
        np.testing.assert_allclose(model.world_gaps(points),
                                   (np.linalg.norm(world[0] - target), 0.03))

    def test_hard_ground_keeps_soft_known_xy_only(self) -> None:
        target = np.array((1.0, 2.0, 0.2))
        model = self.model(known={"known": target}, known_slack=0.1,
                           ground=("known",), ground_slack=0.0)
        self.assertEqual(set(model.hard_z), {0})
        points = self.chart(np.array(((1.05, 2.04, 0.0),
                                      (0.0, 0.0, 1.0), (0.0, 0.0, 2.0))))
        residual, _ = model.residual_and_jacobian(
            points, point_offset=0, parameter_count=9, jacobian=False)
        np.testing.assert_allclose(residual, (3.0, 2.4), atol=1e-10)

    def test_rejects_missing_references_and_invalid_slack(self) -> None:
        for kwargs in (
            {"known": {"absent": (0.0, 0.0, 0.0)}},
            {"ground": ("absent",)},
            {"ground": ("ground", "ground")},
            {"known_slack": -0.1},
            {"ground_slack": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)

    def test_persisted_float32_rotation_is_accepted_but_shear_and_reflection_refuse(self):
        exact = self.rotation.copy()
        self.rotation = exact.astype(np.float32).astype(np.float64)
        self.model()
        sheared = exact.copy()
        sheared[0, 1] += 1e-3
        self.rotation = sheared
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            self.model()
        reflected = exact.copy()
        reflected[0] *= -1.0
        self.rotation = reflected
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            self.model()
