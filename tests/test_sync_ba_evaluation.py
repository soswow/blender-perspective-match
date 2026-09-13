"""Trial scoring and sparse line derivatives preserve the full BA objective."""
from itertools import product
from unittest import TestCase, mock

import numpy as np

from match_perspective import core
from match_perspective.core import sync
from match_perspective.core.sync import ba


def problem(*, lock_scale=True, lock_rotation=False, lock_translation=False, distorted=False):
    cal = core.Calibration(core.CameraIntrinsics(700., 710., 320., 240., 640, 480),
                           np.eye(3), np.array([0., 0., -5.]),
                           division_lambda=0.02 if distorted else 0.)
    sims = {'other': sync.SimilarityTransform(1.2, ba._rodrigues(np.array([.05, -.03, .02])),
                                             np.array([.8, .1, .2]))}
    points = {'a': np.array([-.4, .2, .1]), 'b': np.array([.5, .3, .2]),
              'c': np.array([.1, -.5, .3]), 'd': np.array([.2, .4, .6])}
    line_points = {'la': np.array([-.5, .2, .4]), 'lb': np.array([.5, .2, .4])}
    directions = {key: np.array([0., 1., 0.]) for key in line_points}
    observations = [sync.SyncObservation(camera, key, 315.+i*7, 241.-i*9, weight=1.+i*.2)
                    for camera in ('anchor', 'other') for i, key in enumerate(points)]
    strokes = [(key, p, directions[key], sync.SyncLineObservation(camera, key, 270.+i, 210., 310., 290.))
               for camera in ('anchor', 'other') for key, p in line_points.items() for i in range(3)]
    free = [] if lock_scale and lock_rotation and lock_translation else ['other']
    kwargs = dict(free_match_ids=free, free_landmark_ids=list(points), fixed_landmarks={},
                  anchor_id='anchor', matches={key: sync.SyncMatchInput(key, cal) for key in ('anchor', 'other')},
                  observations=observations, line_constraints=strokes, lock_scale=lock_scale,
                  fixed_scales={'other': 1.2}, lock_rotation=lock_rotation,
                  fixed_rotations={'other': sims['other'].rotation}, lock_translation=lock_translation,
                  fixed_translations={'other': sims['other'].translation}, free_line_ids=list(line_points),
                  fixed_line_points={}, fixed_line_directions=directions,
                  ground_landmark_ids=['a'], ground_slack=.1,
                  known_world_priors={'a': points['a'], 'c': points['c']}, known_3d_slack=.2,
                  mirror_pairs=[('a', 'b'), ('la', 'lb')],
                  mirror_plane=(np.zeros(3), np.array([1., 0., 0.])), mirror_slack=.1,
                  mirror_landmark_id='d', free_plane_offset=True,
                  plane_groups=[('a', 'Z', 1), ('b', 'Z', 1), ('la', 'Z', 1)], plane_slack=.1,
                  location_match_ids={'anchor'})
    params = ba._pack_ba_params(free, list(points), sims, points, lock_scale=lock_scale,
                               lock_rotation=lock_rotation, lock_translation=lock_translation,
                               free_line_ids=list(line_points), free_line_points=line_points)
    return np.r_[params, .02], kwargs


class BAEvaluationTests(TestCase):
    def test_score_only_preserves_residuals_and_protected_constraints(self):
        for scale, rotation, translation, distorted in product((False, True), repeat=4):
            with self.subTest(scale=scale, rotation=rotation, translation=translation, distorted=distorted):
                params, kwargs = problem(lock_scale=scale, lock_rotation=rotation,
                                         lock_translation=translation, distorted=distorted)
                residual, _jac, protected = ba._ba_raw_residuals_and_jacobian(params, **kwargs)
                with mock.patch.object(ba, '_rodrigues_partials', side_effect=AssertionError('Unneeded derivatives')), \
                     mock.patch.object(ba, '_unpack_ba_params', wraps=ba._unpack_ba_params) as unpack:
                    actual, jac, mask = ba._ba_raw_residuals_and_jacobian(params, **kwargs, compute_jacobian=False)
                np.testing.assert_array_equal(actual, residual)
                np.testing.assert_array_equal(mask, protected)
                self.assertIsNone(jac)
                self.assertEqual(unpack.call_count, 1)
                weights = ba._robust_weights(residual, 3.)
                weights[protected] = 1.
                np.testing.assert_array_equal(ba._ba_residual_vector(params, **kwargs, huber_delta=3.), residual*weights)

    def test_line_derivatives_unpack_each_parameter_only_once(self):
        for locks, distorted in product(((True, False, False),
                                         (False, True, True),
                                         (True, True, True)), (False, True)):
            with self.subTest(locks=locks, distorted=distorted):
                params, kwargs = problem(lock_scale=locks[0], lock_rotation=locks[1],
                                         lock_translation=locks[2], distorted=distorted)
                stride = ba._similarity_param_stride(lock_scale=locks[0],
                                                     lock_rotation=locks[1],
                                                     lock_translation=locks[2])
                pose_columns = stride if kwargs['free_match_ids'] else 0
                line_start = pose_columns + 3*len(kwargs['free_landmark_ids'])
                columns = list(range(pose_columns)) + list(range(line_start, line_start+6))
                with mock.patch.object(ba, '_unpack_ba_params', wraps=ba._unpack_ba_params) as unpack:
                    residual, jac, _ = ba._ba_raw_residuals_and_jacobian(params, **kwargs)
                # Repeated strokes reuse each pose/line perturbation; Fit Only blocks line motion.
                self.assertEqual(unpack.call_count, 1 + len(columns))
                for column in columns:
                    step = 1e-5 * max(1., abs(params[column]))
                    shifted = params.copy()
                    shifted[column] += step
                    other, _, _ = ba._ba_raw_residuals_and_jacobian(shifted, **kwargs)
                    expected = (other[-24:] - residual[-24:]) / step
                    if column >= line_start:
                        # The non-location camera still scores against the line,
                        # but its stroke does not move that line's 3D midpoint.
                        expected[12:] = 0.
                    np.testing.assert_array_equal(jac[-24:, column], expected)
