"""Pose LM does not evaluate an identical trial residual twice."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from match_perspective.core import sync
from sync_fixtures import _synthetic_scene, _true_landmarks


class PoseResidualReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        matches, observations, true_pose, _private_center, _shared_center = (
            _synthetic_scene(with_ground=False)
        )
        self.anchor = matches[0].calibration
        self.other = matches[1].calibration
        by_landmark = {}
        for observation in observations:
            by_landmark.setdefault(observation.landmark_id, {})[
                observation.match_id
            ] = observation
        self.pairs = [
            (items["anchor"], items["other"])
            for items in by_landmark.values()
        ]
        self.shared_points = list(_true_landmarks().values())
        self.image_points = [
            np.array((items["other"].u, items["other"].v), dtype=np.float64)
            for items in by_landmark.values()
        ]
        self.seed = sync.SimilarityTransform(
            scale=true_pose.scale,
            rotation=true_pose.rotation.copy(),
            translation=true_pose.translation + np.array((0.4, -0.3, 0.2)),
        )

    def assert_one_evaluation_per_pose(self, solve) -> None:
        original = sync.pose._unpack_similarity_pose
        evaluated = []

        def record(values, **kwargs):
            evaluated.append(np.asarray(values).tobytes())
            return original(values, **kwargs)

        with patch.object(sync.pose, "_unpack_similarity_pose", side_effect=record):
            result = solve()

        self.assertIsNotNone(result)
        self.assertGreater(len(evaluated), 8)
        # The returned pose is unpacked once more after the LM loop.
        self.assertEqual(len(evaluated[:-1]), len(set(evaluated[:-1])))

    def test_free_ray_trials_are_evaluated_once(self) -> None:
        self.assert_one_evaluation_per_pose(
            lambda: sync.pose._refine_rigid_from_rays(
                self.seed, self.pairs, self.anchor, self.other,
                max_iterations=3,
            )
        )

    def test_pnp_trials_are_evaluated_once(self) -> None:
        self.assert_one_evaluation_per_pose(
            lambda: sync.pose._pnp_similarity(
                "other", self.shared_points, self.image_points, self.other,
                initial=self.seed, max_iterations=3,
            )
        )

    def test_mixed_trials_are_evaluated_once(self) -> None:
        self.assert_one_evaluation_per_pose(
            lambda: sync.pose._refine_rigid_mixed(
                self.seed, self.pairs, self.shared_points, self.image_points,
                self.anchor, self.other, max_iterations=3,
            )
        )

    def test_pair_registration_builds_each_camera_ray_once(self) -> None:
        with (
            patch.object(
                sync.pose, "camera_ray_private",
                wraps=sync.pose.camera_ray_private,
            ) as camera_ray,
            patch.object(
                sync.pose, "_axis_aligned_rotations",
                return_value=(np.eye(3),),
            ),
            patch.object(
                sync.pose, "_refine_rigid_from_rays",
                return_value=self.seed,
            ) as refine,
            patch.object(sync.pose, "triangulate_midpoint", return_value=None),
        ):
            sync.pose._compute_relative_from_pairs(
                self.pairs, self.anchor, self.other
            )

        self.assertEqual(camera_ray.call_count, 2 * len(self.pairs))
        self.assertEqual(refine.call_count, 2)
        self.assertIs(
            refine.call_args_list[0].kwargs["pair_ray_data"],
            refine.call_args_list[1].kwargs["pair_ray_data"],
        )


class PairJobSchedulingTests(unittest.TestCase):
    def test_gil_enabled_runs_pairs_serially_in_input_order(self) -> None:
        for gil_probe in (None, lambda: True):
            with self.subTest(gil_probe=gil_probe):
                seen = []

                def work(item):
                    seen.append(item)
                    return item * 2

                with (
                    patch.object(sync.pose.sys, "_is_gil_enabled", gil_probe,
                                 create=True),
                    patch.object(sync.pose, "ThreadPoolExecutor") as pool,
                ):
                    result = sync.pose._map_pair_jobs(work, [3, 1, 2])

                self.assertEqual(result, [6, 2, 4])
                self.assertEqual(seen, [3, 1, 2])
                pool.assert_not_called()

    def test_gil_free_keeps_ordered_pool_map(self) -> None:
        seen = []

        def work(item):
            seen.append(item)
            return item * 2

        pool = MagicMock()
        pool.__enter__.return_value = pool
        pool.map.side_effect = lambda func, items: map(func, items)
        with (
            patch.object(sync.pose.sys, "_is_gil_enabled", lambda: False,
                         create=True),
            patch.object(sync.pose, "_pair_worker_count", return_value=3),
            patch.object(sync.pose, "ThreadPoolExecutor", return_value=pool)
            as executor,
        ):
            result = sync.pose._map_pair_jobs(work, [3, 1, 2])

        self.assertEqual(result, [6, 2, 4])
        self.assertEqual(seen, [3, 1, 2])
        executor.assert_called_once_with(max_workers=3)
        pool.map.assert_called_once()


if __name__ == "__main__":
    unittest.main()
