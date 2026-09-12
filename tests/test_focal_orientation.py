"""Independent oracle checks for the tilted-anchor focal regression."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from match_perspective.core import geometry, sync
from match_perspective.core.focal_bundle import fit_independent_focals
from tools.synthetic_sync import focal_orientation
from tools.synthetic_sync.geometry import project


class OrientationFixtureTests(TestCase):
    def test_fixture_world_priors_expose_the_missing_global_rotation(self):
        case = focal_orientation.generate()
        points, cameras = focal_orientation.anchor_gauge_start(case)
        request = case["request"]
        self.assertEqual(len(request["plane_groups"]), 4)
        self.assertEqual(len(request["mirror_pairs"]), 3)
        self.assertGreater(np.ptp([points[p][0] for p, _, _ in request["plane_groups"]]), .05)
        normal = np.asarray(request["mirror_plane"][1])
        normal /= np.linalg.norm(normal)
        a, b = request["mirror_pairs"][0]
        delta = np.asarray(points[a]) - np.asarray(points[b])
        self.assertGreater(np.linalg.norm(np.cross(delta, normal)), .05)
        for pick in request["observations"]:
            predicted = project([points[pick["landmark_id"]]], cameras[pick["match_id"]])[0][0]
            self.assertLess(np.linalg.norm(predicted - (pick["u"], pick["v"])), 1e-9)

    def test_free_control_has_identical_pixels_and_stored_anchor(self):
        constrained = focal_orientation.generate()
        free = focal_orientation.generate(constrained=False)
        for field in ("cameras", "points", "observations"):
            self.assertEqual(constrained["request"][field], free["request"][field])
        self.assertEqual(constrained["truth"], free["truth"])
        self.assertFalse(free["request"]["plane_groups"])
        self.assertFalse(free["request"]["mirror_pairs"])

    def test_production_bundle_fits_tilted_world_constraints(self):
        case, result = self._fit(constrained=True)
        self.assertTrue(result.accepted, result.reason)
        anchor = case["request"]["anchor_id"]
        true = {item["id"]: item for item in case["truth"]["cameras"]}
        fitted = result.calibrations[anchor]
        np.testing.assert_allclose(fitted.camera_center, true[anchor]["center"], atol=1e-9)
        self.assertLess(np.linalg.norm(fitted.rotation_w2c - true[anchor]["rotation"]), 1e-3)
        self.assertGreater(np.linalg.norm(fitted.rotation_w2c -
                                          case["request"]["cameras"][0]["rotation"]), .2)
        self.assertLess(np.linalg.norm(result.sync_result.similarities[anchor].rotation - np.eye(3)), 1e-9)
        points = result.sync_result.landmarks
        self.assertLess(np.ptp([points[key][0] for key, axis, _ in case["request"]["plane_groups"]
                                if axis == "X"]), 1e-4)
        origin, normal = (np.asarray(value) for value in case["request"]["mirror_plane"])
        for left, right in case["request"]["mirror_pairs"]:
            reflection = points[left] - 2 * normal * float(normal @ (points[left] - origin))
            self.assertLess(np.linalg.norm(reflection - points[right]), 1e-4)
        self._check_holdouts(case, result, world_rotation=np.eye(3))
        # Reusable trial reports must use the returned private anchor pose.
        from tools.synthetic_sync.focal_constraint_trial import _outcome_record
        recorded = _outcome_record(result, case["request"]["cameras"])["fitted"]
        np.testing.assert_allclose(recorded["cameras"][anchor]["rotation"],
                                   fitted.rotation_w2c, atol=1e-10)

    def test_fixed_orientation_control_reproduces_focal_bound_refusal(self):
        with patch("match_perspective.core.focal_bundle.orientation_basis",
                   return_value=np.empty((0, 3))):
            _case, result = self._fit(constrained=True)
        self.assertFalse(result.accepted)
        self.assertIn("search bound", result.reason)

    def test_production_bundle_keeps_free_anchor_gauge(self):
        case, result = self._fit(constrained=False)
        self.assertTrue(result.accepted, result.reason)
        anchor = case["request"]["anchor_id"]
        np.testing.assert_allclose(result.calibrations[anchor].rotation_w2c,
                                   case["request"]["cameras"][0]["rotation"], atol=1e-10)
        self._check_holdouts(case, result, world_rotation=focal_orientation.rotation())

    def test_two_point_axis_group_releases_one_observable_rotation(self):
        case = focal_orientation.generate()
        case["request"]["plane_groups"] = case["request"]["plane_groups"][:2]
        case["request"]["mirror_pairs"] = []
        case["request"]["mirror_plane"] = None
        diagnostics = []
        case, result = self._fit(case=case, diagnostic_callback=diagnostics.append)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["orientation_parameters"], 1)
        ids = [item[0] for item in case["request"]["plane_groups"]]
        self.assertLess(abs(result.sync_result.landmarks[ids[0]][0] -
                            result.sync_result.landmarks[ids[1]][0]), 1e-4)
        self._check_holdouts_rigid_aligned(case, result)

    def _fit(self, *, constrained=True, case=None, diagnostic_callback=None):
        case = case if case is not None else focal_orientation.generate(constrained=constrained)
        request = case["request"]
        starts, world_cameras = focal_orientation.anchor_gauge_start(case)
        calibrations = {}
        similarities = {}
        for camera in request["cameras"]:
            camera_id = camera["id"]
            k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                          camera["cy"], camera["width"], camera["height"])
            calibration = geometry.Calibration(k, np.asarray(camera["rotation"]),
                                               np.asarray(camera["center"]))
            calibrations[camera_id] = calibration
            target = world_cameras[camera_id]
            rotation = np.asarray(target["rotation"]).T @ calibration.rotation_w2c
            similarities[camera_id] = sync.SimilarityTransform(
                1., rotation, np.asarray(target["center"]) - rotation @ calibration.camera_center)
        similarities[request["anchor_id"]] = sync.SimilarityTransform()
        initial = sync.SyncSolveResult(
            similarities=similarities,
            landmarks={key: np.asarray(value) for key, value in starts.items()},
            mean_reprojection_px=0., per_match_rmse_px={}, per_landmark_rmse_px={},
            message="Oracle anchor-gauge start", success=True)
        observations = [sync.SyncObservation(**item) for item in request["observations"]]
        result = fit_independent_focals(
            calibrations, observations, initial, anchor_id=request["anchor_id"],
            pick_sigma_px=case["pick_sigma_px"], fx_span=.25,
            plane_groups=request["plane_groups"], plane_slack=0.,
            mirror_pairs=request["mirror_pairs"], mirror_plane=request["mirror_plane"],
            mirror_slack=0., diagnostic_callback=diagnostic_callback)
        return case, result

    def _check_holdouts_rigid_aligned(self, case, result):
        """Use training points alone to map independent truth into the free gauge."""
        ids = sorted(case["truth"]["points"])
        truth = np.asarray([case["truth"]["points"][key] for key in ids])
        fitted = np.asarray([result.sync_result.landmarks[key] for key in ids])
        mean_truth, mean_fitted = truth.mean(axis=0), fitted.mean(axis=0)
        u, singular, vt = np.linalg.svd((truth - mean_truth).T @ (fitted - mean_fitted))
        reflection = np.eye(3)
        reflection[-1, -1] = np.linalg.det(u @ vt)
        turn = vt.T @ reflection @ u.T
        scale = float(np.sum(singular * np.diag(reflection)) / np.sum((truth - mean_truth)**2))
        self.assertGreater(scale, 0)
        errors = []
        for camera in case["truth"]["cameras"]:
            camera_id = camera["id"]
            calibration = result.calibrations[camera_id]
            similarity = result.sync_result.similarities[camera_id]
            fitted_camera = dict(camera,
                fx=calibration.intrinsics.fx, fy=calibration.intrinsics.fy,
                rotation=calibration.rotation_w2c @ similarity.rotation.T,
                center=similarity.scale * similarity.rotation @ calibration.camera_center +
                       similarity.translation)
            for point_id, point in case["truth"]["holdouts"].items():
                aligned = mean_fitted + scale * turn @ (np.asarray(point) - mean_truth)
                pixel = project([aligned], fitted_camera)[0][0]
                errors.append(np.linalg.norm(pixel -
                    case["truth"]["holdout_pixels"][point_id][camera_id]))
        self.assertLess(max(errors), .01)

    def _check_holdouts(self, case, result, *, world_rotation):
        anchor = case["request"]["anchor_id"]
        center = np.asarray(case["truth"]["cameras"][0]["center"])
        cameras = {item["id"]: item for item in case["truth"]["cameras"]}
        errors = []
        for camera_id, truth_camera in cameras.items():
            calibration = result.calibrations[camera_id]
            similarity = result.sync_result.similarities[camera_id]
            global_rotation = calibration.rotation_w2c @ similarity.rotation.T
            global_center = similarity.scale * similarity.rotation @ calibration.camera_center + similarity.translation
            fitted_camera = dict(truth_camera, fx=calibration.intrinsics.fx,
                                 fy=calibration.intrinsics.fy,
                                 rotation=global_rotation, center=global_center)
            for point_id, point in case["truth"]["holdouts"].items():
                q = center + world_rotation @ (np.asarray(point) - center)
                pixel = project([q], fitted_camera)[0][0]
                target = case["truth"]["holdout_pixels"][point_id][camera_id]
                errors.append(np.linalg.norm(pixel - target))
        self.assertLess(max(errors), .01)
