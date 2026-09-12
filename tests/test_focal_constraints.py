"""Independent-FOV plane and mirror fixture and fitting contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from unittest import TestCase, mock

import numpy as np

from match_perspective.core import geometry, lens_refine, sync
from match_perspective.core.focal_bundle import (
    FocalBundleOutcome, _fits_constrained_noise_model,
)
from match_perspective.core.focal_constraints import PointFocalConstraints
from match_perspective.core.sync.constants import MIRROR_PAIR_HARD_GAP, PLANE_HARD_SLACK
from tools.synthetic_sync import focal_constraints as fixtures


class FocalConstraintFixtureTests(TestCase):
    def test_frozen_cases_match_deterministic_factory_and_validate(self):
        def compare(saved, generated, path="case"):
            self.assertIs(type(saved), type(generated), path)
            if isinstance(saved, dict):
                self.assertEqual(saved.keys(), generated.keys(), path)
                for key in saved:
                    compare(saved[key], generated[key], f"{path}.{key}")
            elif isinstance(saved, list):
                self.assertEqual(len(saved), len(generated), path)
                for index, (left, right) in enumerate(zip(saved, generated)):
                    compare(left, right, f"{path}[{index}]")
            elif isinstance(saved, float):
                # Allow cross-NumPy roundoff, without relaxing geometry oracles.
                self.assertTrue(math.isclose(saved, generated, rel_tol=1e-14, abs_tol=1e-14),
                                f"{path}: {saved!r} != {generated!r}")
            else:
                self.assertEqual(saved, generated, path)

        for name in fixtures.CASE_NAMES:
            with self.subTest(name=name):
                path = fixtures.ROOT / f"{name}.json"
                frozen = json.loads(path.read_text())
                compare(frozen, fixtures.generate(name))
                fixtures.validate(frozen)
                counts = {camera["id"]: 0 for camera in frozen["request"]["cameras"]}
                for pick in frozen["request"]["observations"]:
                    counts[pick["match_id"]] += 1
                self.assertGreaterEqual(min(counts.values()), 8)
                self.assertFalse(frozen["expectation"]["one_view_members"])
                self.assertEqual(frozen["truth"]["scene"],
                                 "ideal_unobstructed_landmark_scaffold")

    def test_relation_removals_keep_identical_evidence_and_truth(self):
        pairs = (("free-hard", "free-hard-removed"),
                 ("free-soft", "free-removed"),
                 ("axis-hard", "axis-hard-removed"),
                 ("axis-soft", "axis-removed"),
                 ("mirror-biased-soft", "mirror-removed"),
                 ("free-hard-exact", "free-hard-removed-exact"),
                 ("axis-hard-exact", "axis-hard-removed-exact"),
                 ("mirror-hard-offcenter-exact", "mirror-removed-exact"))
        for constrained_name, removed_name in pairs:
            with self.subTest(constrained=constrained_name):
                constrained = fixtures.generate(constrained_name)
                removed = fixtures.generate(removed_name)
                for field in ("cameras", "points", "observations"):
                    self.assertEqual(constrained["request"][field], removed["request"][field])
                self.assertEqual(constrained["truth"], removed["truth"])
                self.assertFalse(removed["request"]["plane_groups"])
                self.assertFalse(removed["request"]["mirror_pairs"])

    def test_exact_copies_keep_geometry_and_zero_pick_perturbation(self):
        for name in ("free-hard", "axis-hard", "mirror-hard-offcenter",
                     "mirror-through-anchor"):
            with self.subTest(name=name):
                noisy = fixtures.generate(name)
                exact = fixtures.generate(f"{name}-exact")
                self.assertEqual(noisy["truth"], exact["truth"])
                self.assertEqual(noisy["request"]["cameras"], exact["request"]["cameras"])
                self.assertEqual(noisy["request"]["plane_groups"], exact["request"]["plane_groups"])
                self.assertEqual(noisy["request"]["mirror_pairs"], exact["request"]["mirror_pairs"])
                self.assertEqual(noisy["pick_sigma_px"], exact["pick_sigma_px"])
                for pick in exact["request"]["observations"]:
                    truth = exact["truth"]["oracle_pixels"][pick["landmark_id"]][pick["match_id"]]
                    self.assertEqual([pick["u"], pick["v"]], truth)

    def test_biased_mirror_controls_preserve_picks_but_change_constraint(self):
        hard = fixtures.generate("mirror-biased-hard")
        soft = fixtures.generate("mirror-biased-soft")
        exact = fixtures.generate("mirror-hard-offcenter")
        self.assertEqual(hard["request"]["observations"], soft["request"]["observations"])
        self.assertEqual(hard["request"]["observations"], exact["request"]["observations"])
        self.assertEqual(hard["truth"], soft["truth"])
        self.assertEqual(hard["truth"], exact["truth"])
        self.assertFalse(hard["expectation"]["reference_truth_satisfies_request"])
        self.assertTrue(soft["expectation"]["reference_truth_satisfies_request"])
        self.assertFalse(hard["expectation"]["reference_truth_exact"])
        self.assertFalse(soft["expectation"]["reference_truth_exact"])
        self.assertEqual(hard["expectation"]["behavior"], "conditional_metric_bias")
        self.assertEqual(soft["expectation"]["behavior"], "conditional_metric_bias")
        self.assertFalse(hard["expectation"]["metric_truth_independently_known_to_solver"])
        self.assertEqual(hard["request"]["mirror_slack"], 0.0)
        self.assertGreater(soft["request"]["mirror_slack"], 0.0)

    def test_biased_mirror_has_exact_rescaled_pixel_and_hard_plane_witness(self):
        for name in ("mirror-biased-hard", "mirror-biased-soft"):
            with self.subTest(name=name):
                case = fixtures.generate(name)
                witness = fixtures.biased_mirror_scale_witness(case)
                self.assertGreater(abs(witness["scale"] - 1.0), 0.01)
                for left, right in case["request"]["mirror_pairs"]:
                    reflected = fixtures._reflect(witness["landmarks"][left],
                                                  case["request"]["mirror_plane"])
                    self.assertLess(np.linalg.norm(reflected - witness["landmarks"][right]), 1e-9)
                for point_id, view_pixels in case["truth"]["oracle_pixels"].items():
                    for camera_id, expected in view_pixels.items():
                        pixel = fixtures.project([witness["landmarks"][point_id]],
                                                 witness["cameras"][camera_id])[0][0]
                        self.assertLess(np.linalg.norm(pixel - expected), 1e-9)

    def test_tilted_supplied_mirror_is_a_reference_orientation_mismatch_control(self):
        exact = fixtures.generate("mirror-hard-offcenter")
        tilted = fixtures.generate("mirror-tilted-hard")
        self.assertEqual(exact["truth"], tilted["truth"])
        self.assertEqual(exact["request"]["observations"], tilted["request"]["observations"])
        self.assertFalse(tilted["expectation"]["reference_truth_satisfies_request"])
        self.assertEqual(tilted["expectation"]["behavior"], "orientation_conflict_probe")
        first = np.asarray(exact["request"]["mirror_plane"][1])
        second = np.asarray(tilted["request"]["mirror_plane"][1])
        self.assertGreater(np.linalg.norm(first - second), 0.1)

    def test_anchor_chart_and_mirror_scale_controls(self):
        off = fixtures.generate("mirror-hard-offcenter")
        through = fixtures.generate("mirror-through-anchor")
        for case, expected in ((off, "anchor-plane-offset"), (through, "free")):
            anchor = case["request"]["anchor_id"]
            truth = next(c for c in case["truth"]["cameras"] if c["id"] == anchor)
            stored = next(c for c in case["request"]["cameras"] if c["id"] == anchor)
            self.assertEqual(truth["rotation"], stored["rotation"])
            self.assertEqual(truth["center"], stored["center"])
            self.assertGreater(np.linalg.norm(np.asarray(truth["rotation"]) - np.eye(3)), 0.1)
            self.assertEqual(case["expectation"]["scale_gauge"], expected)
            origin, normal = (np.asarray(item) for item in case["request"]["mirror_plane"])
            signed = float(normal @ (np.asarray(truth["center"]) - origin))
            if expected == "free":
                self.assertLess(abs(signed), 1e-10)
            else:
                self.assertGreater(abs(signed), 0.5)

    def test_axis_normal_transforms_into_nonidentity_anchor_chart(self):
        case = fixtures.generate("axis-hard")
        anchor = case["truth"]["cameras"][0]
        rotation = np.asarray(anchor["rotation"])
        center = np.asarray(anchor["center"])
        baseline = np.linalg.norm(np.asarray(case["truth"]["cameras"][1]["center"]) - center)
        normal_chart = rotation @ np.array((1.0, 0.0, 0.0))
        self.assertGreater(np.linalg.norm(normal_chart - (1, 0, 0)), 0.1)
        coordinates = [normal_chart @ (rotation @ (np.asarray(case["truth"]["points"][item[0]]) - center) / baseline)
                       for item in case["request"]["plane_groups"]]
        self.assertLess(np.ptp(coordinates), 1e-12)

    def test_reordered_nonanchor_inputs_leave_reference_assessment_unchanged(self):
        case = fixtures.generate("axis-hard")
        original = deepcopy(case)
        anchor, *others = case["request"]["cameras"]
        case["request"]["cameras"] = [anchor, *reversed(others)]
        case["request"]["observations"].reverse()
        fixtures.validate(case)
        self.assertEqual(case["truth"], original["truth"])
        fit = dict(cameras={item["id"]: item for item in case["truth"]["cameras"]},
                   landmarks=case["truth"]["points"])
        first = fixtures.assess(original, fit)
        second = fixtures.assess(case, fit)
        self.assertEqual(first, second)

    def test_other_private_poses_are_not_reference_geometry(self):
        case = fixtures.generate("mirror-hard-offcenter")
        altered = deepcopy(case)
        for camera in altered["request"]["cameras"][1:]:
            camera["center"] = (np.asarray(camera["center"]) + (0.7, -0.4, 0.2)).tolist()
            camera["rotation"] = fixtures.look_at(camera["center"], (0.1, -0.3, 0.7))
        fixtures.validate(altered)
        self.assertEqual(case["truth"], altered["truth"])
        fit = dict(cameras={item["id"]: item for item in case["truth"]["cameras"]},
                   landmarks=case["truth"]["points"])
        self.assertEqual(fixtures.assess(case, fit), fixtures.assess(altered, fit))

    def test_truth_assessment_has_zero_withheld_error_and_reports_bias(self):
        for name in fixtures.CASE_NAMES:
            with self.subTest(name=name):
                case = fixtures.generate(name)
                fit = dict(cameras={item["id"]: item for item in case["truth"]["cameras"]},
                           landmarks=case["truth"]["points"])
                scored = fixtures.assess(case, fit)
                self.assertAlmostEqual(scored["scale_ratio"], 1.0, delta=1e-12)
                self.assertLess(scored["withheld_max_px"], 1e-10)
                self.assertLess(scored["center_rmse_world"], 1e-10)
                self.assertLess(scored["landmark_rmse_world"], 1e-10)
                self.assertLess(scored["max_rotation_error_deg"], 1e-5)
                self.assertLess(max(abs(value) for value in scored["focal_relative"].values()), 1e-12)
                base_name = name.removesuffix("-exact")
                if base_name == "mirror-biased-hard" or base_name == "mirror-biased-soft":
                    self.assertAlmostEqual(scored["mirror_midpoint_offset_world"], -0.08, delta=1e-10)
                    self.assertAlmostEqual(scored["mirror_max_gap_world"], 0.16, delta=1e-10)
                    self.assertLess(scored["mirror_max_gap_shifted_world"], 1e-10)
                elif base_name.startswith("mirror-") and base_name not in {"mirror-removed", "mirror-tilted-hard"}:
                    self.assertLess(scored["mirror_max_gap_world"], 1e-10)

    def test_free_scale_assessment_fits_one_scale_from_training_points_only(self):
        case = fixtures.generate("free-hard")
        anchor = np.asarray(case["truth"]["cameras"][0]["center"])
        fitted_cameras = deepcopy(case["truth"]["cameras"])
        for camera in fitted_cameras:
            camera["center"] = (anchor + 1.3 * (np.asarray(camera["center"]) - anchor)).tolist()
        fitted_points = {key: (anchor + 1.3 * (np.asarray(point) - anchor)).tolist()
                         for key, point in case["truth"]["points"].items()}
        score = fixtures.assess(case, dict(
            cameras={item["id"]: item for item in fitted_cameras},
            landmarks=fitted_points))
        self.assertAlmostEqual(score["scale_ratio"], 1.3, delta=1e-12)
        self.assertAlmostEqual(score["alignment_scale"], 1.3, delta=1e-12)
        self.assertLess(score["withheld_max_px"], 1e-10)
        self.assertLess(score["landmark_rmse_world"], 1e-10)

    def test_invalid_oracle_and_anchor_are_detected(self):
        case = fixtures.generate("axis-hard")
        bad_oracle = deepcopy(case)
        bad_oracle["truth"]["oracle_pixels"]["base_00"]["view_0"][0] += 4.0
        with self.assertRaises(ValueError):
            fixtures.validate(bad_oracle)
        bad_anchor = deepcopy(case)
        bad_anchor["request"]["cameras"][0]["rotation"][0][0] += 0.01
        with self.assertRaises(ValueError):
            fixtures.validate(bad_anchor)


def _compiled(case):
    truth = case["truth"]
    anchor = next(item for item in truth["cameras"]
                  if item["id"] == case["request"]["anchor_id"])
    rotation = np.asarray(anchor["rotation"], float)
    center = np.asarray(anchor["center"], float)
    baseline = max(np.linalg.norm(np.asarray(item["center"]) - center)
                   for item in truth["cameras"] if item["id"] != anchor["id"])
    ids = sorted(truth["points"])
    points = np.asarray([rotation @ (np.asarray(truth["points"][key]) - center) / baseline
                         for key in ids])
    request = case["request"]
    model = PointFocalConstraints.from_inputs(
        ids, anchor_rotation=rotation, anchor_center=center,
        baseline_world=baseline, plane_groups=request["plane_groups"],
        plane_slack=request["plane_slack"], mirror_pairs=request["mirror_pairs"],
        mirror_plane=request["mirror_plane"], mirror_slack=request["mirror_slack"])
    return model, points


class PointFocalConstraintModelTests(TestCase):
    def test_live_mirror_reference_has_joint_jacobian_and_no_metric_scale_prior(self):
        case = fixtures.generate("mirror-hard-offcenter-exact")
        old, old_points = _compiled(case)
        reference_id = "reference"
        ids = sorted(case["truth"]["points"]) + [reference_id]
        left, right = old.mirror_pairs[0]
        points = np.vstack((old_points, (old_points[left] + old_points[right]) / 2))
        anchor = case["truth"]["cameras"][0]
        normal = np.asarray(case["request"]["mirror_plane"][1])
        model = PointFocalConstraints.from_inputs(
            ids, anchor_rotation=np.asarray(anchor["rotation"]),
            anchor_center=np.asarray(anchor["center"]), baseline_world=8.0,
            plane_groups=[], plane_slack=0.0,
            mirror_pairs=case["request"]["mirror_pairs"],
            mirror_plane=(np.full(3, np.nan), normal), mirror_slack=0.15,
            mirror_landmark_id=reference_id)
        self.assertEqual(model.mirror_reference_index, len(ids) - 1)
        self.assertFalse(model.free_baseline)
        self.assertLess(model.world_gaps(points)[1], 1e-9)
        self.assertLess(model.world_gaps(points * 2.3)[1], 1e-9)
        parameter_count = points.size + 1
        offset_column = points.size
        residual, jac = model.residual_and_jacobian(
            points, point_offset=0, parameter_count=parameter_count,
            mirror_offset_column=offset_column, jacobian=True)
        self.assertLess(np.max(abs(residual)), 1e-8)
        for column in range(points.size - 3, parameter_count):
            shifted = points.copy()
            offset = 0.0
            step = 1e-7
            if column == offset_column:
                offset = step
            else:
                shifted.reshape(-1)[column] += step
            changed, _ = model.residual_and_jacobian(
                shifted, point_offset=0, parameter_count=parameter_count,
                mirror_offset=offset, mirror_offset_column=offset_column,
                jacobian=False)
            np.testing.assert_allclose(jac[:, column], (changed - residual) / step,
                                       rtol=2e-6, atol=1e-4)
        moved = points.copy()
        moved[-1] += 0.1 * model.mirror_normal
        self.assertGreater(model.world_gaps(moved)[1], 0.1)

    def test_live_mirror_reference_is_covariant_under_world_translation(self):
        case = fixtures.generate("mirror-hard-offcenter-exact")
        anchor = case["truth"]["cameras"][0]
        normal = np.asarray(case["request"]["mirror_plane"][1])
        ids = sorted(case["truth"]["points"]) + ["reference"]
        shift = np.asarray((2500.0, -3100.0, 750.0))
        common = dict(point_ids=ids, anchor_rotation=np.asarray(anchor["rotation"]),
                      baseline_world=8.0, plane_groups=[], plane_slack=0.0,
                      mirror_pairs=case["request"]["mirror_pairs"],
                      mirror_slack=0.0, mirror_landmark_id="reference")
        first = PointFocalConstraints.from_inputs(
            **common, anchor_center=np.asarray(anchor["center"]),
            mirror_plane=(np.zeros(3), normal))
        shifted = PointFocalConstraints.from_inputs(
            **common, anchor_center=np.asarray(anchor["center"]) + shift,
            mirror_plane=(np.full(3, 1e9), normal))
        self.assertFalse(first.free_baseline)
        self.assertFalse(shifted.free_baseline)
        points = np.random.default_rng(19).normal(size=(len(ids), 3))
        np.testing.assert_allclose(
            first.residual_and_jacobian(points, point_offset=0,
                                        parameter_count=points.size, jacobian=False)[0],
            shifted.residual_and_jacobian(points, point_offset=0,
                                          parameter_count=points.size, jacobian=False)[0])

    def test_live_mirror_reference_rejects_missing_and_paired_landmarks(self):
        case = fixtures.generate("mirror-hard-offcenter")
        anchor = case["truth"]["cameras"][0]
        common = dict(point_ids=sorted(case["truth"]["points"]),
                      anchor_rotation=np.asarray(anchor["rotation"]),
                      anchor_center=np.asarray(anchor["center"]), baseline_world=8.0,
                      plane_groups=[], plane_slack=0.0,
                      mirror_pairs=case["request"]["mirror_pairs"],
                      mirror_plane=case["request"]["mirror_plane"], mirror_slack=0.0)
        with self.assertRaisesRegex(ValueError, "reference landmark needs two-view picks"):
            PointFocalConstraints.from_inputs(**common, mirror_landmark_id="missing")
        with self.assertRaisesRegex(ValueError, "reference landmark ID is invalid"):
            PointFocalConstraints.from_inputs(**common, mirror_landmark_id=[])
        with self.assertRaisesRegex(ValueError, "cannot be a mirror pair member"):
            PointFocalConstraints.from_inputs(
                **common, mirror_landmark_id=case["request"]["mirror_pairs"][0][0])

    def test_conditional_pixel_noise_operator_excludes_prior_rows_as_clicks(self):
        # Two raw pixels determine one scalar; a hard geometric prior shrinks
        # the pixel influence without becoming a third independent click.
        pixel_jacobian = np.ones((2, 1))
        influence = np.full((1, 2), 0.25)
        operator = np.eye(2) - pixel_jacobian @ influence
        np.testing.assert_allclose(np.linalg.svd(operator, compute_uv=False), (1.0, 0.5))
        self.assertTrue(_fits_constrained_noise_model(
            np.array((0.2, -0.3)), pixel_jacobian, influence, 0.35))
        self.assertFalse(_fits_constrained_noise_model(
            np.array((5.0, -5.0)), pixel_jacobian, influence, 0.35))

    def test_axis_free_and_mirror_gauges_match_fixed_anchor_geometry(self):
        for name, expected in (("axis-hard", False), ("free-hard", False),
                               ("mirror-hard-offcenter", True),
                               ("mirror-through-anchor", False)):
            with self.subTest(name=name):
                model, points = _compiled(fixtures.generate(name))
                self.assertEqual(model.free_baseline, expected)
                residual, _jac = model.residual_and_jacobian(
                    points, point_offset=0, parameter_count=points.size, jacobian=False)
                self.assertLess(np.max(abs(residual)), 1e-7)

    def test_float32_through_anchor_mirror_keeps_unit_baseline_gauge(self):
        case = fixtures.generate("mirror-through-anchor")
        anchor = case["truth"]["cameras"][0]
        plane = case["request"]["mirror_plane"]
        rounded_plane = tuple(np.asarray(value, np.float32).astype(float) for value in plane)
        ids = sorted(case["truth"]["points"])
        model = PointFocalConstraints.from_inputs(
            ids, anchor_rotation=np.asarray(anchor["rotation"]),
            anchor_center=np.asarray(anchor["center"], np.float32).astype(float),
            baseline_world=8.0, plane_groups=[], plane_slack=0.0,
            mirror_pairs=case["request"]["mirror_pairs"],
            mirror_plane=rounded_plane, mirror_slack=0.0)
        self.assertFalse(model.free_baseline)
        off, _points = _compiled(fixtures.generate("mirror-hard-offcenter"))
        self.assertTrue(off.free_baseline)

    def test_mirror_gauge_is_translation_invariant_and_preserves_input_normal(self):
        for name, expected in (("mirror-hard-offcenter", True),
                               ("mirror-through-anchor", False)):
            with self.subTest(name=name):
                case = fixtures.generate(name)
                anchor = case["truth"]["cameras"][0]
                plane = case["request"]["mirror_plane"]
                normal = 4.0 * np.asarray(plane[1], float)
                original = normal.copy()
                shift = np.asarray((250000.0, -370000.0, 110000.0))
                ids = sorted(case["truth"]["points"])
                common = dict(point_ids=ids, anchor_rotation=np.asarray(anchor["rotation"]),
                              baseline_world=8.0, plane_groups=[], plane_slack=0.0,
                              mirror_pairs=case["request"]["mirror_pairs"], mirror_slack=0.0)
                first = PointFocalConstraints.from_inputs(
                    **common, anchor_center=np.asarray(anchor["center"]),
                    mirror_plane=(np.asarray(plane[0]), normal))
                shifted = PointFocalConstraints.from_inputs(
                    **common, anchor_center=np.asarray(anchor["center"]) + shift,
                    mirror_plane=(np.asarray(plane[0]) + shift, normal))
                np.testing.assert_array_equal(normal, original)
                self.assertEqual(first.free_baseline, expected)
                self.assertEqual(shifted.free_baseline, expected)
                self.assertAlmostEqual(first.mirror_distance, shifted.mirror_distance,
                                       delta=1e-10)

    def test_inactive_groups_and_unused_mirror_empty_add_no_prior_or_scale(self):
        case = fixtures.generate("free-hard")
        anchor = case["truth"]["cameras"][0]
        ids = sorted(case["truth"]["points"])
        point_set = {key: index for index, key in enumerate(ids)}
        free_members = [item for item in case["request"]["plane_groups"]
                        if item[0] in point_set][:3]
        axis_members = [(ids[0], "X", 3)]
        mirror = fixtures.generate("mirror-hard-offcenter")["request"]["mirror_plane"]
        model = PointFocalConstraints.from_inputs(
            ids, anchor_rotation=np.asarray(anchor["rotation"]),
            anchor_center=np.asarray(anchor["center"]), baseline_world=8.0,
            plane_groups=free_members + axis_members, plane_slack=0.0,
            mirror_pairs=[], mirror_plane=mirror, mirror_slack=0.0)
        self.assertFalse(model.active)
        self.assertFalse(model.free_baseline)
        residual, jac = model.residual_and_jacobian(
            np.zeros((len(ids), 3)), point_offset=0,
            parameter_count=3 * len(ids), jacobian=True)
        self.assertEqual(residual.size, 0)
        self.assertEqual(jac.shape, (0, 3 * len(ids)))

    def test_axis_and_mirror_analytic_jacobians_match_independent_differences(self):
        for name in ("axis-hard", "mirror-hard-offcenter", "mirror-biased-soft"):
            with self.subTest(name=name):
                model, points = _compiled(fixtures.generate(name))
                parameters = points.size + int(model.free_mirror_offset)
                offset_col = points.size if model.free_mirror_offset else None
                residual, jac = model.residual_and_jacobian(
                    points, point_offset=0, parameter_count=parameters,
                    mirror_offset_column=offset_col, jacobian=True)
                assert jac is not None
                columns = [0, 2, points.size // 2, points.size - 1]
                if offset_col is not None:
                    columns.append(offset_col)
                for column in columns:
                    shifted = points.copy()
                    offset = 0.0
                    step = 1e-7
                    if column == offset_col:
                        offset = step
                    else:
                        shifted.reshape(-1)[column] += step
                    changed, _ = model.residual_and_jacobian(
                        shifted, point_offset=0, parameter_count=parameters,
                        mirror_offset=offset, mirror_offset_column=offset_col,
                        jacobian=False)
                    np.testing.assert_allclose(jac[:, column], (changed - residual) / step,
                                               rtol=2e-6, atol=1e-4)

    def test_final_world_gaps_detect_broken_hard_relations(self):
        axis, axis_points = _compiled(fixtures.generate("axis-hard"))
        plane_good, mirror_good = axis.world_gaps(axis_points)
        self.assertLess(plane_good, PLANE_HARD_SLACK)
        self.assertEqual(mirror_good, 0.0)
        broken_axis = axis_points.copy()
        member = axis.axis_groups[0][1][1]
        broken_axis[member] += 0.02 * axis.axis_groups[0][0] / axis.baseline_world
        self.assertGreater(axis.world_gaps(broken_axis)[0], PLANE_HARD_SLACK)

        mirror, mirror_points = _compiled(fixtures.generate("mirror-hard-offcenter"))
        self.assertLess(mirror.world_gaps(mirror_points)[1], MIRROR_PAIR_HARD_GAP)
        broken_mirror = mirror_points.copy()
        partner = mirror.mirror_pairs[0][1]
        broken_mirror[partner] += 0.02 * mirror.mirror_normal / mirror.baseline_world
        self.assertGreater(mirror.world_gaps(broken_mirror)[1], MIRROR_PAIR_HARD_GAP)

    def test_public_point_route_initializes_from_picks_then_fits_relations(self):
        for name in ("free-hard", "axis-hard", "mirror-hard-offcenter"):
            with self.subTest(name=name):
                case = fixtures.generate(name)
                truth_cameras = {item["id"]: item for item in case["truth"]["cameras"]}
                matches = []
                similarities = {}
                for camera in case["request"]["cameras"]:
                    k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                                  camera["cy"], camera["width"], camera["height"])
                    calibration = geometry.Calibration(k, np.asarray(camera["rotation"]),
                                                       np.asarray(camera["center"]))
                    matches.append(lens_refine.MatchLensInput(camera["id"], {}, k,
                                                               base_calibration=calibration))
                    true = truth_cameras[camera["id"]]
                    rotation = np.asarray(true["rotation"]).T @ np.asarray(camera["rotation"])
                    center = np.asarray(true["center"])
                    similarities[camera["id"]] = sync.SimilarityTransform(
                        1.0, rotation, center - rotation @ camera["center"])
                initial = sync.SyncSolveResult(
                    similarities=similarities,
                    landmarks={key: np.asarray(value) for key, value in case["truth"]["points"].items()},
                    mean_reprojection_px=0.0, per_match_rmse_px={}, per_landmark_rmse_px={},
                    message="mocked initial", success=True)
                observations = [sync.SyncObservation(**item) for item in case["request"]["observations"]]
                with (mock.patch.object(lens_refine, "_run_sync", return_value=initial) as run_sync,
                      mock.patch.object(lens_refine, "fit_independent_focals",
                                        return_value=FocalBundleOutcome(False, "mocked bundle")) as run_bundle):
                    outcome = lens_refine.refine_lenses_from_landmarks(
                        matches, observations, anchor_id=case["request"]["anchor_id"],
                        share_lens=False, estimate_focal_from_points=True,
                        pick_sigma_px=case["pick_sigma_px"],
                        plane_groups=case["request"]["plane_groups"],
                        plane_slack=case["request"]["plane_slack"],
                        mirror_pairs=case["request"]["mirror_pairs"],
                        mirror_plane=case["request"]["mirror_plane"],
                        mirror_slack=case["request"]["mirror_slack"])
                self.assertEqual(outcome.refusal_reason, "mocked bundle")
                self.assertIsNone(run_sync.call_args.kwargs["plane_groups"])
                self.assertIsNone(run_sync.call_args.kwargs["mirror_pairs"])
                self.assertIsNone(run_sync.call_args.kwargs["mirror_plane"])
                self.assertEqual(run_bundle.call_args.kwargs["plane_groups"],
                                 case["request"]["plane_groups"])
                self.assertEqual(run_bundle.call_args.kwargs["mirror_pairs"],
                                 case["request"]["mirror_pairs"])
                self.assertEqual(run_bundle.call_args.kwargs["mirror_plane"],
                                 case["request"]["mirror_plane"])

    def test_public_point_route_refuses_unsupported_inputs_before_sync(self):
        case = fixtures.generate("mirror-hard-offcenter")
        request = case["request"]
        matches = []
        for camera in request["cameras"]:
            k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                          camera["cy"], camera["width"], camera["height"])
            matches.append(lens_refine.MatchLensInput(
                camera["id"], {}, k, base_calibration=geometry.Calibration(
                    k, np.asarray(camera["rotation"]), np.asarray(camera["center"]))))
        observations = [sync.SyncObservation(**item) for item in request["observations"]]
        one_view = [item for item in observations
                    if item.landmark_id != "base_00" or item.match_id == "view_0"]
        ground = [replace(item, on_ground=True) if item.landmark_id == "base_00" else item
                  for item in observations]
        common = dict(anchor_id=request["anchor_id"], share_lens=False,
                      estimate_focal_from_points=True, pick_sigma_px=case["pick_sigma_px"],
                      mirror_pairs=request["mirror_pairs"],
                      mirror_plane=request["mirror_plane"])
        variants = (
            ("one-view", one_view, {}, "at least two camera picks"),
            ("missing Mirror Empty", observations, {"mirror_plane": None}, "Mirror Empty"),
            ("unknown mirror member", observations,
             {"mirror_pairs": [("base_00", "absent")]}, "without two-view picks"),
            ("line", observations, {"line_observations": [object()]}, "Invalid line landmarks"),
            ("Known 3D", observations, {"known_world": {"base_00": np.zeros(3)}},
             "Known 3D"),
            ("ground", ground, {}, "On Ground"),
            ("pose lock", observations, {"lock_translation": True}, "pose locks"),
        )
        for label, picks, changes, expected in variants:
            with self.subTest(label=label), mock.patch.object(lens_refine, "_run_sync") as run_sync:
                result = lens_refine.refine_lenses_from_landmarks(
                    matches, picks, **{**common, **changes})
                self.assertIn(expected, result.refusal_reason)
                run_sync.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
