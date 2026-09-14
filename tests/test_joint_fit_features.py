"""Deterministic fixed-focal joint-fit contracts with exact numerical inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from match_perspective.core import focal_bundle, geometry
from match_perspective.core.sync import (
    SimilarityTransform, SyncLineObservation, SyncMatchInput, SyncObservation,
    SyncSolveResult,
)
from match_perspective.core.sync.request import SyncSolveRequest, json_values
from tools.synthetic_sync.budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[1]
LEDGER_ROOT = Path("/tmp/pm-joint-fit-tests")
RUN_ID = os.environ.get("PM_JOINT_FEATURE_RUN_ID", os.urandom(8).hex())


def _rotation(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack((right, up, forward))


def _pixel(point: np.ndarray, calibration: geometry.Calibration,
           similarity: SimilarityTransform) -> np.ndarray:
    center = similarity.transform_point(calibration.camera_center)
    rotation = calibration.rotation_w2c @ similarity.rotation.T
    q = rotation @ (point - center)
    if q[2] <= 0:
        raise ValueError("Fixture point lies behind a camera")
    k = calibration.intrinsics
    ideal = np.array(((k.fx*q[0]/q[2] + k.cx, k.fy*q[1]/q[2] + k.cy)))
    return geometry.distort_points(
        ideal.reshape(1, 2), k.fx, k.fy, k.cx, k.cy,
        calibration.division_lambda, calibration.brown_conrady)[0]


def _fixture(*, distortion: bool = False, fit_only: bool = False,
             known_line: bool = False) -> tuple[SyncSolveRequest, SyncSolveResult,
                                               dict[str, np.ndarray], dict[str, np.ndarray]]:
    target = np.array((0.0, 0.0, 0.45))
    centers = {
        "anchor": np.array((1.5, -2.5, 2.8)),
        "side": np.array((-1.8, -2.1, 2.5)),
        "detail": np.array((0.9, 2.4, 2.6)),
    }
    private_centers = {
        "anchor": centers["anchor"].copy(),
        "side": np.array((-0.4, 0.2, 0.3)),
        "detail": np.array((0.6, -0.5, 0.1)),
    }
    truth = {
        "p0": np.array((-0.65, -0.45, 0.0)),
        "p1": np.array((0.65, -0.40, 0.0)),
        "p2": np.array((-0.4, 0.55, 0.0)),
        "p3": np.array((0.5, 0.45, 0.3)),
        "p4": np.array((-0.3, -0.1, 0.8)),
        "p5": np.array((0.3, -0.15, 1.05)),
        "p6": np.array((0.0, 0.6, 0.65)),
    }
    perturb = {
        "p0": np.array((0.03, -0.02, 0.04)),
        "p1": np.array((-0.02, 0.02, 0.03)),
        "p2": np.array((0.025, 0.02, 0.05)),
        "p3": np.array((-0.025, 0.02, -0.03)),
        "p4": np.array((0.01, 0.025, -0.025)),
        "p5": np.array((-0.02, -0.015, 0.02)),
        "p6": np.array((0.02, -0.02, 0.03)),
    }
    similarities = {}
    matches = []
    observations = []
    for camera_id in centers:
        k = geometry.CameraIntrinsics(
            880.0, 1040.0 if distortion else 880.0,
            640.0, 360.0, 1280, 720)
        calibration = geometry.Calibration(
            k, _rotation(centers[camera_id], target), private_centers[camera_id],
            brown_conrady=(0.035, -0.012, 0.003, -0.002, 0.0) if distortion else ())
        matches.append(SyncMatchInput(camera_id, calibration))
        similarity = SimilarityTransform(
            translation=centers[camera_id] - private_centers[camera_id])
        similarities[camera_id] = similarity
        for point_id, point in truth.items():
            uv = _pixel(point, calibration, similarity)
            observations.append(SyncObservation(
                camera_id, point_id, float(uv[0]), float(uv[1]),
                on_ground=point_id in {"p0", "p1", "p2"}))
    line_observations = []
    known_lines = None
    line_segments = {}
    if known_line:
        left = np.array((-0.75, 0.75, 0.15))
        right = np.array((0.75, 0.75, 0.55))
        known_lines = {"edge": (left, right)}
        line_segments = deepcopy(known_lines)
        for match in matches:
            camera_id = match.match_id
            sample_left = left * 0.8 + right * 0.2
            sample_right = left * 0.2 + right * 0.8
            uv0 = _pixel(sample_left, match.calibration, similarities[camera_id])
            uv1 = _pixel(sample_right, match.calibration, similarities[camera_id])
            line_observations.append(SyncLineObservation(
                camera_id, "edge", float(uv0[0]), float(uv0[1]),
                float(uv1[0]), float(uv1[1])))
    seed_similarities = deepcopy(similarities)
    seed_similarities["side"].translation += np.array((0.055, -0.035, 0.035))
    seed_similarities["detail"].translation += np.array((-0.05, 0.035, -0.04))
    initial = SyncSolveResult(
        similarities=seed_similarities,
        landmarks={key: point + perturb[key] for key, point in truth.items()},
        line_segments=line_segments, mean_reprojection_px=0.0,
        per_match_rmse_px={}, per_landmark_rmse_px={}, message="synthetic seed")
    request = SyncSolveRequest(
        matches=matches, observations=observations, anchor_id="anchor",
        known_world={"p0": truth["p0"], "p5": truth["p5"]},
        known_3d_slack=0.0, ground_slack=0.0,
        location_match_ids={"anchor", "side"} if fit_only else set(centers),
        line_observations=line_observations, known_lines=known_lines,
    )
    return request, initial, truth, centers


def _metadata() -> dict:
    source = {}
    paths = sorted((ROOT / "core").rglob("*.py")) + [
        Path(__file__), ROOT / "tools/synthetic_sync/budget.py"]
    for path in paths:
        source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_sha256": source, "python": sys.version,
            "numpy": np.__version__, "platform": platform.platform()}


def _restored_outcome(record: dict) -> SimpleNamespace:
    calibrations = {}
    for key, source in record["calibrations"].items():
        intrinsics = SimpleNamespace(**source["intrinsics"])
        calibrations[key] = SimpleNamespace(
            intrinsics=intrinsics,
            brown_conrady=tuple(source["brown_conrady"]),
        )
    source = record["sync_result"]
    sync_result = None
    if source is not None:
        sync_result = SimpleNamespace(
            landmarks={key: np.asarray(value, dtype=float)
                       for key, value in source["landmarks"].items()},
            line_segments={key: np.asarray(value, dtype=float)
                           for key, value in source["line_segments"].items()},
            similarities={key: SimpleNamespace(
                scale=float(value["scale"]),
                rotation=np.asarray(value["rotation"], dtype=float),
                translation=np.asarray(value["translation"], dtype=float))
                for key, value in source["similarities"].items()},
            per_match_rmse_px=source["per_match_rmse_px"],
            point_rmse_px=source.get("point_rmse_px"),
            line_rmse_px=source.get("line_rmse_px"),
        )
    return SimpleNamespace(
        accepted=record["accepted"], reason=record["reason"],
        initial_rmse_px=record["initial_rmse_px"],
        fitted_rmse_px=record["fitted_rmse_px"],
        calibrations=calibrations, sync_result=sync_result,
    )


def _fit(label: str, request: SyncSolveRequest, initial: SyncSolveResult):
    exact_input = {"request": request.to_record(), "initial": json_values(initial)}
    metadata = _metadata()
    source_key = hashlib.sha256(json.dumps(
        metadata, sort_keys=True).encode()).hexdigest()[:16]
    archive = LEDGER_ROOT / "features-source" / source_key
    for name, digest in metadata["source_sha256"].items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT / name, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    ledger = LEDGER_ROOT / f"features-{source_key}-{RUN_ID}.jsonl"
    with ExperimentBudget(
        ledger, metadata=metadata, max_calls=6,
        wall_seconds=300.0, per_call_seconds=60.0,
    ) as budget:
        with budget.attempt(label, exact_input) as attempt:
            outcome = focal_bundle.refine_fixed_focals(request, initial)
            def finite(value: float) -> float | None:
                return float(value) if math.isfinite(value) else None
            attempt.complete({
                "accepted": outcome.accepted, "reason": outcome.reason,
                "initial_rmse_px": finite(outcome.initial_rmse_px),
                "fitted_rmse_px": finite(outcome.fitted_rmse_px),
                "calibrations": json_values(outcome.calibrations),
                "sync_result": json_values(outcome.sync_result),
            })
    return outcome


class JointFixedFocalFeaturesTests(TestCase):
    def test_known_3d_and_ground_with_oblique_anchor_and_private_centers(self) -> None:
        request, initial, truth, centers = _fixture()
        anchor = request.matches[0].calibration
        self.assertGreater(abs(float(anchor.rotation_w2c[2, 0])), 0.2)
        self.assertFalse(np.allclose(request.matches[1].calibration.camera_center,
                                     centers["side"]))
        result = _fit("hard-point-ground", request, initial)
        self.assertTrue(result.accepted, result.reason)
        fitted = result.sync_result
        self.assertLess(result.fitted_rmse_px, result.initial_rmse_px)
        for point_id in ("p0", "p5"):
            np.testing.assert_allclose(fitted.landmarks[point_id], truth[point_id], atol=1e-7)
        for point_id in ("p0", "p1", "p2"):
            self.assertAlmostEqual(float(fitted.landmarks[point_id][2]), 0.0, places=7)
        for point_id in truth:
            np.testing.assert_allclose(fitted.landmarks[point_id], truth[point_id], atol=0.02)

    def test_fit_only_seed_pose_cannot_change_shared_geometry(self) -> None:
        request, initial, _truth, _centers = _fixture(fit_only=True)
        first = _fit("fit-only-base", request, initial)
        changed = deepcopy(initial)
        changed.similarities["detail"].translation += np.array((0.12, -0.08, 0.07))
        second = _fit("fit-only-pose-perturbed", request, changed)
        self.assertTrue(first.accepted, first.reason)
        self.assertTrue(second.accepted, second.reason)
        for key in first.sync_result.landmarks:
            np.testing.assert_allclose(
                first.sync_result.landmarks[key], second.sync_result.landmarks[key],
                atol=1e-8)
        np.testing.assert_allclose(
            first.sync_result.similarities["side"].translation,
            second.sync_result.similarities["side"].translation, atol=1e-8)
        np.testing.assert_allclose(
            first.sync_result.similarities["side"].rotation,
            second.sync_result.similarities["side"].rotation, atol=1e-8)
        self.assertAlmostEqual(first.sync_result.similarities["side"].scale,
                               second.sync_result.similarities["side"].scale, places=12)
        self.assertLess(second.sync_result.per_match_rmse_px["detail"], 0.1)

    def test_fixed_known_line_supports_aspect_and_brown_distortion(self) -> None:
        request, initial, truth, _centers = _fixture(
            distortion=True, known_line=True)
        result = _fit("known-line-distorted", request, initial)
        self.assertTrue(result.accepted, result.reason)
        self.assertLess(result.fitted_rmse_px, 0.1)
        self.assertLess(result.sync_result.line_rmse_px, 0.1)
        for point_id, expected in truth.items():
            np.testing.assert_allclose(result.sync_result.landmarks[point_id],
                                       expected, atol=0.02)
        np.testing.assert_allclose(
            result.sync_result.line_segments["edge"], request.known_lines["edge"])
        for match in request.matches:
            fitted = result.calibrations[match.match_id]
            original = match.calibration
            self.assertEqual(fitted.intrinsics.fx, original.intrinsics.fx)
            self.assertEqual(fitted.intrinsics.fy, original.intrinsics.fy)
            self.assertEqual(fitted.brown_conrady, original.brown_conrady)
            for point in (np.array((-0.47, 0.11, 0.37)),
                          np.array((0.18, -0.37, 0.92))):
                truth_similarity = SimilarityTransform(
                    translation=_centers[match.match_id] - original.camera_center)
                expected = _pixel(point, original, truth_similarity)
                actual = _pixel(point, fitted,
                                result.sync_result.similarities[match.match_id])
                self.assertLess(float(np.linalg.norm(actual - expected)), 0.1)

    def test_repeated_fixed_fit_is_stable(self) -> None:
        request, initial, _truth, _centers = _fixture()
        first = _fit("repeat-1", request, initial)
        self.assertTrue(first.accepted, first.reason)
        continued_request = replace(request, matches=[
            SyncMatchInput(match.match_id, first.calibrations[match.match_id])
            for match in request.matches])
        second = _fit("repeat-2", continued_request, first.sync_result)
        self.assertTrue(second.accepted, second.reason)
        self.assertAlmostEqual(first.fitted_rmse_px, second.fitted_rmse_px, places=12)
        for key in first.sync_result.landmarks:
            np.testing.assert_allclose(
                first.sync_result.landmarks[key], second.sync_result.landmarks[key],
                rtol=0.0, atol=1e-12)
