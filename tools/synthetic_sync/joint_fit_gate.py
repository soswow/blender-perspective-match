"""Bounded generic lock, relation, graph-size and metric-plane joint-fit controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
if "match_perspective" not in sys.modules:
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    sys.modules["match_perspective"] = package

from .budget import ExperimentBudget
from test_joint_fit_features import _fixture, _pixel
from match_perspective.core import focal_bundle, geometry
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core.sync import (SimilarityTransform, SyncLineObservation,
                                         SyncMatchInput, SyncObservation)
from match_perspective.core.sync.request import json_values
from match_perspective.core.sync.projection import _rodrigues


def _sources() -> tuple[dict, Path]:
    paths = sorted((ROOT / "core").rglob("*.py")) + [
        ROOT / "tests/test_joint_fit_features.py", Path(__file__),
        ROOT / "tools/synthetic_sync/budget.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    metadata = {"sources": hashes, "python": sys.version,
                "numpy": np.__version__, "platform": platform.platform()}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    archive = Path("/tmp/pm-joint-fit-tests/round3/source") / key
    for path in paths:
        destination = archive / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == hashes[str(path.relative_to(ROOT))]
    return metadata, archive


def _lock_translation():
    request, initial, truth, centers = _fixture()
    request.matches = [SyncMatchInput(match.match_id,
                       replace(match.calibration, camera_center=centers[match.match_id].copy()))
                       for match in request.matches]
    initial.similarities = {
        "anchor": SimilarityTransform(),
        "side": SimilarityTransform(rotation=_rodrigues(np.array((0.008, -0.012, 0.006)))),
        "detail": SimilarityTransform(rotation=_rodrigues(np.array((-0.01, 0.004, -0.008)))),
    }
    request.lock_translation = True
    return request, initial, truth, centers


def _lock_translation_scale():
    request, initial, truth, centers = _lock_translation()
    initial.similarities["side"].scale = 0.93
    initial.similarities["detail"].scale = 1.05
    return request, initial, truth, centers


def _relations_and_lock_pose():
    request, initial, truth, centers = _fixture(known_line=True)
    known = request.known_lines["edge"]
    mirror = tuple(np.array((-end[0], end[1], end[2])) for end in known)
    request.line_observations += [
        SyncLineObservation(match.match_id, "mirror_edge", *(
            float(value) for point in (
                _pixel(0.8 * mirror[0] + 0.2 * mirror[1], match.calibration,
                       SimilarityTransform(translation=centers[match.match_id]-match.calibration.camera_center)),
                _pixel(0.2 * mirror[0] + 0.8 * mirror[1], match.calibration,
                       SimilarityTransform(translation=centers[match.match_id]-match.calibration.camera_center)))
            for value in point))
        for match in request.matches]
    reference = np.array((0.0, 0.3, 0.2))
    request.known_world["mirror_ref"] = reference
    initial.landmarks["mirror_ref"] = reference.copy()
    request.mirror_plane = (reference.copy(), np.array((1.0, 0.0, 0.0)))
    request.mirror_landmark_id = "mirror_ref"
    request.mirror_pairs = [("edge", "mirror_edge")]
    request.plane_groups = [(key, "Z", 1) for key in ("p0", "p1", "p2")]
    reference_parallel = (mirror[0] + np.array((0.0, 0.2, 0.1)),
                          mirror[1] + np.array((0.0, 0.2, 0.1)))
    request.known_lines["parallel_ref"] = reference_parallel
    request.parallel_pairs = [("mirror_edge", "parallel_ref")]
    initial.line_segments["parallel_ref"] = reference_parallel
    initial.line_segments["mirror_edge"] = tuple(
        end + np.array((0.01, -0.015, 0.008)) for end in mirror)
    detail = next(item for item in request.matches if item.match_id == "detail")
    locked = SimilarityTransform(translation=centers["detail"]-detail.calibration.camera_center)
    initial.similarities["detail"] = locked
    request.fixed_similarities = {"detail": deepcopy(locked)}
    return request, initial, truth, centers


def _large():
    request, initial, truth, centers = _fixture()
    rng = np.random.default_rng(9173)
    for index in range(76):
        key = f"extra_{index:03d}"
        point = np.array((rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                          rng.uniform(0.12, 1.05)))
        truth[key] = point
        initial.landmarks[key] = point + rng.normal(0.0, 0.014, 3)
        for match in request.matches:
            camera_id = match.match_id
            truth_sim = SimilarityTransform(
                translation=centers[camera_id]-match.calibration.camera_center)
            uv = _pixel(point, match.calibration, truth_sim)
            request.observations.append(SyncObservation(camera_id, key, float(uv[0]), float(uv[1])))
    assert len(truth) > focal_bundle.MAX_POINTS
    return request, initial, truth, centers


def _planar_metric():
    request, initial, _truth, centers = _fixture()
    request.matches = [item for item in request.matches if item.match_id in {"anchor", "side"}]
    initial.similarities = {key: value for key, value in initial.similarities.items()
                            if key in {"anchor", "side"}}
    grid = {f"metric_{x}_{y}": np.array((x * 0.34, y * 0.30, 0.0))
            for x in (-1, 0, 1) for y in (-1, 0, 1)}
    request.known_world = deepcopy(grid)
    request.observations = []
    initial.landmarks = deepcopy(grid)
    for match in request.matches:
        truth_sim = SimilarityTransform(
            translation=centers[match.match_id]-match.calibration.camera_center)
        for key, point in grid.items():
            uv = _pixel(point, match.calibration, truth_sim)
            request.observations.append(SyncObservation(match.match_id, key,
                                                         float(uv[0]), float(uv[1]),
                                                         on_ground=True))
    request.matches = [SyncMatchInput(match.match_id, replace(
        match.calibration, intrinsics=replace(match.calibration.intrinsics,
                                               fx=760.0, fy=760.0)))
        for match in request.matches]
    return request, initial, grid, centers


def _fit_only_focal():
    request, initial, truth, centers = _fixture(fit_only=True)
    request.readonly_match_ids = {"detail"}
    request.matches = [SyncMatchInput(match.match_id, replace(
        match.calibration, intrinsics=replace(match.calibration.intrinsics,
                                               fx=760.0, fy=760.0)))
        if match.match_id == "detail" else match for match in request.matches]
    return request, initial, truth, centers


def _known_line_plane():
    request, initial, truth, centers = _fixture(known_line=True)
    segment = (np.array((-0.5, 0.75, 0.35)),
               np.array((0.55, 0.75, 0.95)))
    for match in request.matches:
        true_sim = SimilarityTransform(
            translation=centers[match.match_id]-match.calibration.camera_center)
        uv0 = _pixel(0.15 * segment[0] + 0.85 * segment[1],
                     match.calibration, true_sim)
        uv1 = _pixel(0.85 * segment[0] + 0.15 * segment[1],
                     match.calibration, true_sim)
        request.line_observations.append(SyncLineObservation(
            match.match_id, "plane_edge", float(uv0[0]), float(uv0[1]),
            float(uv1[0]), float(uv1[1])))
    initial.line_segments["plane_edge"] = tuple(
        point + np.array((0.015, 0.04, -0.02)) for point in segment)
    request.plane_groups = [("edge", "Y", 2), ("plane_edge", "Y", 2)]
    return request, initial, truth, centers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("lock_translation", "relations_lock_pose",
                                         "large", "planar_metric",
                                         "lock_translation_scale", "fit_only_focal",
                                         "known_line_plane"))
    mode = parser.parse_args().case
    request, initial, truth, centers = {
        "lock_translation": _lock_translation,
        "lock_translation_scale": _lock_translation_scale,
        "relations_lock_pose": _relations_and_lock_pose,
        "large": _large,
        "planar_metric": _planar_metric,
        "fit_only_focal": _fit_only_focal,
        "known_line_plane": _known_line_plane,
    }[mode]()
    metadata, archive = _sources()
    source_key = archive.name
    ledger = Path("/tmp/pm-joint-fit-tests/round3") / f"gates-{source_key}.jsonl"
    exact_input = {"mode": mode, "request": request.to_record(),
                   "initial": json_values(initial)}
    diagnostic = {}
    with ExperimentBudget(ledger, metadata=metadata, max_calls=6,
                          wall_seconds=600.0, per_call_seconds=120.0) as budget:
        with budget.attempt(mode, exact_input) as attempt:
            if mode in {"planar_metric", "fit_only_focal"}:
                outcome = focal_bundle.fit_independent_focals(
                    {match.match_id: match.calibration for match in request.matches},
                    request.observations, initial, anchor_id=request.anchor_id,
                    known_world=request.known_world, ground_slack=0.0,
                    fx_span=0.3, diagnostic_callback=diagnostic.update,
                    location_match_ids=request.location_match_ids,
                    readonly_match_ids=request.readonly_match_ids,
                    frozen_focal_ids={"anchor", "side"} if mode == "fit_only_focal" else None)
            else:
                outcome = focal_bundle.refine_fixed_focals(
                    request, initial, diagnostic_callback=diagnostic.update)
            result = outcome.sync_result
            score = (JointFitScorer(request, initial).score(
                result, calibrations=outcome.calibrations) if result else None)
            withheld = {}
            if result:
                for match in request.matches:
                    camera_id = match.match_id
                    true_cal = match.calibration
                    if mode == "planar_metric" or mode == "fit_only_focal" and camera_id == "detail":
                        true_cal = replace(true_cal, intrinsics=replace(
                            true_cal.intrinsics, fx=880.0, fy=880.0))
                    truth_sim = SimilarityTransform(
                        translation=centers[camera_id]-true_cal.camera_center)
                    withheld[camera_id] = [float(np.linalg.norm(
                        _pixel(point, outcome.calibrations[camera_id],
                               result.similarities[camera_id]) -
                        _pixel(point, true_cal, truth_sim)))
                        for point in (np.array((-0.47, 0.11, 0.37)),
                                      np.array((0.18, -0.37, 0.92)))]
            record = {
                "accepted": outcome.accepted, "reason": outcome.reason,
                "fitted_rmse_px": (outcome.fitted_rmse_px if math.isfinite(outcome.fitted_rmse_px) else None),
                "initial_objective": outcome.initial_objective,
                "fitted_objective": outcome.fitted_objective,
                "score": json_values(score), "diagnostic": json_values(diagnostic),
                "withheld_pixel_errors": withheld,
                "coverage": json_values(outcome.support_coverage),
                "intervals_px": json_values(outcome.intervals_px),
                "calibrations": json_values(outcome.calibrations),
                "result": json_values(result),
            }
            attempt.complete(record)
    summary = {key: record[key] for key in (
        "accepted", "reason", "fitted_rmse_px", "initial_objective",
        "fitted_objective", "coverage", "withheld_pixel_errors")}
    if record["score"]:
        summary["point_rmse_px"] = record["score"]["point_rmse_px"]
        summary["line_rmse_px"] = record["score"]["line_rmse_px"]
        summary["constraint_gaps"] = record["score"]["constraint_gaps"]
    if record["result"]:
        summary["focal_px"] = {key: cal["intrinsics"]["fx"]
                               for key, cal in record["calibrations"].items()}
        summary["root_scales"] = {key: sim["scale"] for key, sim in
                                  record["result"]["similarities"].items()}
        summary["root_translations"] = {key: sim["translation"] for key, sim in
                                        record["result"]["similarities"].items()}
    summary["intervals_px"] = record["intervals_px"]
    summary["stop_reason"] = diagnostic.get("stop_reason")
    summary["ledger"] = str(ledger)
    summary["source_archive"] = str(archive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
