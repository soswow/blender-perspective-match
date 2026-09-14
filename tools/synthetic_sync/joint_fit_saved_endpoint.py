"""Replay one saved Sync initializer through the common fitter without registration."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "match_perspective" not in sys.modules:
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    sys.modules["match_perspective"] = package

from .budget import ExperimentBudget
from match_perspective.core import focal_bundle
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core.sync.request import SyncSolveRequest, json_values, request_fingerprint
from match_perspective.core.sync.types import SimilarityTransform, SyncSolveResult


def _request(record):
    # A saved v5 continuation request carries an initializer seed that this
    # zero-registration replay supplies explicitly. The v4 evidence is exact.
    if record["version"] == 5:
        record = dict(record)
        inputs = dict(record["inputs"])
        inputs.pop("initial_solution")
        record.update(version=4, inputs=inputs, sha256=request_fingerprint(inputs))
    return SyncSolveRequest.from_record(record)


def _initial(record):
    values = dict(record)
    values["similarities"] = {
        key: SimilarityTransform(float(item["scale"]),
                                 np.asarray(item["rotation"], float),
                                 np.asarray(item["translation"], float))
        for key, item in values["similarities"].items()}
    values["landmarks"] = {key: np.asarray(item, float)
                            for key, item in values["landmarks"].items()}
    values["line_segments"] = {key: tuple(np.asarray(end, float) for end in segment)
                               for key, segment in values["line_segments"].items()}
    allowed = {field.name for field in fields(SyncSolveResult)}
    result = SyncSolveResult(**{key: value for key, value in values.items() if key in allowed})
    result.joint_mirror_offset_m = float(values.get("joint_mirror_offset_m", 0.0))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = {
        "request": json.loads(args.request.read_text()),
        "initial": json.loads(args.initial.read_text()),
        "inputs": json.loads(args.inputs.read_text()),
    }
    if "sync_result" in records["initial"]:
        records["initial"] = records["initial"]["sync_result"]
    source_paths = sorted((ROOT / "core").rglob("*.py")) + [Path(__file__),
                                                            ROOT / "tools/synthetic_sync/budget.py"]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    source_key = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:16]
    for path in source_paths:
        destination = args.out / "source" / source_key / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == source_hashes[str(path.relative_to(ROOT))]
    metadata = {"source_key": source_key, "source_hashes": source_hashes,
                "python": sys.version, "numpy": np.__version__,
                "platform": platform.platform()}
    request = _request(records["request"])
    initial = _initial(records["initial"])
    options = records["inputs"]
    calibrations = {item.match_id: item.calibration for item in request.matches}
    scorer = JointFitScorer(request, initial, calibrations=calibrations)
    diagnostic = {}
    with ExperimentBudget(args.out / "attempts.jsonl", metadata=metadata,
                          max_calls=4, wall_seconds=300.0, per_call_seconds=120.0) as budget:
        with budget.attempt("saved-endpoint", {"records": records, "source_key": source_key}) as attempt:
            outcome = focal_bundle.fit_independent_focals(
                calibrations, scorer.point_observations, initial,
                anchor_id=request.anchor_id, pick_sigma_px=options["pick_sigma_px"],
                fx_span=options["fx_span"], plane_groups=request.plane_groups,
                plane_slack=request.plane_slack, mirror_pairs=request.mirror_pairs,
                mirror_plane=request.mirror_plane, mirror_slack=request.mirror_slack,
                mirror_landmark_id=request.mirror_landmark_id,
                line_observations=request.line_observations,
                parallel_pairs=request.parallel_pairs,
                fixed_similarities=request.fixed_similarities,
                location_match_ids=request.location_match_ids,
                readonly_match_ids=request.readonly_match_ids,
                known_world=request.known_world,
                known_3d_slack=request.known_3d_slack,
                ground_slack=request.ground_slack,
                known_lines=request.known_lines,
                lock_rotation=request.lock_rotation,
                lock_translation=request.lock_translation,
                share_lens=options["share_lens"],
                diagnostic_callback=diagnostic.update)
            score = (scorer.score(
                outcome.sync_result, calibrations=outcome.calibrations)
                     if outcome.sync_result is not None else None)
            internal_objective = diagnostic.get("final_objective")
            public_objective = score.objective if score and score.valid else None
            relative_objective_delta = (
                abs(public_objective - internal_objective) /
                max(public_objective, internal_objective, 1.0)
                if public_objective is not None and internal_objective is not None else None)
            output = {"outcome": json_values(outcome), "score": json_values(score),
                      "diagnostic": json_values(diagnostic), "source_key": source_key,
                      "internal_objective": internal_objective,
                      "public_objective": public_objective,
                      "relative_objective_delta": relative_objective_delta}
            attempt.complete(output)
    (args.out / "endpoint.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({"accepted": outcome.accepted, "reason": outcome.reason,
                      "score_valid": score.valid if score else None,
                      "score_reason": score.reason if score else None,
                      "internal_objective": internal_objective,
                      "public_objective": public_objective,
                      "relative_objective_delta": relative_objective_delta,
                      "constraint_gaps": score.constraint_gaps if score else None,
                      "source_key": source_key}))


if __name__ == "__main__":
    main()
