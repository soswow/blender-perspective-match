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
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "match_perspective" not in sys.modules:
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    sys.modules["match_perspective"] = package

from .budget import ExperimentBudget
from match_perspective.core import focal_bundle, lens_refine
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core.sync.request import SyncSolveRequest, json_values, request_fingerprint
from match_perspective.core.sync.solve import solution_result_from_seed
from match_perspective.core.sync.types import SimilarityTransform, SyncSolveResult


def _finite_json(value):
    """Replace nonfinite diagnostic scalars while preserving solver decisions."""
    value = json_values(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


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


def _mirror_line_witnesses(request, result):
    """Measure reflected line direction and offset at stored finite helpers."""
    if request.mirror_plane is None:
        return []
    normal = np.asarray(request.mirror_plane[1], float)
    normal /= np.linalg.norm(normal)
    origin = (np.asarray(result.landmarks[request.mirror_landmark_id], float)
              if request.mirror_landmark_id is not None else
              np.asarray(request.mirror_plane[0], float))
    rows = []
    for left, right in request.mirror_pairs or ():
        if left not in result.line_segments or right not in result.line_segments:
            continue
        first = np.asarray(result.line_segments[left], float)
        second = np.asarray(result.line_segments[right], float)
        first_midpoint = np.mean(first, axis=0)
        second_midpoint = np.mean(second, axis=0)
        first_direction = first[1] - first[0]
        second_direction = second[1] - second[0]
        first_direction /= np.linalg.norm(first_direction)
        second_direction /= np.linalg.norm(second_direction)
        reflected_direction = first_direction - 2.0 * (normal @ first_direction) * normal
        reflected_midpoint = first_midpoint - 2.0 * (
            normal @ (first_midpoint - origin)) * normal
        sine = float(np.linalg.norm(np.cross(second_direction, reflected_direction)))
        rows.append(dict(
            pair=[left, right], direction_sine=sine,
            direction_degrees=float(np.degrees(np.arcsin(min(sine, 1.0)))),
            midpoint_witness_gap_m=float(np.linalg.norm(
                np.cross(second_direction, reflected_midpoint - second_midpoint))),
        ))
    return rows


def _archive_records(args):
    """Copy exact JSON inputs beside the output before replaying them."""
    archive = args.out / "inputs"
    paths = {"request.json": args.request, "initial.json": args.initial,
             "options.json": args.inputs}
    hashes = {}
    for name, source in paths.items():
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Archived {name} differs from supplied input")
        else:
            shutil.copyfile(source, destination)
        hashes[name] = digest
    return hashes


def _wrapper_matches(calibrations):
    """Build point-focal inputs that retain each saved private calibration."""
    return [lens_refine.MatchLensInput(
        match_id=match_id, line_bundles={}, intrinsics=calibration.intrinsics,
        base_calibration=calibration)
        for match_id, calibration in calibrations.items()]


def _public_wrapper(request, initial, calibrations, scorer, options):
    """Replay point FOV refinement from one mocked saved startup, never registration."""
    missing = set(calibrations) - set(initial.similarities)
    if missing:
        raise ValueError("Saved startup is missing cameras: " + ", ".join(sorted(missing)))
    progress = []
    matches = _wrapper_matches(calibrations)
    calls = {"startup": 0, "bundle": 0}
    fit = lens_refine.fit_independent_focals

    def saved_start(*_args, **_kwargs):
        calls["startup"] += 1
        if calls["startup"] > 1:
            raise RuntimeError("Saved-start replay cannot restart registration")
        return initial

    def one_bundle(*args, **kwargs):
        calls["bundle"] += 1
        if calls["bundle"] > 1:
            raise RuntimeError("Saved-start replay permits only one numerical bundle")
        return fit(*args, **kwargs)

    with mock.patch.object(lens_refine, "_run_sync", side_effect=saved_start) as run_sync, \
         mock.patch.object(lens_refine, "fit_independent_focals", side_effect=one_bundle):
        result = lens_refine.refine_lenses_from_landmarks(
            matches, scorer.point_observations, anchor_id=request.anchor_id,
            known_world=request.known_world,
            line_observations=request.line_observations,
            known_lines=request.known_lines, derived_lines=request.derived_lines,
            parallel_pairs=request.parallel_pairs,
            fx_span=options["fx_span"],
            lock_rotation=request.lock_rotation,
            lock_translation=request.lock_translation,
            share_lens=options["share_lens"],
            estimate_focal_from_points=True,
            pick_sigma_px=options["pick_sigma_px"],
            fixed_similarities=request.fixed_similarities,
            ground_slack=request.ground_slack,
            known_3d_slack=request.known_3d_slack,
            mirror_pairs=request.mirror_pairs,
            mirror_plane=request.mirror_plane,
            mirror_slack=request.mirror_slack,
            mirror_landmark_id=request.mirror_landmark_id,
            plane_groups=request.plane_groups,
            plane_slack=request.plane_slack,
            location_match_ids=request.location_match_ids,
            readonly_match_ids=request.readonly_match_ids,
            progress_callback=lambda _done, _total, label: progress.append(label))
    if run_sync.call_count != 1:
        raise RuntimeError(
            f"Saved-start wrapper made {run_sync.call_count} startup calls; expected one")
    score = (scorer.score(result.sync_result, calibrations=result.calibrations)
             if result.improved else None)
    objective_consistent = bool(
        score is not None and score.valid and
        np.isclose(result.final_cost, score.objective, rtol=1.e-10, atol=1.e-10))
    return result, score, progress, objective_consistent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-wrapper", action="store_true",
                        help="Run Refine Lenses from the saved startup with registration mocked.")
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--per-call-seconds", type=float, default=120.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = {
        "request": json.loads(args.request.read_text()),
        "initial": json.loads(args.initial.read_text()),
        "inputs": json.loads(args.inputs.read_text()),
    }
    initial_is_request_seed = False
    if "sync_result" in records["initial"]:
        records["initial"] = records["initial"]["sync_result"]
    elif (records["initial"].get("format") == "perspective-match-sync-request" and
          records["initial"].get("inputs", {}).get("initial_solution") is not None):
        records["initial"] = records["initial"]["inputs"]["initial_solution"]
        initial_is_request_seed = True
    input_hashes = _archive_records(args)
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
                "input_hashes": input_hashes, "public_wrapper": args.public_wrapper,
                "python": sys.version, "numpy": np.__version__,
                "platform": platform.platform()}
    request = _request(records["request"])
    initial = (solution_result_from_seed(request.initial_solution)
               if initial_is_request_seed else _initial(records["initial"]))
    if initial is None:
        raise ValueError("Saved solution seed lacks complete applied diagnostics")
    options = records["inputs"]
    calibrations = {item.match_id: item.calibration for item in request.matches}
    scorer = JointFitScorer(request, initial, calibrations=calibrations)
    initial_score = scorer.score(initial, calibrations=calibrations)
    diagnostic = {}
    request_record = {"records": records, "source_key": source_key,
                      "input_hashes": input_hashes, "public_wrapper": args.public_wrapper}
    with ExperimentBudget(args.out / "attempts.jsonl", metadata=metadata,
                          max_calls=args.max_calls, wall_seconds=args.wall_seconds,
                          per_call_seconds=args.per_call_seconds) as budget:
        label = "saved-start-public-wrapper" if args.public_wrapper else "saved-endpoint"
        with budget.attempt(label, request_record) as attempt:
            if args.public_wrapper:
                result, score, progress, objective_consistent = _public_wrapper(
                    request, initial, calibrations, scorer, options)
                output = {"result": json_values(result), "score": json_values(score),
                          "progress": progress,
                          "mocked_startup_calls": 1,
                          "original_weight_objective_consistent": objective_consistent,
                          "source_key": source_key, "input_hashes": input_hashes}
            else:
                outcome = focal_bundle.fit_independent_focals(
                    calibrations, scorer.point_observations, initial,
                    anchor_id=request.anchor_id, pick_sigma_px=options["pick_sigma_px"],
                    fx_span=options["fx_span"], plane_groups=request.plane_groups,
                    fixed_focals=bool(options.get("fixed_focals", False)),
                    plane_slack=request.plane_slack, mirror_pairs=request.mirror_pairs,
                    mirror_plane=request.mirror_plane, mirror_slack=request.mirror_slack,
                    mirror_pair_slack=request.mirror_pair_slack,
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
                    derived_lines=request.derived_lines,
                    lock_rotation=request.lock_rotation,
                    lock_translation=request.lock_translation,
                    share_lens=options["share_lens"],
                    diagnostic_callback=diagnostic.update)
                # Preserve the numerical endpoint before public scoring or
                # report serialization can fail independently of the solve.
                (args.out / "solver-endpoint.json").write_text(json.dumps(
                    _finite_json(outcome), indent=2, allow_nan=False))
                score = (scorer.score(
                    outcome.sync_result, calibrations=outcome.calibrations)
                         if outcome.sync_result is not None else None)
                internal_objective = diagnostic.get("final_objective")
                public_objective = score.objective if score and score.valid else None
                relative_objective_delta = (
                    abs(public_objective - internal_objective) /
                    max(public_objective, internal_objective, 1.0)
                    if public_objective is not None and internal_objective is not None else None)
                output = {"outcome": _finite_json(outcome), "score": _finite_json(score),
                          "initial_score": _finite_json(initial_score),
                          "initial_mirror_line_witnesses": _mirror_line_witnesses(
                              request, initial),
                          "final_mirror_line_witnesses": _mirror_line_witnesses(
                              request, outcome.sync_result)
                              if outcome.sync_result is not None else [],
                          "diagnostic": _finite_json(diagnostic), "source_key": source_key,
                          "input_hashes": input_hashes,
                          "internal_objective": internal_objective,
                          "public_objective": public_objective,
                          "relative_objective_delta": relative_objective_delta}
            attempt.complete(output)
    endpoint = "wrapper.json" if args.public_wrapper else "endpoint.json"
    (args.out / endpoint).write_text(json.dumps(output, indent=2, allow_nan=False))
    if args.public_wrapper:
        print(json.dumps({"improved": result.improved, "refusal_reason": result.refusal_reason,
                          "score_valid": score.valid if score else None,
                          "public_objective": score.objective if score and score.valid else None,
                          "original_weight_objective_consistent": objective_consistent,
                          "mocked_startup_calls": 1,
                          "source_key": source_key}))
    else:
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
