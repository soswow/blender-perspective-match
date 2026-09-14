"""Capped oracle-start and noise-free controls for generated continuation evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.solver import environment, load_core, result_record
from tools.synthetic_sync.sync_continuation import (
    _assess_sequence_case, _finite_json, _validate_sequence_case,
)
from tools.synthetic_sync.geometry import project


def _archive_sources(out: Path) -> str:
    paths = sorted((ROOT / "core").rglob("*.py"))
    paths += sorted((ROOT / "tools" / "synthetic_sync").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT)
        payload = path.read_bytes()
        digest.update(str(relative).encode() + b"\0" + payload)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Numerical source changed: {relative}")
        if not target.exists():
            shutil.copyfile(path, target)
    return digest.hexdigest()


def _start_at_truth(request, initial, case):
    """Place each represented camera, point and infinite line at oracle truth."""
    from match_perspective.core.sync import SimilarityTransform

    truth = {item["id"]: item for item in case["truth"]["cameras"]}
    anchor_id = request.anchor_id
    calibrations = {item.match_id: deepcopy(item.calibration) for item in request.matches}
    for item in request.matches:
        key = item.match_id
        camera = truth[key]
        source = calibrations[key]
        source.intrinsics.fx = camera["fx"]
        source.intrinsics.fy = camera["fy"]
        if key == anchor_id:
            source.rotation_w2c = np.asarray(camera["rotation"], float)
            source.camera_center = np.asarray(camera["center"], float)
        item.calibration = source
    similarities = {anchor_id: SimilarityTransform()}
    for key, source in calibrations.items():
        if key == anchor_id:
            continue
        camera = truth[key]
        root_rotation = np.asarray(camera["rotation"], float).T @ source.rotation_w2c
        center = np.asarray(camera["center"], float)
        similarities[key] = SimilarityTransform(
            1.0, root_rotation, center - root_rotation @ source.camera_center)
    initial.calibrations = calibrations
    initial.similarities = similarities
    initial.landmarks = {key: np.asarray(value, float)
                         for key, value in case["truth"]["points"].items()}
    initial.line_segments = {
        key: tuple(np.asarray(end, float) for end in segment)
        for key, segment in case["truth"]["lines"].items()}
    initial.landmarks.update({key: 0.5 * (ends[0] + ends[1])
                              for key, ends in initial.line_segments.items()})


def _remove_noise(request, case):
    truth = case["truth"]
    for item in request.observations:
        item.u, item.v = truth["oracle_pixels"][item.landmark_id][item.match_id]
    for item in request.line_observations:
        first, second = truth["line_oracle_pixels"][item.landmark_id][item.match_id]
        item.u1, item.v1 = first
        item.u2, item.v2 = second


def _frame_sensitivity(case: dict, recovered: dict) -> dict:
    """Diagnose drift with a training-only rigid rotation, without granting a gauge."""
    if (case["expectation"]["scale_gauge"] != "anchor-plane-offset" or
            case["request"]["mirror_plane"] is None):
        raise ValueError("Frame sensitivity audit requires a fixed metric mirror")
    truth = case["truth"]
    anchor_id = case["request"]["anchor_id"]
    true_cameras = {item["id"]: item for item in truth["cameras"]}
    true_anchor = np.asarray(true_cameras[anchor_id]["center"], float)
    fitted_anchor = np.asarray(recovered["cameras"][anchor_id]["center"], float)
    ids = sorted(truth["points"])
    source = np.asarray([truth["points"][key] for key in ids]) - true_anchor
    target = np.asarray([recovered["landmarks"][key] for key in ids]) - fitted_anchor
    u, _s, vt = np.linalg.svd(source.T @ target)
    rotation = vt.T @ np.diag((1.0, 1.0, np.linalg.det(vt.T @ u.T))) @ u.T
    before = np.sqrt(np.mean(np.sum((source - target)**2, axis=1)))
    after = np.sqrt(np.mean(np.sum((source @ rotation.T - target)**2, axis=1)))
    errors = {"raw": [], "training_rotation_diagnostic": []}
    for key, world in truth["holdouts"].items():
        point = np.asarray(world, float)
        rotated = fitted_anchor + rotation @ (point - true_anchor)
        for camera_id, camera in recovered["cameras"].items():
            oracle = truth["holdout_pixels"][key][camera_id]
            for label, position in (("raw", point),
                                    ("training_rotation_diagnostic", rotated)):
                pixel, depth = project([position], camera)
                if depth[0] <= 0:
                    raise ValueError("Diagnostic withheld point is behind a camera")
                errors[label].append(float(np.linalg.norm(pixel[0] - oracle)))
    normal_angles = {}
    for name, normal in (("axis_x", np.array((1.0, 0.0, 0.0))),
                         ("mirror", np.asarray(case["request"]["mirror_plane"][1], float))):
        normal /= np.linalg.norm(normal)
        normal_angles[name] = float(np.degrees(np.arccos(np.clip(
            normal @ (rotation @ normal), -1.0, 1.0))))
    return dict(
        training_rotation_deg=float(np.degrees(np.arccos(np.clip(
            (np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))),
        training_point_rms_before_world=float(before),
        training_point_rms_after_world=float(after),
        normal_rotation_deg=normal_angles,
        raw_withheld_rmse_px=float(np.sqrt(np.mean(np.square(errors["raw"])))),
        training_rotation_withheld_rmse_px=float(np.sqrt(np.mean(np.square(
            errors["training_rotation_diagnostic"])))),
        note="Training rotation is diagnostic only; fixed mirror and axis planes forbid it.",
    )


def run(source: Path, out: Path, mode: str) -> None:
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Archive must remain outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    source_hash = _archive_sources(out)
    load_core()
    from match_perspective.core import focal_bundle
    from match_perspective.core.joint_fit_score import JointFitScorer, supported_joint_request
    from match_perspective.core.sync.request import SyncSolveRequest, json_values
    from match_perspective.core.sync.solve import solution_result_from_seed

    case_file = source / "joint-case.json"
    report_file = source / "refine-result.json"
    case = json.loads(case_file.read_text())
    _validate_sequence_case(case)
    report = json.loads(report_file.read_text())
    request = SyncSolveRequest.from_record(report["post_request"])
    seed = request.initial_solution
    if seed is None or seed.evidence_sha256 != request.evidence_sha256():
        raise ValueError("Source endpoint is not certified")
    initial = solution_result_from_seed(seed)
    if initial is None or seed.diagnostics is None:
        raise ValueError("Source endpoint lacks applied diagnostics")
    frozen = seed.diagnostics.joint_point_weights
    if not frozen:
        raise ValueError("Source endpoint lacks frozen point weights")

    truth = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    if mode == "oracle_noisy":
        _start_at_truth(request, initial, case)
    elif mode == "endpoint_clean":
        _remove_noise(request, case)
        # Changed picks have a new evidence identity. Recompute effective
        # weights, as the product does, instead of reusing certified old picks.
        frozen = None
        for match in request.matches:
            camera = truth[match.match_id]
            match.calibration.intrinsics.fx = camera["fx"]
            match.calibration.intrinsics.fy = camera["fy"]
            initial.calibrations[match.match_id] = deepcopy(match.calibration)
    else:
        raise ValueError(mode)
    request.initial_solution = None
    supported, coverage = supported_joint_request(request, initial)
    scorer = JointFitScorer(supported, initial,
                            calibrations=initial.calibrations,
                            frozen_point_weights=frozen)
    start_score = scorer.score(initial, calibrations=initial.calibrations)
    if not start_score.valid or scorer.weight_refusal:
        raise ValueError(f"Start cannot score: {start_score.reason or scorer.weight_refusal}")
    start_record = result_record(initial, case["truth"]["cameras"],
                                 calibrations=initial.calibrations)
    input_payload = _finite_json(json_values(dict(
        request=request, initial=initial, frozen_point_weights=frozen,
        coverage=coverage, start_score=start_score)))
    (out / f"{mode}-input.json").write_text(json.dumps(input_payload, indent=2) + "\n")

    metadata = dict(case="continuation-accuracy-control", source_sha256=source_hash,
                    source_case_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),
                    source_result_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest(),
                    max_sync_calls=0, max_bundle_calls=4, environment=environment(ROOT))
    raw = {}
    original_fit = focal_bundle.fit_independent_focals

    def capture(*args, **kwargs):
        if raw:
            raise RuntimeError("Control must call one inner bundle")
        result = original_fit(*args, **kwargs)
        raw["serialized"] = _finite_json(json_values(result))
        raw["live"] = result
        return result

    with ExperimentBudget(out / "ledger.jsonl", metadata=metadata,
                          max_calls=4, wall_seconds=300, per_call_seconds=120) as budget:
        with budget.attempt(f"bundle:{mode}", input_payload) as attempt:
            with patch.object(focal_bundle, "fit_independent_focals", capture):
                outcome = focal_bundle.refine_fixed_focals(
                    request, initial, frozen_point_weights=frozen)
            attempt.complete(dict(wrapper=_finite_json(json_values(outcome)),
                                  inner=raw["serialized"]))
    summary = dict(mode=mode, start_score=_finite_json(json_values(start_score)),
                   start_assessment=_assess_sequence_case(case, start_record),
                   wrapper_accepted=outcome.accepted, wrapper_reason=outcome.reason,
                   inner_accepted=raw["live"].accepted,
                   inner_reason=raw["live"].reason, coverage=coverage)
    for label, fitted in (("wrapper", outcome), ("inner", raw["live"])):
        if fitted.sync_result is None:
            continue
        record = result_record(fitted.sync_result, case["truth"]["cameras"],
                               calibrations=fitted.calibrations)
        summary[label] = dict(
            score=_finite_json(json_values(scorer.score(
                fitted.sync_result, calibrations=fitted.calibrations))),
            assessment=_assess_sequence_case(case, record), recovered=record)
    (out / f"{mode}-result.json").write_text(json.dumps(
        _finite_json(summary), indent=2) + "\n")
    print(json.dumps(dict(mode=mode, start_objective=start_score.objective,
                          start_withheld_px=summary["start_assessment"]["withheld_rmse_px"],
                          wrapper_accepted=outcome.accepted, wrapper_reason=outcome.reason,
                          inner_accepted=raw["live"].accepted,
                          inner_reason=raw["live"].reason,
                          end_objective=(summary.get("inner") or {}).get("score", {}).get("objective"),
                          end_withheld_px=(summary.get("inner") or {}).get("assessment", {}).get("withheld_rmse_px")),
                     indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle_noisy", "endpoint_clean", "frame_audit"), required=True)
    args = parser.parse_args()
    if args.mode == "frame_audit":
        case = json.loads((args.source / "joint-case.json").read_text())
        result = json.loads((args.out / "oracle_noisy-result.json").read_text())
        report = _frame_sensitivity(case, result["inner"]["recovered"])
        (args.out / "frame-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if os.environ.get("PM_ACCURACY_CHILD") != "1":
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=130, check=False, env=dict(os.environ, PM_ACCURACY_CHILD="1"))
            print(process.stdout, end="")
            exit_code = process.returncode
        except subprocess.TimeoutExpired as error:
            print(error.stdout or "", end="")
            exit_code = 124
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.mode}-process-exit.json").write_text(json.dumps(
            dict(exit_code=exit_code, elapsed_s=time.monotonic() - started),
            indent=2) + "\n")
        raise SystemExit(exit_code)
    run(args.source, args.out, args.mode)


if __name__ == "__main__":
    main()
