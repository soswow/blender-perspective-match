"""Bounded replay of incomplete and imperfect point-FOV picks from saved starts.

The fitter only sees the request and saved Sync state. Independent truth is
evaluated after each decision has been committed to the experiment ledger.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.no_vp_bootstrap import assess
from tools.synthetic_sync.solver import environment, load_core, result_record, solver_arguments

load_core()
from test_focal_bundle import _inputs
from match_perspective.core import geometry, lens_refine, sync
from match_perspective.core.focal_bundle import fit_independent_focals

SOURCES = ("core/focal_bundle.py", "core/lens_refine.py", "tests/test_focal_bundle.py",
           "tools/synthetic_sync/solver.py", "tools/synthetic_sync/no_vp_bootstrap.py",
           "tools/synthetic_sync/independent_focal_reliability.py")
FROZEN = ROOT / "tools/synthetic_sync/cases/independent-focal-production"


def _subset(case, size, mode):
    observations = case["request"]["observations"]
    by_point = {}
    for item in observations:
        by_point.setdefault(item["landmark_id"], []).append(item)
    # The middle camera sees all points. Build compact or spread subsets from
    # its raw picks, while preferring three-view overlap to preserve eligibility.
    centers = {key: np.mean([[o["u"], o["v"]] for o in obs], axis=0)
               for key, obs in by_point.items()}
    keys = sorted(by_point)
    if mode == "cluster":
        image_center = np.mean(list(centers.values()), axis=0)
        keys.sort(key=lambda key: (np.linalg.norm(centers[key] - image_center), key))
    else:
        chosen = [min(keys)]
        while len(chosen) < len(keys):
            nxt = max((key for key in keys if key not in chosen),
                      key=lambda key: (min(np.linalg.norm(centers[key] - centers[old])
                                           for old in chosen), key))
            chosen.append(nxt)
        keys = chosen
    if size == 12:
        triple = [key for key in keys if len(by_point[key]) == 3]
        left = [key for key in keys if len(by_point[key]) == 2 and
                any(o["match_id"] == "view_0" for o in by_point[key])]
        right = [key for key in keys if len(by_point[key]) == 2 and
                 any(o["match_id"] == "view_2" for o in by_point[key])]
        if len(triple) >= 6 and len(left) >= 2 and len(right) >= 2:
            keys = triple[:6] + left[:3] + right[:3]
    selected = set(keys[:size])
    request = deepcopy(case["request"])
    request["points"] = [p for p in request["points"] if p["id"] in selected]
    request["observations"] = [o for o in observations if o["landmark_id"] in selected]
    return request


def _trials(followup=False):
    # Freeze this short representative matrix before fitting. Exact and noisy
    # controls share underlying geometry; full and partial views are paired.
    initial_matrix = (
        ("no-vp-mixed-guessedK", 8, "spread", "none"),
        ("no-vp-mixed-guessedK", 12, "spread", "none"),
        ("no-vp-mixed-guessedK", 16, "spread", "none"),
        ("no-vp-mixed-guessedK", 23, "spread", "none"),
        ("noisy-mixed", 12, "spread", "none"),
        ("noisy-mixed", 16, "spread", "none"),
        ("noisy-mixed", 23, "spread", "none"),
        ("noisy-shared", 16, "spread", "none"),
        ("noisy-shared", 23, "spread", "none"),
        ("four-view", 12, "spread", "none"),
        ("four-view", 16, "spread", "none"),
        ("noisy-mixed", 16, "cluster", "none"),
        ("noisy-mixed", 23, "spread", "bad_pick"),
        ("noisy-mixed", 23, "spread", "alternate_fov"),
    )
    followup_matrix = (
        ("no-vp-mixed-guessedK", 12, "spread", "none"),
        ("noisy-mixed", 12, "spread", "none"),
        ("noisy-shared", 12, "spread", "none"),
        ("four-view", 12, "spread", "none"),
        ("noisy-mixed", 16, "spread", "bad_pick"),
        ("noisy-mixed", 23, "spread", "bad_pick"),
        ("noisy-shared", 23, "spread", "bad_pick"),
        ("noisy-mixed", 16, "spread", "alternate_fov"),
        ("noisy-shared", 23, "spread", "alternate_fov"),
        ("four-view", 16, "spread", "alternate_fov"),
    )
    for name, size, mode, change in (followup_matrix if followup else initial_matrix):
        case, matches, _observations = _inputs(name)
        initial = _saved_initial(case, matches, name)
        request = _subset(case, size, mode)
        if change == "bad_pick":
            # A deterministic single wrong correspondence on the peripheral
            # camera, with the correct picks in other views left untouched.
            picked = next(o for o in request["observations"] if o["match_id"] == "view_2")
            picked["u"] += 20.0
            picked["v"] -= 15.0
        if change == "alternate_fov":
            for c in request["cameras"]:
                c["fx"] *= 1.15
                c["fy"] *= 1.15
        label = f"{name}-{size}-{mode}-{change}"
        yield label, case, request, initial


def _saved_initial(case, matches, name):
    from match_perspective.core import sync as core_sync
    run = "run-02" if name == "no-vp-mixed-guessedK" else "run-03"
    rows = [json.loads(line) for line in (FROZEN / run / "sync-ledger.jsonl").read_text().splitlines()]
    attempt = next(row["attempt"] for row in rows
                   if row.get("kind") == "started" and row["label"] == name)
    record = next(row["result"] for row in rows
                  if row.get("kind") == "completed" and row["attempt"] == attempt)
    similarities = {}
    for item in matches:
        source = item.base_calibration
        world = record["cameras"][item.match_id]
        world_r = np.asarray(world["rotation"], float)
        world_c = np.asarray(world["center"], float)
        rotation = world_r.T @ source.rotation_w2c
        similarities[item.match_id] = core_sync.SimilarityTransform(
            1.0, rotation, world_c - rotation @ source.camera_center)
    return core_sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value, float) for key, value in record["landmarks"].items()},
        mean_reprojection_px=record["reported_rmse_px"],
        per_match_rmse_px={}, per_landmark_rmse_px={}, message="Frozen fresh Sync start",
        success=record["success"])


def _run_fit(request, initial):
    cameras = request["cameras"]
    calibrations = {}
    for camera in cameras:
        k = geometry.CameraIntrinsics(camera["fx"], camera["fy"], camera["cx"],
                                      camera["cy"], camera["width"], camera["height"])
        calibrations[camera["id"]] = geometry.Calibration(
            k, np.asarray(camera["rotation"], float), np.asarray(camera["center"], float))
    observations = [sync.SyncObservation(**o) for o in request["observations"]]
    initial.landmarks = {key: value for key, value in initial.landmarks.items()
                         if key in {o.landmark_id for o in observations}}
    fitted = fit_independent_focals(calibrations, observations, initial,
                                   anchor_id=request["anchor_id"], pick_sigma_px=1.0)
    record = result_record(fitted.sync_result, cameras,
                           calibrations=fitted.calibrations) if fitted.accepted else dict(
        success=False, message=fitted.reason, cameras={}, landmarks={},
        reported_rmse_px=(fitted.fitted_rmse_px if np.isfinite(fitted.fitted_rmse_px) else None))
    return dict(accepted=fitted.accepted, reason=fitted.reason,
                fitted_rmse_px=(fitted.fitted_rmse_px if np.isfinite(fitted.fitted_rmse_px) else None),
                intervals_px=fitted.intervals_px, record=record)


def run(out, *, followup=False):
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
        for relative in SOURCES:
            archive.add(ROOT / relative, arcname=relative)
    metadata = dict(source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in SOURCES}, environment=environment(ROOT),
                    saved_starts="independent-focal-production/run-03/sync-ledger.jsonl",
                    limits=dict(inner_sync_calls=0, bundle_attempts=(10 if followup else 14),
                                per_bundle_seconds=30, cumulative_bundle_seconds=300))
    (out / "source-runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with ExperimentBudget(out / "bundle-ledger.jsonl", metadata=metadata,
                          max_calls=(10 if followup else 14), per_call_seconds=30,
                          wall_seconds=300) as budget:
        for label, case, request, initial in _trials(followup=followup):
            with budget.attempt(label, dict(request=request,
                                            initial=result_record(initial, case["request"]["cameras"]))) as attempt:
                fitted = _run_fit(request, initial)
                attempt.complete(fitted)
            # The independent oracle is deliberately downstream of the saved
            # decision. Trim its point requirement to the supplied subset.
            oracle = deepcopy(case)
            selected = {p["id"] for p in request["points"]}
            oracle["request"] = request
            oracle["truth"]["points"] = {k: v for k, v in oracle["truth"]["points"].items()
                                           if k in selected}
            oracle["expectation"]["required_points"] = sorted(selected)
            assessment = assess(oracle, fitted["record"])
            truth_fx = {c["id"]: c["fx"] for c in case["truth"]["cameras"]}
            interval_coverage = {k: low <= truth_fx[k] <= high
                                 for k, (low, high) in fitted["intervals_px"].items()}
            report = dict(label=label, assessment=assessment,
                          interval_coverage=interval_coverage)
            (out / f"{label}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(label, fitted["accepted"], fitted["reason"],
                  assessment["classification"], interval_coverage, flush=True)


def fresh_sync(out, *, remaining=False):
    """Measure actual startup support for representative sparse and bad-pick inputs."""
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
        for relative in SOURCES:
            archive.add(ROOT / relative, arcname=relative)
    metadata = dict(source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in SOURCES}, environment=environment(ROOT),
                    limits=dict(max_inner_sync=(2 if remaining else 6), per_sync_seconds=120,
                                cumulative_sync_seconds=360, outer_seconds=420))
    (out / "source-runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    labels = {
        "no-vp-mixed-guessedK-12-spread-none",
        "noisy-mixed-12-spread-none",
        "noisy-shared-12-spread-none",
        "noisy-mixed-16-spread-none",
        "noisy-mixed-23-spread-bad_pick",
        "noisy-mixed-23-spread-alternate_fov",
    }
    if remaining:
        labels = {"noisy-mixed-16-spread-none", "noisy-mixed-23-spread-alternate_fov"}
    with ExperimentBudget(out / "sync-ledger.jsonl", metadata=metadata,
                          max_calls=(2 if remaining else 6), per_call_seconds=120,
                          wall_seconds=360) as budget:
        seen = set()
        for label, case, request, _saved in list(_trials(followup=True)) + list(_trials()):
            if label not in labels or label in seen:
                continue
            seen.add(label)
            arguments = solver_arguments(request)
            with budget.attempt(label, request) as attempt:
                initial = sync.solve_landmark_sync(**arguments)
                record = result_record(initial, request["cameras"])
                attempt.complete(record)
            selected = {p["id"] for p in request["points"]}
            oracle = deepcopy(case)
            oracle["request"] = request
            oracle["truth"]["points"] = {k: v for k, v in oracle["truth"]["points"].items()
                                           if k in selected}
            oracle["expectation"]["required_points"] = sorted(selected)
            assessment = assess(oracle, record)
            (out / f"{label}.json").write_text(json.dumps(
                dict(label=label, assessment=assessment), indent=2, allow_nan=False) + "\n")
            print(label, initial.success, len(initial.similarities), len(initial.landmarks),
                  initial.message, flush=True)


def _fresh_record(label):
    base = ROOT / "tools/synthetic_sync/cases/independent-focal-reliability"
    for run in ("fresh-sync-01", "fresh-sync-02"):
        rows = [json.loads(line) for line in (base / run / "sync-ledger.jsonl").read_text().splitlines()]
        for started in rows:
            if started.get("kind") != "started" or started["label"] != label:
                continue
            completed = next(row for row in rows if row.get("kind") == "completed" and
                             row["attempt"] == started["attempt"])
            return started["request"], completed["result"]
    raise KeyError(label)


def _reconstruct_initial(request, record):
    similarities = {}
    for camera in request["cameras"]:
        world = record["cameras"][camera["id"]]
        source_r = np.asarray(camera["rotation"], float)
        source_c = np.asarray(camera["center"], float)
        world_r = np.asarray(world["rotation"], float)
        world_c = np.asarray(world["center"], float)
        rotation = world_r.T @ source_r
        similarities[camera["id"]] = sync.SimilarityTransform(
            1.0, rotation, world_c - rotation @ source_c)
    return sync.SyncSolveResult(
        similarities=similarities,
        landmarks={key: np.asarray(value, float) for key, value in record["landmarks"].items()},
        mean_reprojection_px=record["reported_rmse_px"],
        per_match_rmse_px={}, per_landmark_rmse_px={},
        message=record["message"], success=record["success"])


def fresh_bundle(out):
    """Run the public lens route from fresh, fully ledgered Sync states."""
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
        for relative in SOURCES:
            archive.add(ROOT / relative, arcname=relative)
    metadata = dict(source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in SOURCES}, environment=environment(ROOT),
                    sync_ledgers=["fresh-sync-01", "fresh-sync-02"],
                    limits=dict(bundle_attempts=6, per_bundle_seconds=30,
                                cumulative_bundle_seconds=180, outer_seconds=240))
    (out / "source-runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    labels = (
        "no-vp-mixed-guessedK-12-spread-none", "noisy-mixed-12-spread-none",
        "noisy-shared-12-spread-none", "noisy-mixed-16-spread-none",
        "noisy-mixed-23-spread-bad_pick", "noisy-mixed-23-spread-alternate_fov",
    )
    cases = {label: case for label, case, _request, _saved in
             list(_trials(followup=True)) + list(_trials())}
    with ExperimentBudget(out / "bundle-ledger.jsonl", metadata=metadata,
                          max_calls=6, per_call_seconds=30, wall_seconds=180) as budget:
        for label in labels:
            request, record = _fresh_record(label)
            initial = _reconstruct_initial(request, record)
            arguments = solver_arguments(request)
            matches = []
            for item in arguments["matches"]:
                source = item.calibration
                matches.append(lens_refine.MatchLensInput(
                    item.match_id, {}, source.intrinsics, base_calibration=source,
                    freeze_focal=True))
            observations = arguments["observations"]
            with budget.attempt(label, dict(request=request, initial=record)) as attempt:
                with mock.patch.object(lens_refine, "_run_sync", return_value=initial):
                    result = lens_refine.refine_lenses_from_landmarks(
                        matches, observations, anchor_id=request["anchor_id"],
                        estimate_focal_from_points=True, pick_sigma_px=1.0)
                fitted_record = (result_record(result.sync_result, request["cameras"],
                                                calibrations=result.calibrations) if result.improved else
                    dict(success=False, message=result.refusal_reason, cameras={},
                         landmarks={}, reported_rmse_px=None))
                fitted = dict(accepted=result.improved, reason=result.refusal_reason,
                              intervals_px=result.focal_intervals, record=fitted_record)
                attempt.complete(fitted)
            oracle = deepcopy(cases[label])
            selected = {p["id"] for p in request["points"]}
            oracle["request"] = request
            oracle["truth"]["points"] = {k: v for k, v in oracle["truth"]["points"].items()
                                           if k in selected}
            oracle["expectation"]["required_points"] = sorted(selected)
            assessment = assess(oracle, fitted_record)
            truth_fx = {c["id"]: c["fx"] for c in oracle["truth"]["cameras"]}
            coverage = {k: low <= truth_fx[k] <= high
                        for k, (low, high) in result.focal_intervals.items()}
            (out / f"{label}.json").write_text(json.dumps(dict(
                label=label, assessment=assessment, interval_coverage=coverage),
                indent=2, allow_nan=False) + "\n")
            print(label, result.improved, result.refusal_reason,
                  assessment["classification"], coverage, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--fresh-sync", action="store_true")
    parser.add_argument("--remaining", action="store_true")
    parser.add_argument("--fresh-bundle", action="store_true")
    args = parser.parse_args()
    if args.fresh_bundle:
        fresh_bundle(args.out)
    elif args.fresh_sync:
        fresh_sync(args.out, remaining=args.remaining)
    else:
        run(args.out, followup=args.followup)


if __name__ == "__main__":
    main()
