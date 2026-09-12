"""Trace one frozen no-VP startup solve under a bounded numerical ledger.

The trace records fitted stage state only. Independent truth is used afterward
by ``no_vp_bootstrap.assess`` and never enters candidate selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.synthetic_sync.budget import ExperimentBudget
from tools.synthetic_sync.no_vp_bootstrap import CASE_DIR, ROOT, assess
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import load_core, solve


LEDGER = CASE_DIR / "no-vp-startup-followup-ledger.jsonl"
METADATA = dict(purpose="frozen no-VP true-K startup repair", calls=12,
                per_call_seconds=180, active_seconds=720)


def source_hash() -> str:
    digest = hashlib.sha256()
    harness = ROOT / "tools" / "synthetic_sync"
    paths = [*(ROOT / "core").rglob("*.py"),
             *(harness / name for name in (
                 "solver.py", "evaluation.py", "geometry.py", "scenarios.py",
                 "no_vp_bootstrap.py", "budget.py")),
             Path(__file__)]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_key() -> dict:
    """Include versions and numerical thread settings in cache identity."""
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    return dict(python=sys.version, numpy=np.__version__, opencv=opencv,
                platform=platform.platform(), machine=platform.machine(),
                threads={key: os.environ.get(key) for key in thread_vars})


def _similarities(items):
    return {key: dict(scale=float(value.scale),
                      rotation=value.rotation.tolist(),
                      translation=value.translation.tolist())
            for key, value in items.items()}


def traced_solve(request):
    load_core()
    pose = importlib.import_module("match_perspective.core.sync.pose")
    stages = importlib.import_module("match_perspective.core.sync.solve")
    trace = []
    originals = {}

    def wrap(module, name, function):
        original = getattr(module, name)
        originals[(module, name)] = original
        setattr(module, name, function(original))

    def pair_wrapper(original):
        def call(pairs, anchor, other, **kwargs):
            result = original(pairs, anchor, other, **kwargs)
            rmse = (pose._pair_reprojection_rmse(result, pairs, anchor, other)
                    if result is not None else None)
            trace.append(dict(stage="pair", n=len(pairs), rmse_px=rmse,
                              solution=_similarities({"pair": result}) if result is not None else None))
            return result
        return call

    def registration_wrapper(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            trace.append(dict(stage="registration", matches=_similarities(result[0] or {}),
                              detail=result[1]))
            return result
        return call

    def state_wrapper(name):
        def factory(original):
            def call(state):
                trace.append(dict(stage=name+" before", matches=_similarities(state.similarities),
                                  landmarks={key: value.tolist() for key, value in state.landmarks.items()},
                                  active=state.free_match_ids.copy(), skipped=state.skipped_unregistered.copy()))
                result = original(state)
                trace.append(dict(stage=name+" after", matches=_similarities(state.similarities),
                                  landmarks={key: value.tolist() for key, value in state.landmarks.items()},
                                  active=state.free_match_ids.copy(), skipped=state.skipped_unregistered.copy(),
                                  rmse=state.pre_ba_match_rmse.copy()))
                return result
            return call
        return factory

    def downweight_wrapper(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            trace.append(dict(stage="downweight", observation_ids=[str(key) for key in result[1]]))
            return result
        return call

    def selection_wrapper(original):
        def call(candidates, **kwargs):
            result = original(candidates, **kwargs)
            trace.append(dict(stage="graph candidates",
                              scores=[dict(pair_rmse=float(pair), graph_rmse=float(graph),
                                           selected=pose_value is result)
                                      for pair, graph, pose_value in candidates]))
            return result
        return call

    wrap(pose, "_solve_relative_from_pairs", pair_wrapper)
    wrap(pose, "_select_registration_candidate", selection_wrapper)
    wrap(stages, "_register_from_relative_pose", registration_wrapper)
    wrap(stages, "_peel_cameras_above_rmse", state_wrapper("peel"))
    wrap(stages, "_resect_skipped_matches", state_wrapper("resect"))
    wrap(stages, "_auto_downweight_outlier_observations", downweight_wrapper)
    try:
        record = solve(request)
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)
    return dict(record=record, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", default="no-vp-shared-trueK")
    parser.add_argument("--reverse-order", action="store_true",
                        help="Reverse camera and pick input order as an invariance control")
    args = parser.parse_args()
    case = read_case(CASE_DIR / f"{args.case}.json")
    request = deepcopy(case["request"])
    if args.reverse_order:
        request["cameras"].reverse()
        request["observations"].reverse()
    key = dict(request=request, source_hash=source_hash(), runtime=runtime_key())
    with ExperimentBudget(LEDGER, metadata=METADATA, max_calls=12,
                          per_call_seconds=180, wall_seconds=720) as budget:
        result = budget.cached(args.label, key)
        if result is None:
            with budget.attempt(args.label, key) as attempt:
                result = traced_solve(request)
                attempt.complete(result)
    result["assessment"] = assess(case, result["record"])
    out = CASE_DIR / f"{args.label}-trace.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(out)
    print(result["assessment"]["classification"])
    for stage in result["trace"]:
        print(stage["stage"], stage.get("n"), stage.get("rmse_px"),
              stage.get("active"), stage.get("skipped"), stage.get("rmse"))


if __name__ == "__main__":
    main()
