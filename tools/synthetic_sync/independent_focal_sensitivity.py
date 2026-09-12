"""Read-only focal sensitivity of saved independent-focal candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.independent_focal import (
    CASES, HERE, decode, initial_state, input_only, jacobian_pattern,
    observation_arrays, source_identity,
)
from tools.synthetic_sync.solver import fingerprint


LIMITS = dict(residual=120, jacobian=4, seconds=20.)
ASSUMED_COORDINATE_SIGMA_PX = 0.5


def encode(request: dict, state: dict, record: dict) -> np.ndarray:
    cameras = [record["cameras"][c["id"]] for c in request["cameras"]]
    first_center = np.asarray(cameras[1]["center"], float)
    denominator = first_center @ state["direction"]
    if denominator <= 0:
        raise ValueError("Candidate moved beyond the baseline tangent chart")
    tangent = [(first_center @ t) / denominator for t in state["tangents"]]
    return np.concatenate([
        np.log([c["fx"] / r["fx"] for c, r in zip(cameras, request["cameras"])]),
        Rotation.from_matrix(cameras[1]["rotation"]).as_rotvec(), tangent,
        Rotation.from_matrix(cameras[2]["rotation"]).as_rotvec(),
        cameras[2]["center"],
        np.asarray([record["landmarks"][key] for key in state["point_ids"]]).ravel(),
    ])


def diagnose(request: dict, baseline: dict, candidate: dict, counters: dict) -> dict:
    state = initial_state(request, baseline)
    x = encode(request, state, candidate)
    cidx, pidx, uv = observation_arrays(request, state)
    cxcy = np.asarray([[c["cx"], c["cy"]] for c in request["cameras"]], float)
    pattern = jacobian_pattern(cidx, pidx, len(state["point_ids"]))

    def residual(parameters):
        counters["residual"] += 1
        if counters["residual"] > LIMITS["residual"] or time.monotonic()-counters["started"] > LIMITS["seconds"]:
            raise RuntimeError("Sensitivity evaluation cap exceeded")
        focal, rotations, centers, points = decode(parameters, request, state)
        predicted = np.empty_like(uv)
        for camera in range(3):
            selected = cidx == camera
            xyz = (rotations[camera] @ (points[pidx[selected]]-centers[camera]).T).T
            z = np.maximum(xyz[:, 2], 1e-6)
            predicted[selected] = focal[camera] * xyz[:, :2] / z[:, None] + cxcy[camera]
            predicted[selected] += np.maximum(1e-6-xyz[:, 2], 0)[:, None] * 1e3
        return (predicted-uv).ravel()

    counters["jacobian"] += 1
    if counters["jacobian"] > LIMITS["jacobian"]:
        raise RuntimeError("Sensitivity Jacobian cap exceeded")
    jac = approx_derivative(residual, x, method="2-point", sparsity=pattern).toarray()
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    tolerance = np.finfo(float).eps * max(jac.shape) * singular[0]
    is_null = singular <= tolerance
    focal_sigma = []
    for focal_index in range(3):
        if np.any(abs(vh[is_null, focal_index]) > 1e-6):
            focal_sigma.append(None)
        else:
            focal_sigma.append(float(ASSUMED_COORDINATE_SIGMA_PX *
                np.linalg.norm(vh[~is_null, focal_index] / singular[~is_null])))
    fit = residual(x)
    return dict(request_sha256=fingerprint(request),
        candidate_focal_px=[float(c["fx"]) for c in candidate["cameras"].values()],
        fitted_rmse_px=float(np.sqrt(np.mean(np.square(fit.reshape(-1, 2)).sum(axis=1)))),
        singular_values=singular.tolist(), numerical_rank=int((~is_null).sum()),
        parameter_count=jac.shape[1],
        focal_log_sigma_at_0_5px=focal_sigma,
        focal_approx_95pct_upper_relative_excursion=[
            float(math.expm1(1.96 * sigma)) if sigma is not None and sigma < 100 else None
            for sigma in focal_sigma],
        coordinate_sigma_assumed_px=ASSUMED_COORDINATE_SIGMA_PX)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.in_dir / "sensitivity.json"
    ledger = args.in_dir / "sensitivity-ledger.jsonl"
    if output.exists() or ledger.exists():
        raise FileExistsError("Sensitivity evidence cannot be silently rerun")
    source = source_identity()
    digest = hashlib.sha256()
    digest.update(source["source_sha256"].encode())
    digest.update(Path(__file__).read_bytes())
    source["source_sha256"] = digest.hexdigest()
    counters = dict(residual=0, jacobian=0, started=time.monotonic())
    rows = []
    with ledger.open("x") as log:
        for name in CASES:
            case = json.loads((HERE / "cases" / f"{name}.json").read_text())
            artifact = json.loads((args.in_dir / f"{name}.json").read_text())
            baseline = json.loads((HERE / "cases" / "unknown-focal-continuation" /
                f"{name}-scale-1.json").read_text())["record"]
            request = case["request"]
            request, baseline = input_only(request, baseline)
            if (artifact["request_sha256"] != fingerprint(request) or
                baseline["request_sha256"] != fingerprint(request)):
                raise ValueError("Saved candidate request changed")
            trial = dict(name=name, request=request, baseline=baseline,
                candidate=artifact["record"], source_runtime=source,
                limits=LIMITS, coordinate_sigma_assumed_px=ASSUMED_COORDINATE_SIGMA_PX)
            log.write(json.dumps(dict(kind="started", trial=trial), sort_keys=True,
                allow_nan=False)+"\n")
            log.flush()
            os.fsync(log.fileno())
            result = diagnose(request, baseline, artifact["record"], counters)
            log.write(json.dumps(dict(kind="completed", name=name, result=result,
                counts={k:v for k,v in counters.items() if k!="started"}),
                sort_keys=True, allow_nan=False)+"\n")
            log.flush()
            os.fsync(log.fileno())
            rows.append(dict(name=name, **result))
    report = dict(rows=rows, limits=LIMITS, source_runtime=source,
        counts={k:v for k,v in counters.items() if k!="started"})
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
