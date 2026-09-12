"""Bounded, input-only independent-focal bundle-adjustment prototype.

The saved Sync result seeds camera poses and free 3D. Synthetic truth is loaded
only after optimization and input-side selection, for independent assessment.
This is an experiment, not the add-on's lens search or an acceptance policy.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.no_vp_bootstrap import assess, validate_fixture
from tools.synthetic_sync.scenarios import read_case
from tools.synthetic_sync.solver import environment, fingerprint


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINES = HERE / "cases" / "unknown-focal-continuation"
CASES = (
    "no-vp-mixed-guessedK", "no-vp-shared-guessedK",
    "no-vp-weak-baseline-guessedK", "no-vp-pure-rotation-guessedK",
)
# Declared before any exploratory optimizer call. The runner enforces both
# per-case and whole-corpus caps, including finite-difference residual calls.
LIMITS = dict(per_case_residual=3000, per_case_jacobian=80,
              per_case_seconds=90., total_residual=12000,
              total_jacobian=320, total_seconds=360.)
FOCAL_SCALE_BOUNDS = (0.6, 1.6)


def source_identity() -> dict:
    paths = sorted((ROOT / "core").rglob("*.py"))
    paths += [HERE / name for name in (
        "independent_focal.py", "no_vp_bootstrap.py", "solver.py",
        "evaluation.py", "geometry.py", "scenarios.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    return dict(source_sha256=digest.hexdigest(), environment=environment(ROOT),
        executable=sys.executable, python=platform.python_version(),
        numpy=np.__version__, scipy=__import__("scipy").__version__,
        opencv=opencv, threads={key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})


def input_only(request: dict, baseline: dict) -> tuple[dict, dict]:
    """Extract numerical input without passing the oracle to optimization."""
    ids = [camera["id"] for camera in request["cameras"]]
    points = [point["id"] for point in request["points"]]
    if len(ids) != 3:
        raise ValueError("Prototype requires three cameras")
    if request.get("anchor_id") != ids[0]:
        raise ValueError("Prototype requires the first camera to be the anchor")
    if any(not math.isfinite(c["fx"]) or c["fx"] <= 0 or c["fx"] != c["fy"] or
           c.get("division_lambda", 0) or c.get("brown_conrady") or c.get("distortion")
           for c in request["cameras"]):
        raise ValueError("Prototype requires known square pixels and zero distortion")
    if (request.get("fixed_similarities") or request.get("lock_rotation") or
        request.get("lock_translation") or request.get("lines") or
        request.get("line_observations") or request.get("mirror_pairs") or
        request.get("parallel_pairs") or request.get("plane_groups") or
        request.get("readonly_match_ids") or
        (request.get("location_match_ids") is not None and
         set(request["location_match_ids"]) != set(ids)) or
        any(p.get("ground") or p.get("known") is not None for p in request["points"])):
        raise ValueError("Prototype requires unconstrained free 2D point picks")
    if len(ids) != 3 or set(baseline["cameras"]) != set(ids) or set(baseline["landmarks"]) != set(points):
        raise ValueError("Prototype requires three fully supported cameras and points")
    return deepcopy(request), deepcopy(baseline)


def initial_state(request: dict, baseline: dict) -> dict:
    """Put saved camera/point estimates in anchor coordinates and unit baseline."""
    cameras = [baseline["cameras"][c["id"]] for c in request["cameras"]]
    anchor_r = np.asarray(cameras[0]["rotation"], float)
    anchor_c = np.asarray(cameras[0]["center"], float)
    baseline_length = np.linalg.norm(np.asarray(cameras[1]["center"], float)-anchor_c)
    if baseline_length <= 1e-8:
        raise ValueError("Initial camera baseline is zero; no finite scale gauge")
    centers = [anchor_r @ (np.asarray(c["center"], float)-anchor_c) / baseline_length for c in cameras]
    rotations = [np.asarray(c["rotation"], float) @ anchor_r.T for c in cameras]
    point_ids = sorted(baseline["landmarks"])
    points = np.asarray([anchor_r @ (np.asarray(baseline["landmarks"][key], float)-anchor_c) /
                         baseline_length for key in point_ids])
    direction = centers[1] / np.linalg.norm(centers[1])
    axis = np.eye(3)[np.argmin(abs(direction))]
    tangent_a = np.cross(direction, axis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(direction, tangent_a)
    return dict(point_ids=point_ids, centers=centers, rotations=rotations, points=points,
                direction=direction, tangents=np.vstack([tangent_a, tangent_b]),
                baseline_length=baseline_length)


def parameterize(request: dict, state: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.concatenate([
        np.zeros(3),
        Rotation.from_matrix(state["rotations"][1]).as_rotvec(),
        np.zeros(2),
        Rotation.from_matrix(state["rotations"][2]).as_rotvec(),
        state["centers"][2],
        state["points"].ravel(),
    ])


def decode(x: np.ndarray, request: dict, state: dict) -> tuple[list[float], list[np.ndarray], list[np.ndarray], np.ndarray]:
    from scipy.spatial.transform import Rotation

    focals = [float(c["fx"] * math.exp(x[i])) for i, c in enumerate(request["cameras"])]
    direction = state["direction"] + x[6] * state["tangents"][0] + x[7] * state["tangents"][1]
    direction /= np.linalg.norm(direction)
    rotations = [np.eye(3), Rotation.from_rotvec(x[3:6]).as_matrix(),
                 Rotation.from_rotvec(x[8:11]).as_matrix()]
    centers = [np.zeros(3), direction, x[11:14]]
    points = x[14:].reshape(-1, 3)
    return focals, rotations, centers, points


def observation_arrays(request: dict, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_index = {c["id"]: i for i, c in enumerate(request["cameras"])}
    point_index = {key: i for i, key in enumerate(state["point_ids"])}
    observations = request["observations"]
    return (np.asarray([camera_index[o["match_id"]] for o in observations], int),
            np.asarray([point_index[o["landmark_id"]] for o in observations], int),
            np.asarray([[o["u"], o["v"]] for o in observations], float))


def jacobian_pattern(camera_indices: np.ndarray, point_indices: np.ndarray, n_points: int):
    """Each pick depends only on its focal, camera pose and free point."""
    from scipy.sparse import lil_matrix

    pattern = lil_matrix((2 * len(camera_indices), 14 + 3*n_points), dtype=int)
    for i, (camera, point) in enumerate(zip(camera_indices, point_indices)):
        columns = [camera, *range(14+3*point, 17+3*point)]
        if camera == 1:
            columns += list(range(3, 8))
        elif camera == 2:
            columns += list(range(8, 14))
        pattern[2*i:2*i+2, columns] = 1
    return pattern.tocsr()


class EvalCap(BaseException):
    pass


class WorkMeter:
    def __init__(self, total: dict):
        self.total = total
        self.local = dict(residual=0, jacobian=0)
        self.started = time.monotonic()

    def tick(self, kind: str):
        self.local[kind] += 1
        self.total[kind] += 1
        elapsed = time.monotonic()-self.started
        if (self.local[kind] > LIMITS[f"per_case_{kind}"] or
            self.total[kind] > LIMITS[f"total_{kind}"] or
            elapsed > LIMITS["per_case_seconds"] or
            time.monotonic()-self.total["started"] > LIMITS["total_seconds"]):
            raise EvalCap(f"{kind} or wall-time optimizer cap exceeded")


def optimize(request: dict, baseline: dict, total: dict) -> dict:
    """Fit free focal/poses/points from picks; keep only one global gauge."""
    from scipy.optimize import least_squares
    from scipy.optimize._numdiff import approx_derivative

    state = initial_state(request, baseline)
    cidx, pidx, uv = observation_arrays(request, state)
    cxcy = np.asarray([[c["cx"], c["cy"]] for c in request["cameras"]], float)
    meter = WorkMeter(total)
    pattern = jacobian_pattern(cidx, pidx, len(state["point_ids"]))

    def residual(x):
        meter.tick("residual")
        focal, rotations, centers, points = decode(x, request, state)
        predicted = np.empty_like(uv)
        for camera in range(3):
            selector = cidx == camera
            xyz = (rotations[camera] @ (points[pidx[selector]]-centers[camera]).T).T
            z = np.maximum(xyz[:, 2], 1e-6)
            predicted[selector] = focal[camera] * xyz[:, :2] / z[:, None] + cxcy[camera]
            # Nonpositive depth cannot satisfy a pinhole observation.
            predicted[selector] += np.maximum(1e-6-xyz[:, 2], 0)[:, None] * 1e3
        return (predicted-uv).ravel()

    def jacobian(x):
        meter.tick("jacobian")
        return approx_derivative(residual, x, method="2-point", sparsity=pattern)

    x0 = parameterize(request, state)
    lower = np.full(len(x0), -np.inf)
    upper = np.full(len(x0), np.inf)
    lower[:3] = math.log(FOCAL_SCALE_BOUNDS[0])
    upper[:3] = math.log(FOCAL_SCALE_BOUNDS[1])
    result = least_squares(residual, x0, jac=jacobian, bounds=(lower, upper),
        method="trf", x_scale="jac", max_nfev=80, ftol=1e-10,
        xtol=1e-10, gtol=1e-10)
    focal, rotations, centers, points = decode(result.x, request, state)
    rmse = float(np.sqrt(np.mean(np.square(result.fun.reshape(-1, 2)).sum(axis=1))))
    # The gauge is removed, but the raw spectrum still depends on the chosen
    # parameter units. Do not tune a pass/fail cutoff to these frozen cases.
    singular = np.linalg.svd(result.jac.toarray() if hasattr(result.jac, "toarray") else result.jac,
                             compute_uv=False)
    cameras = {}
    for i, source in enumerate(request["cameras"]):
        cameras[source["id"]] = dict(source, fx=focal[i], fy=focal[i],
            rotation=rotations[i].tolist(), center=centers[i].tolist())
    record = dict(success=bool(result.success), message=result.message,
        reported_rmse_px=rmse, cameras=cameras,
        landmarks={key: points[i].tolist() for i, key in enumerate(state["point_ids"])},
        line_segments={})
    return dict(record=record, input_selection=dict(
        all_cameras=set(cameras)=={c["id"] for c in request["cameras"]},
        all_points=set(record["landmarks"])=={p["id"] for p in request["points"]},
        optimizer_converged=bool(result.success), fitted_rmse_px=rmse,
        baseline_fitted_rmse_px=baseline["reported_rmse_px"],
        singular_values=singular.tolist(),
        min_over_max_singular=float(singular[-1]/singular[0]),
        focal_at_bound=[bool(abs(result.x[i]-lower[i]) < 1e-4 or
                             abs(result.x[i]-upper[i]) < 1e-4) for i in range(3)]),
        optimizer=dict(nfev=result.nfev, njev=result.njev, meter=meter.local,
                       elapsed_s=time.monotonic()-meter.started,
                       initial_baseline_length=state["baseline_length"]))


def run_one(name: str, total: dict, source: dict, out: Path) -> dict:
    case_path = HERE / "cases" / f"{name}.json"
    baseline_path = BASELINES / f"{name}-scale-1.json"
    case = read_case(case_path)
    validate_fixture(case)
    saved = json.loads(baseline_path.read_text())
    request, baseline = input_only(case["request"], saved["record"])
    if saved["request_sha256"] != fingerprint(request) or baseline["request_sha256"] != fingerprint(request):
        raise ValueError("Saved baseline does not match the frozen request")
    trial = dict(name=name, request=request, baseline=baseline, source_runtime=source,
                 limits=LIMITS, focal_scale_bounds=FOCAL_SCALE_BOUNDS)
    result_path = out / f"{name}.json"
    if result_path.exists():
        raise FileExistsError("Refuse to silently overwrite an optimizer result")
    ledger = out / "optimizer-ledger.jsonl"
    with ledger.open("a") as log:
        log.write(json.dumps(dict(kind="started", trial=trial), sort_keys=True, allow_nan=False)+"\n")
        log.flush()
        os.fsync(log.fileno())
        try:
            fitted = optimize(request, baseline, total)
        except BaseException as exc:
            log.write(json.dumps(dict(kind="failed", name=name,
                                      error_type=type(exc).__name__, error=str(exc)))+"\n")
            log.flush()
            os.fsync(log.fileno())
            raise
        log.write(json.dumps(dict(kind="completed", name=name, result=fitted,
                                  total_work={k:v for k,v in total.items() if k!="started"}),
                             sort_keys=True, allow_nan=False)+"\n")
        log.flush()
        os.fsync(log.fileno())
    # The oracle enters only here, after the input-only candidate is fixed.
    assessment = assess(case, fitted["record"])
    report = dict(name=name, request_sha256=fingerprint(request),
        baseline_sha256=hashlib.sha256(json.dumps(baseline, sort_keys=True,
            allow_nan=False).encode()).hexdigest(), source_runtime=source,
        limits=LIMITS, focal_scale_bounds=FOCAL_SCALE_BOUNDS,
        **fitted, assessment=assessment)
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return dict(name=name, fitted_rmse_px=fitted["input_selection"]["fitted_rmse_px"],
                focal_relative_error=assessment["focal_relative_error"],
                classification=assessment["classification"],
                geometry_passed=assessment["independent_geometry"]["passed"],
                min_over_max_singular=fitted["input_selection"]["min_over_max_singular"],
                optimizer=fitted["optimizer"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "optimizer-ledger.jsonl").exists():
        raise FileExistsError("Use a new output directory; this bounded trial does not restart")
    total = dict(residual=0, jacobian=0, started=time.monotonic())
    source = source_identity()
    names = [args.case] if args.case else CASES
    rows = [run_one(name, total, source, args.out) for name in names]
    (args.out / "summary.json").write_text(json.dumps(dict(rows=rows,
        limits=LIMITS, source_runtime=source,
        total_work={k:v for k,v in total.items() if k!="started"}),
        indent=2, allow_nan=False)+"\n")
    print(json.dumps(rows, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
