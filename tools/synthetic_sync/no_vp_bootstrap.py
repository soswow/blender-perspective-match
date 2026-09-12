"""Frozen, budgeted no-VP 2D-to-2D startup controls.

The generator and oracle are independent of Perspective Match. The numerical
runner executes the baseline true/guessed-K matrix under the shared experiment
ledger; it does not perform an outer focal search or select candidates by truth.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import look_at, project, visible
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import environment, fingerprint, solve


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent / "cases"
KINDS = ("shared", "mixed", "weak_baseline", "pure_rotation")
INTRINSICS = {"shared": (780., 780., 780.),
              "mixed": (700., 850., 1000.),
              "weak_baseline": (780., 780., 780.),
              "pure_rotation": (780., 780., 780.)}
EXPECTED = {"shared": "solve", "mixed": "solve",
            "weak_baseline": "diagnose", "pure_rotation": "reject"}
GUESS_FACTORS = {"shared": (1.25, 1.25, 1.25),
                 "mixed": (1.25, 0.8, 1.15),
                 "weak_baseline": (1.25, 1.25, 1.25),
                 "pure_rotation": (1.25, 1.25, 1.25)}
FOCAL_TOLERANCE = 0.02
POINT_FRACTION = 0.05
HOLDOUT_RMSE_PX = 1.0
ROTATION_DEG = 1.0
CENTER_FRACTION = 0.02


def make_case(kind: str, intrinsics: str) -> dict:
    """Create exact free-scale evidence without metric or orientation priors."""
    if kind not in KINDS or intrinsics not in {"trueK", "guessedK"}:
        raise ValueError((kind, intrinsics))
    case = generate("free_scale", seed=0, noise_px=0.)
    case.update(name=f"no-vp-{kind.replace('_', '-')}-{intrinsics}",
                family="no_vp_bootstrap", kind=kind, intrinsics=intrinsics)
    truth = case["truth"]
    points = truth["points"]
    cameras = truth["cameras"]
    first_center = np.asarray(cameras[0]["center"], dtype=float)
    for i, (camera, focal) in enumerate(zip(cameras, INTRINSICS[kind])):
        center = np.asarray(camera["center"], dtype=float)
        if kind == "weak_baseline":
            center = first_center + 0.025 * (center-first_center)
        elif kind == "pure_rotation":
            center = first_center.copy()
        target = ([0.17*i, 0.08*i, 0.7] if kind == "pure_rotation" else [0., 0., 0.7])
        camera.update(center=center.tolist(), rotation=look_at(center, target),
                      fx=focal, fy=focal, cx=480., cy=360.)
    mesh = truth["mesh"]
    observations = []
    for point_id, position in sorted(points.items()):
        for camera in cameras:
            if visible(position, camera, mesh):
                uv = project([position], camera)[0][0]
                observations.append(dict(match_id=camera["id"], landmark_id=point_id,
                                         u=float(uv[0]), v=float(uv[1]), weight=1.))
    support = {point_id: {o["match_id"] for o in observations if o["landmark_id"] == point_id}
               for point_id in points}
    observations = [o for o in observations if len(support[o["landmark_id"]]) >= 2]
    retained = {o["landmark_id"] for o in observations}
    truth["points"] = {key: value for key, value in points.items() if key in retained}
    case["request"]["points"] = [dict(id=key, ground=False, known=None) for key in sorted(retained)]
    case["request"]["observations"] = observations
    # Stored private cameras are deliberately neither the true camera nor a
    # consistent relative-pose seed; this includes the anchor.
    stored = []
    for i, camera in enumerate(cameras):
        private_center = np.array([7. + 1.7*i, -8. + 0.4*i, 5. + 0.8*i])
        private = deepcopy(camera)
        private.update(center=private_center.tolist(),
                       rotation=look_at(private_center, [0.3*i, -0.2, 0.4]),
                       fx=camera["fx"] * (GUESS_FACTORS[kind][i] if intrinsics == "guessedK" else 1.),
                       fy=camera["fy"] * (GUESS_FACTORS[kind][i] if intrinsics == "guessedK" else 1.))
        stored.append(private)
    case["request"]["cameras"] = stored
    case["request"]["fixed_similarities"] = {}
    case["request"]["lines"] = []
    case["request"]["line_observations"] = []
    checks = []
    # Regenerate withheld visibility after changing poses, using the independent
    # mesh oracle and excluding all supplied training points.
    original = generate("free_scale", 0, 0.)["truth"]["checks"]
    for item in original:
        views = [c["id"] for c in cameras if visible(item["position"], c, mesh)]
        if views:
            checks.append(dict(id=item["id"], position=item["position"], views=views))
    truth["checks"] = checks
    case["expectation"] = dict(outcome="solve", cameras=[c["id"] for c in cameras],
        excluded_cameras=[], gauge="similarity", holdout_rmse_px=HOLDOUT_RMSE_PX,
        rotation_deg=ROTATION_DEG, center_fraction=CENTER_FRACTION,
        required_points=sorted(retained), point_fraction=POINT_FRACTION)
    case["diagnostic_expectation"] = EXPECTED[kind]
    case["focal_relative_error_limit"] = FOCAL_TOLERANCE
    case["source"] = dict(exact=True, seeded_generator="free_scale-0",
        centered_principal_point=True, distortion="zero", vanishing_points="none",
        known_3d="none", ground="none", pose_locks="none")
    validate_fixture(case)
    return case


def validate_fixture(case: dict) -> None:
    """Catch any accidental truth, metric, or VP leakage before a numerical run."""
    req, truth = case["request"], case["truth"]
    assert len(req["cameras"]) == 3 and not req["fixed_similarities"]
    assert not req["lines"] and not req["line_observations"]
    assert not req["mirror_pairs"] and not req["parallel_pairs"] and not req["plane_groups"]
    assert all(not p["ground"] and p["known"] is None for p in req["points"])
    assert all(c["cx"] == 480. and c["cy"] == 360. and c["fx"] == c["fy"]
               for c in req["cameras"])
    assert set(truth["points"]) == {o["landmark_id"] for o in req["observations"]}
    assert all(len([o for o in req["observations"] if o["landmark_id"] == p["id"]]) >= 2
               for p in req["points"])
    assert len(req["points"]) >= 8
    assert all(sum(c["id"] in item["views"] for item in truth["checks"]) >= 6
               for c in truth["cameras"])
    for stored, actual in zip(req["cameras"], truth["cameras"]):
        assert np.linalg.norm(np.asarray(stored["center"])-actual["center"]) > 2.
    xyz = np.asarray(list(truth["points"].values()))
    assert np.linalg.matrix_rank(xyz-xyz.mean(axis=0), tol=1e-8) == 3
    withheld = np.asarray([c["position"] for c in truth["checks"]])
    assert np.linalg.norm(xyz[:, None] - withheld, axis=2).min() > 1e-6


def assess(case: dict, record: dict) -> dict:
    """Separate fitted success, independent geometry, and useful refusal."""
    geometry = evaluate(case, record)
    focal = {}
    actual = {c["id"]: c for c in case["truth"]["cameras"]}
    for key, camera in record.get("cameras", {}).items():
        focal[key] = abs(camera["fx"]-actual[key]["fx"]) / actual[key]["fx"]
    focal_ok = bool(focal) and set(focal) == set(case["expectation"]["cameras"]) and all(
        np.isfinite(v) and v <= case["focal_relative_error_limit"] for v in focal.values())
    if record.get("exception"):
        classification = "exception"
    elif record["success"] and case["kind"] == "pure_rotation":
        classification = "unsupported_depth_acceptance"
    elif record["success"]:
        classification = "accurate_acceptance" if geometry["passed"] and focal_ok else "false_precise_acceptance"
    else:
        classification = "useful_refusal" if record.get("message", "").strip() else "unexplained_refusal"
    return dict(classification=classification, focal_relative_error=focal, focal_within_limit=focal_ok,
                independent_geometry=geometry)


def frozen_cases() -> list[dict]:
    return [make_case(kind, intrinsics) for kind in KINDS for intrinsics in ("trueK", "guessedK")]


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_frozen() -> list[Path]:
    files = []
    for case in frozen_cases():
        path = CASE_DIR / f"{case['name']}.json"
        write_case(case, path)
        files.append(path)
    return files


def run(out: Path) -> dict:
    """Run the frozen true/guessed-K matrix under the common call ledger."""
    from tools.synthetic_sync.budget import ExperimentBudget

    out.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    metadata = dict(environment=environment(ROOT), python=platform.python_version(),
        numpy=np.__version__, opencv=opencv,
        generator_sha256=source_sha256(Path(__file__)),
        solver_sha256=source_tree_sha256(list((ROOT / "core" / "sync").glob("*.py"))),
        focal_solver_sha256=source_sha256(ROOT / "core" / "lens_refine.py"),
        budget_sha256=source_sha256(ROOT / "tools" / "synthetic_sync" / "budget.py"),
        numerical_core_sha256=source_tree_sha256(list((ROOT / "core").rglob("*.py"))),
        harness_sha256=source_tree_sha256([
            ROOT / "tools" / "synthetic_sync" / name
            for name in ("solver.py", "evaluation.py", "geometry.py", "scenarios.py")
        ]),
        options=dict(max_calls=16, per_call_seconds=180, wall_seconds=720,
                     lens_search_implemented=False, proposed_shared_search_span=0.25,
                     proposed_shared_span_includes_truth=True,
                     matrix=[f"{kind}/{intr}" for kind in KINDS for intr in ("trueK", "guessedK")]))
    rows = []
    with ExperimentBudget(out / "ledger.jsonl", metadata=metadata,
                          max_calls=16, per_call_seconds=180, wall_seconds=720) as budget:
        for kind in KINDS:
            for intr in ("trueK", "guessedK"):
                case_path = CASE_DIR / f"no-vp-{kind.replace('_', '-')}-{intr}.json"
                case = read_case(case_path)
                validate_fixture(case)
                label = case["name"]
                cached = budget.cached(label, case["request"])
                if cached is None:
                    with budget.attempt(label, case["request"]) as attempt:
                        record = solve(case["request"])
                        attempt.complete(record)
                else:
                    record = cached
                assessment = assess(case, record)
                row = dict(name=label, case_file=str(case_path.relative_to(ROOT)),
                           request_sha256=fingerprint(case["request"]), record=record,
                           assessment=assessment)
                rows.append(row)
                (out / f"{label}-result.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
                print(label, assessment["classification"], record.get("reported_rmse_px"),
                      "elapsed", record.get("elapsed_s"), flush=True)
                if kind == "shared" and intr == "trueK" and assessment["classification"] != "accurate_acceptance":
                    break
            if kind == "shared" and rows[0]["assessment"]["classification"] != "accurate_acceptance":
                break
    summary = dict(metadata=metadata, rows=[dict(name=r["name"], classification=r["assessment"]["classification"],
                    reported_rmse_px=r["record"].get("reported_rmse_px"), elapsed_s=r["record"].get("elapsed_s"),
                    focal_relative_error=r["assessment"]["focal_relative_error"],
                    violations=r["assessment"]["independent_geometry"]["violations"])
                    for r in rows])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.freeze:
        for path in write_frozen():
            print(path.relative_to(ROOT))
    if args.run:
        print(json.dumps(run(args.run), indent=2, allow_nan=False))
    if not args.freeze and not args.run:
        parser.error("Choose --freeze or --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
