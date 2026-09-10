"""Controlled evidence-placement experiment on the synthetic overhead fixture."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, surface_points, visible
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import read_case, write_case
from tools.synthetic_sync.solver import fingerprint, solve


TARGET = "view_2"
CANDIDATES = ("outer_surface", "raised_center")


def candidates(case):
    """Two predeclared locations, never selected by their measured solve quality."""
    if case["family"] != "overhead":
        raise ValueError("This bounded experiment requires the overhead fixture")
    mesh = case["truth"]["mesh"]
    positions = {
        "outer_surface": surface_points(mesh, [(0.06, 0.86)])[5],
        "raised_center": surface_points(mesh, [(0.46, 0.54)])[11],
    }
    occupied = list(case["truth"]["points"].values()) + [p["position"] for p in case["truth"]["checks"]]
    for name, point in positions.items():
        if np.min(np.linalg.norm(np.asarray(occupied)-point, axis=1)) < 1e-6:
            raise ValueError(f"{name}: candidate overlaps training or withheld geometry")
        if not all(visible(point, camera, mesh) for camera in case["truth"]["cameras"]):
            raise ValueError(f"{name}: candidate must be visible in all three views")
    return positions


def resample_picks(case, seed):
    """Replace pick noise while keeping cameras, geometry and the graph identical."""
    if case["request"]["line_observations"]:
        raise ValueError("Noise resampling here covers point-only fixtures")
    result = deepcopy(case)
    rng = np.random.default_rng(seed)
    cameras = {c["id"]: c for c in case["truth"]["cameras"]}
    for pick in result["request"]["observations"]:
        ideal = project([case["truth"]["points"][pick["landmark_id"]]], cameras[pick["match_id"]])[0][0]
        pick["u"], pick["v"] = map(float, ideal + rng.normal(0, case["noise_px"], 2))
    result["name"] = f"{case['name']}-noise-{seed}"
    return result


def scale_pick_noise(case, factor):
    """Scale the same error vector; do not move cameras or change observations."""
    result = deepcopy(case)
    cameras = {c["id"]: c for c in case["truth"]["cameras"]}
    for pick in result["request"]["observations"]:
        ideal = project([case["truth"]["points"][pick["landmark_id"]]], cameras[pick["match_id"]])[0][0]
        uv = ideal + factor * (np.array([pick["u"], pick["v"]]) - ideal)
        pick["u"], pick["v"] = map(float, uv)
    result["noise_px"] *= factor
    result["expectation"]["holdout_rmse_px"] = max(1.0, 6 * result["noise_px"])
    result["name"] += f"-noise-scale-{factor:g}"
    return result


def add_landmark(case, candidate, *, include_target):
    """Add a free landmark with two or three noisy observations; never a 3D pin."""
    point = candidates(case)[candidate]
    result = deepcopy(case)
    item_id = "extra_" + candidate
    if item_id in result["truth"]["points"]:
        raise ValueError("Candidate already exists")
    result["request"]["points"].append(dict(id=item_id, ground=False, known=None))
    result["truth"]["points"][item_id] = point
    # Per-camera seeds keep supporting picks byte-identical in both variants.
    for camera in result["truth"]["cameras"]:
        if camera["id"] == TARGET and not include_target:
            continue
        key = f"{fingerprint(case['request'])}:{candidate}:{camera['id']}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        ideal = project([point], camera)[0][0]
        uv = ideal + np.random.default_rng(seed).normal(0, case["noise_px"], 2)
        result["request"]["observations"].append(dict(match_id=camera["id"], landmark_id=item_id,
            u=float(uv[0]), v=float(uv[1]), weight=1.0))
    variant = candidate + ("_three_picks" if include_target else "_support_only")
    result["name"] += "-" + variant
    return result, variant


def compact_metrics(case, result, assessment):
    """Preserve failures and score location-specific error on the unchanged checks."""
    camera = assessment["cameras"].get(TARGET)
    complete = result["success"] and set(case["expectation"]["cameras"]) <= set(result["cameras"])
    metrics = dict(passed=assessment["passed"], complete=bool(complete),
        fitted_rmse_px=result["reported_rmse_px"], elapsed_s=result["elapsed_s"],
        holdout_rmse_px=None, holdout_max_px=None, height_groups={}, violations=assessment["violations"])
    if camera and camera["actual_uv"] is not None:
        metrics.update(holdout_rmse_px=camera["holdout_rmse_px"], holdout_max_px=camera["holdout_max_px"])
        points = [p for p in case["truth"]["checks"] if TARGET in p["views"]]
        errors = np.linalg.norm(np.asarray(camera["expected_uv"])-camera["actual_uv"], axis=1)
        heights = np.array([p["position"][2] for p in points])
        for z in sorted(set(heights)):
            selected = errors[heights == z]
            metrics["height_groups"][str(z)] = dict(count=len(selected), rms=float(np.sqrt(np.mean(selected**2))))
    return metrics


def summarize(rows):
    """Use resampled trials for comparisons; keep the originally flagged draws separate."""
    trials = [row for row in rows if row["group"] == "resampled"]
    variants = sorted({row["variant"] for row in trials})
    lookup = {(row["trial"], row["variant"]): row for row in trials}
    output = {}
    for variant in variants:
        selected = [row for row in trials if row["variant"] == variant]
        usable = [row for row in selected if row["metrics"]["complete"] and row["metrics"]["holdout_rmse_px"] is not None]
        values = [row["metrics"]["holdout_rmse_px"] for row in usable]
        entry = dict(trials=len(selected), incomplete=len(selected)-len(usable),
            accuracy_flags=sum(not row["metrics"]["passed"] for row in selected),
            median_px=float(np.median(values)) if values else None,
            p90_px=float(np.percentile(values, 90)) if values else None)
        for reference in ("baseline", variant.replace("_three_picks", "_support_only")):
            if reference == variant or reference not in variants:
                continue
            pairs = [(row, lookup[(row["trial"], reference)]) for row in usable
                     if lookup[(row["trial"], reference)]["metrics"]["complete"]
                     and lookup[(row["trial"], reference)]["metrics"]["holdout_rmse_px"] is not None]
            ratios = [a["metrics"]["holdout_rmse_px"]/max(b["metrics"]["holdout_rmse_px"], 1e-9) for a,b in pairs]
            entry["vs_" + reference] = dict(pairs=len(pairs), improved=sum(r < 1 for r in ratios),
                median_ratio=float(np.median(ratios)) if ratios else None)
        output[variant] = entry
    return output


def write_summary(rows, path):
    summary = summarize(rows)
    (path / "summary.json").write_text(json.dumps(dict(summary=summary, runs=rows), indent=2, allow_nan=False)+"\n")
    lines = ["# Overhead evidence-placement experiment", "",
        "Fresh noise draws use fixed geometry. Originally flagged cases are reported separately.", "",
        "A new free landmark costs three picks (two supporting views and the overhead view). The support-only control costs two. No Known 3D truth is supplied.", "",
        "| Variant | Trials | Incomplete | Accuracy flags | Median withheld px | P90 px | Improved vs baseline | Median paired ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    for name, m in summary.items():
        comparison = m.get("vs_baseline", {})
        improved = f"{comparison['improved']}/{comparison['pairs']}" if comparison else "—"
        lines.append(f"| {name} | {m['trials']} | {m['incomplete']} | {m['accuracy_flags']} | {number(m['median_px'])} | {number(m['p90_px'])} | {improved} | {number(comparison.get('median_ratio'))} |")
    lines += ["", "Ratios below 1 improve withheld projection. These are descriptive small-sample results, not confidence bounds.", "",
        "## Originally flagged draws and noise scaling", "", "| Case | Variant | Fitted px | Withheld px | Maximum px |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["group"] != "resampled":
            m = row["metrics"]
            lines.append(f"| {row['trial']} | {row['variant']} | {m['fitted_rmse_px']:.3f} | {number(m['holdout_rmse_px'])} | {number(m['holdout_max_px'])} |")
    lines += ["", "Per-height errors, coverage and violations are in summary.json. Each exact case/result has its own JSON; report.html shows all withheld overlays."]
    (path / "summary.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, action="append", help="Saved overhead cases; defaults to the two frozen fixtures in cases/")
    parser.add_argument("--trials", type=int, default=8, help="Fresh noise draws per fixed geometry")
    parser.add_argument("--noise-seed", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.noise_seed < 0:
        parser.error("--trials must be positive and --noise-seed nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory to preserve earlier experiments")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = args.case or [Path(__file__).with_name("cases") / f"overhead-{seed}.json" for seed in (0,2)]
    bases = [read_case(p) for p in paths]
    if len({case["name"] for case in bases}) != len(bases):
        parser.error("Case names must be unique")
    protocol = dict(target=TARGET, candidates={case["name"]: candidates(case) for case in bases},
        base_requests={case["name"]: fingerprint(case["request"]) for case in bases},
        trials=args.trials, noise_seed=args.noise_seed, use_cache=False,
        noise_seeds_reused_across_geometries=True,
        decision="Promote a placement rule only if it lowers median paired withheld RMS by at least 20%, improves at least 70% of fresh-noise pairs, loses no cameras, and does not increase P90 error; validate on other layouts before product use.")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    rows, reports = [], []
    for base in bases:
        trials = [("original", base)] + [("resampled", resample_picks(base, seed)) for seed in range(args.noise_seed, args.noise_seed+args.trials)]
        for group, trial in trials:
            variants = [(trial, "baseline")]
            for name in CANDIDATES:
                variants += [add_landmark(trial, name, include_target=flag) for flag in (False, True)]
            if group == "original":
                variants += [(scale_pick_noise(trial, factor), f"noise_scale_{factor:g}") for factor in (0, .5)]
            for case, variant in variants:
                write_case(case, args.out / (case["name"]+".json"))
                result = solve(case["request"], use_cache=False)
                assessment = evaluate(case, result)
                (args.out / (case["name"]+"-result.json")).write_text(json.dumps(dict(result=result, assessment=assessment), indent=2, allow_nan=False)+"\n")
                metrics = compact_metrics(case, result, assessment)
                rows.append(dict(group=group, trial=trial["name"], variant=variant, metrics=metrics))
                reports.append((case, result, assessment))
                write_summary(rows, args.out)
                print(f"{case['name']}: fit={metrics['fitted_rmse_px']:.3f}px withheld={metrics['holdout_rmse_px']} complete={metrics['complete']}", flush=True)
    write_report(reports, args.out / "report.html")
    # Accuracy flags are measured outcomes, not an experiment execution error.
    return 1 if any(result.get("exception") for _,result,_ in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
