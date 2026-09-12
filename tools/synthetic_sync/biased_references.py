"""Bounded controls for locally biased Known 3D point references."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import generate, write_case
from tools.synthetic_sync.solver import environment, fingerprint, solve


SEED = 17
KNOWN_IDS = ("point_44", "point_45", "point_46", "point_47")
BIASED_IDS = KNOWN_IDS[:2]
UNBIASED_IDS = KNOWN_IDS[2:]
BIAS_VECTOR = np.array([0.16, -0.06, 0.05], dtype=np.float64)
SOFT_SLACK = 0.20
CONDITIONS = (
    ("unbiased-hard", False, 0.0),
    ("unbiased-soft", False, SOFT_SLACK),
    ("biased-hard", True, 0.0),
    ("biased-soft", True, SOFT_SLACK),
)


def biased_reference_case(condition: str, *, noise_px: float = 0.0) -> dict:
    """Build one predeclared reference condition from shared truth and picks."""
    try:
        biased, slack = next((b, s) for name, b, s in CONDITIONS if name == condition)
    except StopIteration as error:
        raise ValueError(f"Unknown biased-reference condition: {condition}") from error
    if noise_px not in (0.0, 0.3):
        raise ValueError("The bounded protocol permits only exact picks or its fixed 0.3 px draw")

    case = generate("ground", seed=SEED, noise_px=noise_px)
    request, truth, expected = (case[key] for key in ("request", "truth", "expectation"))
    ground_ids = {point["id"] for point in request["points"] if point["ground"]}
    retained = ground_ids | set(KNOWN_IDS)
    request["points"] = [point for point in request["points"] if point["id"] in retained]
    request["observations"] = [
        observation for observation in request["observations"]
        if observation["landmark_id"] in retained
    ]
    truth["points"] = {key: value for key, value in truth["points"].items() if key in retained}
    for point in request["points"]:
        if point["id"] not in KNOWN_IDS:
            continue
        reference = np.asarray(truth["points"][point["id"]], dtype=np.float64)
        if biased and point["id"] in BIASED_IDS:
            reference = reference + BIAS_VECTOR
        point["known"] = reference.tolist()
    request["known_3d_slack"] = slack
    expected.update(
        gauge="anchor",
        required_points=list(KNOWN_IDS),
        point_fraction=0.02,
        ground_max_distance=1.0e-5,
    )
    noise_label = "exact" if noise_px == 0.0 else "noise-0p3"
    case.update(
        name=f"biased-reference-{noise_label}-{condition}",
        family="biased_references",
        reference_experiment=dict(
            condition=condition,
            biased=biased,
            seed=SEED,
            noise_px=noise_px,
            known_ids=list(KNOWN_IDS),
            biased_ids=list(BIASED_IDS),
            unbiased_ids=list(UNBIASED_IDS),
            bias_vector=BIAS_VECTOR.tolist(),
            bias_magnitude=float(np.linalg.norm(BIAS_VECTOR)),
            known_3d_slack=slack,
            metric_anchor="view_0 identity plus six unbiased hard On Ground references",
            interpretation=(
                "Hard biased references deliberately conflict with the exact image evidence. "
                "Accuracy flags in that condition describe model conflict, not automatically a solver defect."
            ),
        ),
    )
    return case


def experiment_cases(*, include_noisy: bool = False) -> list[dict]:
    """Return four exact controls, optionally repeated for one fixed noisy draw."""
    noise_levels = (0.0, 0.3) if include_noisy else (0.0,)
    return [
        biased_reference_case(condition, noise_px=noise_px)
        for noise_px in noise_levels
        for condition, _biased, _slack in CONDITIONS
    ]


def _rms(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def fitted_pick_metrics(case: dict, record: dict) -> dict:
    """Reproject supplied points independently and retain evidence-group errors."""
    by_point = {point["id"]: point for point in case["request"]["points"]}
    grouped: dict[str, list[float]] = defaultdict(list)
    by_camera: dict[str, list[float]] = defaultdict(list)
    support_by_camera: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    support_by_point: dict[str, list[str]] = defaultdict(list)
    for observation in case["request"]["observations"]:
        point_id, camera_id = observation["landmark_id"], observation["match_id"]
        point = record.get("landmarks", {}).get(point_id)
        camera = record.get("cameras", {}).get(camera_id)
        if point is None or camera is None:
            continue
        uv = project([point], camera)[0][0]
        error = float(np.linalg.norm(uv - [observation["u"], observation["v"]]))
        if by_point[point_id]["ground"]:
            group = "ground"
        elif point_id in BIASED_IDS:
            group = "biased_known"
        elif point_id in UNBIASED_IDS:
            group = "unbiased_known"
        else:
            group = "other"
        grouped[group].append(error)
        grouped["all"].append(error)
        by_camera[camera_id].append(error)
        support_by_camera[camera_id][group] += 1
        support_by_point[point_id].append(camera_id)
    return dict(
        rms_px={key: _rms(values) for key, values in sorted(grouped.items())},
        rms_by_camera_px={key: _rms(values) for key, values in sorted(by_camera.items())},
        support_by_camera={
            key: dict(sorted(values.items())) for key, values in sorted(support_by_camera.items())
        },
        support_by_point={key: sorted(values) for key, values in sorted(support_by_point.items())},
    )


def reference_metrics(case: dict, record: dict) -> dict:
    """Measure reconstructed points against truth, priors, and hard ground."""
    request_points = {point["id"]: point for point in case["request"]["points"]}
    point_metrics = {}
    for key in KNOWN_IDS:
        reconstructed = record.get("landmarks", {}).get(key)
        if reconstructed is None:
            point_metrics[key] = dict(missing=True)
            continue
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        truth = np.asarray(case["truth"]["points"][key], dtype=np.float64)
        prior = np.asarray(request_points[key]["known"], dtype=np.float64)
        point_metrics[key] = dict(
            reconstructed=reconstructed.tolist(),
            truth=truth.tolist(),
            prior=prior.tolist(),
            truth_error=float(np.linalg.norm(reconstructed - truth)),
            prior_gap=float(np.linalg.norm(reconstructed - prior)),
        )
    ground_z = {
        key: float(record["landmarks"][key][2])
        for key, point in request_points.items()
        if point["ground"] and key in record.get("landmarks", {})
    }
    return dict(
        points=point_metrics,
        biased_truth_rms=_rms([point_metrics[key]["truth_error"] for key in BIASED_IDS if not point_metrics[key].get("missing")]),
        biased_prior_gap_rms=_rms([point_metrics[key]["prior_gap"] for key in BIASED_IDS if not point_metrics[key].get("missing")]),
        unbiased_truth_rms=_rms([point_metrics[key]["truth_error"] for key in UNBIASED_IDS if not point_metrics[key].get("missing")]),
        ground_max_abs_z=max(map(abs, ground_z.values()), default=None),
        ground_z=ground_z,
    )


def measure(case: dict, record: dict) -> dict:
    """Combine the existing oracle with point-fit and reference diagnostics."""
    assessment = evaluate(case, record)
    message = record.get("message", "")
    return dict(
        condition=case["reference_experiment"]["condition"],
        noise_px=case["noise_px"],
        request_sha256=fingerprint(case["request"]),
        success=record.get("success", False),
        message=message,
        flags=dict(
            accuracy_violations=assessment["violations"],
            known_slack_exceeded="known 3D slack" in message,
            ground_slack_exceeded="ground slack" in message,
            weak_line_ids=record.get("weak_line_ids", []),
        ),
        reported_rmse_px=record.get("reported_rmse_px"),
        fitted_picks=fitted_pick_metrics(case, record),
        references=reference_metrics(case, record),
        withheld=assessment.get("cameras", {}),
        oracle_passed=assessment["passed"],
        alignment=assessment.get("alignment"),
    )


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def required_controls_pass(rows: list[dict]) -> bool:
    """Unbiased references must solve and pass the independent oracle."""
    return all(
        row["success"] and row["oracle_passed"]
        for row in rows if row["condition"] in {"unbiased-hard", "unbiased-soft"}
    ) and {row["condition"] for row in rows} >= {"unbiased-hard", "unbiased-soft"}


def interpretation(rows: list[dict]) -> str:
    """Describe observed deltas and actual oracle status, without canned verdicts."""
    exact = {row["condition"]: row for row in rows if row["noise_px"] == 0.0}
    hard, soft = exact["biased-hard"], exact["biased-soft"]
    hard_hold = max((m["holdout_rmse_px"] for k, m in hard["withheld"].items() if k != "view_0"), default=None)
    soft_hold = max((m["holdout_rmse_px"] for k, m in soft["withheld"].items() if k != "view_0"), default=None)
    hard_truth = hard["references"]["biased_truth_rms"]
    soft_truth = soft["references"]["biased_truth_rms"]
    if None in (hard_truth, soft_truth, hard_hold, soft_hold):
        return "Required geometry is missing; see case-level flags. No improvement conclusion is available."
    maximum_truth_error = max(
        (soft["references"]["points"][key]["truth_error"] for key in BIASED_IDS
         if not soft["references"]["points"][key].get("missing")),
        default=float("inf"),
    )
    # The point oracle's own flags are authoritative; avoid a second derived threshold.
    point_flags = [issue for issue in soft["flags"]["accuracy_violations"]
                   if any(issue.startswith(key + ":") for key in BIASED_IDS)]
    point_status = "pass" if not point_flags and np.isfinite(maximum_truth_error) else "flag"
    clean = exact["unbiased-hard"]
    clean_soft = exact["unbiased-soft"]
    baseline_delta = (
        clean_soft["references"]["biased_truth_rms"] - clean["references"]["biased_truth_rms"]
        if clean_soft["references"]["biased_truth_rms"] is not None and clean["references"]["biased_truth_rms"] is not None
        else None
    )
    return (
        f"On exact picks, biased-point truth RMS changed {hard_truth:.3f} → {soft_truth:.3f} scene units "
        f"(improvement {hard_truth-soft_truth:.3f}) and the worst non-anchor withheld-camera RMS changed "
        f"{hard_hold:.3f} → {soft_hold:.3f} px (improvement {hard_hold-soft_hold:.3f}) from biased hard to biased soft. "
        f"Soft biased points {point_status} the unchanged direct-point oracle; the overall soft biased "
        f"case {'passes' if soft['oracle_passed'] else 'flags'} its unchanged oracle "
        f"({'; '.join(soft['flags']['accuracy_violations']) or 'no violations'}). "
        f"Unbiased controls {'both pass' if required_controls_pass(rows) else 'include a failure'}; "
        f"their biased-group truth RMS differs by {_number(baseline_delta, 6)} scene units. "
        "A low fitted-pick error alone cannot establish geometric accuracy."
    )


def markdown_report(rows: list[dict], protocol: dict) -> str:
    """Render the compact experiment record checked into the repository."""
    lines = [
        "# Locally biased Known 3D references",
        "",
        "## Protocol",
        "",
        f"One generated anchor-gauge scene (seed {SEED}) uses the same truth and picks in every condition. "
        f"Two of four nearby Known 3D points receive a local `{BIAS_VECTOR.tolist()}` scene-unit offset "
        f"(magnitude {np.linalg.norm(BIAS_VECTOR):.3f}); the other two remain truthful. Six unbiased On Ground "
        "observations stay constrained hard at Z=0, so the metric/world frame does not come from aligning the answer to truth. "
        "`view_1` and `view_2` remain fully solved cameras and are checked on withheld 3D object samples.",
        "",
        f"The soft setting is `{SOFT_SLACK:.2f}` scene units. Slack is a spring scale, not a bound. "
        "The four predeclared conditions are unbiased/hard, unbiased/soft, biased/hard, and biased/soft. "
        "The ordinary oracle limits remain unchanged; hard wrong references are intentional model conflict.",
        "",
        f"Baseline revision `{protocol['environment']['revision']}`; Python {protocol['environment']['python']}, "
        f"NumPy {protocol['environment']['numpy']}, {protocol['environment']['platform']}. "
        f"Four solves took {protocol['total_solve_seconds']:.2f} s in total. "
        "Case filenames are `tools/synthetic_sync/cases/biased-reference-exact-*.json`.",
        "",
        "Replay from the repository root with "
        "`python3 tools/synthetic_sync/run.py --case "
        "tools/synthetic_sync/cases/biased-reference-exact-biased-soft.json --out /tmp/pm-biased-replay` "
        "(change the filename for the other conditions; a biased oracle flag gives `run.py` a nonzero exit). "
        "To regenerate all four conditions, use "
        "`python3 tools/synthetic_sync/biased_references.py --out /tmp/pm-biased-comparison`.",
        "",
        "## Results",
        "",
        "| Picks | Condition | Fit all px | Fit biased px | Fit truthful px | Biased truth RMS | Biased prior gap | Truthful ref RMS | view_1 holdout px | view_2 holdout px | Ground max | Oracle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {'exact' if row['noise_px'] == 0 else '0.3 px'} | {row['condition']} | "
            f"{_number(row['fitted_picks']['rms_px'].get('all'))} | {_number(row['fitted_picks']['rms_px'].get('biased_known'))} | "
            f"{_number(row['fitted_picks']['rms_px'].get('unbiased_known'))} | "
            f"{_number(row['references']['biased_truth_rms'])} | {_number(row['references']['biased_prior_gap_rms'])} | "
            f"{_number(row['references']['unbiased_truth_rms'])} | "
            f"{_number(row['withheld'].get('view_1', {}).get('holdout_rmse_px'))} | "
            f"{_number(row['withheld'].get('view_2', {}).get('holdout_rmse_px'))} | "
            f"{_number(row['references']['ground_max_abs_z'], 6)} | {'pass' if row['oracle_passed'] else 'flag'} |"
        )
    lines += [
        "",
        "Every Known 3D point has three-view support. The camera evidence is 9 point picks in `view_0`, "
        "9 in `view_1`, and 8 in `view_2`; the last view sees four rather than five ground points.",
        "",
        "### Biased point coordinates",
        "",
        "| Condition | Point | Truth XYZ | CAD prior XYZ | Reconstructed XYZ | Truth error | Prior gap |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        if row["condition"] not in {"biased-hard", "biased-soft"}:
            continue
        for key in BIASED_IDS:
            point = row["references"]["points"][key]
            xyz = lambda values: "[" + ", ".join(f"{value:.4f}" for value in values) + "]"
            lines.append(
                f"| {row['condition']} | {key} | {xyz(point['truth'])} | {xyz(point['prior'])} | "
                f"{xyz(point['reconstructed'])} | {point['truth_error']:.4f} | {point['prior_gap']:.4f} |"
            )
    lines += [
        "",
        "### Solver messages and flags",
        "",
    ]
    for row in rows:
        issues = "; ".join(row["flags"]["accuracy_violations"]) or "none"
        lines.append(f"- **{row['condition']}:** {row['message']} Flags: {issues}.")
    lines += [
        "",
        "## Interpretation",
        "",
        interpretation(rows),
        "",
        "The JSON results retain per-camera support, per-group fitted residuals, every Known 3D truth/prior gap, "
        "withheld camera pose/projection metrics, request fingerprints, solver messages, and all oracle flags.",
        "",
        "This is one seed, one local bias, one slack setting, exact picks and ideal calibrated pinhole cameras; it "
        "does not establish sensitivity over other geometries or noise. Known 3D slack softens every Known 3D "
        "point, including the two truthful references. The solver messages here report fit/constraints but do not "
        "warn of the intentional reference conflict. This numerical runner omits Blender preparation: "
        "Diagnose already warns when a Known 3D Empty differs from its stored anchor pick by more than 5 px "
        "(`scene.known_anchor_pick_warnings`). The experiment does not establish a gap in that existing check "
        "or a calibrated mismatch diagnostic.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-noisy", action="store_true", help="Add the single fixed 0.3 px draw (eight solves total)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        environment=environment(),
        solve_count=8 if args.include_noisy else 4,
        parameter_conditions=4,
        seed=SEED,
        bias_vector=BIAS_VECTOR.tolist(),
        soft_slack=SOFT_SLACK,
        gauge="anchor; no fitted alignment",
        total_solve_seconds=0.0,
    )
    rows = []
    for case in experiment_cases(include_noisy=args.include_noisy):
        write_case(case, args.out / f"{case['name']}.json")
        record = solve(case["request"])
        measured = measure(case, record)
        (args.out / f"{case['name']}-result.json").write_text(
            json.dumps(dict(result=record, metrics=measured), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(measured)
        protocol["total_solve_seconds"] += record["elapsed_s"]
        worst = max(
            (metrics["holdout_rmse_px"] for key, metrics in measured["withheld"].items() if key != "view_0"),
            default=float("nan"),
        )
        print(
            f"{case['name']}: fit={record['reported_rmse_px']:.3f}px "
            f"biased_truth={_number(measured['references']['biased_truth_rms'])} "
            f"holdout={worst:.3f}px oracle={'pass' if measured['oracle_passed'] else 'flag'}",
            flush=True,
        )
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "biased-reference-results.md").write_text(markdown_report(rows, protocol), encoding="utf-8")
    return 1 if any(not row["success"] for row in rows) or not required_controls_pass(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
