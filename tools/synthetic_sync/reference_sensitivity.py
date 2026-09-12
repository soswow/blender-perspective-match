"""Bounded, truth-free Known 3D prior-release diagnostic on frozen controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.biased_references import CONDITIONS, biased_reference_case
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import read_case, write_case
from tools.synthetic_sync.solver import environment, fingerprint, solve


EXACT_CASES = Path(__file__).resolve().parent / "cases"


def released_request(request: dict) -> dict:
    """Remove point priors only; retain every observation and other request field."""
    released = deepcopy(request)
    for point in released["points"]:
        point["known"] = None
    return released


def _rms(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _fitted_errors(request: dict, record: dict) -> dict:
    points = {point["id"]: point for point in request["points"]}
    errors: dict[str, list[float]] = defaultdict(list)
    support: dict[str, int] = defaultdict(int)
    for observation in request["observations"]:
        point_id, camera_id = observation["landmark_id"], observation["match_id"]
        position = record.get("landmarks", {}).get(point_id)
        camera = record.get("cameras", {}).get(camera_id)
        if position is None or camera is None:
            continue
        uv, depth = project([position], camera)
        if depth[0] <= 0 or not np.isfinite(uv).all():
            continue
        error = float(np.linalg.norm(uv[0] - [observation["u"], observation["v"]]))
        group = ("ground" if points[point_id]["ground"] else
                 "known" if points[point_id]["known"] is not None else "other")
        for key in ("all", group,
                    "point:" + point_id, "camera:" + camera_id):
            errors[key].append(error)
            support[key] += 1
    return {key: {"rms_px": _rms(errors[key]), "picks": support[key]}
            for key in sorted(errors)}


def _supported_observations(request: dict, record: dict) -> set[int]:
    supported = set()
    for index, observation in enumerate(request["observations"]):
        point = record.get("landmarks", {}).get(observation["landmark_id"])
        camera = record.get("cameras", {}).get(observation["match_id"])
        if point is None or camera is None:
            continue
        uv, depth = project([point], camera)
        if depth[0] > 0 and np.isfinite(uv).all():
            supported.add(index)
    return supported


def stored_anchor_prior_pick_errors(request: dict) -> dict:
    """Reproduce the existing 5 px anchor-pick guard from synthetic stored inputs."""
    anchor_id = request["anchor_id"]
    camera = next(camera for camera in request["cameras"] if camera["id"] == anchor_id)
    priors = {point["id"]: point["known"] for point in request["points"] if point["known"] is not None}
    errors = {}
    for observation in request["observations"]:
        key = observation["landmark_id"]
        if observation["match_id"] != anchor_id or key not in priors:
            continue
        uv, depth = project([priors[key]], camera)
        errors[key] = (None if depth[0] <= 0 or not np.isfinite(uv).all() else
                       float(np.linalg.norm(uv[0] - [observation["u"], observation["v"]])))
    return dict(sorted(errors.items()))


def measure_sensitivity(request: dict, baseline: dict, released: dict) -> dict:
    """Compare two results using only numerical evidence and solved geometry."""
    known = {point["id"]: np.asarray(point["known"], dtype=float)
             for point in request["points"] if point["known"] is not None}
    original_fit = _fitted_errors(request, baseline)
    released_fit = _fitted_errors(request, released)
    original_picks = _supported_observations(request, baseline)
    released_picks = _supported_observations(request, released)
    fit = {key: {
        "baseline_rms_px": original_fit.get(key, {}).get("rms_px"),
        "released_rms_px": released_fit.get(key, {}).get("rms_px"),
        "baseline_picks": original_fit.get(key, {}).get("picks", 0),
        "released_picks": released_fit.get(key, {}).get("picks", 0),
    } for key in sorted(set(original_fit) | set(released_fit))}
    references = {}
    for key, prior in sorted(known.items()):
        first = baseline.get("landmarks", {}).get(key)
        second = released.get("landmarks", {}).get(key)
        references[key] = {
            "baseline_prior_gap": None if first is None else float(np.linalg.norm(np.asarray(first) - prior)),
            "released_prior_gap": None if second is None else float(np.linalg.norm(np.asarray(second) - prior)),
            "solution_displacement": None if first is None or second is None else
                float(np.linalg.norm(np.asarray(second) - np.asarray(first))),
        }
    cameras = {}
    samples = baseline.get("landmarks", {})
    for key in sorted(set(baseline.get("cameras", {})) & set(released.get("cameras", {}))):
        before, after = baseline["cameras"][key], released["cameras"][key]
        shifts = []
        for position in samples.values():
            uv0, depth0 = project([position], before)
            uv1, depth1 = project([position], after)
            if min(depth0[0], depth1[0]) > 0 and np.isfinite(uv0).all() and np.isfinite(uv1).all():
                shifts.append(float(np.linalg.norm(uv1[0] - uv0[0])))
        cameras[key] = {
            "baseline_landmark_samples": len(shifts),
            "projection_shift_rms_px": _rms(shifts),
            "projection_shift_max_px": max(shifts, default=None),
            "center_displacement": float(np.linalg.norm(np.asarray(after["center"]) - before["center"])),
        }
    support_loss = {
        "cameras": sorted(set(baseline.get("cameras", {})) - set(released.get("cameras", {}))),
        "landmarks": sorted(set(baseline.get("landmarks", {})) - set(released.get("landmarks", {}))),
        "fitted_picks": len(original_picks - released_picks),
        "gained_picks": len(released_picks - original_picks),
        "missing_references": {
            "baseline": sorted(set(known) - set(baseline.get("landmarks", {}))),
            "released": sorted(set(known) - set(released.get("landmarks", {}))),
        },
    }
    reference_views = {}
    for key in sorted(known):
        views = []
        for supported in (original_picks, released_picks):
            views.append(sorted({request["observations"][index]["match_id"]
                                 for index in supported
                                 if request["observations"][index]["landmark_id"] == key}))
        reference_views[key] = {"baseline": views[0], "released": views[1]}
    ground_ids = {point["id"] for point in request["points"] if point["ground"]}
    ground_support = {
        "baseline": sorted(ground_ids & set(baseline.get("landmarks", {}))),
        "released": sorted(ground_ids & set(released.get("landmarks", {}))),
    }
    anchor_id = request["anchor_id"]
    anchor_present = (anchor_id in baseline.get("cameras", {}) and
                      anchor_id in released.get("cameras", {}))
    comparable = (bool(baseline.get("success")) and bool(released.get("success"))
                  and not support_loss["cameras"] and not support_loss["landmarks"]
                  and support_loss["fitted_picks"] == 0 and support_loss["gained_picks"] == 0
                  and bool(known) and anchor_present
                  and not any(support_loss["missing_references"].values())
                  and min(map(len, ground_support.values())) >= 3
                  and all(min(map(len, views.values())) >= 2 for views in reference_views.values()))
    return {
        "request_sha256": fingerprint(request),
        "known_reference_count": len(known),
        "stored_anchor_prior_pick_px": stored_anchor_prior_pick_errors(request),
        "baseline_success": bool(baseline.get("success")),
        "released_success": bool(released.get("success")),
        "support_loss": support_loss,
        "reference_views": reference_views,
        "hard_ground_frame_support": ground_support,
        "anchor_present": anchor_present,
        "comparable": comparable,
        "references": references,
        "fit": fit,
        "cameras": cameras,
    }


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _report(rows: list[dict], protocol: dict) -> str:
    accurate_warning_counts = [sum(value is None or value > 5.0
                                   for value in row["diagnostic"]["stored_anchor_prior_pick_px"].values())
                               for row in rows if row["condition"].startswith("unbiased-")]
    biased_warning_counts = [sum(value is None or value > 5.0
                                 for value in row["diagnostic"]["stored_anchor_prior_pick_px"].values())
                             for row in rows if row["condition"].startswith("biased-")]
    comparable_count = sum(row["diagnostic"]["comparable"] for row in rows)
    release_oracle_passes = sum(row["released_oracle"]["passed"] for row in rows)
    guard_separates_cases = (len(accurate_warning_counts) == len(biased_warning_counts) == 4
                             and all(count == 0 for count in accurate_warning_counts)
                             and all(count == 2 for count in biased_warning_counts))
    incremental_conclusion = (
        "These cases do not demonstrate additional detection beyond the existing guard. "
        if guard_separates_cases else
        "The existing guard does not separate all these controls; inspect the per-case evidence. "
    )
    lines = [
        "# Known 3D prior-release sensitivity pilot", "",
        "## Protocol", "",
        "Eight predeclared cases: four frozen exact conditions and the same four with the generator's fixed "
        "0.3 px noise draw. Each original solve is paired with one solve after removing all Known 3D point "
        "priors. The points retain their image picks; cameras, calibration, hard ground, roles, and every "
        "other request field are unchanged. The diagnostic sees only the request and two result records. "
        "No truth or condition metadata enters its signal, and no threshold produces a warning/classification.", "",
        f"Revision `{protocol['environment']['revision']}`; {protocol['solve_count']} numerical solves; "
        f"{protocol['total_solve_seconds']:.2f} s solver time. No alignment to truth was fitted. "
        "The requested anchor plus hard On Ground references define the comparison frame.", "",
        "Replay from the repository root: `python3 tools/synthetic_sync/reference_sensitivity.py "
        "--out /tmp/pm-reference-sensitivity-replay` (choose an absent output directory). "
        "Each case, request, result pair, and exact per-case diagnostic is retained there.", "",
        "## Results", "",
        "| Picks | Condition | Stored anchor >5 px | Fit original → release px | Max reference fit original → release px | "
        "Max prior gap original → release | Max camera projection shift px | Lost cameras/points/picks | "
        "Comparable | Original → release oracle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        signal = row["diagnostic"]
        gap0 = max((item["baseline_prior_gap"] for item in signal["references"].values()
                    if item["baseline_prior_gap"] is not None), default=None)
        gap1 = max((item["released_prior_gap"] for item in signal["references"].values()
                    if item["released_prior_gap"] is not None), default=None)
        camera_shift = max((item["projection_shift_max_px"] for item in signal["cameras"].values()
                            if item["projection_shift_max_px"] is not None), default=None)
        ref_fit0 = max((signal["fit"].get("point:" + key, {}).get("baseline_rms_px")
                        for key in signal["references"]
                        if signal["fit"].get("point:" + key, {}).get("baseline_rms_px") is not None), default=None)
        ref_fit1 = max((signal["fit"].get("point:" + key, {}).get("released_rms_px")
                        for key in signal["references"]
                        if signal["fit"].get("point:" + key, {}).get("released_rms_px") is not None), default=None)
        loss = signal["support_loss"]
        anchor_warning_count = sum(value is None or value > 5.0
                                   for value in signal["stored_anchor_prior_pick_px"].values())
        lines.append(
            f"| {row['picks']} | {row['condition']} | "
            f"{anchor_warning_count} | "
            f"{_number(signal['fit'].get('all', {}).get('baseline_rms_px'))} → "
            f"{_number(signal['fit'].get('all', {}).get('released_rms_px'))} | "
            f"{_number(ref_fit0)} → {_number(ref_fit1)} | "
            f"{_number(gap0)} → {_number(gap1)} | {_number(camera_shift)} | "
            f"{len(loss['cameras'])}/{len(loss['landmarks'])}/{loss['fitted_picks']} | "
            f"{'yes' if signal['comparable'] else 'no'} | "
            f"{'pass' if row['baseline_oracle']['passed'] else 'flag'} → "
            f"{'pass' if row['released_oracle']['passed'] else 'flag'} |"
        )
    lines += ["", "### Per-reference prior gaps after release (scene units)", "",
              "| Picks | Condition | Reference gaps (sorted ID order) |", "| --- | --- | --- |"]
    for row in rows:
        values = row["diagnostic"]["references"]
        cells = ", ".join(f"{key}: {_number(value['released_prior_gap'], 4)}" for key, value in values.items())
        lines.append(f"| {row['picks']} | {row['condition']} | {cells} |")
    lines += ["", "### Per-camera fit and projection movement", "",
              "| Picks | Condition | Camera | Fitted picks original → release | Fit RMS original → release px | "
              "Projection shift RMS / max px |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        signal = row["diagnostic"]
        for key, camera in signal["cameras"].items():
            fit = signal["fit"].get("camera:" + key, {})
            lines.append(
                f"| {row['picks']} | {row['condition']} | {key} | "
                f"{fit.get('baseline_picks', 0)} → {fit.get('released_picks', 0)} | "
                f"{_number(fit.get('baseline_rms_px'))} → {_number(fit.get('released_rms_px'))} | "
                f"{_number(camera['projection_shift_rms_px'])} / "
                f"{_number(camera['projection_shift_max_px'])} |"
            )
    lines += ["", "## Interpretation", "",
              "Reference gaps and camera movement show sensitivity to removing the priors. "
              "They do not by themselves identify a faulty reference: ambiguity, weak support, or shared "
              "calibration error can cause the same movement. The existing product warning compares each "
              "Known 3D prior with its stored anchor pick and flags deltas above 5 px; its numerical counterpart "
              "is shown above. A release trial can add multiview fit and movement evidence, but these cases "
              "already have anchor picks and therefore test incremental value against that cheaper guard. "
              "Compare accurate exact and noisy controls before interpreting a biased case. "
              "The per-case JSON retains every reference displacement, "
              "per-reference/per-camera fitted residual, projection movement on baseline reconstructed "
              "landmarks, support count/loss, and independent withheld-truth assessment.", "",
              "This is one constructed scene, one local bias, one fixed noise draw, and ideal pinhole calibration. "
              "The withheld oracle is used only after the signal is computed; its flags assess whether the "
              "observed sensitivity was useful, not whether a particular prior is intrinsically wrong. "
              "The stored-anchor calculation follows the numerical comparison in the existing warning; "
              "this pilot does not exercise Blender's UI or scene collection.", "",
              f"Across the four accurate controls, the existing 5 px guard warning counts are "
              f"{accurate_warning_counts}; across the four biased controls they are {biased_warning_counts}. "
              f"Release trials are comparable in {comparable_count}/{len(rows)} cases and pass the independent "
              f"oracle in {release_oracle_passes}/{len(rows)}. The measured prior gaps and camera movement "
              "quantify model dependence. " + incremental_conclusion +
              "This pilot does not justify a new product warning or automatic reference selection.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {"environment": environment(), "solve_count": 0, "total_solve_seconds": 0.0,
                "controls": [name for name, _biased, _slack in CONDITIONS], "noise_px": [0.0, 0.3]}
    rows = []
    for noise_px, picks in ((0.0, "exact"), (0.3, "noise-0p3")):
        for condition, _biased, _slack in CONDITIONS:
            case = (read_case(EXACT_CASES / f"biased-reference-exact-{condition}.json")
                    if noise_px == 0 else biased_reference_case(condition, noise_px=noise_px))
            request = case["request"]
            without_priors = released_request(request)
            case_path = args.out / f"{picks}-{condition}-case.json"
            write_case(case, case_path)
            baseline = solve(request)
            released = solve(without_priors)
            signal = measure_sensitivity(request, baseline, released)
            row = {"picks": picks, "condition": condition, "diagnostic": signal,
                   "baseline_oracle": evaluate(case, baseline), "released_oracle": evaluate(case, released)}
            record = {"case_file": case_path.name, "released_request": without_priors,
                      "baseline_result": baseline, "released_result": released, **row}
            (args.out / f"{picks}-{condition}-result.json").write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            rows.append(row)
            protocol["solve_count"] += 2
            protocol["total_solve_seconds"] += baseline["elapsed_s"] + released["elapsed_s"]
            print(f"{picks} {condition}: baseline={baseline['success']} release={released['success']} "
                  f"loss={signal['support_loss']} fit={_number(signal['fit'].get('all', {}).get('baseline_rms_px'))}"
                  f"→{_number(signal['fit'].get('all', {}).get('released_rms_px'))}", flush=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (args.out / "reference-sensitivity-results.md").write_text(_report(rows, protocol), encoding="utf-8")
    unbiased_rows = [row for row in rows if row["condition"].startswith("unbiased-")]
    return 0 if (all(row["diagnostic"]["baseline_success"] and row["diagnostic"]["released_success"]
                     for row in rows)
                 and all(row["baseline_oracle"]["passed"] and row["released_oracle"]["passed"]
                         for row in unbiased_rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
