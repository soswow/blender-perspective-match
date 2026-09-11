"""Constraint cases in which Fit Only observations must not supply geometry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.constraints import add_mirror_support_stroke, constraint_case
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import solve


def one_sided_fit_only(family):
    """The second view can fit the cloud but cannot seed its mirror-only features."""
    case = constraint_case(family)
    case["name"] = "fit-only-" + family
    case["request"].update(location_match_ids=["view_0", "view_2"], readonly_match_ids=["view_1"])
    for kind in ("points", "lines"):
        case["expectation"]["excluded_"+kind] = case["expectation"]["required_"+kind]
        case["expectation"]["required_"+kind] = []
    return case


def weak_line_with_fit_only_stroke():
    """A new Fit Only stroke may fit that camera, but must not repair the line."""
    base = read_case(Path(__file__).parent / "cases/mirror-lines-weak.json")
    base["name"] = "weak-line-fit-only"
    base["request"].update(location_match_ids=["view_0", "view_1"], readonly_match_ids=["view_2"])
    # Fix the supporting pose so an allowed camera adjustment cannot explain 3D drift.
    base["request"]["fixed_similarities"] = generate("locked_bridge", base["seed"])["request"]["fixed_similarities"]
    extra = add_mirror_support_stroke(base)
    extra["expectation"] = deepcopy(base["expectation"])
    return base, extra


def parallel_line_with_fit_only_stroke(stroke_offset_px=0.3):
    """A fourth camera's imperfect stroke must not move an axis-constrained line."""
    base = generate("mixed_lines")
    base["name"] = "parallel-line-fit-only"
    request, truth = base["request"], base["truth"]
    for camera, actual in zip(request["cameras"][1:], truth["cameras"][1:]):
        rotation = np.asarray(actual["rotation"]).T @ np.asarray(camera["rotation"])
        request["fixed_similarities"][camera["id"]] = dict(scale=1.0, rotation=rotation.tolist(),
            translation=(np.asarray(actual["center"])-rotation@camera["center"]).tolist())
    for camera_list in (request["cameras"], truth["cameras"]):
        camera = deepcopy(camera_list[2])
        camera["id"] = "fit_view"
        camera_list.append(camera)
    for observation in list(request["observations"]):
        if observation["match_id"] == "view_2":
            request["observations"].append(dict(observation, match_id="fit_view"))
    for point in truth["checks"]:
        if "view_2" in point["views"]:
            point["views"].append("fit_view")
    request.update(location_match_ids=["view_0", "view_1", "view_2"], readonly_match_ids=["fit_view"])
    base["expectation"]["cameras"].append("fit_view")
    base["expectation"].update(required_lines=["edge_1"], line_fraction=0.02, line_angle_deg=1.0)
    extra = deepcopy(base)
    extra["name"] += "-extra-stroke"
    extra["measurement_edit"] = dict(kind="translate_stroke", offset_px=[stroke_offset_px, 0.0])
    stroke = next(o for o in request["line_observations"] if o["match_id"] == "view_2" and o["landmark_id"] == "edge_1")
    extra["request"]["line_observations"].append(dict(stroke, match_id="fit_view",
        u1=stroke["u1"]+stroke_offset_px, u2=stroke["u2"]+stroke_offset_px))
    return base, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stroke-offset-px", type=float, default=0.3, help="Explicit bias for the parallel-line Fit Only stroke")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [one_sided_fit_only(family) for family in ("mirror_points", "mirror_lines")]
    pairs = [weak_line_with_fit_only_stroke(), parallel_line_with_fit_only_stroke(args.stroke_offset_px)]
    cases += [case for pair in pairs for case in pair]
    baseline_names = {extra["name"]: base["name"] for base, extra in pairs}
    results = {}
    reports = []
    for case in cases:
        write_case(case, args.out / (case["name"]+".json"))
        result = solve(case["request"])
        assessment = evaluate(case, result)
        baseline = results.get(baseline_names.get(case["name"]))
        if baseline is not None:
            for key in case["expectation"]["required_lines"]:
                before, after = baseline["line_segments"].get(key), result["line_segments"].get(key)
                if before is None or after is None or not np.allclose(before, after, atol=1e-7, rtol=0):
                    assessment["violations"].append(f"{key}: a Fit Only stroke changed reconstructed geometry")
            assessment["passed"] = not assessment["violations"]
        results[case["name"]] = result
        (args.out / (case["name"]+"-result.json")).write_text(
            json.dumps(dict(result=result, assessment=assessment), indent=2, allow_nan=False)+"\n")
        reports.append((case, result, assessment))
        print(f"{'PASS' if assessment['passed'] else 'FAIL'} {case['name']}: {assessment['violations']}", flush=True)
    write_report(reports, args.out / "report.html")
    return 0 if all(assessment["passed"] for _, _, assessment in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
