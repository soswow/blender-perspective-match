"""Paired plane, mirror and independently fixed parallel-direction controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import stroke_plane
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import read_case, write_case
from tools.synthetic_sync.solver import environment, solve


CASE_PATH = Path(__file__).parent / "cases/plane-mirror-parallel.json"


def interaction_cases():
    """Eight frozen controls; each pair removes only the parallel links."""
    base = read_case(CASE_PATH)
    cases = []
    for plane_mode, locked in (("hard", True), ("soft", True), ("none", True), ("hard", False)):
        case = deepcopy(base)
        case["name"] = f"interaction-{plane_mode}-{'locked' if locked else 'unlocked'}"
        if plane_mode == "soft":
            case["request"]["plane_slack"] = .02
        if plane_mode == "none":
            case["request"]["plane_groups"] = []
        if plane_mode != "hard":
            case["expectation"].pop("plane_max_distance")
            case["expectation"]["outcome"] = "warn"
            # The independent fixed-direction reference still has weak depth.
            # This exception never waives the separate parallel-direction check.
            case["expectation"]["weak_lines"] = ["side_edge_left", "side_edge_right"]
        if not locked:
            case["request"]["fixed_similarities"] = {}
        case["interaction"]["plane_mode"] = plane_mode
        cases.append(case)
        removed = deepcopy(case)
        removed["name"] += "-without-parallel"
        removed["request"]["parallel_pairs"] = []
        cases.append(removed)
    return cases


def _direction(ends):
    values = np.asarray(ends, dtype=float)
    delta = values[1] - values[0]
    length = np.linalg.norm(delta)
    if not np.isfinite(values).all() or length < 1e-12:
        raise ValueError("A constraint requires a finite nonzero line")
    return delta / length


def quantize_request_float32(case):
    """Round stored numerical inputs without changing independent truth or limits."""
    def quantize(value):
        if isinstance(value, float):
            return float(np.float32(value))
        if isinstance(value, list):
            return [quantize(item) for item in value]
        if isinstance(value, dict):
            return {key: quantize(item) for key, item in value.items()}
        return value

    result = deepcopy(case)
    result["name"] += "-float32"
    result["request"] = quantize(result["request"])
    result["interaction"]["input_precision"] = "float32"
    return result


def interaction_measurements(case, record):
    """Measure declared relations independently of Sync and its weak-line labels."""
    segments = record["line_segments"]
    rows = dict(parallel={}, mirror={})
    for left, right in case["request"]["parallel_pairs"]:
        if left not in segments or right not in segments:
            rows["parallel"][f"{left}/{right}"] = None
            continue
        sine = float(np.linalg.norm(np.cross(_direction(segments[left]), _direction(segments[right]))))
        rows["parallel"][f"{left}/{right}"] = dict(sine=sine, angle_deg=float(np.degrees(np.arcsin(np.clip(sine, 0, 1)))))
    origin, normal = (np.asarray(value, dtype=float) for value in case["request"]["mirror_plane"])
    normal /= np.linalg.norm(normal)
    for left, right in case["request"]["mirror_pairs"]:
        if left not in segments or right not in segments:
            rows["mirror"][f"{left}/{right}"] = None
            continue
        ends = np.asarray(segments[left])
        reflected = ends - 2 * ((ends - origin) @ normal)[:, None] * normal
        right_ends = np.asarray(segments[right])
        direction = _direction(right_ends)
        distance = float(np.max(np.linalg.norm(np.cross(reflected - right_ends[0], direction), axis=1)))
        rows["mirror"][f"{left}/{right}"] = dict(max_distance=distance)
    return rows


def evaluate_interactions(case, record):
    """Keep camera/line truth checks and add hard parallel/reflection invariants."""
    assessment = evaluate(case, record)
    assessment["interaction_failures"] = []
    try:
        rows = interaction_measurements(case, record)
    except (ValueError, IndexError) as error:
        assessment["violations"].append(str(error))
        assessment["passed"] = False
        return assessment
    assessment["interactions"] = rows
    for relation, field, limit_name in (("parallel", "sine", "parallel_max_sine"), ("mirror", "max_distance", "mirror_max_distance")):
        limit = case["interaction"][limit_name]
        for label, metrics in rows[relation].items():
            if metrics is None or not np.isfinite(metrics[field]) or metrics[field] > limit:
                message = f"{label}: {relation} {field} {None if metrics is None else metrics[field]} exceeds {limit}"
                assessment["violations"].append(message)
                assessment["interaction_failures"].append(dict(relation=relation, pair=label, message=message))
    assessment["passed"] = not assessment["violations"]
    return assessment


def independent_direction_reference(case):
    """Fit the reflected noisy strokes at the independently supplied CAD direction."""
    reference = case["interaction"]["parallel_reference"]
    known = next(line["known"] for line in case["request"]["lines"] if line["id"] == reference)
    direction = _direction(known)
    cameras = {camera["id"]: camera for camera in case["truth"]["cameras"]}
    planes = []
    for pick in case["request"]["line_observations"]:
        if pick["landmark_id"] == reference:
            continue
        equation = stroke_plane(pick, cameras[pick["match_id"]])
        if pick["landmark_id"] == "side_edge_left":
            equation[0] *= -1  # Constructed reflection across X=0.
        planes.append(equation)
    planes = np.asarray(planes)
    if case["interaction"]["plane_mode"] == "hard":
        plane = case["truth"]["planes"][0]
        normal = np.asarray(plane["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        point = normal * (normal @ plane["origin"])
        basis = np.cross(normal, direction)[:, None]
    else:
        point = np.zeros(3)
        basis = np.linalg.svd(direction[None, :])[2][1:].T
    matrix = planes[:, :3] @ basis
    rhs = -(planes[:, :3] @ point + planes[:, 3])
    point += basis @ np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    reference_ends = np.asarray(case["truth"]["lines"]["side_edge_right"])
    offset = float(np.max(np.linalg.norm(np.cross(reference_ends - point, direction), axis=1)))
    return dict(point=point.tolist(), direction=direction.tolist(), offset=offset,
                matrix_condition=float(np.linalg.cond(matrix)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--float32", action="store_true", help="Round request floats while retaining independent truth and limits")
    parser.add_argument("--case", type=Path, help="Assess one existing exact case without running the solver")
    parser.add_argument("--result", type=Path, help="Existing numerical/Blender result, optionally wrapped in a result field")
    args = parser.parse_args()
    if bool(args.case) != bool(args.result):
        parser.error("--case and --result must be supplied together")
    if args.case and args.float32:
        parser.error("Existing-result assessment preserves the supplied case; omit --float32")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(environment(), indent=2) + "\n")
    reports = []
    cases = [read_case(args.case)] if args.case else interaction_cases()
    if args.float32:
        cases = [quantize_request_float32(case) for case in cases]
    for case in cases:
        write_case(case, args.out / (case["name"] + ".json"))
        if args.result:
            saved = json.loads(args.result.read_text())
            record = saved.get("result", saved)
        else:
            record = solve(case["request"])
        assessment = evaluate_interactions(case, record)
        assessment["independent_direction_reference"] = independent_direction_reference(case)
        (args.out / (case["name"] + "-result.json")).write_text(json.dumps(dict(result=record, assessment=assessment), indent=2, allow_nan=False) + "\n")
        reports.append((case, record, assessment))
        print(f"{'PASS' if assessment['passed'] else 'FAIL'} {case['name']}: {assessment['violations']}", flush=True)
    write_report(reports, args.out / "report.html")
    return 0 if all(assessment["passed"] for _, _, assessment in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
