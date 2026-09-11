"""Independent geometry controls for shared-plane points and lines."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import project, visible
from tools.synthetic_sync.roles import parallel_line_with_fit_only_stroke
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, write_case
from tools.synthetic_sync.solver import environment, solve


FAMILIES = ("axis_buckets", "free_tilted", "free_ground", "plane_line")


def plane_case(family, seed=0, noise_px=0.0):
    """Declare physically consistent planes without production fitting helpers."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown shared-plane family: {family}")
    case = generate("mixed_lines" if family == "plane_line" else "ground", seed, noise_px)
    case.update(name=f"plane-{family}-{seed}", family=family)
    request, truth, expected = (case[key] for key in ("request", "truth", "expectation"))
    expected.update(required_points=[], required_lines=[], excluded_points=[], excluded_lines=[],
                    point_fraction=0.02, line_fraction=0.02, line_angle_deg=1.0)
    truth["planes"] = []
    if family in {"free_tilted", "free_ground"}:
        # Six surface points on z = 0.7 + 0.2*y + 0.1*x; front and right faces
        # provide independent horizontal directions, with overlapping views.
        positions = [[x, -.9, .52+.1*x] for x in (-.8, 0., .8)]
        positions += [[1.2, y, .82+.2*y] for y in (-.6, 0., .6)]
        if family == "free_ground":
            # z = y + 2 meets the floor along y = -2 and the front face
            # along z = 1.1. All four members share an exact physical plane.
            positions = [[-.8,-.9,1.1], [.8,-.9,1.1]]
            request["plane_groups"] = [(key, "FREE", 1) for key in ("ground_0", "ground_2")]
            expected["required_points"].extend(("ground_0", "ground_2"))
            expected["ground_max_distance"] = 1e-6
        rng = np.random.default_rng(seed+19000)
        for i, position in enumerate(positions):
            key = f"tilted_{i}"
            request["points"].append(dict(id=key, ground=False, known=None))
            truth["points"][key] = position
            count = 0
            for camera in truth["cameras"]:
                if not visible(position, camera, truth["mesh"]):
                    continue
                uv = project([position], camera)[0][0]+rng.normal(0, noise_px, 2)
                request["observations"].append(dict(match_id=camera["id"], landmark_id=key,
                    u=float(uv[0]), v=float(uv[1]), weight=1.0))
                count += 1
            if count < 2:
                raise ValueError("Tilted-plane control needs two visible picks per member")
            request["plane_groups"].append((key, "FREE", 1))
            expected["required_points"].append(key)
        origin, normal = ([0,-2,0], [0,-1,1]) if family == "free_ground" else ([0,0,.7], [-.1,-.2,1.])
        truth["planes"].append(dict(axis="FREE", bucket=1, origin=origin, normal=normal))
    else:
        for point in request["points"]:
            position = truth["points"][point["id"]]
            if family == "axis_buckets" and abs(position[1]+.9) < 1e-8:
                request["plane_groups"].append((point["id"], "Y", 1))
                expected["required_points"].append(point["id"])
            if abs(position[0]-1.2) < 1e-8:
                request["plane_groups"].append((point["id"], "X", 1))
                expected["required_points"].append(point["id"])
        truth["planes"].append(dict(axis="X", bucket=1, origin=[1.2,0,0], normal=[1,0,0]))
        if family == "axis_buckets":
            truth["planes"].append(dict(axis="Y", bucket=1, origin=[0,-.9,0], normal=[0,1,0]))
        else:
            # Remove the independent axis-direction constraint: the plane and
            # strokes must agree on the same infinite line.
            request["parallel_pairs"] = []
            request["plane_groups"].append(("edge_1", "X", 1))
            expected["required_lines"] = ["edge_1"]
    return case


def plane_measurements(case, result):
    """Distance to known construction planes, not to planes fitted by Sync."""
    rows = {}
    for plane in case["truth"].get("planes", []):
        axis, bucket = plane["axis"], plane["bucket"]
        ids = [key for key, a, b in case["request"]["plane_groups"] if (a,b) == (axis,bucket)]
        coordinates = []
        for key in ids:
            if key in result["line_segments"]:
                coordinates.extend(result["line_segments"][key])
            elif key in result["landmarks"]:
                coordinates.append(result["landmarks"][key])
        normal = np.asarray(plane["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        distance = np.abs((np.asarray(coordinates)-plane["origin"])@normal) if coordinates else []
        rows[f"{axis}#{bucket}"] = dict(members=ids, reconstructed_samples=len(coordinates),
            true_plane_max_distance=float(max(distance)) if len(distance) else None)
    return rows


def fit_only_plane_stroke():
    """Paired fixed-camera control: excluded strokes must not reshape a plane line."""
    base, extra = parallel_line_with_fit_only_stroke()
    for case in (base, extra):
        case["name"] = "plane-"+case["name"]
        case["request"]["parallel_pairs"] = []
        case["request"]["plane_groups"] = [("edge_1", "X", 1)] + [
            (key, "X", 1) for key, point in case["truth"]["points"].items() if abs(point[0]-1.2) < 1e-8]
        case["truth"]["planes"] = [dict(axis="X", bucket=1, origin=[1.2,0,0], normal=[1,0,0])]
    return base, extra


def remove_planes(case):
    """Retain all evidence and truth; these controls are solvable without planes."""
    if case["family"] not in FAMILIES:
        raise ValueError("Plane removal requires a shared-plane control case")
    result = deepcopy(case)
    result["name"] += "-without-plane"
    result["request"]["plane_groups"] = []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-px", type=float, default=0.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"protocol.json").write_text(json.dumps(environment(), indent=2)+"\n")
    cases = []
    for family in FAMILIES:
        case = plane_case(family, args.seed, args.noise_px)
        control = remove_planes(case)
        # These are already solvable without the plane: removal is a safety
        # control, not a claim that planes contributed indispensable evidence.
        cases.extend((case, control))
    base, extra = fit_only_plane_stroke()
    cases.extend((base, extra))
    reports, results = [], {}
    for case in cases:
        write_case(case, args.out/(case["name"]+".json"))
        result = solve(case["request"])
        assessment = evaluate(case, result)
        assessment["planes"] = plane_measurements(case, result)
        if case is extra:
            before = results[base["name"]]["line_segments"].get("edge_1")
            after = result["line_segments"].get("edge_1")
            if before is None or after is None or not np.allclose(before, after, rtol=0, atol=1e-7):
                assessment["violations"].append("Fit Only stroke changed the plane-constrained line")
            assessment["passed"] = not assessment["violations"]
        results[case["name"]] = result
        (args.out/(case["name"]+"-result.json")).write_text(json.dumps(
            dict(result=result, assessment=assessment), indent=2, allow_nan=False)+"\n")
        reports.append((case,result,assessment))
        print(case["name"], "PASS" if assessment["passed"] else "FAIL", assessment["violations"], flush=True)
    write_report(reports, args.out/"report.html")
    return 0 if all(assessment["passed"] for _,_,assessment in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
