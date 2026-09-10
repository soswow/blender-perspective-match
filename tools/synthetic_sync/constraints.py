"""Cases where removing a constraint must remove recoverable information."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import intersect_stroke_planes, project, stroke_plane, visible
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, write_case
from tools.synthetic_sync.solver import solve


FAMILIES = ("known_lines", "mirror_points", "mirror_lines")


def constraint_case(family, seed=0, noise_px=0.0):
    """Use physical visibility and independent truth, with deliberately missing picks."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown constraint family: {family}")
    case = generate("ground", seed, noise_px)
    case.update(family=family, name=f"constraint-{family}-{seed}")
    request, truth, expected = (case[key] for key in ("request", "truth", "expectation"))
    expected.update(required_points=[], required_lines=[], excluded_points=[], excluded_lines=[],
                    point_fraction=0.02, line_fraction=0.02, line_angle_deg=1.0)
    cameras = {c["id"]: c for c in truth["cameras"]}
    rng = np.random.default_rng(seed + 8000)

    def point_pick(key, point, camera_id):
        camera = cameras[camera_id]
        if not visible(point, camera, truth["mesh"]):
            raise ValueError(f"{key}: attempted hidden point pick")
        uv = project([point], camera)[0][0] + rng.normal(0, noise_px, 2)
        request["observations"].append(dict(match_id=camera_id, landmark_id=key, u=float(uv[0]), v=float(uv[1]), weight=1.0))

    def line_pick(key, ends, camera_id, fraction=0.0):
        camera = cameras[camera_id]
        a, b = np.asarray(ends)
        stroke = [a+(0.08+fraction)*(b-a), b-(0.13+fraction)*(b-a)]
        if not all(visible(p, camera, truth["mesh"]) for p in stroke):
            return False
        uv = project(stroke, camera)[0] + rng.normal(0, noise_px, (2,2))
        request["line_observations"].append(dict(match_id=camera_id, landmark_id=key,
            u1=float(uv[0,0]), v1=float(uv[0,1]), u2=float(uv[1,0]), v2=float(uv[1,1]), weight=1.0))
        return True

    if family == "known_lines":
        # Only the second camera sees strokes. Without Known 3D, no other view
        # or metric evidence can determine where these single-view lines live.
        request["cameras"] = request["cameras"][:2]
        truth["cameras"] = truth["cameras"][:2]
        expected["cameras"] = ["view_0", "view_1"]
        request["points"], request["observations"], truth["points"] = [], [], {}
        mesh = truth["mesh"]
        edges = sorted({tuple(sorted((a,b))) for face in mesh["faces"] for a,b in zip(face,face[1:]+face[:1])})
        for index, (a,b) in enumerate(edges):
            key = f"cad_edge_{index:02d}"
            ends = [mesh["vertices"][a], mesh["vertices"][b]]
            if line_pick(key, ends, "view_1"):
                request["lines"].append(dict(id=key, known=ends))
                truth["lines"][key] = ends
        if len(request["lines"]) < 6:
            raise ValueError("Need six visible Known 3D edges in this test")
        # CAD line endpoints are evidence; evaluate withheld face interiors.
        vertices = np.asarray(mesh["vertices"])
        truth["checks"] = [p for p in truth["checks"] if np.min(np.linalg.norm(vertices-p["position"],axis=1)) > 1e-6]
        for point in truth["checks"]:
            point["views"] = [key for key in point["views"] if key in expected["cameras"]]
        truth["checks"] = [p for p in truth["checks"] if p["views"]]
    else:
        request["mirror_plane"] = [[0,0,0],[1,0,0]]
        if family == "mirror_points":
            for i, (y,z) in enumerate([(-.7,.3),(.1,.8),(.6,1.2)]):
                pair = []
                for side, x, camera_id in [("left",-1.2,"view_0"),("right",1.2,"view_1")]:
                    key = f"detail_{side}_{i}"
                    point = [x,y,z]
                    request["points"].append(dict(id=key, ground=False, known=None))
                    truth["points"][key] = point
                    point_pick(key, point, camera_id)
                    pair.append(key)
                request["mirror_pairs"].append(pair)
                expected["required_points"].extend(pair)
        else:
            pair = []
            for side, x, camera_id, fraction in [("left",-1.2,"view_0",0),("right",1.2,"view_1",.07)]:
                key = "side_edge_" + side
                ends = [[x,-.65,.2],[x,.6,1.2]]
                request["lines"].append(dict(id=key, known=None))
                truth["lines"][key] = ends
                if not line_pick(key, ends, camera_id, fraction):
                    raise ValueError("Mirrored line must be physically visible")
                pair.append(key)
            request["mirror_pairs"].append(pair)
            expected["required_lines"].extend(pair)
    return case


def remove_constraint(case):
    """Keep observations identical; update the expected loss of information."""
    result = deepcopy(case)
    result["name"] += "-without-constraint"
    request, expected = result["request"], result["expectation"]
    expected.pop("weak_lines",None)
    expected["outcome"] = "solve"
    if case["family"] == "known_lines":
        for line in request["lines"]:
            line["known"] = None
        expected["outcome"] = "reject"
    else:
        request["mirror_pairs"], request["mirror_plane"] = [], None
        expected["excluded_points"] = expected["required_points"]
        expected["excluded_lines"] = expected["required_lines"]
        expected["required_points"], expected["required_lines"] = [], []
    return result


def add_mirror_support_stroke(case):
    """Observe the existing right-side line in the already-posed third camera."""
    if case["family"] != "mirror_lines":
        raise ValueError("Requires the mirrored-line fixture")
    result = deepcopy(case)
    result["expectation"].pop("weak_lines",None)
    result["expectation"]["outcome"] = "solve"
    key = "side_edge_right"
    camera = next(c for c in case["truth"]["cameras"] if c["id"] == "view_2")
    if any(o["landmark_id"] == key and o["match_id"] == "view_2" for o in case["request"]["line_observations"]):
        raise ValueError("Support stroke already exists")
    a,b = np.asarray(case["truth"]["lines"][key])
    stroke = [a+.18*(b-a), b-.1*(b-a)]
    if not all(visible(p,camera,case["truth"]["mesh"]) for p in stroke):
        raise ValueError("Support stroke must be visible")
    uv = project(stroke,camera)[0] + np.random.default_rng(case["seed"]+9100).normal(0,case["noise_px"],(2,2))
    result["request"]["line_observations"].append(dict(match_id="view_2", landmark_id=key,
        u1=float(uv[0,0]),v1=float(uv[0,1]),u2=float(uv[1,0]),v2=float(uv[1,1]),weight=1.0))
    result["name"] += "-extra-view"
    return result


def mirror_line_reference(case, cameras):
    """Independent reconstruction of the initial two-stroke mirror fixture."""
    by_id = {c["id"]: c for c in cameras}
    planes = []
    for observation in case["request"]["line_observations"]:
        if observation["match_id"] not in by_id:
            continue
        plane = stroke_plane(observation, by_id[observation["match_id"]])
        if observation["landmark_id"] == "side_edge_right":
            plane[0] *= -1  # The fixture's independently known mirror is X=0.
        planes.append(plane)
    point, direction = intersect_stroke_planes(planes)
    ends = np.asarray(case["truth"]["lines"]["side_edge_left"])
    unit = ends[1]-ends[0]
    unit /= np.linalg.norm(unit)
    angle = float(np.degrees(np.arccos(np.clip(abs(direction@unit),0,1))))
    support = max(float(np.degrees(np.arccos(np.clip(abs(a[:3]@b[:3]),0,1))))
                  for i,a in enumerate(planes) for b in planes[i+1:])
    return dict(support_angle_deg=support, direction_error_deg=angle, point=point.tolist(),direction=direction.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for family in FAMILIES if args.family == "all" else (args.family,):
        case = constraint_case(family, args.seed, args.noise_px)
        variants = [case, remove_constraint(case)]
        if family == "mirror_lines":
            variants.append(add_mirror_support_stroke(case))
        for variant in variants:
            write_case(variant, args.out / (variant["name"]+".json"))
            result = solve(variant["request"])
            assessment = evaluate(variant, result)
            if family == "mirror_lines" and variant["request"]["mirror_pairs"] and result["success"]:
                assessment["independent_reference"] = dict(
                    true_cameras=mirror_line_reference(variant,variant["truth"]["cameras"]),
                    solved_cameras=mirror_line_reference(variant,list(result["cameras"].values())),
                )
            (args.out / (variant["name"]+"-result.json")).write_text(json.dumps(dict(result=result, assessment=assessment), indent=2, allow_nan=False)+"\n")
            reports.append((variant,result,assessment))
            print(f"{'PASS' if assessment['passed'] else 'FAIL'} {variant['name']}: {assessment['violations']}", flush=True)
    write_report(reports, args.out / "report.html")
    return 0 if all(a["passed"] for _,_,a in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
