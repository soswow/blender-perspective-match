"""Physically visible five-camera graphs with controlled reconstruction bridges."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.geometry import look_at, project, surface_points, visible
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, write_case
from tools.synthetic_sync.solver import environment, solve

FAMILIES = ("chain", "loop", "broken_link", "fit_only_bridge", "locked_bridge")


def graph_case(family="chain", seed=0, noise_px=0.0):
    """Use separate picks on each graph edge; only neighbor cameras share a point."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown graph family {family}")
    case = generate("ground", seed, noise_px)
    case.update(family="graph_"+family, name=f"graph-{family}-{seed}")
    request, truth, expected = (case[key] for key in ("request", "truth", "expectation"))
    rng = np.random.default_rng(seed+24000)
    truth["cameras"], request["cameras"] = [], []
    for index, angle in enumerate(np.radians([-140, -75, -10, 55, 120])):
        center = np.array([7*np.cos(angle), 7*np.sin(angle), 3.4+.15*(index % 3)]) + rng.uniform(-.04,.04,3)
        focal = [620., 800., 1150., 740., 950.][index] + rng.uniform(-10,10)
        actual = dict(id=f"view_{index}", width=960, height=720, fx=focal, fy=focal,
            cx=472.+3*index, cy=355.-2*index, center=center.tolist(), rotation=look_at(center,[0,0,.7]))
        truth["cameras"].append(actual)
        stored = deepcopy(actual)
        if index:
            stored.update(center=[-2.,-4.,2.2], rotation=look_at([-2.,-4.,2.2],[.3,.1,.4]))
        request["cameras"].append(stored)

    edges = [(i,i+1) for i in range(4)]
    if family == "loop":
        edges.append((4,0))
    request["points"], request["observations"], truth["points"] = [], [], {}
    edge_records = []
    for edge_index, (left, right) in enumerate(edges):
        cameras = [truth["cameras"][i] for i in (left,right)]
        # Each edge gets different physical points, so no accidental third-view
        # observation supplies a shortcut through the intended overlap graph.
        fractions = rng.uniform(.12,.88,(20,2))
        surface = surface_points(truth["mesh"], fractions)
        floor = [[2.65*np.cos(a),2.65*np.sin(a),0.] for a in np.linspace(0,2*np.pi,28,endpoint=False)+.017*edge_index]
        groups = []
        for ground, candidates, count in ((False,surface,8),(True,floor,4)):
            candidates = [p for p in candidates if all(visible(p,c,truth["mesh"]) for c in cameras)]
            if len(candidates) < count:
                raise ValueError(f"Edge {left}-{right} lacks enough visible {'floor' if ground else 'surface'} picks")
            # Spread exact evidence around its visible region; this is a fixture
            # design using truth, not a product recommendation based on picks.
            selected = [candidates.pop(int(rng.integers(len(candidates))))]
            while len(selected) < count:
                distance = np.linalg.norm(np.asarray(candidates)[:,None]-np.asarray(selected),axis=2).min(axis=1)
                selected.append(candidates.pop(int(np.argmax(distance))))
            groups.extend((p,ground) for p in selected)
        ids = []
        for index, (position,ground) in enumerate(groups):
            key = f"link_{edge_index}_point_{index}"
            request["points"].append(dict(id=key,ground=ground,known=None))
            truth["points"][key] = position
            ids.append(key)
            for camera in cameras:
                uv = project([position],camera)[0][0] + rng.normal(0,noise_px,2)
                request["observations"].append(dict(match_id=camera["id"],landmark_id=key,
                    u=float(uv[0]),v=float(uv[1]),weight=1.0))
        edge_records.append(dict(cameras=[c["id"] for c in cameras], landmarks=ids))

    if family == "broken_link":
        removed = set(edge_records[2]["landmarks"])
        request["observations"] = [o for o in request["observations"] if o["landmark_id"] not in removed]
        request["points"] = [p for p in request["points"] if p["id"] not in removed]
        del edge_records[2]
    request.update(location_match_ids=[c["id"] for c in request["cameras"]], readonly_match_ids=[])
    if family == "fit_only_bridge":
        request["location_match_ids"].remove("view_2")
        request["readonly_match_ids"] = ["view_2"]
    if family == "locked_bridge":
        actual, stored = truth["cameras"][2], request["cameras"][2]
        rotation = np.asarray(actual["rotation"]).T @ np.asarray(stored["rotation"])
        request["fixed_similarities"]["view_2"] = dict(scale=1., rotation=rotation.tolist(),
            translation=(np.asarray(actual["center"])-rotation@stored["center"]).tolist())
    all_ids = [c["id"] for c in truth["cameras"]]
    expected.update(cameras=all_ids[:3] if family in {"broken_link","fit_only_bridge"} else all_ids,
                    excluded_cameras=all_ids[3:] if family in {"broken_link","fit_only_bridge"} else [])
    # Label geometry from intended participation, before running the solver.
    # One permitted posed ground ray meets a known plane; free points need two.
    permitted = set(expected["cameras"]) & set(request["location_match_ids"])
    expected.update(required_points=[],excluded_points=[],point_fraction=.02)
    for point in request["points"]:
        views = {o["match_id"] for o in request["observations"]
                 if o["landmark_id"] == point["id"] and o["match_id"] in permitted}
        supported = len(views) >= (1 if point["ground"] else 2)
        expected["required_points" if supported else "excluded_points"].append(point["id"])
    truth["checks"] = []
    for index, position in enumerate(truth["mesh"]["vertices"]+surface_points(truth["mesh"],[(.13,.51),(.52,.87),(.86,.46)])):
        views = [c["id"] for c in truth["cameras"] if visible(position,c,truth["mesh"])]
        if views:
            truth["checks"].append(dict(id=f"check_{index:03d}",position=position,views=views))
    case["graph"] = dict(edges=edge_records, surface_picks_per_edge=8, ground_picks_per_edge=4)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES,"all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--noise-px", type=float, default=0.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    args.out.mkdir(parents=True,exist_ok=False)
    families = FAMILIES if args.family == "all" else (args.family,)
    (args.out / "protocol.json").write_text(json.dumps(dict(families=families,seed=args.seed,count=args.count,
        noise_px=args.noise_px,environment=environment(),
        contract="all supported cameras accurate on withheld object; broken/Fit Only bridge excludes downstream cameras"),indent=2)+"\n")
    reports = []
    for seed in range(args.seed,args.seed+args.count):
        for family in families:
            case = graph_case(family,seed,args.noise_px)
            write_case(case,args.out/(case["name"]+".json"))
            result = solve(case["request"])
            assessment = evaluate(case,result)
            (args.out/(case["name"]+"-result.json")).write_text(json.dumps(dict(result=result,assessment=assessment),indent=2,allow_nan=False)+"\n")
            reports.append((case,result,assessment))
            print(f"{'PASS' if assessment['passed'] else 'FAIL'} {case['name']}: {assessment['violations']}",flush=True)
    write_report(reports,args.out/"report.html")
    return 0 if all(a["passed"] for _,_,a in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
