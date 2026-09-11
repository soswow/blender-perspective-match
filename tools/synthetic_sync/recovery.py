"""Observe constraint changes during the production recovered-camera 3D update."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import environment, load_core, solve


def recovery_case(seed=0, noise_px=.3, shift_px=180.):
    """Bias one camera's off-ground picks while retaining its true floor evidence."""
    case = generate("ground",seed,noise_px)
    case.update(name=f"recovery-{seed}",family="recovery")
    case["measurement_edit"] = dict(kind="shift_non_ground",camera="view_2",offset_px=[shift_px,0.])
    ground_ids = {p["id"] for p in case["request"]["points"] if p["ground"]}
    for observation in case["request"]["observations"]:
        if observation["match_id"] == "view_2" and observation["landmark_id"] not in ground_ids:
            observation["u"] += shift_px
    return case


def constraint_record(state):
    """Measure actual constraint gaps, without treating fitted pixels as truth."""
    ground = {key:float(point[2]) for key,point in state.landmarks.items()
              if any(o.on_ground for o in state.observations_by_landmark_all.get(key,[]))}
    known = {key:float(np.linalg.norm(state.landmarks[key]-point))
             for key,point in state.known_world.items() if key in state.landmarks}
    mirror = {}
    if state.mirror_plane is not None:
        origin,normal = (np.asarray(value,dtype=float) for value in state.mirror_plane)
        normal = normal/np.linalg.norm(normal)
        for left,right in state.mirror_pairs or []:
            if left in state.landmarks and right in state.landmarks:
                point = state.landmarks[left]
                reflected = point - 2*np.dot(point-origin,normal)*normal
                mirror[left+"|"+right] = float(np.linalg.norm(state.landmarks[right]-reflected))
    buckets = {}
    for key,axis,bucket in getattr(state,"plane_groups",[]) or []:
        if key in state.landmarks:
            buckets.setdefault((axis,bucket),[]).append(state.landmarks[key])
    planes = {}
    for (axis,bucket),points in buckets.items():
        active = len(points) >= (4 if axis == "FREE" else 2)
        distance = None
        if active:
            centered = np.asarray(points)-np.mean(points,axis=0)
            if axis == "FREE":
                _u,singular,vt = np.linalg.svd(centered)
                active = bool(singular[1] > 1e-10)
                distance = float(np.max(np.abs(centered@vt[-1]))) if active else None
            else:
                distance = float(np.max(np.abs(centered[:,"XYZ".index(axis)])))
        planes[f"{axis}#{bucket}"] = dict(members=len(points),active=active,max_distance=distance)
    return dict(ground_z=ground,known_gap=known,mirror_gap=mirror,plane_spread=planes,
                landmarks={key:point.tolist() for key,point in state.landmarks.items()})


def observe_recovery(case, *, freeze=False, stage_control_recovered=None):
    """Observe recovery, or explicitly force/freeze that stage for a control."""
    load_core()
    module = importlib.import_module("match_perspective.core.sync.solve")
    original = module._thaw_recovered_location
    traces = []

    def stage(state):
        if stage_control_recovered is not None:
            if not set(stage_control_recovered) <= set(state.similarities):
                raise ValueError("Stage control requires already posed cameras")
            state.recovered = list(stage_control_recovered)
        eligible = sorted(set(state.recovered) & set(state.location_match_ids or []))
        record = dict(recovered=list(state.recovered),eligible=eligible,before=constraint_record(state),
                      ground_slack=state.ground_slack,mirror_slack=state.mirror_slack,freeze_control=freeze,
                      stage_control_recovered=stage_control_recovered)
        if not freeze:
            original(state)
        record["after"] = constraint_record(state)
        record["kept_joint_geometry"] = getattr(state,"kept_joint_geometry",False)
        traces.append(record)

    with patch.object(module,"_thaw_recovered_location",side_effect=stage):
        result = solve(case["request"])
    return result,traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--noise-px",type=float,default=.3)
    parser.add_argument("--shift-px",type=float,default=180.)
    parser.add_argument("--compare-freeze",action="store_true",help="Explicitly bypass the final 3D update in a second control solve")
    parser.add_argument("--stage-control-recovered",action="append",help="Stage-only control: mark an already posed camera recovered (repeatable)")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if not np.isfinite(args.shift_px):
        parser.error("shift must be finite")
    case = read_case(args.case) if args.case else recovery_case(args.seed,args.noise_px,args.shift_px)
    args.out.mkdir(parents=True,exist_ok=False)
    write_case(case,args.out/"case.json")
    (args.out/"protocol.json").write_text(json.dumps(dict(environment=environment(),compare_freeze=args.compare_freeze,
        stage_control_recovered=args.stage_control_recovered,
        measurement="record constraints immediately before and after recovered-camera 3D update",
        interpretation="biased picks create conflicting evidence; accuracy flags alone do not establish a defect"),indent=2)+"\n")
    reports = []
    failed = False
    for freeze in ([False,True] if args.compare_freeze else [False]):
        result,trace = observe_recovery(case,freeze=freeze,stage_control_recovered=args.stage_control_recovered)
        assessment = evaluate(case,result)
        label = "freeze-control" if freeze else "production"
        (args.out/(label+".json")).write_text(json.dumps(dict(result=result,assessment=assessment,trace=trace),indent=2,allow_nan=False)+"\n")
        display_case = deepcopy(case)
        display_case["name"] += "-"+label
        reports.append((display_case,result,assessment))
        failed |= bool(result.get("exception"))
        print(label,result["message"],flush=True)
        for item in trace:
            print("eligible",item["eligible"],"ground |Z| before/after",
                max(map(abs,item["before"]["ground_z"].values()),default=0.),
                max(map(abs,item["after"]["ground_z"].values()),default=0.),flush=True)
        print("accuracy flags",assessment["violations"],flush=True)
    write_report(reports,args.out/"report.html")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
