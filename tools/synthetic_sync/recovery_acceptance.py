"""Audit line-bearing inputs at the recovered-camera 3D acceptance boundary."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.evaluation import evaluate
from tools.synthetic_sync.run import write_report
from tools.synthetic_sync.scenarios import generate, read_case, write_case
from tools.synthetic_sync.solver import environment, load_core, solve


def recovered_line_case(
    seed: int = 0,
    noise_px: float = 0.3,
    shift_px: float = 180.0,
) -> dict:
    """Make one still recover from hard ground after conflicting point picks."""
    case = generate("mixed_lines", seed, noise_px)
    case.update(name=f"recovered-lines-{seed}", family="recovery_line_acceptance")
    case["measurement_edit"] = {
        "kind": "shift_non_ground_points",
        "camera": "view_2",
        "offset_px": [shift_px, 0.0],
        "line_strokes": "unchanged",
    }
    ground_ids = {item["id"] for item in case["request"]["points"] if item["ground"]}
    for observation in case["request"]["observations"]:
        if (
            observation["match_id"] == "view_2"
            and observation["landmark_id"] not in ground_ids
        ):
            observation["u"] += shift_px
    case["expectation"].update(
        required_lines=["edge_1"],
        line_angle_deg=1.0,
        line_fraction=0.02,
    )
    return case


def without_lines(case: dict) -> dict:
    """Retain identical point evidence while removing every line input."""
    control = deepcopy(case)
    control["name"] += "-without-lines"
    control["request"]["lines"] = []
    control["request"]["line_observations"] = []
    control["request"]["parallel_pairs"] = []
    control["truth"]["lines"] = {}
    control["expectation"].pop("required_lines", None)
    control["expectation"].pop("line_angle_deg", None)
    control["expectation"].pop("line_fraction", None)
    return control


def _state_record(state) -> dict:
    point_ids = set(state.observations_by_landmark)
    return {
        "recovered": list(state.recovered),
        "landmark_ids": list(state.landmark_ids),
        "point_observation_ids": sorted(point_ids),
        "line_segment_ids": sorted(state.line_segments),
        "landmark_ids_without_point_observations": sorted(set(state.landmark_ids) - point_ids),
        "kept_joint_geometry": bool(getattr(state, "kept_joint_geometry", False)),
    }


def observe(case: dict, *, freeze: bool) -> tuple[dict, list[dict]]:
    """Run production or bypass only the recovered-camera 3D update."""
    load_core()
    module = importlib.import_module("match_perspective.core.sync.solve")
    original = module._thaw_recovered_location
    traces: list[dict] = []

    def stage(state):
        record = {"freeze_control": freeze, "before": _state_record(state)}
        traces.append(record)
        try:
            if not freeze:
                original(state)
        finally:
            record["after"] = _state_record(state)

    with patch.object(module, "_thaw_recovered_location", side_effect=stage):
        result = solve(case["request"])
    return result, traces


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-px", type=float, default=0.3)
    parser.add_argument("--shift-px", type=float, default=180.0)
    parser.add_argument(
        "--expect-crash",
        action="store_true",
        help="Historical mode: require the production run to raise at the recovered update",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = (
        read_case(args.case)
        if args.case
        else recovered_line_case(args.seed, args.noise_px, args.shift_px)
    )
    no_lines = without_lines(case)
    args.out.mkdir(parents=True, exist_ok=False)
    case_path = args.out / "case.json"
    no_lines_path = args.out / "case-without-lines.json"
    write_case(case, case_path)
    write_case(no_lines, no_lines_path)
    protocol = {
        "environment": environment(),
        "case_sha256": _checksum(case_path),
        "runs": ["production", "freeze-control", "no-lines-control"],
        "expect_crash": args.expect_crash,
        "freeze_boundary": "bypass only _thaw_recovered_location; retain final free-line rebuild",
        "interpretation": (
            "Default mode requires three successful solves and reports contradictory "
            "accuracy flags separately; --expect-crash preserves the historical boundary."
        ),
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    reports = []
    records = []
    for label, selected, freeze in (
        ("production", case, False),
        ("freeze-control", case, True),
        ("no-lines-control", no_lines, False),
    ):
        result, trace = observe(selected, freeze=freeze)
        assessment = evaluate(selected, result)
        record = {"result": result, "assessment": assessment, "trace": trace}
        (args.out / f"{label}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        display = deepcopy(selected)
        display["name"] += "-" + label
        reports.append((display, result, assessment))
        records.append(record)
        tail = result.get("exception", "").splitlines()[-1] if result.get("exception") else None
        recovered = trace[0]["before"]["recovered"] if trace else []
        print(
            label,
            "exception=" + repr(tail),
            "recovered=" + repr(recovered),
            flush=True,
        )
    write_report(reports, args.out / "report.html")

    production, frozen, no_line = records
    controls_pass_boundary = all(
        item["result"].get("success") and not item["result"].get("exception")
        for item in (frozen, no_line)
    )
    exception = production["result"].get("exception", "")
    production_matches = (
        "KeyError: 'edge_0'" in exception and "_triangulate_landmarks" in exception
        if args.expect_crash
        else production["result"].get("success") and not exception
    )
    return 0 if production_matches and controls_pass_boundary else 1


if __name__ == "__main__":
    raise SystemExit(main())
