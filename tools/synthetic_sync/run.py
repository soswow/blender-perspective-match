"""Generate/replay Sync cases. Run with --help; no Blender or OpenCV required."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.synthetic_sync.scenarios import FAMILIES, generate, read_case, write_case
from tools.synthetic_sync.solver import solve
from tools.synthetic_sync.evaluation import evaluate


def write_report(results: list, path: Path, reference_images=None) -> None:
    """Portable HTML with independent withheld-point overlays; no external resources."""
    sections = []
    for case, record, assessment in results:
        title = f"{case['name']} · {'PASS' if assessment['passed'] else 'FAIL'}"
        bits = [f"<h2>{escape(title)}</h2><p>Fitted-pick RMSE: {record['reported_rmse_px']:.3f}px. "
                f"Runtime: {record['elapsed_s']:.2f}s. Gauge: {escape(assessment['gauge'])}.</p>"]
        bits.append(f"<p>{escape(record['message'])}</p>")
        if record.get("exception"):
            bits.append(f"<pre>{escape(record['exception'])}</pre>")
        bits.append("<ul>" + "".join(f"<li>{escape(item)}</li>" for item in assessment["violations"]) + "</ul>")
        for key, metrics in assessment["cameras"].items():
            camera = next(c for c in case["truth"]["cameras"] if c["id"] == key)
            bits.append(f"<h3>{escape(key)} · withheld object RMS {metrics['holdout_rmse_px']:.3f}px · "
                        f"max {metrics['holdout_max_px']:.3f}px · {metrics['check_count']} checks</h3>")
            bits.append(f"<svg viewBox='0 0 {camera['width']} {camera['height']}'>")
            plate = (reference_images or {}).get(key)
            if plate and plate.is_file():
                data = base64.b64encode(plate.read_bytes()).decode("ascii")
                bits.append(f"<image width='{camera['width']}' height='{camera['height']}' href='data:image/png;base64,{data}'/>")
            for index, (x, y) in enumerate(metrics["expected_uv"]):
                if metrics["actual_uv"] is not None:
                    u, v = metrics["actual_uv"][index]
                    bits.append(f"<path d='M{x:.3f},{y:.3f} L{u:.3f},{v:.3f}' stroke='#ff7171'/><circle cx='{u:.3f}' cy='{v:.3f}' r='3' fill='#ff7171'/>")
                bits.append(f"<circle cx='{x:.3f}' cy='{y:.3f}' r='5' fill='none' stroke='#71dabc'/>")
            bits.append("</svg>")
        sections.append("\n".join(bits))
    path.write_text("<!doctype html><meta charset='utf-8'><title>Synthetic Sync checks</title>"
        "<style>body{font:16px system-ui;max-width:960px;margin:40px auto;background:#18212c;color:#eef3f7}"
        "svg{width:100%;max-width:600px;background:#0e1620;border:1px solid #546574}h2{margin-top:50px}</style>"
        "<h1>Synthetic Sync: withheld object geometry</h1><p>Green rings: true projection. Red: recovered camera. "
        "These points were never supplied to the solver. A small fitted-pick error alone is insufficient.</p>"
        + "\n".join(sections), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="ground")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="Consecutive seeds per family")
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument("--case", type=Path, help="Replay the exact saved JSON, ignoring generation options")
    parser.add_argument("--out", type=Path, required=True, help="Directory for cases, metrics and HTML")
    parser.add_argument("--permute", action="store_true", help="Also reverse input order without changing evidence")
    parser.add_argument("--warm", action="store_true", help="Replay each request twice with pair caching enabled")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    families = FAMILIES if args.family == "all" else [args.family]
    cases = [read_case(args.case)] if args.case else [generate(f, seed, args.noise_px)
        for f in families for seed in range(args.seed, args.seed + args.count)]
    results = []
    for case in cases:
        variants = [case]
        if args.permute:
            other = deepcopy(case)
            other["name"] += "-reversed"
            for key in ("cameras", "points", "observations", "lines", "line_observations"):
                other["request"][key].reverse()
            variants.append(other)
        for variant in variants:
            write_case(variant, args.out / (variant["name"] + ".json"))
            previous = None
            for attempt in range(2 if args.warm else 1):
                record = solve(variant["request"], use_cache=args.warm)
                assessment = evaluate(variant, record)
                if previous is not None:
                    if set(previous["cameras"]) != set(assessment["cameras"]):
                        assessment["violations"].append("Cache replay changed camera coverage")
                    for key in set(previous["cameras"]) & set(assessment["cameras"]):
                        before, after = previous["cameras"][key]["actual_uv"], assessment["cameras"][key]["actual_uv"]
                        if before is not None and after is not None:
                            delta = float(np.max(np.linalg.norm(np.asarray(before)-after, axis=1)))
                            assessment["cameras"][key]["cache_replay_delta_px"] = delta
                            if delta > 0.01:
                                assessment["violations"].append(f"{key}: cache replay changed withheld projection by {delta:.4g}px")
                    assessment["passed"] = not assessment["violations"]
                previous = assessment
                suffix = "-warm" if attempt else ""
                (args.out / (variant["name"] + suffix + "-result.json")).write_text(
                    json.dumps(dict(result=record, assessment=assessment), indent=2, allow_nan=False) + "\n")
                results.append((variant, record, assessment))
                worst = max((m["holdout_rmse_px"] for m in assessment["cameras"].values()), default=0.0)
                print(f"{'PASS' if assessment['passed'] else 'FAIL'} {variant['name']}{suffix}: "
                      f"fit={record['reported_rmse_px']:.3f}px holdout_worst_camera={worst:.3f}px {record['elapsed_s']:.2f}s", flush=True)
                for issue in assessment["violations"]:
                    print("  " + issue, flush=True)
    write_report(results, args.out / "report.html")
    return 0 if all(item[2]["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
