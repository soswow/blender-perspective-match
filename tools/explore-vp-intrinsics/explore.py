#!/usr/bin/env python3
"""Explore VP-line error over FOV × principal-point (optional λ) space.

Throwaway diagnostic — not part of the Blender add-on. Loads one match from a
``.blend``, then BFS-samples intrinsics around the current seed, pruning
directions whose VP line RMS exceeds ``--max-error``. Opens an interactive
Plotly 3D scatter in the browser by default.

Example:

  python3 tools/explore-vp-intrinsics/explore.py scene.blend "PM_MyStill" \\
      --max-error 18
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_BLENDER = os.environ.get(
    "BLENDER_BIN",
    "/Applications/Blender 5.1.app/Contents/MacOS/blender",
)


def _load_core():
    """Import ``core.py`` without pulling in the bpy-dependent package init."""
    module_name = "pm_core_explore"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "core.py",
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses on newer Python need the module present in sys.modules
    # before exec_module runs.
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


core = _load_core()


@dataclass(frozen=True)
class SampleKey:
    """Discrete grid key in PP-offset / HFOV / λ space."""

    ppx_i: int
    ppy_i: int
    fov_i: int
    lam_i: int


@dataclass
class Sample:
    """One evaluated intrinsics trial."""

    pp_offset_x: float
    pp_offset_y: float
    hfov_degrees: float
    division_lambda: float
    line_rms: float
    angle_degrees: float
    expanded: bool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BFS-explore FOV / PP offset space for one Perspective Match, "
            "colored by VP line residual RMS"
        ),
    )
    parser.add_argument("blend", type=Path, help="Path to a .blend with PM matches")
    parser.add_argument(
        "match",
        help="Match name as shown in the Perspective Match dropdown",
    )
    parser.add_argument(
        "--blender",
        default=DEFAULT_BLENDER,
        help=f"Blender binary (default: {DEFAULT_BLENDER})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Reuse an existing extract JSON instead of opening the .blend",
    )
    parser.add_argument(
        "--keep-json",
        type=Path,
        default=None,
        help="Write the extracted match JSON here",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=25.0,
        help="Do not expand BFS neighbors when line RMS exceeds this (px)",
    )
    parser.add_argument(
        "--pp-radius",
        type=float,
        default=250.0,
        help="Max |PP offset − seed| in pixels along each axis",
    )
    parser.add_argument(
        "--fov-span",
        type=float,
        default=20.0,
        help="Max |HFOV − seed| in degrees",
    )
    parser.add_argument("--pp-step", type=float, default=25.0, help="PP grid step (px)")
    parser.add_argument("--fov-step", type=float, default=2.0, help="HFOV grid step (°)")
    parser.add_argument(
        "--lambda",
        dest="division_lambda",
        type=float,
        default=None,
        help="Fix division λ (default: value from the match)",
    )
    parser.add_argument(
        "--vary-lambda",
        action="store_true",
        help="Also BFS over λ (still plotted as PP×FOV; λ stored in CSV)",
    )
    parser.add_argument(
        "--lambda-span",
        type=float,
        default=0.15,
        help="Max |λ − seed| when --vary-lambda",
    )
    parser.add_argument(
        "--lambda-step",
        type=float,
        default=0.05,
        help="λ grid step when --vary-lambda",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Hard cap on evaluated samples",
    )
    parser.add_argument(
        "--metric",
        choices=("line_rms", "angle"),
        default="line_rms",
        help="Error used for pruning and plot color (default: line_rms)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional export path (.html interactive, or .png static)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV of all evaluated samples",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive browser plot (useful with --out/--csv)",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help=(
            "Also plot BFS frontier samples with error > --max-error "
            "(gray). Hidden by default — prune only stops expansion, "
            "those points are still evaluated."
        ),
    )
    parser.add_argument(
        "--slide-axis",
        choices=SLIDE_AXIS_IDS,
        default="lambda",
        help=(
            "Which parameter the slider starts on (default: lambda). "
            "In the plot, a dropdown can switch among λ / FOV / PP X / PP Y; "
            "λ takes the place of the axis you slide."
        ),
    )
    return parser.parse_args(argv)


def extract_match_json(
    blend: Path,
    match_name: str,
    blender: str,
    out_json: Path,
) -> dict[str, Any]:
    """Open the .blend in background Blender and dump one match to JSON."""
    extract_script = TOOL_DIR / "extract_match.py"
    if not Path(blender).is_file() and not Path(blender).exists():
        raise FileNotFoundError(
            f"Blender binary not found: {blender}. Set --blender or BLENDER_BIN."
        )
    command = [
        blender,
        "--factory-startup",
        "-b",
        "--python",
        str(extract_script),
        "--",
        "--blend",
        str(blend.resolve()),
        "--match",
        match_name,
        "--out",
        str(out_json.resolve()),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not out_json.is_file():
        raise RuntimeError(
            "Failed to extract match from .blend.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _bundles_from_payload(payload: dict[str, Any]) -> dict[str, list]:
    """Build axis line bundles, applying the same VP-mode filtering as the add-on."""
    lines = payload["lines"]
    bundles = {
        axis: [
            core.LineSegment(
                float(segment["x1"]),
                float(segment["y1"]),
                float(segment["x2"]),
                float(segment["y2"]),
            )
            for segment in lines.get(axis, [])
        ]
        for axis in ("x", "y", "z")
    }
    vp_mode = str(payload.get("vp_mode", "2"))
    if vp_mode == "1":
        bundles["x"] = []
    elif vp_mode == "2":
        bundles["y"] = []
    return bundles


def _error_of(sample: Sample, metric: str) -> float:
    return sample.line_rms if metric == "line_rms" else sample.angle_degrees


def evaluate_intrinsics(
    bundles: dict[str, list],
    *,
    width: int,
    height: int,
    hfov_degrees: float,
    pp_offset_x: float,
    pp_offset_y: float,
    division_lambda: float,
) -> Sample:
    """Locked-focal re-orient at the trial intrinsics; score VP residuals."""
    focal = core.focal_from_hfov(hfov_degrees, width)
    intrinsics = core.CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=0.5 * width + pp_offset_x,
        cy=0.5 * height + pp_offset_y,
        image_width=width,
        image_height=height,
    )
    try:
        calibration = core.refine_camera(
            bundles,
            intrinsics,
            lock_focal=True,
            estimate_principal_point=False,
            estimate_distortion=False,
            initial_division_lambda=division_lambda,
        )
        # Keep the trial λ even if refine would estimate it (distortion off).
        calibration.division_lambda = float(division_lambda)
        line_rms = float(core.vp_line_residual_rms(calibration, bundles))
        angle = float(core.vp_angular_residual_degrees(calibration, bundles))
    except Exception:
        line_rms = 1.0e6
        angle = 180.0
    if not math.isfinite(line_rms):
        line_rms = 1.0e6
    if not math.isfinite(angle):
        angle = 180.0
    return Sample(
        pp_offset_x=pp_offset_x,
        pp_offset_y=pp_offset_y,
        hfov_degrees=hfov_degrees,
        division_lambda=division_lambda,
        line_rms=line_rms,
        angle_degrees=angle,
        expanded=False,
    )


def explore_space(
    payload: dict[str, Any],
    *,
    max_error: float,
    pp_radius: float,
    fov_span: float,
    pp_step: float,
    fov_step: float,
    division_lambda: float,
    vary_lambda: bool,
    lambda_span: float,
    lambda_step: float,
    max_samples: int,
    metric: str,
) -> list[Sample]:
    """BFS from the match seed; expand only while error stays under max_error."""
    bundles = _bundles_from_payload(payload)
    ready_axes = sum(1 for segments in bundles.values() if len(segments) >= 2)
    if ready_axes < 2:
        raise ValueError(
            "Need ≥2 lines on each of two axes to score orientation "
            f"(ready axes={ready_axes}, counts={payload.get('line_counts')})"
        )

    width = int(payload["image_width"])
    height = int(payload["image_height"])
    seed_ppx = float(payload["pp_offset_x"])
    seed_ppy = float(payload["pp_offset_y"])
    seed_fov = float(payload["hfov_degrees"])
    seed_lam = float(division_lambda)

    pp_step = max(float(pp_step), 1.0e-3)
    fov_step = max(float(fov_step), 1.0e-3)
    lambda_step = max(float(lambda_step), 1.0e-6)

    def key_for(ppx: float, ppy: float, fov: float, lam: float) -> SampleKey:
        return SampleKey(
            ppx_i=int(round((ppx - seed_ppx) / pp_step)),
            ppy_i=int(round((ppy - seed_ppy) / pp_step)),
            fov_i=int(round((fov - seed_fov) / fov_step)),
            lam_i=int(round((lam - seed_lam) / lambda_step)) if vary_lambda else 0,
        )

    def decode(key: SampleKey) -> tuple[float, float, float, float]:
        ppx = seed_ppx + key.ppx_i * pp_step
        ppy = seed_ppy + key.ppy_i * pp_step
        fov = seed_fov + key.fov_i * fov_step
        lam = seed_lam + key.lam_i * lambda_step if vary_lambda else seed_lam
        return ppx, ppy, fov, lam

    def in_bounds(ppx: float, ppy: float, fov: float, lam: float) -> bool:
        if abs(ppx - seed_ppx) > pp_radius + 1.0e-6:
            return False
        if abs(ppy - seed_ppy) > pp_radius + 1.0e-6:
            return False
        if abs(fov - seed_fov) > fov_span + 1.0e-6:
            return False
        if fov <= 1.0 or fov >= 179.0:
            return False
        if vary_lambda and abs(lam - seed_lam) > lambda_span + 1.0e-9:
            return False
        return True

    neighbors = [
        (1, 0, 0, 0),
        (-1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, -1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, -1, 0),
    ]
    if vary_lambda:
        neighbors.extend([(0, 0, 0, 1), (0, 0, 0, -1)])

    start_key = key_for(seed_ppx, seed_ppy, seed_fov, seed_lam)
    queue: deque[SampleKey] = deque([start_key])
    seen: set[SampleKey] = {start_key}
    samples: list[Sample] = []
    # Always walk one ring around the seed so a bad starting guess still
    # produces a local map; pruning applies from the second hop onward.
    force_expand: set[SampleKey] = {start_key}

    while queue and len(samples) < max_samples:
        current_key = queue.popleft()
        ppx, ppy, fov, lam = decode(current_key)
        if not in_bounds(ppx, ppy, fov, lam):
            continue
        sample = evaluate_intrinsics(
            bundles,
            width=width,
            height=height,
            hfov_degrees=fov,
            pp_offset_x=ppx,
            pp_offset_y=ppy,
            division_lambda=lam,
        )
        error = _error_of(sample, metric)
        sample.expanded = error <= max_error or current_key in force_expand
        samples.append(sample)
        if not sample.expanded:
            continue
        for d_ppx, d_ppy, d_fov, d_lam in neighbors:
            neighbor = SampleKey(
                ppx_i=current_key.ppx_i + d_ppx,
                ppy_i=current_key.ppy_i + d_ppy,
                fov_i=current_key.fov_i + d_fov,
                lam_i=current_key.lam_i + d_lam,
            )
            if neighbor in seen:
                continue
            n_ppx, n_ppy, n_fov, n_lam = decode(neighbor)
            if not in_bounds(n_ppx, n_ppy, n_fov, n_lam):
                continue
            seen.add(neighbor)
            queue.append(neighbor)

    return samples


def write_csv(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pp_offset_x",
                "pp_offset_y",
                "hfov_degrees",
                "division_lambda",
                "line_rms",
                "angle_degrees",
                "expanded",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "pp_offset_x": sample.pp_offset_x,
                    "pp_offset_y": sample.pp_offset_y,
                    "hfov_degrees": sample.hfov_degrees,
                    "division_lambda": sample.division_lambda,
                    "line_rms": sample.line_rms,
                    "angle_degrees": sample.angle_degrees,
                    "expanded": int(sample.expanded),
                }
            )


def _summarize_samples(
    samples: list[Sample],
    payload: dict[str, Any],
    *,
    metric: str,
    max_error: float,
) -> tuple[Sample, Sample]:
    """Print seed/best stats; return (seed, best)."""
    seed_x = float(payload["pp_offset_x"])
    seed_y = float(payload["pp_offset_y"])
    seed_z = float(payload["hfov_degrees"])
    best = min(samples, key=lambda sample: _error_of(sample, metric))
    seed = min(
        samples,
        key=lambda sample: (
            abs(sample.pp_offset_x - seed_x)
            + abs(sample.pp_offset_y - seed_y)
            + abs(sample.hfov_degrees - seed_z)
        ),
    )
    print(
        f"Seed error={_error_of(seed, metric):.3f} @ FOV {seed.hfov_degrees:.1f}° "
        f"PP ({seed.pp_offset_x:.0f},{seed.pp_offset_y:.0f})  "
        f"best={_error_of(best, metric):.3f} @ FOV {best.hfov_degrees:.1f}° "
        f"PP ({best.pp_offset_x:.0f},{best.pp_offset_y:.0f})  "
        f"expanded={sum(1 for sample in samples if sample.expanded)}/"
        f"{len(samples)}"
    )
    if _error_of(seed, metric) > max_error and len(samples) <= 7:
        print(
            f"Note: seed error exceeds --max-error {max_error:g}; "
            "only the seed ring was forced. Raise --max-error to keep walking."
        )
    return seed, best


# Which parameter the slider steps through. The other three become plot X/Y/Z;
# when the slider is not λ, λ takes that axis's place in the 3D view.
SLIDE_AXIS_IDS = ("lambda", "hfov", "ppx", "ppy")
SLIDE_AXIS_META: dict[str, dict[str, Any]] = {
    "lambda": {
        "label": "λ",
        "prefix": "λ = ",
        "decimals": 6,
        "value_format": ".5f",
        "step_format": ".3f",
        "plot_attrs": ("pp_offset_x", "pp_offset_y", "hfov_degrees"),
        "plot_titles": ("PP offset X (px)", "PP offset Y (px)", "HFOV (°)"),
    },
    "hfov": {
        "label": "FOV",
        "prefix": "HFOV = ",
        "decimals": 4,
        "value_format": ".2f",
        "step_format": ".1f",
        "plot_attrs": ("pp_offset_x", "pp_offset_y", "division_lambda"),
        "plot_titles": ("PP offset X (px)", "PP offset Y (px)", "λ"),
    },
    "ppx": {
        "label": "PP X",
        "prefix": "PP X = ",
        "decimals": 3,
        "value_format": ".1f",
        "step_format": ".0f",
        "plot_attrs": ("division_lambda", "pp_offset_y", "hfov_degrees"),
        "plot_titles": ("λ", "PP offset Y (px)", "HFOV (°)"),
    },
    "ppy": {
        "label": "PP Y",
        "prefix": "PP Y = ",
        "decimals": 3,
        "value_format": ".1f",
        "step_format": ".0f",
        "plot_attrs": ("pp_offset_x", "division_lambda", "hfov_degrees"),
        "plot_titles": ("PP offset X (px)", "λ", "HFOV (°)"),
    },
}


def _axis_value(sample: Sample, axis: str) -> float:
    if axis == "lambda":
        return float(sample.division_lambda)
    if axis == "hfov":
        return float(sample.hfov_degrees)
    if axis == "ppx":
        return float(sample.pp_offset_x)
    if axis == "ppy":
        return float(sample.pp_offset_y)
    raise ValueError(f"Unknown slide axis: {axis}")


def _plot_xyz(sample: Sample, slide_axis: str) -> tuple[float, float, float]:
    attrs = SLIDE_AXIS_META[slide_axis]["plot_attrs"]
    return (
        float(getattr(sample, attrs[0])),
        float(getattr(sample, attrs[1])),
        float(getattr(sample, attrs[2])),
    )


def _unique_axis_values(samples: list[Sample], axis: str) -> list[float]:
    decimals = int(SLIDE_AXIS_META[axis]["decimals"])
    keys = sorted({round(_axis_value(sample, axis), decimals) for sample in samples})
    return [float(key) for key in keys]


def _samples_for_axis_value(
    samples: list[Sample],
    axis: str,
    value: float,
) -> list[Sample]:
    decimals = int(SLIDE_AXIS_META[axis]["decimals"])
    target = round(value, decimals)
    return [
        sample
        for sample in samples
        if round(_axis_value(sample, axis), decimals) == target
    ]


def _seed_legend_name(sample: Sample, metric: str) -> str:
    """Legend/footer label for the cyan seed marker, including error and XYZ."""
    error = _error_of(sample, metric)
    unit = "px" if metric == "line_rms" else "°"
    return (
        f"seed (current match) · {error:.3f}{unit} @ "
        f"PP({sample.pp_offset_x:.0f},{sample.pp_offset_y:.0f}) "
        f"FOV={sample.hfov_degrees:.1f}° · λ={sample.division_lambda:.5f}"
    )


def _best_legend_name(sample: Sample | None, metric: str) -> str:
    """Legend label for the green best marker, including error and params."""
    if sample is None:
        return "best in slice · (none)"
    error = _error_of(sample, metric)
    unit = "px" if metric == "line_rms" else "°"
    return (
        f"best in slice · {error:.3f}{unit} @ "
        f"PP({sample.pp_offset_x:.0f},{sample.pp_offset_y:.0f}) "
        f"FOV={sample.hfov_degrees:.1f}° · λ={sample.division_lambda:.5f}"
    )


def _scatter_traces_for_slice(
    go_module,
    *,
    accepted: list[Sample],
    rejected: list[Sample],
    seed: Sample,
    metric: str,
    max_error: float,
    error_label: str,
    show_rejected: bool,
    show_colorbar: bool,
    slide_axis: str,
    hover_for,
) -> list[Any]:
    """Build the fixed-length trace list used by the base figure and frames."""
    traces: list[Any] = []
    if show_rejected:
        rejected_xyz = [_plot_xyz(sample, slide_axis) for sample in rejected]
        traces.append(
            go_module.Scatter3d(
                x=[point[0] for point in rejected_xyz] or [None],
                y=[point[1] for point in rejected_xyz] or [None],
                z=[point[2] for point in rejected_xyz] or [None],
                mode="markers",
                name=f"rejected (>{max_error:g})",
                text=[hover_for(sample) for sample in rejected] or [""],
                hoverinfo="text",
                marker={
                    "size": 3,
                    "color": "lightgray",
                    "opacity": 0.35,
                },
                visible=True if rejected else False,
            )
        )

    accepted_xyz = [_plot_xyz(sample, slide_axis) for sample in accepted]
    errors = [_error_of(sample, metric) for sample in accepted]
    marker: dict[str, Any] = {
        "size": 4,
        "color": errors or [0.0],
        "cmin": 0.0,
        "cmax": float(max_error),
        "colorscale": "Magma_r",
        "opacity": 0.9,
    }
    if show_colorbar:
        marker["colorbar"] = {
            "title": error_label,
            "x": 1.0,
            "xpad": 8,
        }
    traces.append(
        go_module.Scatter3d(
            x=[point[0] for point in accepted_xyz] or [None],
            y=[point[1] for point in accepted_xyz] or [None],
            z=[point[2] for point in accepted_xyz] or [None],
            mode="markers",
            name=f"accepted (≤{max_error:g})",
            text=[hover_for(sample) for sample in accepted] or [""],
            hoverinfo="text",
            marker=marker,
        )
    )

    seed_xyz = _plot_xyz(seed, slide_axis)
    traces.append(
        go_module.Scatter3d(
            x=[seed_xyz[0]],
            y=[seed_xyz[1]],
            z=[seed_xyz[2]],
            mode="markers",
            name=_seed_legend_name(seed, metric),
            hovertext=[
                f"SEED<br>PP ({seed.pp_offset_x:.1f}, {seed.pp_offset_y:.1f})<br>"
                f"HFOV {seed.hfov_degrees:.2f}° · λ {seed.division_lambda:.5f}<br>"
                f"error {_error_of(seed, metric):.3f}"
            ],
            hoverinfo="text",
            marker={
                "size": 10,
                "color": "cyan",
                "symbol": "diamond",
                "line": {"width": 1, "color": "black"},
            },
            visible=True,
        )
    )

    if accepted:
        plot_best = min(accepted, key=lambda sample: _error_of(sample, metric))
        best_xyz = _plot_xyz(plot_best, slide_axis)
        traces.append(
            go_module.Scatter3d(
                x=[best_xyz[0]],
                y=[best_xyz[1]],
                z=[best_xyz[2]],
                mode="markers",
                name=_best_legend_name(plot_best, metric),
                hovertext=[
                    f"BEST<br>PP ({plot_best.pp_offset_x:.1f}, {plot_best.pp_offset_y:.1f})<br>"
                    f"HFOV {plot_best.hfov_degrees:.2f}° · λ {plot_best.division_lambda:.5f}<br>"
                    f"error {_error_of(plot_best, metric):.3f}"
                ],
                hoverinfo="text",
                marker={
                    "size": 10,
                    "color": "lime",
                    "symbol": "x",
                    "line": {"width": 1, "color": "black"},
                },
                visible=True,
            )
        )
    else:
        traces.append(
            go_module.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=_best_legend_name(None, metric),
                hoverinfo="skip",
                marker={"size": 10, "color": "lime", "symbol": "x"},
                visible=False,
            )
        )
    return traces


def _trace_update_payload(traces: list[Any]) -> dict[str, list[Any]]:
    """Flatten Scatter3d traces into a Plotly updatemenus data payload."""
    payload: dict[str, list[Any]] = {
        "x": [],
        "y": [],
        "z": [],
        "text": [],
        "hovertext": [],
        "name": [],
        "visible": [],
        "marker": [],
    }
    for trace in traces:
        payload["x"].append(list(trace.x) if trace.x is not None else [None])
        payload["y"].append(list(trace.y) if trace.y is not None else [None])
        payload["z"].append(list(trace.z) if trace.z is not None else [None])
        payload["text"].append(list(trace.text) if getattr(trace, "text", None) else [""])
        hover = getattr(trace, "hovertext", None)
        payload["hovertext"].append(list(hover) if hover else [""])
        payload["name"].append(trace.name)
        payload["visible"].append(True if trace.visible is None else trace.visible)
        payload["marker"].append(trace.marker.to_plotly_json())
    return payload


def plot_samples(
    samples: list[Sample],
    payload: dict[str, Any],
    *,
    metric: str,
    max_error: float,
    out_path: Path | None,
    show: bool,
    show_rejected: bool = False,
    slide_axis: str = "lambda",
) -> None:
    """Open an interactive Plotly 3D scatter (orbit / zoom / hover)."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "plotly is required for the interactive viewer. "
            "Install it in this Python (pip install plotly)."
        ) from error

    if not samples:
        raise ValueError("No samples to plot")
    if slide_axis not in SLIDE_AXIS_META:
        raise ValueError(f"Unknown slide axis: {slide_axis}")

    seed, _overall_best = _summarize_samples(
        samples,
        payload,
        metric=metric,
        max_error=max_error,
    )
    error_label = (
        "VP line RMS (px)" if metric == "line_rms" else "VP angular residual (°)"
    )

    def hover_for(sample: Sample) -> str:
        return (
            f"PP offset: ({sample.pp_offset_x:.1f}, {sample.pp_offset_y:.1f}) px<br>"
            f"HFOV: {sample.hfov_degrees:.2f}°<br>"
            f"λ: {sample.division_lambda:.5f}<br>"
            f"line RMS: {sample.line_rms:.3f} px<br>"
            f"angle: {sample.angle_degrees:.3f}°<br>"
            f"expanded: {sample.expanded}"
        )

    accepted_all = [
        sample for sample in samples if _error_of(sample, metric) <= max_error
    ]
    rejected_all = [
        sample for sample in samples if _error_of(sample, metric) > max_error
    ]
    if not accepted_all:
        accepted_all = [seed]
    plot_best = min(accepted_all, key=lambda sample: _error_of(sample, metric))
    if rejected_all and not show_rejected:
        print(
            f"Hidden {len(rejected_all)} rejected frontier samples "
            f"(error > {max_error:g}). Pass --show-rejected to include them."
        )

    def best_for_accepted(accepted: list[Sample]) -> Sample | None:
        if not accepted:
            return None
        return min(accepted, key=lambda sample: _error_of(sample, metric))

    def slice_lists(axis: str, value: float) -> tuple[list[Sample], list[Sample]]:
        accepted = _samples_for_axis_value(accepted_all, axis, value)
        rejected = (
            _samples_for_axis_value(rejected_all, axis, value) if show_rejected else []
        )
        return accepted, rejected

    def padded_range(values: list[float]) -> list[float]:
        low = min(values)
        high = max(values)
        pad = max((high - low) * 0.05, 1.0e-3)
        return [low - pad, high + pad]

    def axis_ranges(axis: str) -> tuple[list[float], list[float], list[float]]:
        pool = accepted_all + (rejected_all if show_rejected else [])
        if not pool:
            pool = [seed]
        xs, ys, zs = [], [], []
        for sample in pool:
            x_coordinate, y_coordinate, z_coordinate = _plot_xyz(sample, axis)
            xs.append(x_coordinate)
            ys.append(y_coordinate)
            zs.append(z_coordinate)
        # Always include seed so marker stays in-frame when outside the slice cloud.
        seed_xyz = _plot_xyz(seed, axis)
        xs.append(seed_xyz[0])
        ys.append(seed_xyz[1])
        zs.append(seed_xyz[2])
        return padded_range(xs), padded_range(ys), padded_range(zs)

    # Axes you can slide when they have more than one sampled value.
    slidable_axes = [
        axis
        for axis in SLIDE_AXIS_IDS
        if len(_unique_axis_values(samples, axis)) > 1
    ]
    if not slidable_axes:
        # Flat cloud — still draw once with the requested mapping.
        slidable_axes = [slide_axis]

    if slide_axis not in slidable_axes:
        slide_axis = slidable_axes[0]

    title_prefix = f"<b>{payload['match_name']}</b>"
    title = (
        f"{title_prefix} · "
        f"best overall {_error_of(plot_best, metric):.2f} @ "
        f"PP({plot_best.pp_offset_x:.0f},{plot_best.pp_offset_y:.0f}) "
        f"FOV={plot_best.hfov_degrees:.1f}° · λ={plot_best.division_lambda:.5f}"
    )
    use_slider = any(len(_unique_axis_values(samples, axis)) > 1 for axis in slidable_axes)

    def footer_annotations(slice_best: Sample | None, active_axis: str) -> list[dict[str, Any]]:
        meta = SLIDE_AXIS_META[active_axis]
        return [
            {
                "text": (
                    f"{title}<br>"
                    f"sliding <b>{meta['label']}</b> · "
                    f"plot X={meta['plot_titles'][0]}, "
                    f"Y={meta['plot_titles'][1]}, "
                    f"Z={meta['plot_titles'][2]}"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.10 if not use_slider else -0.16,
                "xanchor": "center",
                "yanchor": "top",
                "showarrow": False,
                "align": "center",
                "font": {"size": 12},
            },
            {
                "text": (
                    f"<span style='color:#00bcd4'>◆</span> "
                    f"<b>{_seed_legend_name(seed, metric)}</b>"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.16 if not use_slider else -0.22,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "font": {"size": 13, "color": "#1a1a1a"},
            },
            {
                "text": (
                    f"<span style='color:#32cd32'>✚</span> "
                    f"<b>{_best_legend_name(slice_best, metric)}</b>"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.20 if not use_slider else -0.26,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "font": {"size": 13, "color": "#1a1a1a"},
            },
        ]

    def pick_initial_value(axis: str) -> float:
        values = _unique_axis_values(samples, axis)
        seed_value = _axis_value(seed, axis)
        decimals = int(SLIDE_AXIS_META[axis]["decimals"])
        rounded_seed = round(seed_value, decimals)
        if rounded_seed in {round(value, decimals) for value in values}:
            return float(seed_value)
        for value in values:
            accepted, rejected = slice_lists(axis, value)
            if accepted or rejected:
                return value
        return values[0] if values else seed_value

    def build_slider(axis: str, active_value: float) -> dict[str, Any]:
        meta = SLIDE_AXIS_META[axis]
        values = _unique_axis_values(samples, axis)
        decimals = int(meta["decimals"])
        steps = []
        for value in values:
            frame_name = f"{axis}::{value:{meta['value_format']}}"
            steps.append(
                {
                    "args": [
                        [frame_name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": f"{value:{meta['step_format']}}",
                    "method": "animate",
                }
            )
        active = next(
            (
                index
                for index, value in enumerate(values)
                if round(value, decimals) == round(active_value, decimals)
            ),
            0,
        )
        return {
            "active": active,
            "currentvalue": {
                "prefix": str(meta["prefix"]),
                "xanchor": "center",
                "font": {"size": 14},
            },
            "pad": {"t": 30, "b": 4},
            "len": 0.9,
            "x": 0.05,
            "xanchor": "left",
            "y": -0.02,
            "yanchor": "top",
            "steps": steps,
        }

    def scene_for_axis(axis: str) -> dict[str, Any]:
        titles = SLIDE_AXIS_META[axis]["plot_titles"]
        x_range, y_range, z_range = axis_ranges(axis)
        return {
            "xaxis_title": titles[0],
            "yaxis_title": titles[1],
            "zaxis_title": titles[2],
            "aspectmode": "cube",
            "xaxis": {"range": x_range},
            "yaxis": {"range": y_range},
            "zaxis": {"range": z_range},
        }

    # Build every frame for every slidable axis up front (dropdown swaps which
    # slider steps animate which frames, and remaps XYZ).
    all_frames: list[Any] = []
    axis_bootstrap: dict[str, dict[str, Any]] = {}
    for axis in slidable_axes:
        initial_value = pick_initial_value(axis)
        initial_accepted, initial_rejected = slice_lists(axis, initial_value)
        bootstrap_traces = _scatter_traces_for_slice(
            go,
            accepted=initial_accepted if use_slider else accepted_all,
            rejected=initial_rejected if use_slider else (
                rejected_all if show_rejected else []
            ),
            seed=seed,
            metric=metric,
            max_error=max_error,
            error_label=error_label,
            show_rejected=show_rejected,
            show_colorbar=True,
            slide_axis=axis,
            hover_for=hover_for,
        )
        axis_bootstrap[axis] = {
            "value": initial_value,
            "traces": bootstrap_traces,
            "best": best_for_accepted(
                initial_accepted if use_slider else accepted_all
            ),
            "payload": _trace_update_payload(bootstrap_traces),
            "slider": build_slider(axis, initial_value) if use_slider else None,
        }
        if not use_slider:
            continue
        meta = SLIDE_AXIS_META[axis]
        for value in _unique_axis_values(samples, axis):
            accepted, rejected = slice_lists(axis, value)
            slice_best = best_for_accepted(accepted)
            frame_traces = _scatter_traces_for_slice(
                go,
                accepted=accepted,
                rejected=rejected,
                seed=seed,
                metric=metric,
                max_error=max_error,
                error_label=error_label,
                show_rejected=show_rejected,
                show_colorbar=False,
                slide_axis=axis,
                hover_for=hover_for,
            )
            frame_name = f"{axis}::{value:{meta['value_format']}}"
            all_frames.append(
                go.Frame(
                    data=frame_traces,
                    name=frame_name,
                    layout={"annotations": footer_annotations(slice_best, axis)},
                )
            )

    active = axis_bootstrap[slide_axis]
    figure = go.Figure(data=active["traces"])
    if all_frames:
        figure.frames = all_frames
        print(
            "Slide axes: "
            + ", ".join(
                f"{axis}({len(_unique_axis_values(samples, axis))})"
                for axis in slidable_axes
            )
            + f" · starting on {slide_axis}"
        )

    layout_updates: dict[str, Any] = {
        "title": None,
        "scene": scene_for_axis(slide_axis),
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.85)",
            "font": {"size": 11},
        },
        "annotations": footer_annotations(active["best"], slide_axis),
        "margin": {
            "l": 0,
            "r": 40,
            "t": 48,
            "b": 140 if not use_slider else 180,
        },
        "template": "plotly_white",
        "modebar": {"orientation": "v"},
    }
    if use_slider and active["slider"] is not None:
        layout_updates["sliders"] = [active["slider"]]

    # Dropdown: pick which parameter the slider walks; λ fills the vacated plot axis.
    if len(slidable_axes) > 1:
        buttons = []
        trace_indices = list(range(len(active["traces"])))
        for axis in slidable_axes:
            boot = axis_bootstrap[axis]
            layout_args: dict[str, Any] = {
                "scene": scene_for_axis(axis),
                "annotations": footer_annotations(boot["best"], axis),
            }
            if boot["slider"] is not None:
                layout_args["sliders"] = [boot["slider"]]
            buttons.append(
                {
                    "label": f"Slide {SLIDE_AXIS_META[axis]['label']}",
                    "method": "update",
                    "args": [boot["payload"], layout_args, trace_indices],
                }
            )
        layout_updates["updatemenus"] = [
            {
                "type": "dropdown",
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
                "pad": {"r": 8, "t": 0, "b": 0},
                "buttons": buttons,
                "active": slidable_axes.index(slide_axis),
                "bgcolor": "rgba(255,255,255,0.95)",
                "bordercolor": "#888",
                "font": {"size": 12},
            }
        ]

    figure.update_layout(**layout_updates)
    figure_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
        "scrollZoom": True,
    }

    if out_path is not None:
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            figure.write_html(
                str(out_path),
                include_plotlyjs="cdn",
                auto_open=False,
                config=figure_config,
            )
            print(f"Wrote interactive HTML {out_path}")
        elif suffix == ".png":
            try:
                figure.write_image(str(out_path), scale=2)
            except Exception as error:
                raise RuntimeError(
                    "PNG export needs the optional kaleido package "
                    "(pip install kaleido), or use --out something.html"
                ) from error
            print(f"Wrote PNG {out_path}")
        else:
            html_path = out_path.with_suffix(".html") if suffix else out_path.with_suffix(".html")
            figure.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                auto_open=False,
                config=figure_config,
            )
            print(f"Wrote interactive HTML {html_path}")

    if show:
        figure.show(config=figure_config)
        print("Opened interactive 3D scatter in your browser.")
        try:
            input("Press Enter to exit… ")
        except EOFError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    blend = args.blend.expanduser().resolve()
    if args.json is None and not blend.is_file():
        print(f"Blend not found: {blend}", file=sys.stderr)
        return 2

    temporary_json: Path | None = None
    try:
        if args.json is not None:
            payload = json.loads(args.json.expanduser().resolve().read_text(encoding="utf-8"))
        else:
            if args.keep_json is not None:
                json_path = args.keep_json.expanduser().resolve()
            else:
                handle = tempfile.NamedTemporaryFile(
                    prefix="pm-vp-explore-",
                    suffix=".json",
                    delete=False,
                )
                handle.close()
                temporary_json = Path(handle.name)
                json_path = temporary_json
            payload = extract_match_json(
                blend,
                args.match,
                args.blender,
                json_path,
            )
            if args.keep_json is not None:
                print(f"Wrote extract {json_path}")

        division_lambda = (
            float(args.division_lambda)
            if args.division_lambda is not None
            else float(payload["division_lambda"])
        )
        samples = explore_space(
            payload,
            max_error=float(args.max_error),
            pp_radius=float(args.pp_radius),
            fov_span=float(args.fov_span),
            pp_step=float(args.pp_step),
            fov_step=float(args.fov_step),
            division_lambda=division_lambda,
            vary_lambda=bool(args.vary_lambda),
            lambda_span=float(args.lambda_span),
            lambda_step=float(args.lambda_step),
            max_samples=int(args.max_samples),
            metric=args.metric,
        )
        if args.csv is not None:
            write_csv(args.csv.expanduser().resolve(), samples)
            print(f"Wrote CSV {args.csv}")
        if args.no_show and args.out is None and args.csv is None:
            print(
                "Nothing to show: omit --no-show, or add --out / --csv.",
                file=sys.stderr,
            )
            return 2
        plot_samples(
            samples,
            payload,
            metric=args.metric,
            max_error=float(args.max_error),
            out_path=args.out,
            show=not bool(args.no_show),
            show_rejected=bool(args.show_rejected),
            slide_axis=str(args.slide_axis),
        )
    finally:
        if temporary_json is not None and temporary_json.is_file():
            temporary_json.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
