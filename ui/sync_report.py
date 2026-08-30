"""Self-contained HTML reports for landmark-sync diagnostics.

The renderer is deliberately bpy-free so its model and markup can be tested by
normal Python. Blender-specific operators supply scene data, choose the temp
root, and open the resulting ``file://`` URL on the main thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence


PAIR_POINT_REQUIREMENT = 5
METRIC_POINT_REQUIREMENT = 3
HIGH_ERROR_PX = 8.0


def _shared_point_summary(count: int) -> str:
    """Describe pairwise point support without making the threshold a fraction."""
    if count < PAIR_POINT_REQUIREMENT:
        return f"{count} of {PAIR_POINT_REQUIREMENT} shared"
    return f"{count} shared; minimum {PAIR_POINT_REQUIREMENT}"


_REPORT_KEEP_COUNT = 10
_last_report_path: Path | None = None


@dataclass(frozen=True)
class ReportIssue:
    """One user-facing problem or useful piece of context."""

    severity: str
    title: str
    detail: str
    action: str = ""


@dataclass(frozen=True)
class ReportMatch:
    """Registration and observation summary for one match."""

    match_id: str
    label: str
    status: str
    rmse_px: float | None
    point_picks: int
    line_picks: int
    ground_picks: int
    locked: bool = False
    best_reference: str = ""
    best_shared_points: int = 0
    usable_3d_points: int = 0
    known_3d_line_picks: int = 0


@dataclass(frozen=True)
class ReportLandmark:
    """One solved point/line row in the error table."""

    landmark_id: str
    name: str
    kind: str
    rmse_px: float
    matches: tuple[str, ...]
    downweighted: bool = False


@dataclass(frozen=True)
class ReportEdge:
    """Point-overlap edge in the camera graph."""

    match_a: str
    match_b: str
    shared_points: int


@dataclass
class SyncDiagnosticReport:
    """Serializable presentation model shared by HTML and compact Blender UI."""

    operation: str
    generated_at: str
    source_name: str
    outcome: str
    severity: str
    enabled_matches: int
    registered_matches: int
    rmse_px: float
    issues: list[ReportIssue] = field(default_factory=list)
    matches: list[ReportMatch] = field(default_factory=list)
    landmarks: list[ReportLandmark] = field(default_factory=list)
    edges: list[ReportEdge] = field(default_factory=list)
    constraints: dict[str, int] = field(default_factory=dict)
    disabled_matches: int = 0
    excluded_landmarks: int = 0
    locked_matches: int = 0
    notes: list[str] = field(default_factory=list)
    raw_message: str = ""
    bundle_adjusted: bool = False

    @property
    def attention_count(self) -> int:
        return sum(1 for item in self.issues if item.severity in {"error", "warning"})


def friendly_match_name(match_id: str) -> str:
    """Turn an internal PM hierarchy name into a compact report label."""
    name = str(match_id)
    if name.startswith("PM_"):
        name = name[3:]
    if name.endswith("_Origin"):
        name = name[:-7]
    return name.replace("_", " ")


def _match_ids(matches: Sequence[object]) -> list[str]:
    return [str(item.match_id) for item in matches]


def _landmark_names(
    observations: Sequence[object],
    line_observations: Sequence[object],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in (*observations, *line_observations):
        landmark_id = str(item.landmark_id)
        name = str(getattr(item, "landmark_name", "") or "")
        if name:
            names.setdefault(landmark_id, name)
        else:
            names.setdefault(landmark_id, landmark_id[:8])
    return names


def _observation_sets(
    observations: Sequence[object],
    *,
    excluded_pair_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    all_points: dict[str, set[str]] = {}
    pair_points: dict[str, set[str]] = {}
    ground_counts: dict[str, int] = {}
    for observation in observations:
        match_id = str(observation.match_id)
        landmark_id = str(observation.landmark_id)
        all_points.setdefault(match_id, set()).add(landmark_id)
        if landmark_id not in excluded_pair_ids:
            pair_points.setdefault(match_id, set()).add(landmark_id)
        if bool(getattr(observation, "on_ground", False)):
            ground_counts[match_id] = ground_counts.get(match_id, 0) + 1
    return all_points, pair_points, ground_counts


def _line_sets(line_observations: Sequence[object]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for observation in line_observations:
        output.setdefault(str(observation.match_id), set()).add(
            str(observation.landmark_id)
        )
    return output


def _connectivity_edges(
    enabled_ids: Sequence[str],
    pair_points: Mapping[str, set[str]],
) -> list[ReportEdge]:
    edges: list[ReportEdge] = []
    for index, match_a in enumerate(enabled_ids):
        points_a = pair_points.get(match_a, set())
        for match_b in enabled_ids[index + 1 :]:
            count = len(points_a & pair_points.get(match_b, set()))
            if count:
                edges.append(ReportEdge(match_a, match_b, count))
    return edges


def _best_registered_edge(
    match_id: str,
    registered_ids: set[str],
    edges: Sequence[ReportEdge],
) -> tuple[str, int]:
    best_reference = ""
    best_count = 0
    for edge in edges:
        if edge.match_a == match_id and edge.match_b in registered_ids:
            reference, count = edge.match_b, edge.shared_points
        elif edge.match_b == match_id and edge.match_a in registered_ids:
            reference, count = edge.match_a, edge.shared_points
        else:
            continue
        if count > best_count:
            best_reference, best_count = reference, count
    return best_reference, best_count


def build_sync_report(
    *,
    operation: str,
    source_name: str,
    matches: Sequence[object],
    observations: Sequence[object],
    line_observations: Sequence[object],
    result: object,
    anchor_id: str,
    known_world: Mapping[str, object] | None = None,
    known_lines: Mapping[str, object] | None = None,
    parallel_pairs: Sequence[object] = (),
    mirror_pairs: Sequence[object] = (),
    all_match_labels: Mapping[str, str] | None = None,
    disabled_match_ids: Iterable[str] = (),
    fixed_match_ids: Iterable[str] = (),
    excluded_landmarks: int = 0,
    warnings: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> SyncDiagnosticReport:
    """Build a report model from one solver result and its frozen inputs."""
    known_world = known_world or {}
    known_lines = known_lines or {}
    labels = dict(all_match_labels or {})
    enabled_ids = _match_ids(matches)
    disabled_ids = {str(item) for item in disabled_match_ids}
    fixed_ids = {str(item) for item in fixed_match_ids}
    registered_ids = {str(item) for item in getattr(result, "similarities", {})}
    if not bool(getattr(result, "success", False)):
        # Failed core results intentionally carry identity transforms for safe
        # callers; those are not successful registrations for the report.
        registered_ids = {anchor_id} if anchor_id in enabled_ids else set()
    anchor_ground_ids = {
        str(item.landmark_id)
        for item in observations
        if str(item.match_id) == anchor_id
        and bool(getattr(item, "on_ground", False))
    }
    solved_3d_ids = {
        str(item) for item in getattr(result, "landmarks", {})
    } | {str(item) for item in known_world} | anchor_ground_ids
    all_points, pair_points, ground_counts = _observation_sets(
        observations,
        excluded_pair_ids={str(item) for item in known_world},
    )
    lines_by_match = _line_sets(line_observations)
    known_line_ids = {str(item) for item in known_lines}
    edges = _connectivity_edges(enabled_ids, pair_points)

    for match_id in (*enabled_ids, *sorted(disabled_ids)):
        labels.setdefault(match_id, friendly_match_name(match_id))

    inconsistent_by_match: dict[str, list[tuple[str, float]]] = {}
    for match_id, name, error in getattr(result, "inconsistent_picks", ()):
        inconsistent_by_match.setdefault(str(match_id), []).append(
            (str(name), float(error))
        )

    report_matches: list[ReportMatch] = []
    for match_id in dict.fromkeys((*enabled_ids, *sorted(disabled_ids))):
        if match_id in disabled_ids:
            status = "disabled"
        elif match_id == anchor_id:
            status = "anchor"
        elif match_id in registered_ids:
            status = "synced"
        else:
            status = "skipped"
        best_reference, best_count = _best_registered_edge(
            match_id, registered_ids - {match_id}, edges
        )
        picked_ids = all_points.get(match_id, set())
        usable_3d = len(picked_ids & solved_3d_ids)
        known_3d_line_picks = len(
            lines_by_match.get(match_id, set()) & known_line_ids
        )
        rmse = getattr(result, "per_match_rmse_px", {}).get(match_id)
        report_matches.append(
            ReportMatch(
                match_id=match_id,
                label=labels[match_id],
                status=status,
                rmse_px=None if rmse is None else float(rmse),
                point_picks=len(picked_ids),
                line_picks=len(lines_by_match.get(match_id, set())),
                ground_picks=ground_counts.get(match_id, 0),
                locked=match_id in fixed_ids,
                best_reference=labels.get(best_reference, best_reference),
                best_shared_points=best_count,
                usable_3d_points=usable_3d,
                known_3d_line_picks=known_3d_line_picks,
            )
        )

    names = _landmark_names(observations, line_observations)
    matches_by_landmark: dict[str, set[str]] = {}
    point_ids: set[str] = set()
    for observation in observations:
        landmark_id = str(observation.landmark_id)
        point_ids.add(landmark_id)
        matches_by_landmark.setdefault(landmark_id, set()).add(
            labels.get(str(observation.match_id), friendly_match_name(observation.match_id))
        )
    for observation in line_observations:
        matches_by_landmark.setdefault(str(observation.landmark_id), set()).add(
            labels.get(str(observation.match_id), friendly_match_name(observation.match_id))
        )
    downweighted = {
        str(item) for item in getattr(result, "downweighted_landmark_ids", ())
    }
    report_landmarks = [
        ReportLandmark(
            landmark_id=str(landmark_id),
            name=names.get(str(landmark_id), str(landmark_id)[:8]),
            kind="point" if str(landmark_id) in point_ids else "line",
            rmse_px=float(rmse),
            matches=tuple(sorted(matches_by_landmark.get(str(landmark_id), set()))),
            downweighted=str(landmark_id) in downweighted,
        )
        for landmark_id, rmse in getattr(result, "per_landmark_rmse_px", {}).items()
    ]
    report_landmarks.sort(key=lambda item: (-item.rmse_px, item.name.casefold()))

    skipped = [item for item in report_matches if item.status == "skipped"]
    success = bool(getattr(result, "success", False))
    if not success:
        outcome, severity = "Sync failed", "error"
    elif skipped:
        outcome, severity = "Partial sync", "warning"
    elif float(getattr(result, "mean_reprojection_px", 0.0)) > HIGH_ERROR_PX:
        outcome, severity = "Sync completed with high error", "warning"
    else:
        outcome, severity = "Sync complete", "success"

    issues: list[ReportIssue] = []
    for item in skipped:
        if item.best_reference:
            route = (
                f"Best ordinary-point connection: {item.best_reference} "
                f"({_shared_point_summary(item.best_shared_points)})."
            )
        else:
            route = "No ordinary-point connection to a registered match was found."
        metric = (
            f" Metric support in this match: {item.usable_3d_points} point pick(s) "
            "to known/reconstructed 3D and "
            f"{item.known_3d_line_picks} Known 3D line pick(s)."
        )
        mismatches = inconsistent_by_match.get(item.match_id, [])
        if mismatches:
            name, error = max(mismatches, key=lambda entry: entry[1])
            action = f"Re-pick {name} ({error:.0f}px disagreement), then Diagnose again."
        elif (
            item.best_shared_points >= PAIR_POINT_REQUIREMENT
            or item.usable_3d_points >= METRIC_POINT_REQUIREMENT
            or item.known_3d_line_picks >= METRIC_POINT_REQUIREMENT
        ):
            action = (
                "A minimum registration route is present; check landmark identity, "
                "2D/3D spread, camera FOV, and pose locks."
            )
        else:
            pair_missing = PAIR_POINT_REQUIREMENT - item.best_shared_points
            target = f" with {item.best_reference}" if item.best_reference else ""
            point_missing = METRIC_POINT_REQUIREMENT - item.usable_3d_points
            line_missing = METRIC_POINT_REQUIREMENT - item.known_3d_line_picks
            action = (
                f"Add {pair_missing} well-spread ordinary point landmark"
                f"{'s' if pair_missing != 1 else ''}{target}; or {point_missing} "
                "non-collinear 2D↔3D point pick"
                f"{'s' if point_missing != 1 else ''}; or {line_missing} Known 3D "
                f"line pick{'s' if line_missing != 1 else ''}."
            )
        issues.append(
            ReportIssue(
                "error",
                f"{item.label} could not register",
                route + metric,
                action,
            )
        )

    rmse_px = float(getattr(result, "mean_reprojection_px", 0.0))
    if rmse_px > HIGH_ERROR_PX:
        worst = ", ".join(
            f"{item.name} {item.rmse_px:.0f}px" for item in report_landmarks[:3]
        )
        issues.append(
            ReportIssue(
                "warning",
                f"High reprojection error: {rmse_px:.2f}px",
                (f"Worst landmarks: {worst}." if worst else "The joint fit remains noisy."),
                "Review the worst picks and camera FOV; locked poses cannot move.",
            )
        )

    helpful_leave_one_out = [
        (str(name), float(with_rmse), float(without_rmse))
        for name, with_rmse, without_rmse in getattr(result, "leave_one_out", ())
        if float(without_rmse) < float(with_rmse)
    ]
    if helpful_leave_one_out:
        bits = ", ".join(
            f"{name} {with_rmse:.0f}→{without_rmse:.0f}px"
            for name, with_rmse, without_rmse in helpful_leave_one_out[:3]
        )
        issues.append(
            ReportIssue(
                "warning",
                "Some landmarks disproportionately affect the solve",
                bits,
                "Re-check these picks before excluding or downweighting them.",
            )
        )

    for warning in warnings:
        issues.append(
            ReportIssue(
                "warning",
                "Known 3D consistency warning",
                str(warning),
                "Re-run Landmarks from Selected / Use Selected after checking the Empty.",
            )
        )

    if fixed_ids:
        issues.append(
            ReportIssue(
                "info",
                f"{len(fixed_ids)} camera pose(s) locked",
                "Locked cameras still constrain landmarks but cannot move during adjustment.",
                "If error remains high, review whether every locked pose is still trusted.",
            )
        )

    ground_ids = {
        str(item.landmark_id)
        for item in observations
        if bool(getattr(item, "on_ground", False))
    }
    all_line_ids = {str(item.landmark_id) for item in line_observations}
    constraints = {
        "Known 3D points": len(known_world),
        "Known 3D lines": len(known_lines),
        "On Ground points": len(ground_ids),
        "Free lines": len(all_line_ids - known_line_ids),
        "Parallel constraints": len(parallel_pairs),
        "Mirror pairs": len(mirror_pairs),
    }

    return SyncDiagnosticReport(
        operation=str(operation),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source_name=str(source_name or "Untitled.blend"),
        outcome=outcome,
        severity=severity,
        enabled_matches=len(enabled_ids),
        registered_matches=len(registered_ids & set(enabled_ids)),
        rmse_px=rmse_px,
        issues=issues,
        matches=report_matches,
        landmarks=report_landmarks,
        edges=edges,
        constraints=constraints,
        disabled_matches=len(disabled_ids),
        excluded_landmarks=max(int(excluded_landmarks), 0),
        locked_matches=len(fixed_ids),
        notes=[str(item) for item in notes if str(item)],
        raw_message=str(getattr(result, "message", "")),
        bundle_adjusted=bool(getattr(result, "bundle_adjusted", False)),
    )


def compact_status(report: SyncDiagnosticReport, *, opened: bool = False) -> str:
    """Short Blender sidebar summary; the HTML owns the detail."""
    bits = [
        report.operation,
        report.outcome,
        f"{report.registered_matches}/{report.enabled_matches} cameras",
        f"{report.rmse_px:.2f}px",
    ]
    if report.attention_count:
        bits.append(
            f"{report.attention_count} issue"
            f"{'s' if report.attention_count != 1 else ''}"
        )
    bits.append("Report opened" if opened else "Report ready")
    return " · ".join(bits)


def _status_label(status: str) -> str:
    return {
        "anchor": "Anchor",
        "synced": "Synced",
        "skipped": "Skipped",
        "disabled": "Disabled",
    }.get(status, status.title())


def _backbone_edges(report: SyncDiagnosticReport) -> list[ReportEdge]:
    """Maximum-overlap forest for a legible graph instead of an edge hairball."""
    parent = {item.match_id: item.match_id for item in report.matches}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    output: list[ReportEdge] = []
    for edge in sorted(
        report.edges,
        key=lambda item: (-item.shared_points, item.match_a, item.match_b),
    ):
        root_a = find(edge.match_a)
        root_b = find(edge.match_b)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        output.append(edge)
    return output


def _svg_graph(report: SyncDiagnosticReport) -> str:
    nodes = report.matches
    if not nodes:
        return '<p class="muted">No camera graph available.</p>'
    width = 960
    height = 450 if len(nodes) > 4 else 350
    center_x, center_y = width / 2, height / 2
    radius_x = min(360.0, max(180.0, width * 0.38))
    radius_y = min(160.0, max(110.0, height * 0.36))
    positions: dict[str, tuple[float, float]] = {}
    anchor = next((item for item in nodes if item.status == "anchor"), None)
    orbit = [item for item in nodes if item is not anchor]
    if anchor is not None:
        positions[anchor.match_id] = (center_x, center_y)
    if orbit:
        for index, item in enumerate(orbit):
            angle = -math.pi / 2 + 2 * math.pi * index / len(orbit)
            positions[item.match_id] = (
                center_x + radius_x * math.cos(angle),
                center_y + radius_y * math.sin(angle),
            )
    elif anchor is not None:
        positions[anchor.match_id] = (center_x, center_y)

    edge_bits: list[str] = []
    for edge in _backbone_edges(report):
        if edge.match_a not in positions or edge.match_b not in positions:
            continue
        x1, y1 = positions[edge.match_a]
        x2, y2 = positions[edge.match_b]
        strength = "strong" if edge.shared_points >= PAIR_POINT_REQUIREMENT else "weak"
        midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
        edge_bits.append(
            f'<g class="edge {strength}"><line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}"/><text x="{midpoint_x:.1f}" '
            f'y="{midpoint_y - 5:.1f}">{edge.shared_points}</text></g>'
        )

    node_bits: list[str] = []
    for item in nodes:
        x_coord, y_coord = positions[item.match_id]
        short_label = item.label if len(item.label) <= 24 else item.label[:21] + "…"
        lock = " · locked" if item.locked else ""
        node_bits.append(
            f'<g class="node {escape(item.status)}" transform="translate({x_coord:.1f} '
            f'{y_coord:.1f})"><rect x="-92" y="-29" width="184" height="58" '
            f'rx="11"/><text class="node-name" text-anchor="middle" y="-4">'
            f'{escape(short_label)}</text><text class="node-state" text-anchor="middle" '
            f'y="16">{escape(_status_label(item.status) + lock)}</text></g>'
        )
    return (
        f'<svg class="camera-graph" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Camera overlap graph">'
        + "".join(edge_bits)
        + "".join(node_bits)
        + "</svg>"
        '<p class="legend">Strongest non-cyclic 2D↔2D links · labels are ordinary '
        'shared points <span class="swatch strong"></span>5+ '
        '<span class="swatch weak"></span>1–4</p>'
    )


def _render_issues(report: SyncDiagnosticReport) -> str:
    if not report.issues:
        return (
            '<div class="empty-state"><strong>No issues found.</strong>'
            '<span>The current sync graph and reprojection error look healthy.</span></div>'
        )
    cards = []
    for issue in report.issues:
        action = (
            f'<p class="action"><strong>Next:</strong> {escape(issue.action)}</p>'
            if issue.action
            else ""
        )
        cards.append(
            f'<article class="issue {escape(issue.severity)}">'
            f'<div class="issue-icon" aria-hidden="true"></div><div>'
            f'<h3>{escape(issue.title)}</h3><p>{escape(issue.detail)}</p>{action}'
            "</div></article>"
        )
    return "".join(cards)


def _render_match_rows(report: SyncDiagnosticReport) -> str:
    rows = []
    for item in report.matches:
        rmse = "—" if item.rmse_px is None else f"{item.rmse_px:.2f}px"
        best = (
            f"{escape(item.best_reference)} · "
            f"{escape(_shared_point_summary(item.best_shared_points))}"
            if item.best_reference
            else "—"
        )
        rows.append(
            f'<tr><td><strong>{escape(item.label)}</strong>'
            f'{"<span class=\"lock\">Locked</span>" if item.locked else ""}</td>'
            f'<td><span class="pill {escape(item.status)}">'
            f'{escape(_status_label(item.status))}</span></td>'
            f'<td class="number">{rmse}</td><td class="number">{item.point_picks}</td>'
            f'<td class="number">{item.line_picks}</td><td class="number">'
            f'{item.usable_3d_points}</td><td>{best}</td></tr>'
        )
    return "".join(rows)


def _render_landmark_rows(report: SyncDiagnosticReport) -> str:
    rows = []
    for item in report.landmarks:
        status = '<span class="pill warning">Downweighted</span>' if item.downweighted else ""
        search = " ".join((item.name, item.kind, *item.matches)).casefold()
        rows.append(
            f'<tr data-search="{escape(search, quote=True)}" data-rmse="{item.rmse_px:.8f}">'
            f'<td><strong>{escape(item.name)}</strong> {status}</td>'
            f'<td>{escape(item.kind.title())}</td><td class="number">'
            f'{item.rmse_px:.2f}px</td><td>{escape(", ".join(item.matches))}</td></tr>'
        )
    if not rows:
        return '<tr><td colspan="4" class="muted">No landmark residuals available.</td></tr>'
    return "".join(rows)


def render_sync_report_html(report: SyncDiagnosticReport) -> str:
    """Render a portable offline HTML diagnostic report."""
    severity_label = {
        "success": "Healthy",
        "warning": "Needs review",
        "error": "Failed",
    }.get(report.severity, report.severity.title())
    constraint_cards = "".join(
        f'<div class="mini-card"><span>{escape(name)}</span><strong>{count}</strong></div>'
        for name, count in report.constraints.items()
    )
    notes = "".join(f"<li>{escape(item)}</li>" for item in report.notes)
    notes_block = f'<ul class="notes">{notes}</ul>' if notes else '<p class="muted">None.</p>'
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
  <title>Perspective Match — {escape(report.outcome)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f4f6f8; --surface:#fff; --surface2:#f8fafc;
      --text:#17202a; --muted:#65717e; --line:#d9e0e7; --accent:#356ae6;
      --success:#157a4b; --success-bg:#e8f7ef; --warning:#9a5b00; --warning-bg:#fff3d6;
      --error:#b4232d; --error-bg:#fdebed; --shadow:0 12px 32px rgba(18,34,52,.08); }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#11161c; --surface:#1a2129;
      --surface2:#202933; --text:#e9eef3; --muted:#a4afba; --line:#36414d;
      --accent:#77a0ff; --success:#65d39d; --success-bg:#17382b; --warning:#ffc867;
      --warning-bg:#443417; --error:#ff8f98; --error-bg:#482126; --shadow:none; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
      font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:32px auto 72px; }}
    header {{ display:flex; gap:24px; align-items:flex-start; justify-content:space-between;
      margin-bottom:20px; }} h1 {{ margin:4px 0 4px; font-size:clamp(28px,4vw,44px); line-height:1.05; }}
    h2 {{ margin:0 0 16px; font-size:21px; }} h3 {{ margin:0 0 4px; font-size:16px; }}
    p {{ margin:0; }} .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.09em;
      font-size:12px; font-weight:750; }} .meta {{ color:var(--muted); }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    button {{ border:1px solid var(--line); background:var(--surface); color:var(--text); padding:9px 13px;
      border-radius:9px; cursor:pointer; font:inherit; }} button:hover {{ border-color:var(--accent); }}
    .hero {{ background:var(--surface); border:1px solid var(--line); border-radius:18px;
      padding:22px; box-shadow:var(--shadow); margin-bottom:18px; }}
    .hero-top {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
    .state {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:6px 10px;
      font-weight:700; }} .state::before {{ content:""; width:9px; height:9px; border-radius:50%; background:currentColor; }}
    .state.success {{ color:var(--success); background:var(--success-bg); }}
    .state.warning {{ color:var(--warning); background:var(--warning-bg); }}
    .state.error {{ color:var(--error); background:var(--error-bg); }}
    .summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:20px; }}
    .stat {{ background:var(--surface2); border:1px solid var(--line); border-radius:12px; padding:14px; }}
    .stat span {{ display:block; color:var(--muted); font-size:12px; }} .stat strong {{ font-size:23px; }}
    section, details.section {{ background:var(--surface); border:1px solid var(--line); border-radius:16px;
      padding:20px; margin-top:18px; box-shadow:var(--shadow); }}
    details.section > summary {{ cursor:pointer; font-size:20px; font-weight:750; }}
    details.section[open] > summary {{ margin-bottom:16px; }}
    .issues {{ display:grid; gap:10px; }} .issue {{ display:grid; grid-template-columns:18px 1fr; gap:12px;
      padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--surface2); }}
    .issue-icon {{ width:11px; height:11px; border-radius:50%; margin-top:6px; background:var(--muted); }}
    .issue.error .issue-icon {{ background:var(--error); }} .issue.warning .issue-icon {{ background:var(--warning); }}
    .issue.info .issue-icon {{ background:var(--accent); }} .issue p {{ color:var(--muted); }}
    .issue .action {{ color:var(--text); margin-top:6px; }}
    .empty-state {{ display:flex; flex-direction:column; gap:3px; padding:18px; border-radius:12px;
      background:var(--success-bg); color:var(--success); }}
    .graph-wrap {{ overflow-x:auto; }} .camera-graph {{ width:100%; min-width:720px; height:auto; }}
    .edge line {{ stroke:var(--line); stroke-width:2; }} .edge.strong line {{ stroke:var(--success); stroke-width:3; }}
    .edge text {{ fill:var(--muted); font-size:12px; paint-order:stroke; stroke:var(--surface); stroke-width:5; }}
    .node rect {{ fill:var(--surface2); stroke:var(--line); stroke-width:2; }}
    .node.anchor rect {{ stroke:var(--accent); }} .node.synced rect {{ stroke:var(--success); }}
    .node.skipped rect {{ stroke:var(--error); }} .node.disabled {{ opacity:.55; }}
    .node text {{ fill:var(--text); }} .node-name {{ font-size:13px; font-weight:700; }}
    .node-state {{ font-size:11px; fill:var(--muted)!important; }} .legend {{ color:var(--muted); text-align:center; }}
    .swatch {{ display:inline-block; width:22px; height:3px; vertical-align:middle; margin:0 6px 2px 15px;
      background:var(--line); }} .swatch.strong {{ background:var(--success); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    table {{ border-collapse:collapse; width:100%; min-width:720px; }} th,td {{ padding:10px 12px;
      border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:var(--surface2); color:var(--muted); font-size:12px;
      text-transform:uppercase; letter-spacing:.04em; }} tbody tr:last-child td {{ border-bottom:0; }}
    tbody tr:hover {{ background:var(--surface2); }} .number {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700;
      background:var(--surface2); border:1px solid var(--line); }} .pill.synced,.pill.anchor {{ color:var(--success); }}
    .pill.skipped {{ color:var(--error); }} .pill.warning {{ color:var(--warning); }}
    .lock {{ color:var(--accent); font-size:11px; margin-left:7px; }}
    .search {{ width:min(420px,100%); border:1px solid var(--line); background:var(--surface2);
      color:var(--text); padding:10px 12px; border-radius:9px; margin-bottom:12px; font:inherit; }}
    .mini-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .mini-card {{ display:flex; justify-content:space-between; gap:12px; padding:12px; background:var(--surface2);
      border:1px solid var(--line); border-radius:10px; }} .mini-card span,.muted {{ color:var(--muted); }}
    .notes {{ margin:0; padding-left:22px; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--surface2);
      border:1px solid var(--line); border-radius:10px; padding:14px; color:var(--muted); }}
    footer {{ color:var(--muted); text-align:center; margin-top:24px; font-size:12px; }}
    @media (max-width:760px) {{ main {{ width:min(100% - 20px,1180px); margin-top:18px; }} header {{ display:block; }}
      .toolbar {{ justify-content:flex-start; margin-top:14px; }} .summary-grid,.mini-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media print {{ body {{ background:#fff; color:#111; }} main {{ width:100%; margin:0; }} .toolbar {{ display:none; }}
      section,details.section,.hero {{ break-inside:avoid; box-shadow:none; }} details.section:not([open]) > * {{ display:block; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">Perspective Match · {escape(report.operation)}</div>
      <h1>{escape(report.outcome)}</h1><p class="meta">{escape(report.source_name)} · {escape(report.generated_at)}</p></div>
    <div class="toolbar"><button type="button" id="copy-report">Copy solver message</button>
      <button type="button" onclick="window.print()">Print / Save PDF</button></div>
  </header>
  <div class="hero"><div class="hero-top"><span class="state {escape(report.severity)}">{escape(severity_label)}</span>
    <span class="meta">{report.attention_count} item{'s' if report.attention_count != 1 else ''} need attention</span></div>
    <div class="summary-grid">
      <div class="stat"><span>Cameras registered</span><strong>{report.registered_matches}/{report.enabled_matches}</strong></div>
      <div class="stat"><span>Overall RMSE</span><strong>{report.rmse_px:.2f}px</strong></div>
      <div class="stat"><span>Landmarks measured</span><strong>{len(report.landmarks)}</strong></div>
      <div class="stat"><span>Pose locks</span><strong>{report.locked_matches}</strong></div>
    </div>
  </div>
  <section><h2>Needs attention</h2><div class="issues">{_render_issues(report)}</div></section>
  <section><h2>Camera connectivity</h2><p class="meta" style="margin-bottom:12px">
    The overlap backbone shows ordinary point landmarks shared between photos. Metric 2D↔3D
    and Known 3D line routes are summarized in the match table and issue cards.</p>
    <div class="graph-wrap">{_svg_graph(report)}</div></section>
  <section><h2>Matches</h2><div class="table-wrap"><table><thead><tr><th>Match</th><th>Status</th>
    <th class="number">RMSE</th><th class="number">Point picks</th><th class="number">Lines</th>
    <th class="number">Usable 3D</th><th>Best registered edge</th></tr></thead><tbody>
    {_render_match_rows(report)}</tbody></table></div></section>
  <section><h2>Landmark errors</h2><input class="search" id="landmark-search" type="search"
    placeholder="Filter landmarks or matches…" aria-label="Filter landmarks">
    <div class="table-wrap"><table id="landmark-table"><thead><tr><th>Landmark</th><th>Kind</th>
    <th class="number"><button type="button" id="sort-rmse">RMSE ↓</button></th><th>Observed in</th></tr></thead>
    <tbody>{_render_landmark_rows(report)}</tbody></table></div></section>
  <details class="section" open><summary>Constraint inventory</summary><div class="mini-grid">{constraint_cards}</div>
    <p class="meta" style="margin-top:12px">{report.disabled_matches} match(es) sync-disabled ·
    {report.excluded_landmarks} landmark(s) excluded from sync ·
    {'Joint bundle adjustment ran' if report.bundle_adjusted else 'No joint bundle adjustment recorded'}</p></details>
  <details class="section"><summary>Run notes</summary>{notes_block}</details>
  <details class="section"><summary>Technical solver message</summary><pre id="technical-message">{escape(report.raw_message)}</pre></details>
  <footer>Generated locally by Perspective Match. No report data was uploaded.</footer>
</main>
<script>
  (() => {{
    const search = document.getElementById('landmark-search');
    const tableBody = document.querySelector('#landmark-table tbody');
    if (search && tableBody) {{
      search.addEventListener('input', () => {{
        const query = search.value.trim().toLocaleLowerCase();
        for (const row of tableBody.rows) {{
          row.hidden = query && !row.dataset.search.includes(query);
        }}
      }});
    }}
    let descending = true;
    document.getElementById('sort-rmse')?.addEventListener('click', event => {{
      const rows = [...tableBody.rows];
      descending = !descending;
      rows.sort((a,b) => (Number(a.dataset.rmse || 0) - Number(b.dataset.rmse || 0)) * (descending ? -1 : 1));
      rows.forEach(row => tableBody.appendChild(row));
      event.currentTarget.textContent = `RMSE ${{descending ? '↓' : '↑'}}`;
    }});
    document.getElementById('copy-report')?.addEventListener('click', async event => {{
      const text = document.getElementById('technical-message')?.textContent || '';
      try {{ await navigator.clipboard.writeText(text); event.currentTarget.textContent = 'Copied'; }}
      catch (_error) {{ event.currentTarget.textContent = 'Select text below to copy'; }}
    }});
  }})();
</script>
</body>
</html>
"""
    return html_text


def _temp_report_directory(temp_root: str | Path | None = None) -> Path:
    if temp_root is None:
        base = Path(tempfile.gettempdir())
    else:
        base = Path(temp_root)
    return base / "match-perspective" / "reports" / f"blender-{os.getpid()}"


def write_temp_report(
    report: SyncDiagnosticReport,
    *,
    temp_root: str | Path | None = None,
    keep: int = _REPORT_KEEP_COUNT,
) -> Path:
    """Write one unique report and retain only recent reports for this process."""
    global _last_report_path
    directory = _temp_report_directory(temp_root)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    operation = "".join(
        character.lower() if character.isalnum() else "-"
        for character in report.operation
    ).strip("-") or "sync"
    path = directory / f"{operation}-{stamp}.html"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(render_sync_report_html(report), encoding="utf-8")
    temporary.replace(path)
    _last_report_path = path

    candidates = sorted(
        directory.glob("*.html"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for stale in candidates[max(int(keep), 1) :]:
        try:
            stale.unlink()
        except OSError:
            pass
    return path


def last_report_path() -> Path | None:
    """Newest report produced in this loaded add-on process, if it still exists."""
    if _last_report_path is None or not _last_report_path.is_file():
        return None
    return _last_report_path


def clear_last_report() -> None:
    """Forget the last report without deleting the browser's temporary file."""
    global _last_report_path
    _last_report_path = None


def export_last_report(destination: str | Path) -> Path:
    """Copy the current self-contained report to a user-selected permanent path."""
    source = last_report_path()
    if source is None:
        raise FileNotFoundError("No sync diagnostic report is available")
    output = Path(destination).expanduser()
    if output.suffix.lower() != ".html":
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return output
