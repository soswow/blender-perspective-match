"""Self-contained HTML reports for landmark-sync diagnostics.

The renderer is deliberately bpy-free so its model and markup can be tested by
normal Python. Blender-specific operators supply scene data, choose the temp
root, and open the resulting ``file://`` URL on the main thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


PAIR_POINT_REQUIREMENT = 5
METRIC_POINT_REQUIREMENT = 3
HIGH_ERROR_PX = 8.0


def _shared_point_summary(count: int) -> str:
    """Describe pairwise point support without making the threshold a fraction."""
    if count < PAIR_POINT_REQUIREMENT:
        return f"{count} of {PAIR_POINT_REQUIREMENT} shared"
    return f"{count} shared; minimum {PAIR_POINT_REQUIREMENT}"


def _edge_tooltip(name_a: str, name_b: str, shared_points: int) -> str:
    """Explain one overlap link for the camera-graph hover card."""
    if shared_points >= PAIR_POINT_REQUIREMENT:
        support = (
            f"{shared_points} shared ordinary points "
            f"(pairwise 2D↔2D needs {PAIR_POINT_REQUIREMENT})."
        )
    else:
        missing = PAIR_POINT_REQUIREMENT - shared_points
        support = (
            f"{shared_points} of {PAIR_POINT_REQUIREMENT} shared ordinary points. "
            f"Add {missing} more for pairwise 2D↔2D."
        )
    return f"{name_a} ↔ {name_b}\n{support}"


_REPORT_KEEP_COUNT = 10
_last_report_path: Path | None = None
_last_report: SyncDiagnosticReport | None = None
_last_report_scene_uid: int | None = None
_last_report_request_sha256: str = ""


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
    point_rmse_px: float | None
    line_rmse_px: float | None
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
    weak_line: bool = False
    line_support_angle_deg: float | None = None
    plane_seeded: bool = False


@dataclass(frozen=True)
class ReportEdge:
    """Point-overlap edge in the camera graph."""

    match_a: str
    match_b: str
    shared_points: int


@dataclass(frozen=True)
class ReportCommonLeaveOneOut:
    """One independent candidate compared on the same surviving evidence."""

    name: str
    kind: str
    removed_relations: tuple[str, ...]
    baseline_objective: float | None
    candidate_objective: float | None
    baseline_point_rmse_px: float | None
    candidate_point_rmse_px: float | None
    baseline_line_rmse_px: float | None
    candidate_line_rmse_px: float | None
    baseline_valid: bool
    candidate_valid: bool
    reason: str
    support: str


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
    rmse_px: float | None
    point_rmse_px: float | None = None
    line_rmse_px: float | None = None
    applied: bool = True
    application_state: str = ""
    stale: bool = False
    evaluation_note: str = ""
    focal_summary: list[str] = field(default_factory=list)
    common_leave_one_out: list[ReportCommonLeaveOneOut] = field(default_factory=list)
    common_leave_one_out_ranked: int = 0
    common_leave_one_out_incomplete: bool = False
    common_leave_one_out_reason: str = ""
    common_leave_one_out_baseline_support: str = ""
    joint_support_summary: str = ""
    joint_support_partial: bool = False
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


def _finite_score(value: object, valid: bool) -> float | None:
    if not valid or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relation_label(value: object, names: Mapping[str, str]) -> str:
    return ": ".join(names.get(part, part) for part in str(value).split(":"))


def _coverage_summary(
    coverage: Mapping[str, object] | None,
    names: Mapping[str, str],
    labels: Mapping[str, str],
) -> str:
    if not coverage:
        return "Coverage unavailable"
    point_fit = int(coverage.get("fitted_point_picks", 0))
    point_requested = int(coverage.get("requested_point_picks", 0))
    line_fit = int(coverage.get("fitted_line_strokes", 0))
    line_requested = int(coverage.get("requested_line_strokes", 0))
    bits = [
        f"{point_fit}/{point_requested} point picks fitted",
        f"{line_fit}/{line_requested} line strokes fitted",
    ]
    for key, title, formatter in (
        ("skipped_camera_ids", "skipped cameras", lambda value: labels.get(value, value)),
        ("skipped_point_ids", "skipped points", lambda value: names.get(value, value)),
        ("skipped_line_ids", "skipped lines", lambda value: names.get(value, value)),
        ("skipped_relation_ids", "skipped relations", lambda value: _relation_label(value, names)),
    ):
        values = coverage.get(key) or ()
        if values:
            bits.append(f"{title}: " + ", ".join(formatter(str(value)) for value in values))
    return "; ".join(bits)


def _joint_coverage_summary(
    coverage: Mapping[str, object], names: Mapping[str, str],
    labels: Mapping[str, str], enabled_ids: Sequence[str],
) -> str:
    skipped = {str(item) for item in coverage.get("skipped_camera_ids", ())}
    fitted = len(set(enabled_ids) - skipped)
    return (
        f"{fitted}/{len(enabled_ids)} cameras fitted; "
        + _coverage_summary(coverage, names, labels)
    )


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
    applied: bool = True,
    application_state: str = "",
    evaluation_note: str = "",
    calibrations: Mapping[str, object] | None = None,
    plane_groups: Sequence[object] = (),
) -> SyncDiagnosticReport:
    """Build a report model from one solver result and its frozen inputs."""
    known_world = known_world or {}
    known_lines = known_lines or {}
    labels = dict(all_match_labels or {})
    enabled_ids = _match_ids(matches)
    disabled_ids = {str(item) for item in disabled_match_ids}
    fixed_ids = {str(item) for item in fixed_match_ids}
    registered_ids = {str(item) for item in getattr(result, "similarities", {})}
    if result is None:
        registered_ids = set()
    elif not bool(getattr(result, "success", False)):
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
        elif result is None:
            status = "unassessed"
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
        point_errors = getattr(result, "per_match_point_rmse_px", None)
        point_rmse = (point_errors.get(match_id) if point_errors is not None else rmse)
        line_rmse = getattr(result, "per_match_line_rmse_px", {}).get(match_id)
        report_matches.append(
            ReportMatch(
                match_id=match_id,
                label=labels[match_id],
                status=status,
                rmse_px=None if rmse is None else float(rmse),
                point_rmse_px=None if point_rmse is None else float(point_rmse),
                line_rmse_px=None if line_rmse is None else float(line_rmse),
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
    joint_coverage = getattr(result, "joint_support_coverage", None)
    joint_support_partial = bool(joint_coverage) and (
        int(joint_coverage.get("fitted_point_picks", 0))
        < int(joint_coverage.get("requested_point_picks", 0))
        or int(joint_coverage.get("fitted_line_strokes", 0))
        < int(joint_coverage.get("requested_line_strokes", 0))
        or any(joint_coverage.get(key) for key in (
            "skipped_camera_ids", "skipped_point_ids", "skipped_line_ids",
            "skipped_relation_ids",
        ))
    )
    joint_support_summary = (
        _joint_coverage_summary(joint_coverage, names, labels, enabled_ids)
        if joint_coverage else ""
    )
    common = getattr(result, "common_leave_one_out", None)
    common_items = []
    if common is not None:
        for item in getattr(common, "items", ()):
            baseline_valid = bool(getattr(item, "baseline_valid", False))
            candidate_valid = bool(getattr(item, "candidate_valid", False))
            common_items.append(ReportCommonLeaveOneOut(
                name=str(getattr(item, "landmark_name", "") or
                         names.get(str(getattr(item, "landmark_id", "")), "Unnamed landmark")),
                kind=str(getattr(item, "kind", "landmark")),
                removed_relations=tuple(
                    _relation_label(relation, names)
                    for relation in getattr(item, "removed_relations", ())
                ),
                baseline_objective=_finite_score(getattr(item, "baseline_objective", None), baseline_valid),
                candidate_objective=_finite_score(getattr(item, "candidate_objective", None), candidate_valid),
                baseline_point_rmse_px=_finite_score(getattr(item, "baseline_point_rmse_px", None), baseline_valid),
                candidate_point_rmse_px=_finite_score(getattr(item, "candidate_point_rmse_px", None), candidate_valid),
                baseline_line_rmse_px=_finite_score(getattr(item, "baseline_line_rmse_px", None), baseline_valid),
                candidate_line_rmse_px=_finite_score(getattr(item, "candidate_line_rmse_px", None), candidate_valid),
                baseline_valid=baseline_valid,
                candidate_valid=candidate_valid,
                reason=str(getattr(item, "reason", "") or ""),
                support=_coverage_summary(getattr(item, "support_coverage", None), names, labels),
            ))
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
    weak_lines = set(getattr(result, "weak_line_ids", ()))
    plane_seeded = set(getattr(result, "plane_seeded_landmark_ids", ()))
    support_angles = getattr(result, "line_support_angles_deg", {})
    report_landmarks = [
        ReportLandmark(
            landmark_id=str(landmark_id),
            name=names.get(str(landmark_id), str(landmark_id)[:8]),
            kind="point" if str(landmark_id) in point_ids else "line",
            rmse_px=float(rmse),
            matches=tuple(sorted(matches_by_landmark.get(str(landmark_id), set()))),
            downweighted=str(landmark_id) in downweighted,
            weak_line=str(landmark_id) in weak_lines,
            line_support_angle_deg=support_angles.get(str(landmark_id)),
            plane_seeded=str(landmark_id) in plane_seeded,
        )
        for landmark_id, rmse in getattr(result, "per_landmark_rmse_px", {}).items()
    ]
    report_landmarks.sort(key=lambda item: (-item.rmse_px, item.name.casefold()))

    skipped = [item for item in report_matches if item.status == "skipped"]
    success = bool(getattr(result, "success", False))
    if result is None:
        outcome, severity = f"{operation} not applied", "warning"
    elif not success:
        outcome, severity = "Sync failed", "error"
    elif skipped:
        outcome, severity = "Partial sync", "warning"
    elif float(getattr(result, "mean_reprojection_px", 0.0)) > HIGH_ERROR_PX:
        outcome, severity = "Sync completed with high error", "warning"
    elif weak_lines:
        outcome, severity = "Sync completed with weak line geometry", "warning"
    else:
        outcome, severity = "Sync complete", "success"
    if operation == "Use Best Fit" and applied:
        outcome, severity = "Provisional fit applied", "warning"
    elif operation == "Refine Lenses" and applied and outcome == "Sync complete":
        outcome = "Refine complete"
    if not applied:
        if operation == "Investigate Problems":
            outcome = "Investigation complete" if success else "Investigation found no solution"
        elif operation == "Refine Lenses":
            outcome = ("Lens settings applied; Sync not applied"
                       if application_state else "Refine not applied")
        elif operation == "Solve Sync":
            outcome = "Sync refused"

    issues: list[ReportIssue] = []
    if joint_support_partial:
        issues.append(ReportIssue(
            "warning", "Joint fit used partial evidence", joint_support_summary,
            "Review the omitted features and constraints before treating the fit as complete.",
        ))
        severity = "warning" if severity == "success" else severity
    joint_refusal_reason = str(getattr(result, "joint_refusal_reason", "") or "")
    if joint_refusal_reason:
        issues.append(ReportIssue(
            "warning", "Joint fit was refused", joint_refusal_reason,
            ("The report describes the retained camera result and why the joint fit was not used."
             if success else "No joint camera geometry was applied; review the refusal before trying again."),
        ))
        severity = "warning" if severity == "success" else severity
    if plane_seeded:
        issues.append(ReportIssue(
            "info", "Some points use a plane to determine depth",
            ", ".join(names.get(key,key) for key in sorted(plane_seeded)),
            "These points use one Solve or Lock Pose view and your shared-plane constraint. "
            "Their pixel error does not independently verify depth. A pick in another permitted view "
            "can check the position against additional image evidence.",
        ))
    if weak_lines:
        details = ", ".join(
            f"{names.get(key,key)} ({float(support_angles[key]):.1f}° support)"
            if key in support_angles else names.get(key,key)
            for key in sorted(weak_lines)
        )
        issues.append(ReportIssue(
            "warning", "Some 3D lines are sensitive to small stroke edits", details,
            "Their strokes and constraints supply nearly coincident supporting planes, including reflected views for mirror pairs. "
            "A low pixel error does not establish precise 3D position or direction. "
            "Try longer strokes or an additional stroke from a more distinct viewing angle. "
            "That camera must use Solve or Lock Pose to contribute to 3D. "
            "This is a geometric weakness check, not a confidence interval.",
        ))
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

    raw_rmse = getattr(result, "mean_reprojection_px", None)
    rmse_px = None if raw_rmse is None else float(raw_rmse)
    if rmse_px is not None and rmse_px > HIGH_ERROR_PX:
        worst = ", ".join(
            f"{item.name} {item.rmse_px:.0f}px"
            for item in [entry for entry in report_landmarks if entry.kind == "point"][:3]
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
        if common is None and float(without_rmse) < float(with_rmse)
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

    if common is not None and bool(getattr(common, "incomplete", False)):
        reason = str(getattr(common, "reason", "") or "Some candidate removals were not evaluated")
        issues.append(ReportIssue(
            "warning", "Investigation incomplete", reason,
            "An untested removal has no result in this report; review the available evidence before changing picks or constraints.",
        ))
        severity = "warning" if severity == "success" else severity
        if operation == "Investigate Problems":
            outcome = "Investigation incomplete"

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
        "Plane memberships": len(plane_groups),
    }

    focal_summary = []
    for match_id, calibration in (calibrations or {}).items():
        hfov = getattr(calibration, "hfov_degrees", None)
        if hfov is not None:
            focal_summary.append(f"{labels.get(match_id, friendly_match_name(match_id))}: {float(hfov):.2f}° HFOV")
    focal_summary.sort()
    point_rmse = getattr(result, "point_rmse_px", None)
    line_rmse = getattr(result, "line_rmse_px", None)
    if not hasattr(result, "point_rmse_px") and observations and rmse_px is not None:
        point_rmse = rmse_px

    return SyncDiagnosticReport(
        operation=str(operation),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source_name=str(source_name or "Untitled.blend"),
        outcome=outcome,
        severity=severity,
        enabled_matches=len(enabled_ids),
        registered_matches=len(registered_ids & set(enabled_ids)),
        rmse_px=rmse_px,
        point_rmse_px=None if point_rmse is None else float(point_rmse),
        line_rmse_px=None if line_rmse is None else float(line_rmse),
        applied=bool(applied),
        application_state=str(application_state),
        evaluation_note=str(evaluation_note),
        focal_summary=focal_summary,
        common_leave_one_out=common_items,
        common_leave_one_out_ranked=max(int(getattr(common, "candidates_ranked", 0)), 0),
        common_leave_one_out_incomplete=bool(getattr(common, "incomplete", False)),
        common_leave_one_out_reason=str(getattr(common, "reason", "") or ""),
        common_leave_one_out_baseline_support=_coverage_summary(
            getattr(common, "baseline_coverage", None), names, labels
        ) if common is not None else "",
        joint_support_summary=joint_support_summary,
        joint_support_partial=joint_support_partial,
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
        _format_error(report.rmse_px),
    ]
    if report.attention_count:
        bits.append(
            f"{report.attention_count} issue"
            f"{'s' if report.attention_count != 1 else ''}"
        )
    bits.append("Report opened" if opened else "Report ready")
    if report.stale:
        bits.append("Stale")
    return " · ".join(bits)


def _format_error(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}px"


def _status_label(status: str) -> str:
    return {
        "anchor": "Anchor",
        "synced": "Synced",
        "skipped": "Skipped",
        "disabled": "Disabled",
        "unassessed": "Not assessed",
    }.get(status, status.title())


def _status_rank(status: str) -> int:
    return {"anchor": 0, "synced": 1, "skipped": 2, "disabled": 3,
            "unassessed": 4}.get(status, 9)


def _vendor_js(name: str) -> str:
    return (_VENDOR_DIR / name).read_text(encoding="utf-8")


def _html_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c")


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


def _camera_layout_positions(
    match_ids: Sequence[str],
    edges: Sequence[ReportEdge],
    *,
    anchor_id: str | None,
) -> dict[str, tuple[float, float]]:
    """Pack the overlap forest into rows so the viewport can fit readable nodes."""
    adjacency: dict[str, list[str]] = {item: [] for item in match_ids}
    for edge in edges:
        if edge.match_a in adjacency and edge.match_b in adjacency:
            adjacency[edge.match_a].append(edge.match_b)
            adjacency[edge.match_b].append(edge.match_a)
    for neighbors in adjacency.values():
        neighbors.sort()

    seen: set[str] = set()
    components: list[list[list[str]]] = []
    seeds = [anchor_id] if anchor_id in adjacency else []
    seeds.extend(item for item in match_ids if item not in seeds)
    for seed in seeds:
        if seed in seen or seed not in adjacency:
            continue
        layers: list[list[str]] = [[seed]]
        depth = {seed: 0}
        seen.add(seed)
        queue = [seed]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_depth = depth[current] + 1
                depth[neighbor] = next_depth
                while len(layers) <= next_depth:
                    layers.append([])
                layers[next_depth].append(neighbor)
                queue.append(neighbor)
        rows: list[list[str]] = []
        for layer in layers:
            for start in range(0, len(layer), 6):
                rows.append(layer[start : start + 6])
        components.append(rows or [[seed]])

    node_width, node_height, gap_x, gap_y = 200.0, 68.0, 32.0, 52.0
    positions: dict[str, tuple[float, float]] = {}
    x_cursor = 0.0
    for layers in components:
        column_width = max(
            len(layer) * node_width + max(len(layer) - 1, 0) * gap_x for layer in layers
        )
        for depth_index, layer in enumerate(layers):
            row_width = len(layer) * node_width + max(len(layer) - 1, 0) * gap_x
            start_x = x_cursor + (column_width - row_width) / 2.0 + node_width / 2.0
            y_coord = depth_index * (node_height + gap_y)
            for index, match_id in enumerate(layer):
                positions[match_id] = (start_x + index * (node_width + gap_x), y_coord)
        x_cursor += column_width + 72.0
    if positions:
        xs = [point[0] for point in positions.values()]
        ys = [point[1] for point in positions.values()]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        positions = {
            key: (x_coord - center_x, y_coord - center_y)
            for key, (x_coord, y_coord) in positions.items()
        }
    return positions


def _camera_graph_payload(report: SyncDiagnosticReport) -> dict[str, object]:
    anchor_id = next(
        (item.match_id for item in report.matches if item.status == "anchor"),
        None,
    )
    backbone = _backbone_edges(report)
    positions = _camera_layout_positions(
        [item.match_id for item in report.matches],
        backbone,
        anchor_id=anchor_id,
    )
    nodes = []
    for item in report.matches:
        short_label = item.label if len(item.label) <= 24 else item.label[:21] + "…"
        lock = " · locked" if item.locked else ""
        x_coord, y_coord = positions.get(item.match_id, (0.0, 0.0))
        nodes.append(
            {
                "id": item.match_id,
                "label": f"{short_label}\n{_status_label(item.status)}{lock}",
                "status": item.status,
                "locked": item.locked,
                "x": round(x_coord, 1),
                "y": round(y_coord, 1),
            }
        )
    labels = {item.match_id: item.label for item in report.matches}
    edges = []
    for edge in backbone:
        edges.append(
            {
                "id": f"{edge.match_a}--{edge.match_b}",
                "source": edge.match_a,
                "target": edge.match_b,
                "shared": edge.shared_points,
                "strong": edge.shared_points >= PAIR_POINT_REQUIREMENT,
                "tooltip": _edge_tooltip(
                    labels.get(edge.match_a, edge.match_a),
                    labels.get(edge.match_b, edge.match_b),
                    edge.shared_points,
                ),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _render_camera_graph(report: SyncDiagnosticReport) -> str:
    if not report.matches:
        return '<p class="muted">No camera graph available.</p>'
    height = min(720, max(380, 52 * len(report.matches)))
    return (
        f'<div class="graph-wrap"><div id="camera-graph" class="camera-graph" '
        f'style="height:{height}px" role="img" aria-label="Camera overlap graph">'
        "</div><img id=\"camera-graph-print\" alt=\"\"></div>"
        f'<script type="application/json" id="camera-graph-data">'
        f"{_html_json(_camera_graph_payload(report))}</script>"
        '<div id="camera-graph-tooltip" class="graph-tooltip" role="tooltip" hidden></div>'
        '<div class="legend"><p>Numbers on the links are ordinary point landmarks shared by those '
        "two photos. Only the strongest loop-free overlap path is drawn.</p>"
        '<ul class="legend-keys">'
        '<li><span class="swatch strong"></span><strong>Green</strong> — 5 or more shared points, '
        "enough for pairwise 2D↔2D registration.</li>"
        '<li><span class="swatch weak"></span><strong>Gray</strong> — 1–4 shared points, below that '
        "minimum on its own.</li></ul>"
        "<p>Hover a number for the two match names. Drag a camera to move it; scroll to zoom.</p></div>"
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
        rmse_sort = "" if item.point_rmse_px is None else f"{item.point_rmse_px:.8f}"
        point_rmse = _format_error(item.point_rmse_px)
        line_rmse = _format_error(item.line_rmse_px)
        best = (
            f"{escape(item.best_reference)} · "
            f"{escape(_shared_point_summary(item.best_shared_points))}"
            if item.best_reference
            else "—"
        )
        rows.append(
            f'<tr data-label="{escape(item.label.casefold(), quote=True)}" '
            f'data-status="{_status_rank(item.status)}" data-rmse="{rmse_sort}" '
            f'data-points="{item.point_picks}" data-lines="{item.line_picks}" '
            f'data-usable="{item.usable_3d_points}">'
            f'<td><strong>{escape(item.label)}</strong>'
            f'{"<span class=\"lock\">Locked</span>" if item.locked else ""}</td>'
            f'<td><span class="pill {escape(item.status)}">'
            f'{escape(_status_label(item.status))}</span></td>'
            f'<td class="number">{point_rmse}</td><td class="number">{line_rmse}</td>'
            f'<td class="number">{item.point_picks}</td>'
            f'<td class="number">{item.line_picks}</td><td class="number">'
            f'{item.usable_3d_points}</td><td>{best}</td></tr>'
        )
    return "".join(rows)


def _render_landmark_rows(report: SyncDiagnosticReport) -> str:
    rows = []
    for item in report.landmarks:
        status = '<span class="pill warning">Downweighted</span>' if item.downweighted else ""
        if item.weak_line:
            angle = f" ({item.line_support_angle_deg:.1f}°)" if item.line_support_angle_deg is not None else ""
            status += f'<span class="pill warning">Weak 3D support{angle}</span>'
        if item.plane_seeded:
            status += '<span class="pill">Plane + one view</span>'
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


def _render_common_leave_one_out(report: SyncDiagnosticReport) -> str:
    if not report.common_leave_one_out_baseline_support:
        return ""
    cards = []
    for item in report.common_leave_one_out:
        relations = ", ".join(item.removed_relations) if item.removed_relations else "None"
        baseline_state = "valid" if item.baseline_valid else "refused"
        candidate_state = "valid" if item.candidate_valid else "refused"
        reason = (f'<p class="meta">Reason: {escape(item.reason)}</p>'
                  if item.reason else "")
        cards.append(
            '<article class="loo-item">'
            f'<h3>{escape(item.name)} <span class="pill">{escape(item.kind.title())}</span></h3>'
            '<div class="mini-grid">'
            f'<div class="mini-card"><span>Combined objective</span><strong>{_format_objective(item.baseline_objective)} → {_format_objective(item.candidate_objective)}</strong></div>'
            f'<div class="mini-card"><span>Point RMSE</span><strong>{_format_error(item.baseline_point_rmse_px)} → {_format_error(item.candidate_point_rmse_px)}</strong></div>'
            f'<div class="mini-card"><span>Line RMSE</span><strong>{_format_error(item.baseline_line_rmse_px)} → {_format_error(item.candidate_line_rmse_px)}</strong></div>'
            '</div>'
            f'<p class="meta">Before: {baseline_state} · After: {candidate_state}</p>'
            f'<p class="meta">Removed relations: {escape(relations)}</p>'
            f'<p class="meta">Remaining support: {escape(item.support)}</p>'
            f'{reason}</article>'
        )
    incomplete = (
        f'<p class="issue warning">Incomplete: {escape(report.common_leave_one_out_reason or "Some removals were not evaluated")}</p>'
        if report.common_leave_one_out_incomplete else ""
    )
    body = "".join(cards) if cards else '<p class="muted">No removals were evaluated.</p>'
    return (
        '<section><h2>Independent removal checks</h2>'
        '<p class="meta">Before and after score the same surviving point and line evidence. '
        'The combined objective also includes constraints. Removing a feature can remove its defining relations. '
        'A lower score shows a different fit on this reduced model; it does not prove a pick or constraint is wrong.</p>'
        f'<p class="meta">Baseline support: {escape(report.common_leave_one_out_baseline_support)} · '
        f'{report.common_leave_one_out_ranked} ranked removal(s)</p>'
        f'{incomplete}<div class="loo-items">{body}</div></section>'
    )


def _format_objective(value: float | None) -> str:
    return "—" if value is None else f"{value:.5g}"


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
    joint_support_block = (
        f'<p class="meta" style="margin-top:12px">Joint fit coverage: {escape(report.joint_support_summary)}</p>'
        if report.joint_support_summary else ""
    )
    application = (report.application_state or
                   ("Applied to this scene" if report.applied else "Not applied to this scene"))
    if report.stale:
        application += " · Stale: scene inputs or cameras changed after this report"
    if report.evaluation_note:
        application += " · " + report.evaluation_note
    focal_block = ""
    if report.focal_summary:
        focal_block = (
            '<details class="section" open><summary>Camera FOV in this result</summary><ul class="notes">'
            + "".join(f"<li>{escape(item)}</li>" for item in report.focal_summary)
            + "</ul></details>"
        )
    common_leave_one_out_block = _render_common_leave_one_out(report)
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
    .graph-wrap {{ overflow:hidden; }} .camera-graph {{ width:100%; min-height:360px; border:1px solid var(--line);
      border-radius:12px; background:var(--surface2); }}
    #camera-graph-print {{ display:none; width:100%; }}
    .graph-tooltip {{ position:fixed; z-index:5; max-width:280px; padding:8px 10px; border-radius:8px;
      border:1px solid var(--line); background:var(--surface); color:var(--text); font-size:12px;
      line-height:1.4; white-space:pre-wrap; box-shadow:var(--shadow); pointer-events:none; }}
    .legend {{ color:var(--muted); margin-top:12px; }} .legend p {{ margin:0 0 8px; }}
    .legend-keys {{ list-style:none; margin:0 0 8px; padding:0; }} .legend-keys li {{ margin:4px 0; }}
    .swatch {{ display:inline-block; width:22px; height:3px; vertical-align:middle; margin:0 8px 2px 0;
      background:var(--line); }} .swatch.strong {{ background:var(--success); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    table {{ border-collapse:collapse; width:100%; min-width:720px; }} th,td {{ padding:10px 12px;
      border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:var(--surface2); color:var(--muted); font-size:12px;
      text-transform:uppercase; letter-spacing:.04em; }} tbody tr:last-child td {{ border-bottom:0; }}
    th button {{ padding:4px 8px; font:inherit; font-size:12px; font-weight:700; text-transform:uppercase;
      letter-spacing:.04em; background:transparent; }}
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
    .loo-items {{ display:grid; gap:12px; margin-top:14px; }} .loo-item {{ border:1px solid var(--line);
      border-radius:12px; padding:14px; background:var(--surface2); }} .loo-item h3 {{ margin:0 0 10px; }}
    .loo-item .mini-grid {{ margin-bottom:10px; }} .loo-item .mini-card {{ flex-direction:column; gap:3px; }}
    .loo-item .mini-card strong {{ font-variant-numeric:tabular-nums; }}
    .loo-item p {{ margin-top:5px; }}
    .notes {{ margin:0; padding-left:22px; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--surface2);
      border:1px solid var(--line); border-radius:10px; padding:14px; color:var(--muted); }}
    footer {{ color:var(--muted); text-align:center; margin-top:24px; font-size:12px; }}
    @media (max-width:760px) {{ main {{ width:min(100% - 20px,1180px); margin-top:18px; }} header {{ display:block; }}
      .toolbar {{ justify-content:flex-start; margin-top:14px; }} .summary-grid,.mini-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media print {{ body {{ background:#fff; color:#111; }} main {{ width:100%; margin:0; }} .toolbar {{ display:none; }}
      section,details.section,.hero {{ break-inside:avoid; box-shadow:none; }} details.section:not([open]) > * {{ display:block; }}
      #camera-graph {{ display:none !important; }} #camera-graph-print {{ display:block; }} }}
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
    <p class="meta" style="margin-top:12px">{escape(application)}</p>
    <div class="summary-grid">
      <div class="stat"><span>Cameras registered</span><strong>{report.registered_matches}/{report.enabled_matches}</strong></div>
      <div class="stat"><span>Point RMSE</span><strong>{_format_error(report.point_rmse_px)}</strong></div>
      <div class="stat"><span>Line RMSE</span><strong>{_format_error(report.line_rmse_px)}</strong></div>
      <div class="stat"><span>Pose locks</span><strong>{report.locked_matches}</strong></div>
    </div>
    {joint_support_block}
  </div>
  <section><h2>Needs attention</h2><div class="issues">{_render_issues(report)}</div></section>
  {common_leave_one_out_block}
  <section><h2>Camera connectivity</h2><p class="meta" style="margin-bottom:12px">
    The overlap backbone shows ordinary point landmarks shared between photos. Metric 2D↔3D
    and Known 3D line routes are summarized in the match table and issue cards.</p>
    {_render_camera_graph(report)}</section>
  <section><h2>Matches</h2><div class="table-wrap"><table id="match-table"><thead><tr>
    <th><button type="button" data-sort="label" data-type="text">Match</button></th>
    <th><button type="button" data-sort="status" data-type="number">Status</button></th>
    <th class="number"><button type="button" data-sort="rmse" data-type="number">Point RMSE</button></th>
    <th class="number">Line RMSE</th>
    <th class="number"><button type="button" data-sort="points" data-type="number">Point picks</button></th>
    <th class="number"><button type="button" data-sort="lines" data-type="number">Lines</button></th>
    <th class="number"><button type="button" data-sort="usable" data-type="number">Usable 3D</button></th>
    <th>Best registered edge</th></tr></thead><tbody>
    {_render_match_rows(report)}</tbody></table></div></section>
  <section><h2>Landmark errors</h2><input class="search" id="landmark-search" type="search"
    placeholder="Filter landmarks or matches…" aria-label="Filter landmarks">
    <div class="table-wrap"><table id="landmark-table"><thead><tr><th>Landmark</th><th>Kind</th>
    <th class="number"><button type="button" id="sort-rmse">RMSE ↓</button></th><th>Observed in</th></tr></thead>
    <tbody>{_render_landmark_rows(report)}</tbody></table></div></section>
  {focal_block}
  <details class="section" open><summary>Constraint inventory</summary><div class="mini-grid">{constraint_cards}</div>
    <p class="meta" style="margin-top:12px">{report.disabled_matches} match(es) sync-disabled ·
    {report.excluded_landmarks} landmark(s) excluded from sync ·
    {'Joint bundle adjustment ran' if report.bundle_adjusted else 'No joint bundle adjustment recorded'}</p></details>
  <details class="section"><summary>Run notes</summary>{notes_block}</details>
  <details class="section"><summary>Technical solver message</summary><pre id="technical-message">{escape(report.raw_message)}</pre></details>
  <footer>Generated locally by Perspective Match. No report data was uploaded.</footer>
</main>
<script>__VENDOR_CYTOSCAPE__</script>
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
    const matchTable = document.getElementById('match-table');
    if (matchTable) {{
      let active = '';
      let matchDescending = false;
      for (const button of matchTable.querySelectorAll('thead button[data-sort]')) {{
        button.addEventListener('click', () => {{
          const key = button.dataset.sort;
          const numeric = button.dataset.type === 'number';
          if (active === key) matchDescending = !matchDescending;
          else {{
            active = key;
            matchDescending = numeric && key !== 'status';
          }}
          const factor = matchDescending ? -1 : 1;
          const rows = [...matchTable.tBodies[0].rows];
          rows.sort((left, right) => {{
            if (!numeric) {{
              return left.dataset[key].localeCompare(right.dataset[key], undefined, {{
                numeric: true, sensitivity: 'base'
              }}) * factor;
            }}
            const leftRaw = left.dataset[key];
            const rightRaw = right.dataset[key];
            const leftVal = leftRaw === '' || leftRaw == null ? NaN : Number(leftRaw);
            const rightVal = rightRaw === '' || rightRaw == null ? NaN : Number(rightRaw);
            const leftNa = Number.isNaN(leftVal);
            const rightNa = Number.isNaN(rightVal);
            if (leftNa && rightNa) return 0;
            if (leftNa) return 1;
            if (rightNa) return -1;
            if (leftVal === rightVal) return 0;
            return (leftVal < rightVal ? -1 : 1) * factor;
          }});
          rows.forEach(row => matchTable.tBodies[0].appendChild(row));
          for (const other of matchTable.querySelectorAll('thead button[data-sort]')) {{
            const base = other.textContent.replace(/ [↑↓]$/, '');
            other.textContent = other === button ? `${{base}} ${{matchDescending ? '↓' : '↑'}}` : base;
          }}
        }});
      }}
    }}
    const graphMount = document.getElementById('camera-graph');
    const graphDataEl = document.getElementById('camera-graph-data');
    if (graphMount && graphDataEl && window.cytoscape) {{
      const payload = JSON.parse(graphDataEl.textContent);
      const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const colors = dark ? {{
        text: '#e9eef3', muted: '#a4afba', line: '#36414d', surface: '#1a2129',
        surface2: '#202933', accent: '#77a0ff', success: '#65d39d', error: '#ff8f98'
      }} : {{
        text: '#17202a', muted: '#65717e', line: '#d9e0e7', surface: '#fff',
        surface2: '#f8fafc', accent: '#356ae6', success: '#157a4b', error: '#b4232d'
      }};
      const cy = cytoscape({{
        container: graphMount,
        elements: [
          ...payload.nodes.map(node => ({{
            data: node,
            classes: node.status,
            position: {{ x: node.x, y: node.y }}
          }})),
          ...payload.edges.map(edge => ({{
            data: edge, classes: edge.strong ? 'strong' : 'weak'
          }}))
        ],
        minZoom: 0.3,
        maxZoom: 2.5,
        wheelSensitivity: 0.35,
        layout: {{ name: 'preset', fit: false, padding: 28 }},
        style: [
          {{ selector: 'node', style: {{
            shape: 'round-rectangle', width: 176, height: 52,
            'background-color': colors.surface2, 'border-width': 2,
            'border-color': colors.line, color: colors.text, label: 'data(label)',
            'text-wrap': 'wrap', 'text-max-width': 176, 'text-valign': 'center',
            'text-halign': 'center', 'font-size': 12, 'font-weight': 700, 'line-height': 1.25
          }} }},
          {{ selector: 'node.disabled', style: {{ opacity: 0.55 }} }},
          {{ selector: 'node.anchor', style: {{ 'border-color': colors.accent }} }},
          {{ selector: 'node.synced', style: {{ 'border-color': colors.success }} }},
          {{ selector: 'node.skipped', style: {{ 'border-color': colors.error }} }},
          {{ selector: 'edge', style: {{
            'curve-style': 'bezier', width: 2, 'line-color': colors.line,
            label: 'data(shared)', 'font-size': 11, color: colors.muted,
            'text-background-color': colors.surface, 'text-background-opacity': 1,
            'text-background-padding': 2, 'text-events': 'yes', 'overlay-padding': 10,
            'overlay-opacity': 0
          }} }},
          {{ selector: 'edge.strong', style: {{ width: 3, 'line-color': colors.success }} }}
        ]
      }});
      const printImg = document.getElementById('camera-graph-print');
      const syncPrint = () => {{
        if (printImg) printImg.src = cy.png({{ full: true, scale: 2, bg: colors.surface }});
      }};
      const fitGraph = () => {{
        cy.resize();
        if (cy.nodes().nonempty()) {{
          cy.fit(cy.elements(), 28);
        }}
        syncPrint();
      }};
      cy.ready(fitGraph);
      window.addEventListener('load', fitGraph);
      window.addEventListener('beforeprint', syncPrint);
      const tip = document.getElementById('camera-graph-tooltip');
      const placeTip = event => {{
        if (!tip) return;
        const pos = event.renderedPosition;
        const box = graphMount.getBoundingClientRect();
        tip.hidden = false;
        tip.textContent = event.target.data('tooltip') || '';
        tip.style.left = `${{box.left + pos.x + 14}}px`;
        tip.style.top = `${{box.top + pos.y + 14}}px`;
      }};
      cy.on('mouseover', 'edge', placeTip);
      cy.on('mousemove', 'edge', placeTip);
      cy.on('mouseout', 'edge', () => {{ if (tip) tip.hidden = true; }});
      cy.on('viewport', () => {{ if (tip) tip.hidden = true; }});
    }}
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
    return html_text.replace("__VENDOR_CYTOSCAPE__", _vendor_js("cytoscape.min.js"), 1)


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
    scene_uid: int | None = None,
    request_sha256: str = "",
) -> Path:
    """Write one unique report and retain only recent reports for this process."""
    global _last_report_path, _last_report, _last_report_scene_uid, _last_report_request_sha256
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
    _last_report = report
    _last_report_scene_uid = scene_uid
    _last_report_request_sha256 = str(request_sha256)

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


def refresh_last_report(scene_uid: int | None, request_sha256: str) -> bool:
    """Mark the last report stale when its scene or numerical inputs differ."""
    report = _last_report
    path = last_report_path()
    if report is None or path is None:
        return False
    stale = (
        _last_report_scene_uid is not None
        and (
            scene_uid != _last_report_scene_uid
            or request_sha256 != _last_report_request_sha256
        )
    )
    if report.stale != stale:
        report.stale = stale
        temporary = path.with_suffix(".tmp")
        temporary.write_text(render_sync_report_html(report), encoding="utf-8")
        temporary.replace(path)
    return stale


def last_report_stale() -> bool:
    return bool(last_report_path() is not None and _last_report is not None and _last_report.stale)


def last_report_path() -> Path | None:
    """Newest report produced in this loaded add-on process, if it still exists."""
    if _last_report_path is None or not _last_report_path.is_file():
        return None
    return _last_report_path


def clear_last_report() -> None:
    """Forget the last report without deleting the browser's temporary file."""
    global _last_report_path, _last_report, _last_report_scene_uid, _last_report_request_sha256
    _last_report_path = None
    _last_report = None
    _last_report_scene_uid = None
    _last_report_request_sha256 = ""


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
