"""Bounded leave-one-out investigation using the common joint objective."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Callable

from .constants import DIAGNOSE_COMMON_MAX_CANDIDATES, DIAGNOSE_COMMON_MAX_SECONDS
from .request import SyncSolveRequest
from .types import SyncCancelled, SyncSolveResult


@dataclass
class CommonLeaveOneOutItem:
    """One omission assessed on exactly the same surviving evidence."""

    landmark_id: str
    landmark_name: str
    kind: str
    removed_relations: list[str]
    baseline_objective: float
    candidate_objective: float = math.inf
    baseline_point_rmse_px: float = math.inf
    candidate_point_rmse_px: float = math.inf
    baseline_line_rmse_px: float = math.inf
    candidate_line_rmse_px: float = math.inf
    baseline_valid: bool = False
    candidate_valid: bool = False
    reason: str = ""
    support_coverage: dict[str, object] = field(default_factory=dict)

    @property
    def objective_improvement(self) -> float:
        """Positive value means the re-fit improved surviving evidence."""
        return self.baseline_objective - self.candidate_objective


@dataclass
class CommonLeaveOneOutOutcome:
    """Completed comparisons and any reason the bounded search stopped early."""

    items: list[CommonLeaveOneOutItem] = field(default_factory=list)
    baseline_coverage: dict[str, object] = field(default_factory=dict)
    candidates_ranked: int = 0
    incomplete: bool = False
    reason: str = ""


def _omit_landmark(request: SyncSolveRequest, landmark_id: str) -> tuple[SyncSolveRequest, list[str]]:
    """Remove one landmark and its direct model relations without mutating input."""
    removed: list[str] = []
    dependent_lines = {item[0] for item in request.derived_lines or ()
                       if landmark_id in item[1:]}
    mirror_pairs = []
    for left, right in request.mirror_pairs or ():
        if landmark_id in (left, right) or dependent_lines & {left, right}:
            removed.append(f"mirror:{left}:{right}")
        else:
            mirror_pairs.append((left, right))
    plane_groups = []
    for member, axis, group in request.plane_groups or ():
        if member == landmark_id or member in dependent_lines:
            removed.append(f"plane:{axis}:{group}:{member}")
        else:
            plane_groups.append((member, axis, group))
    parallel_pairs = []
    for left, right in request.parallel_pairs or ():
        if landmark_id in (left, right) or dependent_lines & {left, right}:
            removed.append(f"parallel:{left}:{right}")
        else:
            parallel_pairs.append((left, right))
    updates = dict(
        observations=[item for item in request.observations
                      if item.landmark_id != landmark_id],
        line_observations=[item for item in request.line_observations or ()
                           if item.landmark_id != landmark_id],
        known_world={key: value for key, value in (request.known_world or {}).items()
                     if key != landmark_id},
        known_lines={key: value for key, value in (request.known_lines or {}).items()
                     if key != landmark_id},
        derived_lines=[item for item in request.derived_lines or ()
                       if landmark_id not in item],
        mirror_pairs=mirror_pairs,
        mirror_landmark_id=(request.mirror_landmark_id if mirror_pairs else None),
        plane_groups=plane_groups,
        parallel_pairs=parallel_pairs,
    )
    if "initial_solution" in request.__dataclass_fields__:
        updates["initial_solution"] = None
    return replace(request, **updates), removed


def _same_support(request: SyncSolveRequest, supported: SyncSolveRequest,
                  coverage: dict[str, object]) -> bool:
    """A counterfactual must retain every observation and relation being tested."""
    return (
        len(supported.matches) == len(request.matches)
        and len(supported.observations) == len(request.observations)
        and len(supported.line_observations or ()) == len(request.line_observations or ())
        and supported.mirror_pairs == request.mirror_pairs
        and supported.mirror_landmark_id == request.mirror_landmark_id
        and supported.plane_groups == request.plane_groups
        and supported.parallel_pairs == request.parallel_pairs
        and supported.derived_lines == request.derived_lines
        and not coverage.get("skipped_camera_ids")
        and not coverage.get("skipped_point_ids")
        and not coverage.get("skipped_line_ids")
        and not coverage.get("skipped_relation_ids")
    )


def common_leave_one_out(
    request: SyncSolveRequest,
    baseline: SyncSolveResult,
    solve_callback: Callable,
    *,
    max_candidates: int = DIAGNOSE_COMMON_MAX_CANDIDATES,
    max_seconds: float = DIAGNOSE_COMMON_MAX_SECONDS,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    scorer_factory: Callable | None = None,
    supported_request_fn: Callable | None = None,
) -> CommonLeaveOneOutOutcome:
    """Compare omitted-landmark re-fits on frozen surviving joint evidence.

    ``solve_callback(filtered_request, reduced_baseline, cancel_check=...)``
    returns a SyncSolveResult, or an outcome with ``sync_result``, ``accepted``
    and ``reason`` attributes. The inner solve must honor its cancellation check.
    """
    if scorer_factory is None or supported_request_fn is None:
        from ..joint_fit_score import JointFitScorer, supported_joint_request
        scorer_factory = scorer_factory or JointFitScorer
        supported_request_fn = supported_request_fn or supported_joint_request
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    limit = min(DIAGNOSE_COMMON_MAX_CANDIDATES, max(0, int(max_candidates)))
    started = time.monotonic()
    deadline = started + max_seconds

    def user_cancelled() -> bool:
        return cancel_check is not None and bool(cancel_check())

    def check_user_cancel() -> None:
        if user_cancelled():
            raise SyncCancelled("Sync cancelled")

    def stop_inner() -> bool:
        return user_cancelled() or time.monotonic() >= deadline

    check_user_cancel()
    supported, baseline_coverage = supported_request_fn(request, baseline)
    outcome = CommonLeaveOneOutOutcome(baseline_coverage=baseline_coverage)
    observations = list(supported.observations) + list(supported.line_observations or ())
    available = {item.landmark_id for item in observations}
    protected = set(supported.known_world or ()) | set(supported.known_lines or ())
    if supported.mirror_landmark_id is not None:
        protected.add(supported.mirror_landmark_id)
    names = {item.landmark_id: item.landmark_name or item.landmark_id
             for item in observations}
    kinds = {item.landmark_id: "line" for item in supported.line_observations or ()}
    kinds.update({item.landmark_id: "point" for item in supported.observations})
    ranked = sorted(
        (key for key in available if key not in protected and
         key in baseline.per_landmark_rmse_px and
         math.isfinite(baseline.per_landmark_rmse_px[key])),
        key=lambda key: (-baseline.per_landmark_rmse_px[key], key),
    )[:limit]
    outcome.candidates_ranked = len(ranked)
    for index, landmark_id in enumerate(ranked):
        check_user_cancel()
        if time.monotonic() >= deadline:
            outcome.incomplete = True
            outcome.reason = "Investigation time limit reached"
            break
        name = names[landmark_id]
        if progress_callback is not None:
            progress_callback(index, len(ranked), f"Checking {name}")
        filtered, removed = _omit_landmark(supported, landmark_id)
        reduced = replace(
            baseline,
            landmarks={key: value for key, value in baseline.landmarks.items()
                       if key != landmark_id},
            line_segments={key: value for key, value in baseline.line_segments.items()
                           if key != landmark_id},
        )
        calibrations = getattr(baseline, "calibrations", None) or {
            item.match_id: item.calibration for item in filtered.matches}
        scorer = scorer_factory(filtered, reduced, calibrations=calibrations)
        before = scorer.score(reduced, calibrations=calibrations)
        item = CommonLeaveOneOutItem(
            landmark_id=landmark_id,
            landmark_name=name,
            kind=kinds[landmark_id],
            removed_relations=removed,
            baseline_objective=before.objective,
            baseline_point_rmse_px=before.point_rmse_px,
            baseline_line_rmse_px=before.line_rmse_px,
            baseline_valid=before.valid,
        )
        if not before.valid:
            item.reason = f"Baseline cannot be scored: {before.reason}"
            outcome.items.append(item)
            outcome.incomplete = True
            outcome.reason = item.reason
            break
        try:
            trial = solve_callback(filtered, reduced, cancel_check=stop_inner)
        except SyncCancelled as exc:
            check_user_cancel()
            item.reason = "Investigation time limit reached" if time.monotonic() >= deadline else str(exc)
            outcome.items.append(item)
            outcome.incomplete = True
            outcome.reason = item.reason
            break
        except Exception as exc:
            check_user_cancel()
            item.reason = ("Investigation time limit reached" if time.monotonic() >= deadline
                           else f"Counterfactual fit failed: {exc}")
            outcome.items.append(item)
            outcome.incomplete = True
            outcome.reason = item.reason
            break
        check_user_cancel()
        candidate = getattr(trial, "sync_result", trial)
        accepted = getattr(trial, "accepted", True)
        if not accepted or candidate is None or not candidate.success:
            item.reason = ("Investigation time limit reached" if time.monotonic() >= deadline
                           else getattr(trial, "reason", "") or "Counterfactual fit refused")
            outcome.items.append(item)
            outcome.incomplete = True
            outcome.reason = item.reason
            break
        candidate_supported, coverage = supported_request_fn(filtered, candidate)
        item.support_coverage = coverage
        if not _same_support(filtered, candidate_supported, coverage):
            item.reason = "Counterfactual fit lost surviving evidence"
            outcome.items.append(item)
            outcome.incomplete = True
            outcome.reason = item.reason
            break
        after = scorer.score(candidate, calibrations=getattr(candidate, "calibrations", None) or calibrations)
        item.candidate_objective = after.objective
        item.candidate_point_rmse_px = after.point_rmse_px
        item.candidate_line_rmse_px = after.line_rmse_px
        item.candidate_valid = after.valid
        item.reason = after.reason
        outcome.items.append(item)
        if not after.valid:
            outcome.incomplete = True
            outcome.reason = f"Counterfactual fit cannot be scored: {after.reason}"
            break
        if time.monotonic() >= deadline:
            outcome.incomplete = True
            outcome.reason = "Investigation time limit reached"
            break
    if progress_callback is not None:
        progress_callback(len(outcome.items), len(ranked), "Investigation complete")
    return outcome
