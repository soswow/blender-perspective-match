"""One current-evidence objective for fixed and free focal joint fits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import geometry
from .derived_lines import derived_line_geometry, validate_derived_lines
from .focal_constraints import PointFocalConstraints
from .focal_line_constraints import (LineFocalConstraints,
                                     LINE_RELATION_DIRECTION_HARD_SINE)
from .focal_lines import canonical_line_point
from .focal_point_priors import PointReferenceConstraints
from .focal_projection import project_stroke_line
from .sync.ba import _auto_downweight_outlier_observations, _balance_observation_weights
from .sync.ba import observations_for_location
from .sync.constants import (GROUND_SLACK_DEFAULT, KNOWN_3D_SLACK_DEFAULT,
                             LINE_PLANE_MIN_SINE,
                             MIRROR_PAIR_EXACT_RELATIVE_TOLERANCE,
                             OUTLIER_WEIGHT_FACTOR, PLANE_HARD_SLACK,
                             WORLD_AXIS_DIRECTIONS)
from .sync.lines import line_support_angles
from .sync.planes import supported_line_planes
from .sync.projection import _project_shared_points, _project_world_line_to_image
from .sync.pose import _snap_to_axis_aligned_rotation
from .sync.request import SyncSolveRequest
from .sync.types import SyncSolveResult, SyncMatchInput, SyncObservation


@dataclass
class JointFitScore:
    objective: float = float("inf")
    point_rmse_px: float = float("inf")
    line_rmse_px: float = float("inf")
    per_match_rmse_px: dict[str, float] = field(default_factory=dict)
    per_match_point_rmse_px: dict[str, float] = field(default_factory=dict)
    per_match_line_rmse_px: dict[str, float] = field(default_factory=dict)
    per_landmark_rmse_px: dict[str, float] = field(default_factory=dict)
    supported_observations: int = 0
    constraint_gaps: dict[str, float] = field(default_factory=dict)
    valid: bool = False
    reason: str = ""


def supported_joint_request(request: SyncSolveRequest,
                            result: SyncSolveResult) -> tuple[SyncSolveRequest, dict[str, object]]:
    """Keep exactly posed/reconstructed evidence and report every exclusion."""
    posed = set(result.similarities)
    point_geometry = set(result.landmarks)
    line_geometry = set(result.line_segments) | set(request.known_lines or {})
    point_observations = [item for item in request.observations
                          if item.match_id in posed and item.landmark_id in point_geometry]
    line_observations = [item for item in request.line_observations or ()
                         if item.match_id in posed and item.landmark_id in line_geometry]
    included_cameras = {item.match_id for item in point_observations} | {
        item.match_id for item in line_observations}
    if request.anchor_id in posed and any(item.match_id == request.anchor_id
                                          for item in request.matches):
        included_cameras.add(request.anchor_id)
    included_points = {item.landmark_id for item in point_observations} | \
        set(request.known_world or {})
    included_derived = [
        tuple(item) for item in request.derived_lines or ()
        if item[1] in included_points and item[2] in included_points
    ]
    included_lines = {item.landmark_id for item in line_observations} | {
        key for key in request.known_lines or () if key in line_geometry}
    included_lines.update(item[0] for item in included_derived)
    included_geometry = included_points | included_lines
    skipped_relations: list[str] = []
    plane_members: dict[tuple[str, int], list[tuple[str, str, int]]] = {}
    for relation in request.plane_groups or ():
        plane_members.setdefault((relation[1], relation[2]), []).append(relation)
    plane_groups: list[tuple[str, str, int]] = []
    for group, relations in plane_members.items():
        if all(item[0] in included_geometry for item in relations):
            plane_groups.extend(relations)
        else:
            skipped_relations.append(f"plane:{group[0]}:{group[1]}")
    mirror_pairs = []
    for left, right in request.mirror_pairs or ():
        if left in included_geometry and right in included_geometry:
            mirror_pairs.append((left, right))
        else:
            skipped_relations.append(f"mirror:{left}:{right}")
    mirror_reference = request.mirror_landmark_id
    if mirror_pairs and mirror_reference is not None and mirror_reference not in included_points:
        skipped_relations.extend(f"mirror:{left}:{right}" for left, right in mirror_pairs)
        mirror_pairs = []
    if not mirror_pairs:
        mirror_reference = None
    parallel_pairs = []
    for left, right in request.parallel_pairs or ():
        if ((left in included_lines or left in WORLD_AXIS_DIRECTIONS) and
                (right in included_lines or right in WORLD_AXIS_DIRECTIONS) and
                (left in included_lines or right in included_lines)):
            parallel_pairs.append((left, right))
        else:
            skipped_relations.append(f"parallel:{left}:{right}")
    coverage = {
        "requested_point_picks": len(request.observations),
        "fitted_point_picks": len(point_observations),
        "requested_line_strokes": len(request.line_observations or ()),
        "fitted_line_strokes": len(line_observations),
        "skipped_camera_ids": sorted({item.match_id for item in request.matches} -
                                     included_cameras),
        "skipped_point_ids": sorted({item.landmark_id for item in request.observations} -
                                    included_points),
        "skipped_line_ids": sorted(
            ({item.landmark_id for item in request.line_observations or ()} |
             {item[0] for item in request.derived_lines or ()}) - included_lines),
        "skipped_relation_ids": sorted(set(skipped_relations)),
    }
    updates = dict(
        matches=[item for item in request.matches if item.match_id in included_cameras],
        observations=point_observations,
        line_observations=line_observations,
        known_world={key: value for key, value in (request.known_world or {}).items()
                     if key in included_points},
        known_lines={key: value for key, value in (request.known_lines or {}).items()
                     if key in included_lines},
        derived_lines=included_derived,
        plane_groups=plane_groups,
        mirror_pairs=mirror_pairs,
        mirror_landmark_id=mirror_reference,
        parallel_pairs=parallel_pairs,
        initial_similarities={key: value for key, value in
                              (request.initial_similarities or {}).items()
                              if key in included_cameras},
        fixed_similarities={key: value for key, value in
                            (request.fixed_similarities or {}).items()
                            if key in included_cameras},
        location_match_ids=({key for key in request.location_match_ids
                             if key in included_cameras}
                            if request.location_match_ids is not None else None),
        readonly_match_ids=({key for key in request.readonly_match_ids
                             if key in included_cameras}
                            if request.readonly_match_ids is not None else None),
    )
    if "initial_solution" in request.__dataclass_fields__:
        updates["initial_solution"] = None
    return replace(request, **updates), coverage


class JointFitScorer:
    """Freeze current-input weights once so incumbent and trial are comparable."""

    def __init__(self, request: SyncSolveRequest, incumbent: SyncSolveResult,
                 *, calibrations: dict[str, geometry.Calibration] | None = None,
                 frozen_point_weights: list[tuple[str, str, float]] | None = None):
        self.request = request
        self.base_calibrations = calibrations or {
            item.match_id: item.calibration for item in request.matches}
        matches = {key: SyncMatchInput(key, cal)
                   for key, cal in self.base_calibrations.items()}
        balanced = _balance_observation_weights(request.observations, matches)
        if frozen_point_weights is not None:
            by_id = {(item.match_id, item.landmark_id): item for item in balanced}
            frozen = {}
            try:
                for camera_id, point_id, weight in frozen_point_weights:
                    key = (camera_id, point_id)
                    if key in frozen or key not in by_id or not np.isfinite(weight) or weight <= 0:
                        raise ValueError
                    base = float(by_id[key].weight)
                    reduced = max(base * OUTLIER_WEIGHT_FACTOR, 1.0e-3)
                    if not (np.isclose(weight, base, rtol=1e-8, atol=1e-10) or
                            np.isclose(weight, reduced, rtol=1e-8, atol=1e-10)):
                        raise ValueError
                    frozen[key] = float(weight)
                if len(frozen) != len(balanced):
                    raise ValueError
                reduced_ids = {item.landmark_id for item in balanced
                               if not np.isclose(
                                   frozen[(item.match_id, item.landmark_id)],
                                   item.weight, rtol=1e-8, atol=1e-10)}
                protected = {key for pair in request.mirror_pairs or () for key in pair}
                if any(item.landmark_id in reduced_ids and
                       (item.protect_outlier or item.landmark_id in protected or
                        np.isclose(frozen[(item.match_id, item.landmark_id)],
                                   item.weight, rtol=1e-8, atol=1e-10))
                       for item in balanced):
                    raise ValueError
            except (TypeError, ValueError, KeyError):
                self.point_observations = balanced
                self.downweighted_landmark_ids = []
                self.weight_refusal = "Certified point weights do not match current evidence"
                return
            self.point_observations = [
                replace(item, weight=frozen[(item.match_id, item.landmark_id)])
                for item in balanced]
            self.downweighted_landmark_ids = sorted(reduced_ids)
            self.weight_refusal = ""
            return
        raw = self._pixel_errors(incumbent, self.base_calibrations,
                                 balanced, request.line_observations or [])
        if isinstance(raw, str):
            self.point_observations = balanced
            self.weight_refusal = raw
            self.downweighted_landmark_ids = []
        else:
            _points, _lines, by_landmark = raw
            prior_rmse = {key: float(np.sqrt(np.mean(np.square(values))))
                          for key, values in by_landmark.items()}
            protected = {key for pair in request.mirror_pairs or () for key in pair}
            self.point_observations, _downweighted = _auto_downweight_outlier_observations(
                balanced, prior_rmse, protected_ids=protected)
            self.downweighted_landmark_ids = sorted(_downweighted)
            self.weight_refusal = ""

    @property
    def effective_point_weights(self) -> list[tuple[str, str, float]]:
        return [(item.match_id, item.landmark_id, float(item.weight))
                for item in self.point_observations]

    @staticmethod
    def _pixel_errors(result: SyncSolveResult,
                      calibrations: dict[str, geometry.Calibration],
                      points: list[SyncObservation], lines) -> tuple[dict, dict, dict] | str:
        by_camera: dict[str, list[SyncObservation]] = {}
        for item in points:
            if item.match_id not in result.similarities:
                return f"Camera {item.match_id} has no fitted pose"
            if item.match_id not in calibrations:
                return f"Camera {item.match_id} has no calibration"
            if item.landmark_id not in result.landmarks:
                return f"Point {item.landmark_id} has no fitted geometry"
            by_camera.setdefault(item.match_id, []).append(item)
        point_errors: dict[tuple[str, str], float] = {}
        by_landmark: dict[str, list[float]] = {}
        for camera_id, items in by_camera.items():
            predicted, valid = _project_shared_points(
                np.asarray([result.landmarks[item.landmark_id] for item in items]),
                calibrations[camera_id], result.similarities[camera_id])
            if not np.all(valid) or not np.isfinite(predicted).all():
                return f"Camera {camera_id} has a point behind the image plane"
            for item, image in zip(items, predicted):
                error = float(np.linalg.norm(image - (item.u, item.v)))
                point_errors[(camera_id, item.landmark_id)] = error
                by_landmark.setdefault(item.landmark_id, []).append(error)
        line_errors: dict[tuple[str, str], np.ndarray] = {}
        for item in lines:
            if item.match_id not in result.similarities:
                return f"Camera {item.match_id} has no fitted pose"
            if item.match_id not in calibrations:
                return f"Camera {item.match_id} has no calibration"
            segment = result.line_segments.get(item.landmark_id)
            if segment is None:
                return f"Line {item.landmark_id} has no fitted geometry"
            first, second = [np.asarray(end, float).reshape(3) for end in segment]
            if (not np.isfinite((first, second)).all() or
                    np.linalg.norm(second - first) < 1e-9):
                return f"Line {item.landmark_id} has invalid fitted geometry"
            endpoints = np.asarray(((item.u1, item.v1), (item.u2, item.v2)))
            calibration = calibrations[item.match_id]
            similarity = result.similarities[item.match_id]
            line = project_stroke_line(
                0.5 * (first + second), second - first,
                calibration.rotation_w2c @ similarity.rotation.T,
                similarity.transform_point(calibration.camera_center),
                calibration.intrinsics, endpoints,
                division_lambda=calibration.division_lambda,
                brown_conrady=calibration.brown_conrady)
            if line is None or not np.isfinite(line).all():
                return f"Line {item.landmark_id} cannot project in {item.match_id}"
            error = endpoints @ line[:2] + line[2]
            line_errors[(item.match_id, item.landmark_id)] = error
            by_landmark.setdefault(item.landmark_id, []).extend(abs(error).tolist())
        return point_errors, line_errors, by_landmark

    def score(self, result: SyncSolveResult, *,
              calibrations: dict[str, geometry.Calibration] | None = None) -> JointFitScore:
        """Evaluate every request observation and active relation, or explain refusal."""
        if self.weight_refusal:
            return JointFitScore(reason=self.weight_refusal)
        request = self.request
        cals = calibrations or getattr(result, "calibrations", None) or self.base_calibrations
        pixels = self._pixel_errors(result, cals, self.point_observations,
                                    request.line_observations or [])
        if isinstance(pixels, str):
            return JointFitScore(reason=pixels)
        point_errors, line_errors, by_landmark = pixels
        matched = {item.match_id for item in self.point_observations} | {
            item.match_id for item in request.line_observations or ()}
        if matched != {item.match_id for item in request.matches} - {request.anchor_id} and \
                matched != {item.match_id for item in request.matches}:
            return JointFitScore(reason="One or more cameras have no supported image evidence")
        point_ids = sorted({item.landmark_id for item in self.point_observations} |
                           set(request.known_world or {}))
        drawn_line_ids = ({item.landmark_id for item in request.line_observations or ()} |
                          set(request.known_lines or {}))
        try:
            derived = validate_derived_lines(
                request.derived_lines, point_ids, other_line_ids=drawn_line_ids)
        except (ValueError, TypeError) as exc:
            return JointFitScore(reason=str(exc))
        line_ids = sorted(drawn_line_ids | {item[0] for item in derived})
        if set(point_ids) & set(line_ids):
            return JointFitScore(reason="A landmark is both a point and a line")
        try:
            anchor_cal = cals[request.anchor_id]
            anchor_sim = result.similarities[request.anchor_id]
            if (abs(anchor_sim.scale - 1.0) > 1e-8 or
                    np.linalg.norm(anchor_sim.rotation - np.eye(3)) > 1e-8 or
                    np.linalg.norm(anchor_sim.translation) > 1e-8):
                return JointFitScore(reason="Anchor gauge is not fixed")
            fixed = request.fixed_similarities or {}
            for key, required in fixed.items():
                actual = result.similarities.get(key)
                if actual is None or (abs(actual.scale - required.scale) > 1e-7 or
                                      np.linalg.norm(actual.rotation - required.rotation) > 1e-7 or
                                      np.linalg.norm(actual.translation - required.translation) > 1e-7):
                    return JointFitScore(reason=f"Locked pose {key} changed")
            for key, actual in result.similarities.items():
                if key == request.anchor_id or key in fixed:
                    continue
                if (request.lock_rotation and np.linalg.norm(
                        actual.rotation - _snap_to_axis_aligned_rotation(actual.rotation)) > 1e-7):
                    return JointFitScore(reason=f"Root rotation lock {key} changed")
                if request.lock_translation and np.linalg.norm(actual.translation) > 1e-7:
                    return JointFitScore(reason=f"Root translation lock {key} changed")
            anchor_r = anchor_cal.rotation_w2c
            anchor_c = anchor_cal.camera_center
            centers = [sim.transform_point(cals[key].camera_center)
                       for key, sim in result.similarities.items() if key != request.anchor_id]
            baseline = max((float(np.linalg.norm(center - anchor_c)) for center in centers),
                           default=0.0)
            if not np.isfinite(baseline) or baseline < 1e-8:
                return JointFitScore(reason="Camera baseline is zero")
            point_chart = np.asarray([anchor_r @
                                      (np.asarray(result.landmarks[key] if key in result.landmarks
                                                  else request.known_world[key]) - anchor_c) / baseline
                                      for key in point_ids], float).reshape(-1, 3)
            lines_world = {}
            for key in line_ids:
                if key in {item[0] for item in derived}:
                    continue
                if key not in result.line_segments:
                    return JointFitScore(reason=f"Line {key} has no fitted geometry")
                first, second = [np.asarray(end, float) for end in result.line_segments[key]]
                direction = second - first
                direction /= np.linalg.norm(direction)
                lines_world[key] = (0.5 * (first + second), direction)
            _derived_segments, derived_world = derived_line_geometry(
                result.landmarks, derived)
            lines_world.update(derived_world)
            line_chart = [(anchor_r @ (lines_world[key][0] - anchor_c) / baseline,
                           anchor_r @ lines_world[key][1]) for key in line_ids]
            point_set, line_set = set(point_ids), set(line_ids)
            mirror_pairs = request.mirror_pairs or []
            point_pairs = [pair for pair in mirror_pairs if set(pair) <= point_set]
            line_pairs = [pair for pair in mirror_pairs if set(pair) <= line_set]
            if len(point_pairs) + len(line_pairs) != len(mirror_pairs):
                return JointFitScore(reason="Mirror relation has missing or mixed geometry")
            if line_pairs and request.mirror_plane is None:
                return JointFitScore(reason="Line mirror relation lacks a plane normal")
            plane_groups = request.plane_groups or []
            if any(key not in point_set | line_set for key, _, _ in plane_groups):
                return JointFitScore(reason="Plane relation has missing geometry")
            ref = PointReferenceConstraints.from_inputs(
                point_ids, anchor_rotation=anchor_r, anchor_center=anchor_c,
                baseline_world=baseline, known_world=request.known_world,
                known_3d_slack=(KNOWN_3D_SLACK_DEFAULT if request.known_3d_slack is None
                                else request.known_3d_slack),
                ground_landmark_ids=sorted({item.landmark_id for item in self.point_observations
                                            if item.on_ground}),
                ground_slack=(GROUND_SLACK_DEFAULT if request.ground_slack is None
                              else request.ground_slack))
            point_rel = PointFocalConstraints.from_inputs(
                point_ids, anchor_rotation=anchor_r, anchor_center=anchor_c,
                baseline_world=baseline,
                plane_groups=[item for item in plane_groups if item[0] in point_set],
                plane_slack=0.0 if request.plane_slack is None else request.plane_slack,
                mirror_pairs=point_pairs, mirror_plane=request.mirror_plane,
                mirror_slack=0.0 if request.mirror_slack is None else request.mirror_slack,
                mirror_pair_slack=(0.0 if request.mirror_pair_slack is None
                                   else request.mirror_pair_slack),
                mirror_landmark_id=request.mirror_landmark_id,
                extra_mirror_pairs=bool(line_pairs))
            line_rel = LineFocalConstraints.from_inputs_with_derived(
                point_ids, line_ids, plane_groups=plane_groups,
                parallel_pairs=request.parallel_pairs, anchor_rotation=anchor_r,
                plane_spring=point_rel.plane_spring, baseline_world=baseline,
                hard_plane=point_rel.hard_plane,
                known_line_ids=set(request.known_lines or {}),
                derived_lines=derived,
                known_line_positions={key: anchor_r @ (
                    0.5 * (np.asarray(segment[0]) + np.asarray(segment[1])) -
                    anchor_c) / baseline for key, segment in
                    (request.known_lines or {}).items()})
            mirror_offset = float(getattr(result, "joint_mirror_offset_m", 0.0)) / baseline
            ref_rows, _ = ref.residual_and_jacobian(
                point_chart, point_offset=0, parameter_count=0, jacobian=False)
            point_rows, _ = point_rel.residual_and_jacobian(
                point_chart, point_offset=0, parameter_count=0,
                mirror_offset=mirror_offset, jacobian=False)
            line_rows = line_rel.residual(point_chart, line_chart)
            line_mirror_rows: list[float] = []
            line_mirror_distance = 0.0
            line_mirror_direction = 0.0
            if line_pairs:
                normal = point_rel.mirror_normal
                assert normal is not None
                householder = np.eye(3) - 2.0 * np.outer(normal, normal)
                distance = (float(normal @ point_chart[point_rel.mirror_reference_index])
                            if point_rel.mirror_reference_index is not None
                            else point_rel.mirror_distance)
                distance += mirror_offset
                line_index = {key: index for index, key in enumerate(line_ids)}
                for left, right in line_pairs:
                    left_p, left_d = line_chart[line_index[left]]
                    right_p, right_d = line_chart[line_index[right]]
                    left_p = canonical_line_point(left_p, left_d)
                    right_p = canonical_line_point(right_p, right_d)
                    reflected_p = householder @ left_p + 2.0 * distance * normal
                    reflected_d = householder @ left_d
                    if reflected_d @ right_d < 0:
                        reflected_d = -reflected_d
                    position = np.cross(right_d, reflected_p - right_p)
                    direction = np.cross(right_d, reflected_d)
                    if point_rel.mirror_pair_slack > 1.0e-12:
                        line_mirror_rows.extend(
                            (point_rel.mirror_pair_spring * position).tolist())
                    line_mirror_distance = max(line_mirror_distance,
                                               baseline * float(np.linalg.norm(position)))
                    line_mirror_direction = max(line_mirror_direction,
                                                float(np.linalg.norm(direction)))
            line_mirror_rows = np.asarray(line_mirror_rows, float)
            known_gap, ground_gap = ref.world_gaps(point_chart)
            hard_ground_gap = max((abs(float(normal @ point_chart[index] - offset)) * baseline
                                   for index, (normal, offset) in ref.hard_z.items()),
                                  default=0.0)
            plane_gap, point_mirror_gap = point_rel.world_gaps(
                point_chart, mirror_offset=mirror_offset)
            line_plane_gap, line_direction_gap = line_rel.world_gaps(point_chart, line_chart)
            known_line_gap = 0.0
            for key, known_segment in (request.known_lines or {}).items():
                if key not in result.line_segments:
                    return JointFitScore(reason=f"Known line {key} has no fitted geometry")
                known_line_gap = max(known_line_gap, float(np.max(np.linalg.norm(
                    np.asarray(result.line_segments[key]) - np.asarray(known_segment), axis=1))))
            gaps = dict(known_3d_m=known_gap, ground_m=ground_gap,
                        point_plane_m=plane_gap, point_mirror_m=point_mirror_gap,
                        line_plane_m=line_plane_gap,
                        line_direction_sine=line_direction_gap,
                        line_mirror_m=line_mirror_distance,
                        line_mirror_direction_sine=line_mirror_direction,
                        known_line_m=known_line_gap)
            exact_tolerance = MIRROR_PAIR_EXACT_RELATIVE_TOLERANCE * max(
                baseline,
                baseline * max((float(np.linalg.norm(point))
                                for point in point_chart), default=0.0),
                baseline * max((float(np.linalg.norm(point))
                                for point, _direction in line_chart), default=0.0),
                1.0)
            mirror_limit = point_rel.mirror_pair_slack + exact_tolerance
            if (ref.hard_xyz and known_gap > 1e-7 or
                    hard_ground_gap > 1e-7 or
                    point_rel.hard_plane and plane_gap > PLANE_HARD_SLACK or
                    point_mirror_gap > mirror_limit or
                    line_rel.hard_plane and line_plane_gap > PLANE_HARD_SLACK or
                    line_direction_gap > LINE_RELATION_DIRECTION_HARD_SINE or
                    line_mirror_distance > mirror_limit or
                    line_mirror_direction > MIRROR_PAIR_EXACT_RELATIVE_TOLERANCE or
                    known_line_gap > 1e-7):
                return JointFitScore(constraint_gaps=gaps, reason="Hard world relation is violated")
            objective = sum(item.weight * point_errors[(item.match_id, item.landmark_id)]**2
                            for item in self.point_observations)
            objective += sum(item.weight * float(line_errors[(item.match_id, item.landmark_id)] @
                                                  line_errors[(item.match_id, item.landmark_id)])
                             for item in request.line_observations or ())
            objective += float(ref_rows @ ref_rows + point_rows @ point_rows +
                               line_rows @ line_rows +
                               line_mirror_rows @ line_mirror_rows)
            by_match: dict[str, list[float]] = {}
            by_match_point: dict[str, list[float]] = {}
            by_match_line: dict[str, list[float]] = {}
            for (camera, _key), error in point_errors.items():
                by_match.setdefault(camera, []).append(error)
                by_match_point.setdefault(camera, []).append(error)
            for (camera, _key), errors in line_errors.items():
                by_match.setdefault(camera, []).extend(abs(errors).tolist())
                by_match_line.setdefault(camera, []).extend(abs(errors).tolist())
            return JointFitScore(
                objective=float(objective),
                point_rmse_px=float(np.sqrt(np.mean(np.square(list(point_errors.values())))))
                if point_errors else 0.0,
                line_rmse_px=float(np.sqrt(np.mean(np.square(
                    np.concatenate(list(line_errors.values())))))) if line_errors else 0.0,
                per_match_rmse_px={key: float(np.sqrt(np.mean(np.square(values))))
                                   for key, values in by_match.items()},
                per_match_point_rmse_px={
                    key: float(np.sqrt(np.mean(np.square(values))))
                    for key, values in by_match_point.items()},
                per_match_line_rmse_px={
                    key: float(np.sqrt(np.mean(np.square(values))))
                    for key, values in by_match_line.items()},
                per_landmark_rmse_px={key: float(np.sqrt(np.mean(np.square(values))))
                                      for key, values in by_landmark.items()},
                supported_observations=len(point_errors) + len(line_errors),
                constraint_gaps=gaps, valid=True)
        except (KeyError, TypeError, ValueError, FloatingPointError,
                np.linalg.LinAlgError) as exc:
            return JointFitScore(reason=f"Joint score could not represent current evidence: {exc}")


def score_joint_fit(request: SyncSolveRequest, result: SyncSolveResult, *,
                    calibrations: dict[str, geometry.Calibration] | None = None,
                    incumbent: SyncSolveResult | None = None) -> JointFitScore:
    """Score one result; pass one incumbent to freeze weights across comparisons."""
    return JointFitScorer(request, incumbent or result,
                          calibrations=calibrations).score(result, calibrations=calibrations)


def joint_line_support_diagnostics(
    request: SyncSolveRequest, result: SyncSolveResult,
    calibrations: dict[str, geometry.Calibration],
) -> tuple[dict[str, float], list[str]]:
    """Recompute line support after the accepted cameras and geometry change."""
    if not result.line_segments:
        return {}, []
    by_line: dict[str, list] = {}
    for item in request.line_observations or ():
        if item.landmark_id in result.line_segments:
            by_line.setdefault(item.landmark_id, []).append(item)
    location_ids = (set(request.location_match_ids)
                    if request.location_match_ids is not None else
                    {item.match_id for item in request.matches})
    location_ids -= set(request.readonly_match_ids or ())
    location_ids.add(request.anchor_id)
    hard_planes = ((0.0 if request.plane_slack is None
                    else request.plane_slack) <= 1.0e-12)
    support_planes = (supported_line_planes(
        result.landmarks, result.line_segments, request.plane_groups,
        request.known_lines or {},
        ground_landmark_ids=[item.landmark_id for item in request.observations
                             if item.on_ground],
        excluded_support_ids=set(result.plane_seeded_landmark_ids))
        if hard_planes else None)
    angles = line_support_angles(
        result.line_segments,
        {key: observations_for_location(items, location_ids)
         for key, items in by_line.items()},
        result.similarities,
        {key: SyncMatchInput(key, cal) for key, cal in calibrations.items()},
        known_lines=request.known_lines,
        mirror_pairs=request.mirror_pairs,
        mirror_normal=(request.mirror_plane[1]
                       if request.mirror_plane is not None else None),
        support_planes=support_planes)
    weak = sorted(key for key, angle in angles.items()
                  if float(np.sin(np.radians(angle))) < LINE_PLANE_MIN_SINE)
    return angles, weak
