"""Lens-input report with optional bounded fit; never apply or save the blend.

blender --factory-startup --disable-autoexec -b --python this.py -- --blend scene.blend
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_cameras import _load_extension


def _stroke_diagnostic(item, source, segment, result, *, endpoint_point_errors=None):
    """Measure one drawn stroke or From Points image chord at the applied endpoint."""
    import numpy as np

    from match_perspective.core.focal_projection import project_stroke_line

    calibration = result.calibrations.get(item.match_id)
    similarity = result.similarities.get(item.match_id)
    if calibration is None or similarity is None or segment is None:
        return None
    first, second = [np.asarray(value, dtype=float).reshape(3) for value in segment]
    endpoints = np.asarray(((item.u1, item.v1), (item.u2, item.v2)), dtype=float)
    stroke = endpoints[1] - endpoints[0]
    stroke_length = float(np.linalg.norm(stroke))
    line = project_stroke_line(
        0.5 * (first + second), second - first,
        calibration.rotation_w2c @ similarity.rotation.T,
        similarity.transform_point(calibration.camera_center),
        calibration.intrinsics, endpoints,
        division_lambda=calibration.division_lambda,
        brown_conrady=calibration.brown_conrady,
    )
    record = dict(
        match_id=item.match_id,
        source=source,
        stroke_length_px=stroke_length,
        endpoint_point_errors_px=endpoint_point_errors,
    )
    if line is None or not np.isfinite(line).all() or stroke_length <= 1.0e-12:
        record["projection_error"] = "line cannot be projected"
        return record
    distances = endpoints @ line[:2] + line[2]
    projected_direction = np.asarray((-line[1], line[0]), dtype=float)
    cosine = abs(float((stroke / stroke_length) @ projected_direction))
    record.update(
        angle_deg=float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))),
        endpoint_distances_px=distances.tolist(),
        endpoint_rms_px=float(np.sqrt(np.mean(distances * distances))),
        midpoint_offset_px=abs(float(np.mean(distances))),
    )
    return record


def _forced_distortion_probe(result, scorer, pick_sigma_px):
    """Search the production coefficient bounds while bypassing only coverage gates."""
    import numpy as np

    from match_perspective import core
    from match_perspective.core import distortion_refine
    from match_perspective.core.sync.projection import _project_shared_points

    by_match = defaultdict(list)
    for item in scorer.point_observations:
        if item.match_id in result.calibrations and item.landmark_id in result.landmarks:
            by_match[item.match_id].append(item)
    forced = dict(result.calibrations)
    forced_full = dict(result.calibrations)
    cameras = {}
    for match_id, calibration in result.calibrations.items():
        items = by_match.get(match_id, [])
        if len(items) < distortion_refine.MIN_DISTORTION_PICKS:
            cameras[match_id] = dict(
                supported_point_picks=len(items),
                reason=f"needs at least {distortion_refine.MIN_DISTORTION_PICKS} supported point picks",
            )
            continue
        observed = np.asarray([(item.u, item.v) for item in items], dtype=float)
        intrinsics = calibration.intrinsics
        radii = np.sqrt(
            ((observed[:, 0] - intrinsics.cx) / intrinsics.fx) ** 2
            + ((observed[:, 1] - intrinsics.cy) / intrinsics.fy) ** 2
        )
        order = np.argsort(radii, kind="stable")
        validation_indices = order[distortion_refine.VALIDATION_STRIDE - 1::
                                   distortion_refine.VALIDATION_STRIDE]
        validation_mask = np.zeros(len(items), dtype=bool)
        validation_mask[validation_indices] = True
        fit_mask = ~validation_mask
        points = np.asarray([result.landmarks[item.landmark_id] for item in items])
        pinhole = distortion_refine._calibration_at_lambda(calibration, 0.0)
        ideal, valid = _project_shared_points(
            points, pinhole, result.similarities[match_id])
        if not np.all(valid) or not np.isfinite(ideal).all():
            cameras[match_id] = dict(
                supported_point_picks=len(items),
                reason="fitted points do not project in front of the camera",
            )
            continue
        weights = np.asarray([max(float(item.weight), 1.0e-12) for item in items])

        def errors_at(value):
            projected = core.distort_points(
                ideal, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                float(value), (),
            )
            return projected - observed

        start = float(calibration.division_lambda)
        low = max(-distortion_refine.MAX_DIVISION_LAMBDA,
                  start - distortion_refine.MAX_DIVISION_LAMBDA_CHANGE)
        high = min(distortion_refine.MAX_DIVISION_LAMBDA,
                   start + distortion_refine.MAX_DIVISION_LAMBDA_CHANGE)
        samples = np.linspace(low, high, 361)
        losses = []
        full_losses = []
        for value in samples:
            if not distortion_refine._division_valid_over_image(intrinsics, float(value)):
                losses.append(float("inf"))
                full_losses.append(float("inf"))
                continue
            all_errors = errors_at(float(value))
            error = all_errors[fit_mask]
            losses.append(float(np.sum(weights[fit_mask] * np.sum(error * error, axis=1))))
            full_losses.append(float(np.sum(weights * np.sum(all_errors * all_errors, axis=1))))
        best_index = int(np.argmin(losses))
        candidate = float(samples[best_index])
        full_best_index = int(np.argmin(full_losses))
        full_candidate = float(samples[full_best_index])
        before = errors_at(start)
        after = errors_at(candidate)

        def rmse(errors, mask):
            return distortion_refine._weighted_rmse(errors[mask], weights[mask])

        before_fit, after_fit = rmse(before, fit_mask), rmse(after, fit_mask)
        before_validation = rmse(before, validation_mask)
        after_validation = rmse(after, validation_mask)
        before_full = distortion_refine._weighted_rmse(before, weights)
        after_full = distortion_refine._weighted_rmse(after, weights)
        absolute_gain = max(
            distortion_refine.MIN_SIGMA_RMSE_GAIN * float(pick_sigma_px), 0.05)

        def supported(before_rmse, after_rmse):
            return (before_rmse - after_rmse >= absolute_gain and
                    after_rmse <= before_rmse *
                    (1.0 - distortion_refine.MIN_RELATIVE_RMSE_GAIN))

        projected_before = observed + before
        projected_after = observed + after
        correction = np.linalg.norm(projected_after - projected_before, axis=1)
        bound_corrections = []
        for bound in (low, high):
            if distortion_refine._division_valid_over_image(intrinsics, bound):
                bound_corrections.extend(np.linalg.norm(
                    errors_at(bound) - before, axis=1).tolist())
        forced[match_id] = distortion_refine._calibration_at_lambda(
            calibration, candidate)
        forced_full[match_id] = distortion_refine._calibration_at_lambda(
            calibration, full_candidate)
        cameras[match_id] = dict(
            supported_point_picks=len(items),
            validation_point_picks=int(np.sum(validation_mask)),
            start_lambda=start,
            best_lambda=candidate,
            best_full_lambda=full_candidate,
            at_search_bound=bool(best_index in (0, len(samples) - 1)),
            start_fit_rmse_px=before_fit,
            best_fit_rmse_px=after_fit,
            start_validation_rmse_px=before_validation,
            best_validation_rmse_px=after_validation,
            start_full_rmse_px=before_full,
            best_full_rmse_px=after_full,
            minimum_full_rmse_px=distortion_refine._weighted_rmse(
                errors_at(full_candidate), weights),
            validation_supports=bool(
                supported(before_fit, after_fit)
                and supported(before_validation, after_validation)
                and supported(before_full, after_full)
            ),
            max_correction_px=float(np.max(correction)),
            max_bound_correction_px=max(bound_corrections, default=0.0),
        )
    score = scorer.score(result, calibrations=forced)
    full_score = scorer.score(result, calibrations=forced_full)
    return dict(
        note="Diagnostic only: radial-coverage gates bypassed; geometry stayed frozen",
        cameras=cameras,
        combined_train_selected_score=dict(
            valid=score.valid, reason=score.reason,
            point_rmse_px=score.point_rmse_px,
            line_rmse_px=score.line_rmse_px,
            objective=score.objective,
        ),
        combined_full_selected_score=dict(
            valid=full_score.valid, reason=full_score.reason,
            point_rmse_px=full_score.point_rmse_px,
            line_rmse_px=full_score.line_rmse_px,
            objective=full_score.objective,
        ),
    )


def _applied_endpoint_report(prep, space, names, *, force_distortion=False):
    """Audit distortion and line residuals without solving or changing the scene."""
    import numpy as np

    from match_perspective import scene
    from match_perspective.core.distortion_refine import refine_division_distortion
    from match_perspective.core.joint_fit_score import JointFitScorer, supported_joint_request
    from match_perspective.core.sync.projection import _project_shared_points
    from match_perspective.core.sync.solve import solution_result_from_seed
    from match_perspective.core.sync.types import SyncLineObservation

    request = scene.collect_sync_request(bpy.context)
    seed = request.initial_solution
    if seed is None:
        return {"available": False, "reason": "no complete applied Sync solution"}
    result = solution_result_from_seed(seed)
    if result is None:
        return {"available": False, "reason": "applied Sync diagnostics are incomplete"}
    supported, coverage = supported_joint_request(request, result)
    frozen = seed.diagnostics.joint_point_weights if seed.diagnostics is not None else None
    scorer = JointFitScorer(
        supported, result, calibrations=result.calibrations,
        frozen_point_weights=frozen,
    )
    score = scorer.score(result, calibrations=result.calibrations)
    distortion = refine_division_distortion(
        result.calibrations, scorer.point_observations, result,
        pick_sigma_px=prep.pick_sigma_px,
    )
    forced_distortion = (
        _forced_distortion_probe(result, scorer, prep.pick_sigma_px)
        if force_distortion else None
    )

    strokes = []
    for item in supported.line_observations or ():
        record = _stroke_diagnostic(
            item, "DRAWN", result.line_segments.get(item.landmark_id), result)
        if record is not None:
            record["landmark_id"] = item.landmark_id
            record["landmark_name"] = names.get(item.landmark_id, item.landmark_id)
            strokes.append(record)

    point_picks = {(item.match_id, item.landmark_id): item
                   for item in scorer.point_observations}
    for line_id, point_a, point_b in supported.derived_lines or ():
        for match_id in sorted(result.similarities):
            first_pick = point_picks.get((match_id, point_a))
            second_pick = point_picks.get((match_id, point_b))
            if first_pick is None or second_pick is None:
                continue
            endpoints = np.asarray(((first_pick.u, first_pick.v),
                                    (second_pick.u, second_pick.v)), dtype=float)
            calibration = result.calibrations[match_id]
            predicted, valid = _project_shared_points(
                np.asarray((result.landmarks[point_a], result.landmarks[point_b])),
                calibration, result.similarities[match_id],
            )
            point_errors = None
            if np.all(valid) and np.isfinite(predicted).all():
                point_errors = np.linalg.norm(predicted - endpoints, axis=1).tolist()
            item = SyncLineObservation(
                match_id, line_id,
                float(endpoints[0, 0]), float(endpoints[0, 1]),
                float(endpoints[1, 0]), float(endpoints[1, 1]),
            )
            record = _stroke_diagnostic(
                item, "FROM_POINTS", result.line_segments.get(line_id), result,
                endpoint_point_errors=point_errors,
            )
            if record is not None:
                record["landmark_id"] = line_id
                record["landmark_name"] = names.get(line_id, line_id)
                record["point_a"] = names.get(point_a, point_a)
                record["point_b"] = names.get(point_b, point_b)
                strokes.append(record)

    line_settings = [dict(
        landmark_id=item.item_id,
        landmark_name=item.name or item.item_id,
        enabled=bool(getattr(item, "use_in_sync", True)),
        source=str(getattr(item, "line_source", "DRAWN") or "DRAWN"),
        point_a=names.get(str(getattr(item, "line_point_a", "NONE")),
                          str(getattr(item, "line_point_a", "NONE"))),
        point_b=names.get(str(getattr(item, "line_point_b", "NONE")),
                          str(getattr(item, "line_point_b", "NONE"))),
    ) for item in space.landmarks if item.kind == "LINE"]
    return dict(
        available=True,
        coverage=coverage,
        score=dict(valid=score.valid, reason=score.reason,
                   point_rmse_px=score.point_rmse_px,
                   line_rmse_px=score.line_rmse_px,
                   objective=score.objective),
        distortion=dict(
            accepted_match_ids=distortion.accepted_match_ids,
            skipped_reasons=distortion.skipped_reasons,
            cameras={key: asdict(value) for key, value in
                     (distortion.diagnostics or {}).items()},
        ),
        forced_distortion=forced_distortion,
        line_settings=line_settings,
        strokes=strokes,
    )


def report(*, force_distortion=False):
    from match_perspective import properties, scene
    prep = scene.collect_lens_refine_inputs(bpy.context)
    space = properties.workspace(bpy.context)
    names = {p.item_id: p.name for p in space.landmarks}
    kinds = {p.item_id: p.kind for p in space.landmarks}
    views = defaultdict(set)
    for pick in prep.observations:
        views[pick.landmark_id].add(pick.match_id)
    relations = {p[0] for p in (prep.plane_groups or [])}
    relations.update(p for pair in (prep.mirror_pairs or []) for p in pair)
    counts = Counter(p.match_id for p in prep.observations)
    images = []
    for root in properties.iter_match_roots():
        settings = root.pm_session
        images.append(dict(match_id=root.name,
            source_path=bpy.path.abspath(settings.image_path) if settings.image_path else None,
            stored_size=[settings.image_width, settings.image_height],
            source_width=settings.source_image_width,
            principal_point=[settings.cx, settings.cy],
            undistorted=bool(settings.view_undistorted)))
    result = dict(same_lens=prep.share_lens, point_focal=prep.estimate_focal_from_points,
        images=images,
        cameras=[dict(name=m.match_id, picks=counts[m.match_id], focal=m.intrinsics.fx,
                      vp_lines=sum(len(v) for v in m.line_bundles.values())) for m in prep.lens_inputs],
        points=len(views), known_points=len(prep.known_world),
        ground_picks=sum(p.on_ground for p in prep.observations),
        line_picks=len(prep.line_observations),
        plane_groups=[(names.get(p, p), axis, bucket) for p, axis, bucket in (prep.plane_groups or [])],
        mirror_pairs=[(names.get(a, a), names.get(b, b)) for a, b in (prep.mirror_pairs or [])],
        parallel_pairs=[(names.get(a, a), names.get(b, b)) for a, b in (prep.parallel_pairs or [])],
        line_landmarks=sorted({names.get(p.landmark_id, p.landmark_id)
                                           for p in prep.line_observations}),
        constrained_landmarks=[dict(id=p, name=names.get(p, p), kind=kinds.get(p), views=sorted(views[p]))
                            for p in sorted(relations)],
        insufficient_points=[dict(id=p, name=names.get(p, p), views=sorted(views[p]))
                             for p in sorted(set(views) | relations)
                             if kinds.get(p) == 'POINT' and len(views[p]) < 2])
    result["applied_endpoint"] = _applied_endpoint_report(
        prep, space, names, force_distortion=force_distortion)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', required=True)
    parser.add_argument('--out')
    parser.add_argument('--fit-seconds', type=float, default=0,
                        help='Optionally run one fit, with cooperative timeout; never apply it')
    parser.add_argument('--force-distortion-probe', action='store_true',
                        help='At the applied endpoint, bypass only radial-coverage gates')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    _load_extension()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).expanduser().resolve()))
    if not 0 <= args.fit_seconds <= 600:
        parser.error('--fit-seconds must be between 0 and 600')
    result = report(force_distortion=args.force_distortion_probe)

    def emit():
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.out:
            Path(args.out).write_text(text + '\n', encoding='utf-8')

    emit()
    if args.fit_seconds:
        from match_perspective import scene
        from match_perspective.core import lens_refine
        from match_perspective.core.sync.request import json_values
        prep = scene.collect_lens_refine_inputs(bpy.context)
        started = time.monotonic()
        result['fit'] = dict(status='started', budget_seconds=args.fit_seconds,
                             input_sha256=prep.source_request_sha256)
        emit()
        if args.out:
            Path(args.out + '.inputs.json').write_text(json.dumps(json_values(prep.solver_kwargs()), indent=2))
        original_sync = lens_refine._run_sync

        def capture_startup(*values, **options):
            initial = original_sync(*values, **options)
            result['fit']['registered_cameras'] = sorted(initial.similarities)
            result['fit']['startup_message'] = initial.message
            if args.out:
                Path(args.out + '.startup.json').write_text(json.dumps(json_values(initial), indent=2))
            emit()
            return initial

        with patch.object(lens_refine, '_run_sync', side_effect=capture_startup):
            fitted = scene.run_lens_refine(
                prep, cancel_check=lambda: time.monotonic() - started > args.fit_seconds)
        result['fit'].update(status='completed', seconds=time.monotonic() - started,
            accepted=fitted.improved, refusal=fitted.refusal_reason,
            message=fitted.message, focal_intervals=fitted.focal_intervals)
        candidate = fitted.candidate
        result['fit']['candidate'] = (dict(
            initial_rmse_px=candidate.initial_rmse_px,
            fitted_rmse_px=candidate.fitted_rmse_px, reason=candidate.reason,
            cameras=sorted(candidate.calibrations)) if candidate else None)
        if args.out:
            Path(args.out + '.fit.json').write_text(
                json.dumps(json_values(fitted), indent=2) + '\n', encoding='utf-8')
        emit()


if __name__ == '__main__':
    main()
