"""Geometry accuracy, separate from the solver's fitted-pick error."""

from __future__ import annotations

import numpy as np

from .geometry import project


def alignment(case: dict, record: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """At most ONE proper global similarity, using training landmarks only."""
    if case["expectation"]["gauge"] == "anchor":
        return 1.0, np.eye(3), np.zeros(3)
    ids = sorted(set(record["landmarks"]) & set(case["truth"]["points"]))
    if len(ids) < 4:
        raise ValueError("Need four reconstructed training landmarks to align an unscaled scene")
    source = np.array([record["landmarks"][key] for key in ids])
    target = np.array([case["truth"]["points"][key] for key in ids])
    a, b = source.mean(axis=0), target.mean(axis=0)
    x, y = source - a, target - b
    if np.linalg.matrix_rank(x, tol=1e-8) < 2:
        raise ValueError("Training landmarks do not constrain a global alignment")
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float((singular @ sign) / np.sum(x*x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid global scale")
    return scale, rotation, b - scale * rotation @ a


def aligned_camera(camera, transform):
    scale, rotation, translation = transform
    return dict(camera, center=(scale * rotation @ camera["center"] + translation).tolist(),
                rotation=(np.asarray(camera["rotation"]) @ rotation.T).tolist())


def evaluate(case: dict, record: dict) -> dict:
    """Require camera coverage and held-out geometry; never trust reported RMSE."""
    expectation = case["expectation"]
    violations = []
    output = dict(passed=False, violations=violations, cameras={}, gauge=expectation["gauge"],
                  expected_outcome=expectation["outcome"], warnings=[])
    if record.get("exception"):
        violations.append("Solver raised an exception; this is not a useful refusal")
        return output
    if expectation["outcome"] == "reject":
        if record["success"]:
            violations.append("Accepted a case explicitly requiring refusal")
        if not record["message"].strip():
            violations.append("Refusal supplied no explanation")
        output["passed"] = not violations
        return output
    if not record["success"]:
        violations.append("Solver rejected a constrained case: " + record["message"])
    expected_weak = set(expectation.get("weak_lines", []))
    reported_weak = set(record.get("weak_line_ids", []))
    for key in sorted(expected_weak-reported_weak):
        violations.append(f"{key}: weak geometry was not reported")
    for key in expectation["excluded_cameras"]:
        if record["success"] and key in record["cameras"]:
            violations.append(f"{key}: disconnected camera was reported as registered")
    stored = {c["id"]: c for c in case["request"]["cameras"]}
    for key, fixed in case["request"]["fixed_similarities"].items():
        if key not in record["cameras"]:
            continue  # Required-camera check below explains this failure.
        wanted = aligned_camera(stored[key], (fixed["scale"], np.asarray(fixed["rotation"]), np.asarray(fixed["translation"])))
        if any(not np.allclose(record["cameras"][key][field], wanted[field], atol=1e-5, rtol=1e-6)
               for field in ("center", "rotation")):
            violations.append(f"{key}: explicitly locked camera moved")
    try:
        transform = alignment(case, record)
    except (ValueError, np.linalg.LinAlgError) as error:
        violations.append(str(error))
        return output
    output["alignment"] = dict(scale=transform[0], rotation=transform[1].tolist(), translation=transform[2].tolist())
    extent = float(np.linalg.norm(np.ptp(np.asarray(case["truth"]["mesh"]["vertices"]), axis=0)))
    output["geometry"] = dict(points={}, lines={})
    for field, kind in (("landmarks", "points"), ("line_segments", "lines")):
        for key in expectation.get("excluded_" + kind, []):
            if key in record[field]:
                violations.append(f"{key}: reconstructed an unconstrained single-view {kind[:-1]}")
    scale, rotation, translation = transform
    for key in expectation.get("required_points", []):
        if key not in record["landmarks"]:
            violations.append(f"{key}: required reconstructed point missing")
            continue
        point = scale * rotation @ record["landmarks"][key] + translation
        error = float(np.linalg.norm(point-case["truth"]["points"][key]) / extent)
        output["geometry"]["points"][key] = dict(error_fraction=error)
        if not np.isfinite(error) or error > expectation["point_fraction"]:
            violations.append(f"{key}: reconstructed point error {error:.4g} of object diagonal")
    for key in expectation.get("required_lines", []):
        if key not in record["line_segments"]:
            violations.append(f"{key}: required reconstructed line missing")
            continue
        ends = scale * np.asarray(record["line_segments"][key]) @ rotation.T + translation
        reference = np.asarray(case["truth"]["lines"][key])
        direction, truth_direction = ends[1]-ends[0], reference[1]-reference[0]
        length, truth_length = np.linalg.norm(direction), np.linalg.norm(truth_direction)
        if not np.isfinite(ends).all() or length < 1e-9 or truth_length < 1e-9:
            violations.append(f"{key}: invalid reconstructed line")
            continue
        direction, truth_direction = direction/length, truth_direction/truth_length
        angle = float(np.degrees(np.arccos(np.clip(abs(direction @ truth_direction),0,1))))
        # Different stroke extents do not identify matching 3D endpoints.
        distances = np.linalg.norm(np.cross(reference-ends.mean(axis=0),direction),axis=1)
        error = float(distances.max()/extent)
        output["geometry"]["lines"][key] = dict(angle_deg=angle, offset_fraction=error)
        if key in expected_weak and key in reported_weak:
            output["warnings"].append(f"{key}: expected weak geometry; actual direction error {angle:.4g}deg, offset {error:.4g} of object diagonal")
        elif angle > expectation["line_angle_deg"] or error > expectation["line_fraction"]:
            violations.append(f"{key}: reconstructed line angle {angle:.4g}deg, offset {error:.4g} of object diagonal")
    truth_cameras = {c["id"]: c for c in case["truth"]["cameras"]}
    for key in expectation["cameras"]:
        if key not in record["cameras"]:
            violations.append(f"{key}: required camera missing")
            continue
        truth = truth_cameras[key]
        estimated = aligned_camera(record["cameras"][key], transform)
        points = [p["position"] for p in case["truth"]["checks"] if key in p["views"]]
        if len(points) < 6:
            violations.append(f"{key}: invalid test case, fewer than six held-out checks")
            continue
        expected_uv, _depth = project(points, truth)
        actual_uv, depth = project(points, estimated)
        valid = bool(np.isfinite(actual_uv).all() and np.all(depth > 0))
        errors = np.linalg.norm(actual_uv - expected_uv, axis=1) if valid else np.full(len(points), 1e9)
        rotation_delta = np.asarray(estimated["rotation"]) @ np.asarray(truth["rotation"]).T
        rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(rotation_delta)-1)/2, -1, 1))))
        center_fraction = float(np.linalg.norm(np.asarray(estimated["center"])-truth["center"]) / extent)
        metrics = dict(holdout_rmse_px=float(np.sqrt(np.mean(errors**2))), holdout_max_px=float(errors.max()),
            rotation_deg=rotation_deg, center_fraction=center_fraction, check_count=len(points),
            all_in_front=valid, expected_uv=expected_uv.tolist(), actual_uv=actual_uv.tolist() if valid else None)
        output["cameras"][key] = metrics
        for name in ("holdout_rmse_px", "rotation_deg", "center_fraction"):
            if not np.isfinite(metrics[name]) or metrics[name] > expectation[name]:
                violations.append(f"{key}: {name} {metrics[name]:.4g} > {expectation[name]:.4g}")
        if not valid:
            violations.append(f"{key}: held-out points are behind camera or non-finite")
    output["passed"] = not violations
    return output
