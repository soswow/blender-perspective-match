"""Bounded known-line/distortion joint-fit probe with independent oracle pixels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
if "match_perspective" not in sys.modules:
    import types
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(ROOT)]
    package.__file__ = str(ROOT / "__init__.py")
    sys.modules["match_perspective"] = package

from .budget import ExperimentBudget
from test_joint_fit_features import _fixture, _pixel

from match_perspective.core import focal_bundle
from match_perspective.core.joint_fit_score import JointFitScorer
from match_perspective.core.sync.request import json_values


def main() -> None:
    sources = sorted((ROOT / "core").rglob("*.py")) + [
        ROOT / "tests/test_joint_fit_features.py", Path(__file__),
        ROOT / "tools/synthetic_sync/budget.py",
    ]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    metadata = {"sources": hashes, "python": sys.version,
                "numpy": np.__version__, "platform": platform.platform()}
    source_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    archive = Path("/tmp/pm-joint-fit-tests/round2/source") / source_key
    for path in sources:
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[str(path.relative_to(ROOT))]

    request, initial, truth, centers = _fixture(distortion=True, known_line=True)
    oracle = json_values(initial)
    from match_perspective.core.sync import SimilarityTransform
    initial_oracle = initial.__class__(
        similarities={key: SimilarityTransform(translation=(centers[key] - match.calibration.camera_center))
                      for key, match in ((item.match_id, item) for item in request.matches)},
        landmarks=truth, line_segments=initial.line_segments,
        mean_reprojection_px=0.0, per_match_rmse_px={}, per_landmark_rmse_px={},
        message="independent oracle")
    scorer = JointFitScorer(request, initial)
    oracle_score = scorer.score(initial_oracle)
    exact_input = {"request": request.to_record(), "initial": oracle,
                   "withheld_points": [[-0.47, 0.11, 0.37], [0.18, -0.37, 0.92]],
                   "oracle_score": json_values(oracle_score)}
    diagnostic = {}
    ledger = Path("/tmp/pm-joint-fit-tests/round2") / f"known-line-{source_key}.jsonl"
    with ExperimentBudget(ledger, metadata=metadata, max_calls=8,
                          wall_seconds=600.0, per_call_seconds=120.0) as budget:
        with budget.attempt("perturbed-known-line-distortion", exact_input) as attempt:
            outcome = focal_bundle.refine_fixed_focals(
                request, initial, diagnostic_callback=diagnostic.update)
            result = outcome.sync_result
            final_score = scorer.score(result, calibrations=outcome.calibrations) if result else None
            withheld = {}
            if result:
                for match in request.matches:
                    camera_id = match.match_id
                    expected_sim = initial_oracle.similarities[camera_id]
                    withheld[camera_id] = [float(np.linalg.norm(
                        _pixel(np.array(point), outcome.calibrations[camera_id],
                               result.similarities[camera_id]) -
                        _pixel(np.array(point), match.calibration, expected_sim)))
                        for point in exact_input["withheld_points"]]
            record = {"accepted": outcome.accepted, "reason": outcome.reason,
                      "initial_rmse_px": (outcome.initial_rmse_px if math.isfinite(outcome.initial_rmse_px) else None),
                      "fitted_rmse_px": (outcome.fitted_rmse_px if math.isfinite(outcome.fitted_rmse_px) else None),
                      "oracle_score": json_values(oracle_score),
                      "final_score": json_values(final_score),
                      "diagnostic": json_values(diagnostic),
                      "withheld_pixel_errors": withheld,
                      "calibrations": json_values(outcome.calibrations),
                      "result": json_values(result)}
            attempt.complete(record)
    print(json.dumps({key: record[key] for key in (
        "accepted", "reason", "initial_rmse_px", "fitted_rmse_px",
        "oracle_score", "final_score", "diagnostic", "withheld_pixel_errors")},
                     indent=2))
    print(f"Ledger: {ledger}\nSource archive: {archive}")


if __name__ == "__main__":
    main()
