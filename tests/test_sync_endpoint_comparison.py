"""Read-only endpoint audits must use certified, persisted fit weights."""

from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from match_perspective.core import joint_fit_score
from match_perspective.core.sync.request import json_values
from match_perspective.core.sync.types import SyncAppliedDiagnostics, SyncSolutionSeed
from test_joint_fit_features import _fixture
import test_joint_fit_score as score_tests
from tools.synthetic_sync.sync_continuation import _compare_private_endpoints


class EndpointComparisonTests(TestCase):
    def _reports(self, directory):
        request, initial, points, centers = _fixture(known_line=True)
        result = score_tests.JointFitScoreTests()._truth(request, initial, points, centers)
        result.calibrations = {item.match_id: item.calibration for item in request.matches}
        scorer = joint_fit_score.JointFitScorer(request, result)
        result.joint_point_weights = scorer.effective_point_weights
        diagnostics = SyncAppliedDiagnostics(**{
            item.name: deepcopy(getattr(result, item.name))
            for item in fields(SyncAppliedDiagnostics)})
        request.initial_solution = SyncSolutionSeed(
            calibrations=deepcopy(result.calibrations),
            similarities=deepcopy(result.similarities),
            landmarks=deepcopy(result.landmarks),
            line_segments=deepcopy(result.line_segments),
            evidence_sha256=request.evidence_sha256(), diagnostics=diagnostics)
        for stage in ("refine", "solve1", "solve2"):
            (directory / f"{stage}-result.json").write_text(json.dumps({
                "result": json_values(result), "post_request": request.to_record()}))
        return result.joint_point_weights

    def test_identical_applied_endpoints_use_the_persisted_objective(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            weights = self._reports(out)
            original = joint_fit_score.JointFitScorer
            with patch.object(joint_fit_score, "JointFitScorer", wraps=original) as scorer:
                report = _compare_private_endpoints(out)
            self.assertEqual(scorer.call_args.kwargs["frozen_point_weights"], weights)
            self.assertTrue(report["effective_point_weights_preserved"])
            self.assertTrue(report["objective_nonincreasing"])
            self.assertTrue(all(item["point_projection_max_px"] == 0
                                and item["line_overlay_max_px"] == 0
                                for item in report["movements"]))

    def test_changed_effective_weights_are_not_a_comparable_sequence(self):
        from match_perspective.core.sync.request import request_fingerprint
        with TemporaryDirectory() as temp:
            out = Path(temp)
            self._reports(out)
            path = out / "solve1-result.json"
            report = json.loads(path.read_text())
            request = report["post_request"]
            request["inputs"]["initial_solution"]["diagnostics"]["joint_point_weights"][0][2] *= 0.5
            request["sha256"] = request_fingerprint(request["inputs"])
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(AssertionError, "Effective point weights changed"):
                _compare_private_endpoints(out)
