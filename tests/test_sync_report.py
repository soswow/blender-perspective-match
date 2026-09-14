"""Structured, portable HTML sync diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "match_perspective.ui.sync_report",
    _ROOT / "ui" / "sync_report.py",
)
assert _SPEC is not None and _SPEC.loader is not None
sync_report = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sync_report
_SPEC.loader.exec_module(sync_report)


def _point(match_id: str, landmark_id: str, name: str = ""):
    return SimpleNamespace(
        match_id=match_id,
        landmark_id=landmark_id,
        landmark_name=name,
        on_ground=False,
    )


class SyncReportTests(unittest.TestCase):
    """The browser report should stay structured, safe, and self-contained."""

    def test_plane_derived_point_explains_its_depth_source(self):
        result = SimpleNamespace(success=True, similarities={"anchor":object(),"side":object()},
            landmarks={"point":object()}, mean_reprojection_px=0., per_match_rmse_px={"side":0.},
            per_landmark_rmse_px={"point":0.}, plane_seeded_landmark_ids=["point"])
        report = sync_report.build_sync_report(operation="Diagnose", source_name="synthetic.blend",
            matches=[SimpleNamespace(match_id="anchor"),SimpleNamespace(match_id="side")],
            observations=[_point("side","point","<surface point>")], line_observations=[], result=result, anchor_id="anchor")
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Plane + one view", html)
        self.assertIn("does not independently verify depth", html)
        self.assertIn("&lt;surface point&gt;", html)
        self.assertNotIn("<surface point>", html)

    def test_low_error_weak_line_is_marked_for_review_and_escaped(self):
        result = SimpleNamespace(success=True, similarities={"anchor":object(),"side":object()},
            landmarks={"line":object()}, mean_reprojection_px=0.2,
            per_match_rmse_px={"side":0.2},per_landmark_rmse_px={"line":0.1},
            weak_line_ids=["line"],line_support_angles_deg={"line":1.2})
        report = sync_report.build_sync_report(operation="Diagnose",source_name="synthetic.blend",
            matches=[SimpleNamespace(match_id="anchor"),SimpleNamespace(match_id="side")],
            observations=[],line_observations=[SimpleNamespace(match_id="side",landmark_id="line",landmark_name="<edge>")],
            result=result,anchor_id="anchor")
        self.assertEqual(report.severity,"warning")
        self.assertIn("weak line",report.outcome)
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Weak 3D support (1.2°)",html)
        self.assertIn("&lt;edge&gt;",html)
        self.assertNotIn("<edge>",html)
        self.assertIn("confidence interval",html)

    def _partial_report(self):
        matches = [
            SimpleNamespace(match_id="anchor"),
            SimpleNamespace(match_id="side"),
            SimpleNamespace(match_id="back"),
        ]
        observations = []
        for index in range(5):
            observations.append(_point("anchor", f"p{index}"))
            observations.append(_point("side", f"p{index}"))
        for index in range(3):
            name = "<img src=x onerror=alert(1)>" if index == 0 else f"Rear {index}"
            observations.append(_point("side", f"rear{index}", name))
            observations.append(_point("back", f"rear{index}", name))
        result = SimpleNamespace(
            similarities={"anchor": object(), "side": object()},
            landmarks={f"p{index}": object() for index in range(5)},
            mean_reprojection_px=9.25,
            per_match_rmse_px={"anchor": 1.0, "side": 3.0},
            per_landmark_rmse_px={"rear0": 14.0, "p0": 2.0},
            message="solver internals stay available",
            success=True,
            downweighted_landmark_ids=["rear0"],
            bundle_adjusted=True,
            leave_one_out=[],
            inconsistent_picks=[],
        )
        return sync_report.build_sync_report(
            operation="Diagnose",
            source_name="sample.blend",
            matches=matches,
            observations=observations,
            line_observations=[],
            result=result,
            anchor_id="anchor",
            all_match_labels={
                "anchor": "Anchor",
                "side": "Side",
                "back": "Back",
                "disabled": "Disabled",
            },
            disabled_match_ids={"disabled"},
            fixed_match_ids={"side"},
            excluded_landmarks=2,
        )

    def test_partial_report_names_best_graph_route_and_deficit(self) -> None:
        report = self._partial_report()
        self.assertEqual(report.outcome, "Partial sync")
        self.assertEqual(report.registered_matches, 2)
        self.assertEqual(report.enabled_matches, 3)
        back = next(item for item in report.matches if item.match_id == "back")
        self.assertEqual(back.status, "skipped")
        self.assertEqual(back.best_reference, "Side")
        self.assertEqual(back.best_shared_points, 3)
        issue = next(item for item in report.issues if "Back" in item.title)
        self.assertIn("3 of 5 shared", issue.detail)
        self.assertIn("Add 2", issue.action)
        self.assertIn("2/3 cameras", sync_report.compact_status(report))

    def test_sufficient_overlap_is_not_presented_as_a_fraction(self) -> None:
        self.assertEqual(
            sync_report._shared_point_summary(13),
            "13 shared; minimum 5",
        )

    def test_graph_uses_strongest_non_cyclic_overlap_backbone(self) -> None:
        report = self._partial_report()
        report.edges = [
            sync_report.ReportEdge("anchor", "side", 9),
            sync_report.ReportEdge("anchor", "back", 5),
            sync_report.ReportEdge("side", "back", 7),
        ]
        backbone = sync_report._backbone_edges(report)
        self.assertEqual([item.shared_points for item in backbone], [9, 7])
        payload = sync_report._camera_graph_payload(report)
        self.assertEqual([edge["shared"] for edge in payload["edges"]], [9, 7])
        self.assertEqual(len(payload["nodes"]), 4)
        xs = [node["x"] for node in payload["nodes"]]
        ys = [node["y"] for node in payload["nodes"]]
        self.assertLess(max(xs) - min(xs), 1200)
        self.assertLess(max(ys) - min(ys), 800)
        self.assertTrue(all("tooltip" in edge for edge in payload["edges"]))
        self.assertIn("pairwise 2D↔2D", payload["edges"][0]["tooltip"])

    def test_failed_result_does_not_count_identity_placeholders_as_registered(self) -> None:
        matches = [
            SimpleNamespace(match_id="anchor"),
            SimpleNamespace(match_id="other"),
        ]
        result = SimpleNamespace(
            similarities={"anchor": object(), "other": object()},
            landmarks={},
            mean_reprojection_px=0.0,
            per_match_rmse_px={},
            per_landmark_rmse_px={},
            message="Need more support",
            success=False,
            downweighted_landmark_ids=[],
            bundle_adjusted=False,
            leave_one_out=[],
            inconsistent_picks=[],
        )
        report = sync_report.build_sync_report(
            operation="Diagnose",
            source_name="sample.blend",
            matches=matches,
            observations=[],
            line_observations=[],
            result=result,
            anchor_id="anchor",
        )
        self.assertEqual(report.registered_matches, 1)
        self.assertEqual(report.outcome, "Sync failed")
        other = next(item for item in report.matches if item.match_id == "other")
        self.assertEqual(other.status, "skipped")

    def test_metric_route_prevents_wrong_add_shared_points_advice(self) -> None:
        matches = [
            SimpleNamespace(match_id="anchor"),
            SimpleNamespace(match_id="other"),
        ]
        observations = []
        for index in range(3):
            observations.extend(
                (_point("anchor", f"free{index}"), _point("other", f"free{index}"))
            )
            observations.append(_point("other", f"known{index}"))
        result = SimpleNamespace(
            similarities={"anchor": object()},
            landmarks={},
            mean_reprojection_px=2.0,
            per_match_rmse_px={"anchor": 1.0},
            per_landmark_rmse_px={},
            message="Other did not register",
            success=True,
            downweighted_landmark_ids=[],
            bundle_adjusted=False,
            leave_one_out=[],
            inconsistent_picks=[],
        )
        report = sync_report.build_sync_report(
            operation="Diagnose",
            source_name="sample.blend",
            matches=matches,
            observations=observations,
            line_observations=[],
            result=result,
            anchor_id="anchor",
            known_world={f"known{index}": object() for index in range(3)},
        )
        other = next(item for item in report.matches if item.match_id == "other")
        self.assertEqual(other.usable_3d_points, 3)
        issue = next(item for item in report.issues if "other" in item.title)
        self.assertIn("minimum registration route is present", issue.action)
        self.assertNotIn("Add 2", issue.action)

    def test_html_is_self_contained_interactive_and_escapes_names(self) -> None:
        html = sync_report.render_sync_report_html(self._partial_report())
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("Camera connectivity", html)
        self.assertIn("landmark-search", html)
        self.assertIn("match-table", html)
        self.assertIn('data-sort="rmse"', html)
        self.assertIn("camera-graph-tooltip", html)
        self.assertIn("<strong>Green</strong>", html)
        self.assertIn("<strong>Gray</strong>", html)
        self.assertIn("preset", html)
        self.assertIn("The Cytoscape Consortium", html)
        self.assertNotIn("__VENDOR_CYTOSCAPE__", html)
        self.assertNotIn('data-sort="best"', html)
        self.assertIn("Print / Save PDF", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
        self.assertNotIn("<img src=x onerror=alert(1)>", html)
        self.assertNotIn("https://", html)

    def test_temp_reports_are_unique_pruned_and_exportable(self) -> None:
        report = self._partial_report()
        with TemporaryDirectory() as directory:
            paths = [
                sync_report.write_temp_report(report, temp_root=directory, keep=2)
                for _index in range(3)
            ]
            self.assertEqual(len(set(paths)), 3)
            self.assertEqual(len(list(paths[-1].parent.glob("*.html"))), 2)
            self.assertEqual(sync_report.last_report_path(), paths[-1])
            exported = sync_report.export_last_report(
                Path(directory) / "permanent-report"
            )
            self.assertEqual(exported.suffix, ".html")
            self.assertTrue(exported.is_file())

    def test_applied_report_separates_point_and_line_error(self) -> None:
        result = SimpleNamespace(
            success=True, similarities={"anchor": object(), "side": object()},
            landmarks={"point": object(), "line": object()},
            mean_reprojection_px=2.0, point_rmse_px=2.0, line_rmse_px=7.5,
            per_match_rmse_px={"anchor": 1.0, "side": 2.0},
            per_match_point_rmse_px={"anchor": 1.0, "side": 2.0},
            per_match_line_rmse_px={"side": 7.5},
            per_landmark_rmse_px={"point": 2.0, "line": 7.5},
        )
        report = sync_report.build_sync_report(
            operation="Solve Sync", source_name="generated.blend", result=result,
            matches=[SimpleNamespace(match_id="anchor"), SimpleNamespace(match_id="side")],
            observations=[_point("side", "point")],
            line_observations=[SimpleNamespace(match_id="side", landmark_id="line")],
            anchor_id="anchor", applied=True,
            calibrations={"side": SimpleNamespace(hfov_degrees=51.25)},
        )
        self.assertEqual(report.point_rmse_px, 2.0)
        self.assertEqual(report.line_rmse_px, 7.5)
        self.assertEqual(report.matches[1].line_rmse_px, 7.5)
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Applied to this scene", html)
        self.assertIn("Point RMSE", html)
        self.assertIn("Line RMSE", html)
        self.assertNotIn("Solve RMSE", html)
        self.assertIn("51.25° HFOV", html)

    def test_refused_report_has_no_invented_errors(self) -> None:
        report = sync_report.build_sync_report(
            operation="Refine Lenses", source_name="generated.blend", result=None,
            matches=[SimpleNamespace(match_id="anchor"), SimpleNamespace(match_id="side")], observations=[],
            line_observations=[], anchor_id="anchor", applied=False,
            evaluation_note="Focal fit refused",
        )
        self.assertIsNone(report.rmse_px)
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Refine not applied", html)
        self.assertIn("Not applied to this scene", html)
        self.assertIn("Focal fit refused", html)
        self.assertIn("<strong>—</strong>", html)
        self.assertIn("Not assessed", html)
        self.assertNotIn("could not register", html)

    def test_partial_legacy_refine_describes_lens_application(self) -> None:
        result = SimpleNamespace(
            success=False, similarities={"anchor": object()}, landmarks={},
            mean_reprojection_px=12.0, per_match_rmse_px={"anchor": 12.0},
            per_landmark_rmse_px={}, message="Sync refused",
        )
        report = sync_report.build_sync_report(
            operation="Refine Lenses", source_name="generated.blend", result=result,
            matches=[SimpleNamespace(match_id="anchor")], observations=[],
            line_observations=[], anchor_id="anchor", applied=False,
            application_state="Lens settings applied; Sync geometry not applied",
            calibrations={"anchor": SimpleNamespace(hfov_degrees=48.0)},
        )
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Lens settings applied; Sync geometry not applied", html)
        self.assertNotIn("Not applied to this scene", html)
        self.assertIn("48.00° HFOV", html)

    def test_best_fit_report_stays_provisional(self) -> None:
        result = SimpleNamespace(
            success=True, similarities={"anchor": object()}, landmarks={},
            mean_reprojection_px=2.0, per_match_rmse_px={"anchor": 2.0},
            per_landmark_rmse_px={}, message="Provisional geometry",
        )
        report = sync_report.build_sync_report(
            operation="Use Best Fit", source_name="generated.blend", result=result,
            matches=[SimpleNamespace(match_id="anchor")], observations=[],
            line_observations=[], anchor_id="anchor", applied=True,
        )
        self.assertEqual(report.outcome, "Provisional fit applied")
        self.assertEqual(report.severity, "warning")

    def test_common_removal_shows_same_support_objective_and_both_errors(self) -> None:
        coverage = dict(
            requested_point_picks=8, fitted_point_picks=7,
            requested_line_strokes=3, fitted_line_strokes=2,
            skipped_camera_ids=["side"], skipped_point_ids=[],
            skipped_line_ids=["edge"], skipped_relation_ids=["parallel:edge:other"],
        )
        item = SimpleNamespace(
            landmark_id="point", landmark_name="<suspect>", kind="point",
            removed_relations=["mirror:point:edge", "plane:X:1:point"],
            baseline_objective=120.0, candidate_objective=80.0,
            baseline_point_rmse_px=2.0, candidate_point_rmse_px=3.0,
            baseline_line_rmse_px=9.0, candidate_line_rmse_px=4.0,
            baseline_valid=True, candidate_valid=True, reason="",
            support_coverage=coverage,
        )
        common = SimpleNamespace(
            items=[item], baseline_coverage=coverage, candidates_ranked=4,
            incomplete=False, reason="",
        )
        result = SimpleNamespace(
            success=True, similarities={"anchor": object(), "side": object()},
            landmarks={"point": object()}, mean_reprojection_px=2.0,
            per_match_rmse_px={"anchor": 2.0, "side": 2.0},
            per_landmark_rmse_px={"point": 2.0},
            common_leave_one_out=common,
            leave_one_out=[("<suspect>", 8.0, 2.0)],
        )
        report = sync_report.build_sync_report(
            operation="Investigate Problems", source_name="generated.blend",
            matches=[SimpleNamespace(match_id="anchor"), SimpleNamespace(match_id="side")],
            observations=[_point("anchor", "point", "<suspect>")],
            line_observations=[SimpleNamespace(match_id="side", landmark_id="edge", landmark_name="<edge>")],
            result=result, anchor_id="anchor", applied=False,
            all_match_labels={"side": "Side camera"},
        )
        html = sync_report.render_sync_report_html(report)
        self.assertIn("Independent removal checks", html)
        self.assertIn("same surviving point and line evidence", html)
        self.assertIn("120 → 80", html)
        self.assertIn("2.00px → 3.00px", html)
        self.assertIn("9.00px → 4.00px", html)
        self.assertIn("mirror: &lt;suspect&gt;: &lt;edge&gt;", html)
        self.assertIn("7/8 point picks fitted", html)
        self.assertIn("2/3 line strokes fitted", html)
        self.assertIn("skipped cameras: Side camera", html)
        self.assertIn("4 ranked removal(s)", html)
        self.assertIn("does not prove a pick or constraint is wrong", html)
        self.assertNotIn("Some landmarks disproportionately affect", html)
        self.assertNotIn("<suspect>", html)

    def test_common_removal_refusal_and_incomplete_reason_are_explicit(self) -> None:
        item = SimpleNamespace(
            landmark_id="line", landmark_name="Line", kind="line",
            removed_relations=[], baseline_objective=5.0,
            candidate_objective=float("inf"),
            baseline_point_rmse_px=1.0, candidate_point_rmse_px=0.0,
            baseline_line_rmse_px=2.0, candidate_line_rmse_px=0.0,
            baseline_valid=True, candidate_valid=False,
            reason="Candidate lost camera support", support_coverage={},
        )
        result = SimpleNamespace(
            success=True, similarities={"anchor": object()}, landmarks={},
            mean_reprojection_px=1.0, per_match_rmse_px={"anchor": 1.0},
            per_landmark_rmse_px={},
            common_leave_one_out=SimpleNamespace(
                items=[item], baseline_coverage=dict(
                    requested_point_picks=2, fitted_point_picks=2,
                    requested_line_strokes=1, fitted_line_strokes=1,
                ), candidates_ranked=1, incomplete=True,
                reason="Time budget reached before all removals",
            ),
        )
        report = sync_report.build_sync_report(
            operation="Investigate Problems", source_name="generated.blend",
            matches=[SimpleNamespace(match_id="anchor")], observations=[],
            line_observations=[], result=result, anchor_id="anchor", applied=False,
        )
        html = sync_report.render_sync_report_html(report)
        self.assertEqual(report.outcome, "Investigation incomplete")
        self.assertEqual(report.severity, "warning")
        self.assertIn("Time budget reached before all removals", html)
        self.assertIn("Before: valid · After: refused", html)
        self.assertIn("Reason: Candidate lost camera support", html)
        self.assertIn("5 → —", html)
        self.assertNotIn("∞", html)

    def test_joint_fit_partial_coverage_and_refusal_are_visible(self) -> None:
        result = SimpleNamespace(
            success=True, similarities={"anchor": object(), "side": object()},
            landmarks={}, mean_reprojection_px=2.0,
            per_match_rmse_px={"anchor": 2.0}, per_landmark_rmse_px={},
            joint_support_coverage=dict(
                requested_point_picks=10, fitted_point_picks=8,
                requested_line_strokes=3, fitted_line_strokes=1,
                skipped_camera_ids=["side"], skipped_point_ids=["point"],
                skipped_line_ids=["edge"],
                skipped_relation_ids=["mirror:point:edge"],
            ),
            joint_refusal_reason="Joint fit did not pass its support gate",
        )
        report = sync_report.build_sync_report(
            operation="Solve Sync", source_name="generated.blend", result=result,
            matches=[SimpleNamespace(match_id="anchor"), SimpleNamespace(match_id="side")],
            observations=[_point("anchor", "point", "Point A")],
            line_observations=[SimpleNamespace(match_id="side", landmark_id="edge", landmark_name="Edge B")],
            anchor_id="anchor", applied=True,
            all_match_labels={"side": "Side camera"},
        )
        html = sync_report.render_sync_report_html(report)
        self.assertTrue(report.joint_support_partial)
        self.assertEqual(report.severity, "warning")
        self.assertIn("1/2 cameras fitted", html)
        self.assertIn("8/10 point picks fitted", html)
        self.assertIn("1/3 line strokes fitted", html)
        self.assertIn("skipped cameras: Side camera", html)
        self.assertIn("skipped relations: mirror: Point A: Edge B", html)
        self.assertIn("Joint fit used partial evidence", html)
        self.assertIn("Joint fit did not pass its support gate", html)
        self.assertIn("Applied to this scene", html)

    def test_last_report_stale_state_is_written_for_open_and_export(self) -> None:
        report = self._partial_report()
        with TemporaryDirectory() as directory:
            path = sync_report.write_temp_report(
                report, temp_root=directory, scene_uid=4, request_sha256="first",
            )
            self.assertFalse(sync_report.refresh_last_report(4, "first"))
            self.assertTrue(sync_report.refresh_last_report(4, "changed"))
            self.assertIn("Stale: scene inputs or cameras changed", path.read_text())
            output = sync_report.export_last_report(Path(directory) / "export.html")
            self.assertIn("Stale:", output.read_text())
            self.assertFalse(sync_report.refresh_last_report(4, "first"))
            self.assertNotIn("Stale: scene inputs", path.read_text())
            sync_report.clear_last_report()


if __name__ == "__main__":
    unittest.main()
