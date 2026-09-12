# Recovered-camera line acceptance investigation

12 September 2026. Synthetic evidence only; no private project was modified.

## Result

A line-bearing solve with a naturally recovered camera failed before the final
3D candidate could be evaluated. The recovered update rebuilt point landmarks
from a mixed point-and-line ID list. Because line IDs have no point-observation
bucket, `_triangulate_landmarks` raised `KeyError: 'edge_0'`. Bypassing only the
recovered 3D update succeeded and still ran the shared final line rebuild. The
same production path also succeeded when all line evidence was removed.

The fix makes `rebuild_landmarks` pass the deterministically sorted point-
observation keys to point triangulation. Free and Known 3D lines continue
through their separate reconstruction path. A focused ordinary-path regression
asserts that the recovered stage is entered, both line segments survive, the
healthy camera remains within its independent withheld-camera limit, and the
free line meets its independent direction and offset limits.

After that fix, the primary production run and its frozen-stage control were
numerically identical. The proposed 3D candidate was rejected by the existing
point-fit guard (`kept_joint_geometry=true`), so this case did **not** reproduce
an accepted update that damages line or plane geometry.

## Exact case and controls

`recovery_acceptance.py` generates the input from `mixed_lines`, seed 0, with
0.3 px noise. It shifts only non-ground point picks in `view_2` by +180 px in U;
the five line strokes are unchanged. Hard ground uses zero slack. Both lines
retain the existing hard `WORLD_AXIS_Z` parallel relations. `view_2` is peeled
and naturally recovered from the floor evidence.

The generated `case.json` SHA-256 is
`ed55ebf86580e53f2e33c40314c74064443b29b853da32f4ce54b31f6f082139`.
The input is preserved in `cases/recovered-lines.json` and read directly by the
regression test. The script also writes exact JSON beside each result.

Three runs vary only the relevant boundary:

| Run | Recovered stage | Lines | Outcome before fix | Outcome after fix |
| --- | --- | --- | --- | --- |
| production | normal | Known + free | `KeyError: 'edge_0'` | success; candidate rejected |
| freeze control | bypassed | Known + free | success | success |
| no-lines control | normal | none | success | success; candidate rejected |

The historical exception ended at:

```text
core/sync/solve.py:1188 in _refine_recovered_location
    state.rebuild_landmarks()
core/sync/solve.py:494 in rebuild_landmarks
    rebuilt = _triangulate_landmarks(...)
core/sync/ba.py:244 in _triangulate_landmarks
    for observation in observations_by_landmark[landmark_id]
KeyError: 'edge_0'
```

Immediately before that call, `landmark_ids` contained `edge_0` and `edge_1`,
`line_segments` contained both IDs, and neither ID existed in
`observations_by_landmark`. The no-lines control had no such unmatched IDs.

## Geometry after the fix

Production and frozen-stage results had exactly the same recorded geometry and
camera checks:

| Independent check | Production | Freeze control | Contract |
| --- | ---: | ---: | ---: |
| free-line direction error | 0.000000° | 0.000000° | at most 1° |
| free-line offset / object diagonal | 0.00133150 | 0.00133150 | at most 0.02 |
| healthy `view_1` withheld RMS | 0.540678 px | 0.540678 px | at most 1.8 px |
| recovered `view_2` withheld RMS | 2.43136 px | 2.43136 px | flagged above 1.8 px |

The recovered-camera error is expected evidence from the deliberately
contradictory point picks; it is reported rather than treated as a passing
accuracy claim. The hard vertical direction remains exact. No plane constraint
was needed to reproduce the crash.

## Static scope and limitation

The recovered BA can optimize free line midpoints and free camera poses, while
its acceptance gate directly scores established point observations. The final
free-line rebuild then uses all location-enabled posed cameras and reapplies
hard parallel and hard supported-plane enforcement. This leaves a plausible
question for a future accepted-candidate case, but static reachability alone is
not a reproduced accuracy defect. Soft-plane deviation would also need an
explicit accuracy contract before it could establish one.

This bounded experiment stopped after 15 numerical solves, including pilots,
the historical controls, the focused regression, and the final three-run
replay. It did not search random seeds, bias line strokes, or change production
acceptance policy.

## Commands

```sh
# Historical revision, before the one-line fix
python3 tools/synthetic_sync/recovery_acceptance.py \
  --case tools/synthetic_sync/cases/recovered-lines.json \
  --expect-crash --out /tmp/recovery-acceptance-old

# Fixed production path and paired controls
python3 tools/synthetic_sync/recovery_acceptance.py \
  --case tools/synthetic_sync/cases/recovered-lines.json \
  --out /tmp/recovery-acceptance-fixed

./scripts/run-unittests.sh \
  test_sync_solve.SolveSyncTests.test_recovered_location_rebuilds_point_ids_separately_from_lines
```

The historical command requires the current diagnostic script and fixture in a
checkout with the old solver. Normal exit zero requires successful production
and control solves, without exceptions. `--expect-crash` instead requires the
specific line-ID KeyError in point triangulation and successful controls;
unrelated exceptions do not count. Neither mode claims the contradictory
camera passes its accuracy threshold; inspect `assessment` in the JSON/report.

The coordinator also ran the final frozen-case unittest against the old solver:
it failed with the intended KeyError, then the fixed solver was restored.
Raw historical/fixed records and that test log are preserved locally under
`.local/recovery-line-integration/`. This adds one solve to the worker's 15;
combined-suite validation is recorded in the continuing decision record.
