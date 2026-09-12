# No-VP 2D-to-2D bootstrap pilot

This bounded pilot tests Sync startup when a user has only corresponding 2D
points. It does not treat solver-triangulated points as independent truth. The
independent synthetic object, cameras and never-picked checks are held outside
the request. The solver receives no VP, Known 3D, ground, pose locks, line or
plane constraint. Every stored camera pose, including the anchor, is unrelated
to its true pose. The image is 960 × 720, the principal point is centered,
distortion is zero, and the 23 picked object points are noncoplanar.

## Frozen experiment and oracle

The eight complete JSON cases are in `cases/no-vp-*.json`: shared focal,
independent mixed focal, weak translation baseline and pure rotation, each with
true intrinsics and guessed intrinsics. Shared guessed focal is 1.25× truth;
mixed guessed factors are 1.25×, 0.8× and 1.15×. No true pose is supplied to
the solver. The weak-baseline case is diagnostic rather than prescribed to
refuse because an exact but small baseline can be recoverable. A pure rotation
has no metric depth information, so even a low-error acceptance cannot establish
3D accuracy. The cases were frozen before the first numerical invocation.

The predeclared accuracy limits are focal relative error ≤2%, withheld
projection RMS ≤1 px per camera, rotation error ≤1°, center error ≤2% of the
object diagonal, and required point error ≤5% of that diagonal. Evaluation
allows one proper global similarity for the free-scale gauge and no per-camera
or per-point correction. Alignment uses synthetic true point positions solely
to fix the gauge; those positions are never solver inputs. Disjoint mesh points
and surface samples supply at least 30 withheld projections per camera. Focal,
camera, point and held-out projection checks use the independent projector.
The focused tests verify that deliberately wrong focal, camera center and point
outputs fail despite a fabricated zero fitted RMSE. A true geometry record
passes after an arbitrary shared similarity.

## Executed result and stop

The first and most favorable control, `no-vp-shared-trueK`, accepted all three
views and 23 landmarks after 3.51 seconds of ledger attempt time. Its reported
RMSE was 1.375 px; the independent reprojected supplied picks have per-view RMS
of 1.623, 1.186 and 0.873 px. The solver message says it recovered the third
camera later and downweighted five observations, although all input picks are
exact. Its focal error is 0% in every view.

After one proper global similarity, held-out RMS was 12.25, 5.28 and 17.28 px
for views 0–2. Rotation errors were 14.01°, 4.73° and 10.72°; center errors
were 49.48%, 16.12% and 26.69% of the object diagonal. One required point
missed by 6.12% of the diagonal. Thus Sync reported success with inaccurate
geometry even with correct intrinsics. This is an extrinsic/structure startup
failure in this frozen case, not evidence that any focal-search method would
repair it.

The predeclared gate stopped the experiment after this result. The guessed-K,
mixed-focal, weak-baseline and pure-rotation cases were **not run**, and the
current shared-lens search was **not run**. Its proposed 25% span would have
included the shared truth; the ordinary default span would not, so this pilot
draws no conclusion about default search bounds. No independent-focal search
was attempted. The one-call ledger is at
`cases/no-vp-bootstrap-ledger.jsonl`; the exact request/result/assessment and
environment are at `cases/no-vp-shared-trueK-result.json`, with a compact
`cases/no-vp-bootstrap-summary.json`. The cap was 16 total real Sync calls,
180 seconds per call and 720 cumulative active seconds, plus a 720-second
outer process timeout. Only 1 of 16 calls and 3.51 of 720 active seconds were
used. Frozen cases remain available for a later, separately budgeted follow-up.

## Reproduce without spending a solve call

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venvs/my/bin/python \
  scripts/run_unittests.py test_synthetic_no_vp_bootstrap test_synthetic_budget
```

The fixture and replay tests read the saved result and ledger; the focused
startup regression added in the follow-up below performs one real Sync call.
To regenerate only case inputs, run
`~/venvs/my/bin/python tools/synthetic_sync/no_vp_bootstrap.py --freeze`.
The numerical runner's `--run OUT` command executes only the baseline
true/guessed-K Sync matrix; it does not implement the proposed outer shared-
or independent-focal search. It must remain under an outer 720-second process
timeout. Its ledger prevents repeating a completed candidate under matching
source/options metadata and rejects a changed source. The saved historical
ledger has its original hashes; later runner code adds recursive core and
harness hashes for new ledgers, without altering that record.

## Follow-up: pair registration repair

The original saved result remains the before-fix record. A bounded stage trace
on that same frozen request found the first failure before bundle adjustment:
the 0↔1 pair (15 common picks) and 1↔2 pair (14) each fit at approximately
`2e-12` px, but the sparse 0↔2 pair (6) fit at `0.519` px. The strongest pair
was 1↔2. Its first anchor bridge was selected through view 2 because 0↔2 had
more image displacement despite much less cross-image spread (57 vs 136 px)
and fewer picks (6 vs 15). Once rooted there, the next direct anchor pose
preempted a bridge through the registered view. The first shared triangulation
had per-view RMSE 22.27, 20.91 and 36.15 px, before any outlier downweighting.
The late resection and joint BA reduced fitted error while leaving inaccurate
withheld geometry.

Two isolated controls confirmed both defects matter: comparing direct and
bridge poses after the weak root did not repair the graph; choosing the
better-spread anchor root alone still let the sparse direct third-camera pose
win. The combined fix first connects the strongest pair through the anchor
member with better image spread/support, then lets a later direct pose compete
with registered-view bridges when the graph has only free 2D points. Ground,
Known 3D, line, mirror, plane and pose-lock registration paths retain their
existing behavior. No acceptance threshold, focal search or frozen input changed.

On the identical true-K request, the corrected solve accepted all three views
and 23 landmarks with fitted RMSE `1.53e-7` px. Independent withheld RMS was
`1.49e-7`, `1.47e-7` and `5.74e-7` px by view after one proper global
similarity; maximum rotation error was `2.42e-6` degrees and maximum center
error `1.02e-8` of the object diagonal. Reversing camera and observation input
order also passed with the same geometry to rounding precision. The separate
mixed-focal case with each view's true K passed at `1.75e-7` px fitted RMSE,
with maximum withheld RMS `7.20e-7` px and maximum rotation error `1.71e-6`
degrees. The saved historical result still fails the new numerical regression's oracle, so it
distinguishes the original false-precise acceptance from the repair.

The exact eleven follow-up attempts, including the two isolated controls, one
diagnostic repeat, an order control and mixed true-K case, are in
`cases/no-vp-startup-followup-ledger.jsonl` (11/12 calls; 41.36/720 active
seconds). `cases/baseline-trace.json`, `cases/final-shared-trueK-trace.json`,
`cases/reversed-order-v1-trace.json` and `cases/final-mixed-trueK-trace.json`
retain the complete stage and independent assessments. The repeat and
isolated-control records remain in the ledger.
Each candidate carried its source hash in the budget key. No guessed-K,
weak-baseline, pure-rotation or focal-search trial was run in this repair.
The old-source baseline trace fails the new regression's accuracy assertion
with `AssertionError`; the guarded source passed it in the final focused run.
Earlier source variants also passed it alongside 46 pairwise, graph, ground,
pose-lock and role tests, and 24 ground, plane and mirror controls.

After these saved attempts, the trace runner's cache key was strengthened to
hash the full numerical `core/` tree and the generator, evaluator, geometry,
solver and budget harness files. It also includes Python, NumPy, OpenCV,
platform and numerical thread settings. The eleven saved ledger records and
assessments remain unchanged. They use their original source keys; replaying a
label under the strengthened key would reserve a new call, so inspect the
saved traces to review this result. No numerical attempt followed this tooling
correction.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venvs/my/bin/python \
  scripts/run_unittests.py test_synthetic_no_vp_bootstrap test_sync_pose test_sync_solve
```
