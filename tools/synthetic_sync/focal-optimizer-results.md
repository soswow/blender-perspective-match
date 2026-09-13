# Focal-bound optimization and crop controls

Checkpoint: 12 September 2026, following `7a2cc6c`.

Current follow-up: **Use Best Fit** and the corrected-principal-point capture
are described at the end. Earlier private measurements in this document used
the calibration stored at those checkpoints; they are not current measurements
of the later corrected input.

## Observed optimizer defect

The joint LM step previously solved for every focal, pose and landmark and then
clipped out-of-range focals. The remaining coordinates still followed a step
computed for an unavailable focal change. Repeated damping could stall at a
substantially worse objective than a bounded optimization method.

The independent linear regression minimizes `(x+y-3)^2 + (x-2)^2` with `x<=1`.
The unconstrained solution is `(2,1)`; clipping gives `(1,1)`, whereas resolving
the other coordinate at the bound gives the constrained solution `(1,2)`.
`tests/test_focal_optimizer.py` also covers upper-bound freezing, lower-bound
release, ordinary unconstrained LM agreement and cancellation between re-solves.

`core/focal_optimizer.py` returns feasible active-set damped steps. The outer
loop still accepts only objective-reducing steps. This is not a general exact
bounded quadratic-program solver or a guarantee of the global camera solution.
No focal range, geometry, noise or uncertainty acceptance gate was relaxed.

## Saved-objective comparison

One read-only private capture supplied inputs and partial initial registration.
Subsequent trials reused that startup, including the existing provisional-pose
completion; no repeated full Sync or camera application was needed. These are
same-input optimizer comparisons, not independent geometry certification.

| Method / bounds | Weighted squared residual | Point RMSE | Line RMSE |
| --- | ---: | ---: | ---: |
| Previous clipped LM / ±75% focal | 33488.84 | 14.01 px | 9.42 px |
| SciPy TRF / same bounds | 17566.58 | 9.81 px | 7.96 px |
| New NumPy step / same bounds | 17566.58 | 9.81 px | 7.96 px |
| SciPy TRF / absolute 5°–130° FOV | 7449.40 | 6.46 px | 4.55 px |

The integrated NumPy path completed 20 iterations in about 1.5 seconds versus
100 iterations / about 7.1 seconds for the old bundle replay. Startup is excluded;
these single-run timings are not a general performance benchmark. Every bounded
endpoint still hit focal limits. Broad-range fitting approached very narrow
FOVs and did not establish correct geometry. A same-span constraint-removal
control reduced point RMSE to 8.28 px but still hit focal bounds; it does not
prove that any particular constraint or pick is wrong.

`compare_focal_optimizers.py` preserves endpoint camera/point data, separate
point/line/prior residuals, depth signs, bound hits and source/input hashes.
Its one-shot local-variable trace is intentionally research tooling: it must
be maintained when the production parameterization changes. SciPy is required
only by the comparison tool. Private inputs and outputs are not repository
fixtures. `probe_focal_startup.py --joint` also retains the production endpoint
on bound/non-convergence refusal, without making it an applicable result.

## Independent crop control

`focal_crop.py` crops the existing live-reference oracle by a known asymmetric
pixel rectangle. Physical cameras and 3D stay fixed; picks, withheld pixels and
principal points translate by exactly the crop offset. All checked observations
remain inside the crop. The paired wrong-calibration case changes only the
input principal points back to image centers.

Correct shifted intrinsics and guessed focals recover withheld projections
within 0.1 px (observed about 0.000021 px before the bounded-step integration).
The centered-principal-point control refuses at a focal bound. This isolates
a plausible failure mechanism; it does not diagnose an arbitrary real crop.
The test uses oracle pose/point startup, not fresh registration, and does not
estimate unknown crop offsets. It also does not cover distortion or deforming
objects.

## Verification and next decisions

The integrated change passes 80 focused numerical tests, including point,
line, plane, live-mirror, weak-evidence and crop controls, plus the two existing
16/32-camera oracle-start capacity checks. A fresh-registration
Blender line fit/apply check reaches about 0.000414 px maximum withheld error.
Native operator callback/RNA checks cover textual point-FOV activity, ordinary
progress normalization and the existing apply/refusal lifecycle. They substitute
window-manager scheduling rather than automating mouse interaction.

Next establish whether original frames or known crop metadata can supply the
fixed principal points. Unknown-offset fitting would need separate weak-case
and withheld-observation experiments. Keep rejected candidates diagnostic until
there is an explicit reversible preview workflow; a lower residual is not a
claim that the cameras are closer to reality.

## Overall orientation follow-up — 13 September 2026

Independent focal fitting now includes a common rotation of the cameras and
reconstructed geometry relative to supplied world-axis planes, mirror normals
and world-axis line directions. Only locally measured rotations are released;
for example, two equal-coordinate points supply one rotational condition.
Free planes and line-to-line parallelism alone retain the starting frame.
The anchor center stays fixed, and accepted anchor orientation is stored in
its private calibration so later Sync input collection preserves it.

`focal_orientation.py` fixes an independent exact oracle and perturbs only the
stored anchor by 12 degrees. With the common rotation disabled, its production
bundle refuses at a focal bound (4.64 px candidate point RMSE). The corrected
fit satisfies the original plane/mirror evidence and unused object projections.
Generated Blender application gives about 0.00049 px maximum withheld error;
its free control gives about 0.0025 px in its original coordinate frame.
Save/reload checks preserve the fitted anchor calibration and evaluated cameras.
A sparse two-point axis group and a tilted case with line planes and world-axis
parallelism have separate numerical regressions. These tests use oracle startup
to isolate fitting and application; they do not test difficult registration.

A later private capture, distinct from the older saved-objective table above,
retained its original plane/mirror constraints and cached initial registration.
Common-rotation fitting reduced its point RMSE from 6.47 px to 1.84 px in one
bundle replay, without another full Sync or scene application. Two focals still
reached their bounds. This is evidence of reduced model conflict, not correct
real-world geometry, nor an accepted solution. Focal limits and fixed principal
points remain separate investigation targets.

Run the native oracle checks in fresh output directories:

```sh
blender --factory-startup --disable-autoexec -b --python-exit-code 1 \
  --python tools/synthetic_sync/verify_focal_orientation_blender.py -- \
  --out /tmp/pm-focal-orientation
blender --factory-startup --disable-autoexec -b --python-exit-code 1 \
  --python tools/synthetic_sync/verify_focal_orientation_blender.py -- \
  --free-control --out /tmp/pm-focal-orientation-free
```

Each command performs one bundle and no Sync registration, then saves and
reopens only its generated scene. CI runs both and the plane-group label check.

A frozen weak-axis case remains feasible with the additional orientation
freedoms but needs 171 iterations, exceeding the former 100-iteration ceiling.
It retains sub-4-pixel withheld shape error. Staged fitting and eliminating the
rotation from the damped step did not give a reliable improvement within the
old budget and were not retained. The production iteration ceiling is now 200;
the existing 30-second bundle time limit and all acceptance checks stay intact.
This buys convergence headroom, not better conditioning or a general speedup.
The crop regression also now uses the returned anchor calibration; its wrong-K
control must refuse, whether at a focal bound or the pick-noise check.

The slow-axis comparison is reproducible without another Sync. The saved-start
tool archives the current core source and exact request/endpoint, records one
budgeted bundle, and evaluates with the returned anchor calibration. Its
production replay converges at 171 iterations with 0.23194 px point RMSE and
1.13 px withheld shape RMSE after alignment using training points only.

```sh
python -m tools.synthetic_sync.focal_constraint_saved_start \
  /tmp/pm-axis-orientation \
  tools/synthetic_sync/cases/focal-constraint-reliability/run-04 \
  --pair weak-axis-hard weak-axis-hard
```

Use a new output directory for `--fixed-frame` or `--max-iterations 100` controls.
Fixed-frame results are historical counterfactuals, not production candidates.

## Provisional application and corrected intrinsics — 13 September 2026

A subsequent user correction removed unintended Manual PP Offsets. Comparing
the complete captures confirms identical picks, line strokes, constraints and
starting focal lengths; principal points changed to the image centers and
Assumed Pick Error changed from 2 to 3 px. The latter changes statistical gates,
so this is not a controlled single-variable comparison.

The corrected production bundle converges in 102 iterations, improving point
RMSE from 6.395 to 1.823 px, with fitted per-camera RMSE about 0.95–2.60 px.
It passes positive-depth, hard geometry and per-camera deterioration checks,
but one focal reaches the +80% search bound. There is no independent truth for
the real geometry. A complete provisional candidate is available, not validated
calibration. The capture took about 210 seconds; replaying only the initialized
bundle took about 6.9 seconds. No user blend was saved.

Before that correction, a bounded four-fit diagnostic budget included a SciPy
TRF comparison, continuation and endpoint sensitivity check on the older input,
then one corrected-input production replay. Continuation beyond 200 SciPy
evaluations reduced that older objective by only about 0.001%; no additional
optimizer change was justified. The full corrected capture was a separate
integration check, not part of those four agent trials. Do not count the old
and new captures as independent geometry-validation evidence.

`FocalBundleOutcome.candidate` and `LensRefineResult.candidate` now retain an
improved completed fit only after physical and per-camera checks. Automatic
acceptance remains separate. **Use Best Fit** applies that candidate through
the same scene identity, input fingerprint and rollback boundary as accepted
fits, keeps its calibration warning and reports before/after point RMSE. It
does not publish local confidence intervals. Cancellation, time limits, invalid geometry,
no improvement and failed startup do not produce this option. Actual Blender
Undo needs an interactive check; the native headless regression exercises the
registered operator, stale edits and rollback, with controlled numerical output.

The diagnostic capture now preserves the entire fit as `.fit.json` alongside
input/startup sidecars. `probe_focal_startup.py --joint` likewise serializes the
candidate in its outcome. These private artifacts avoid another expensive
registration and must remain outside the repository. Their reuse helps future
investigations, but has not measured token savings or certified a real lens.

## Complete-objective candidate eligibility — 13 September 2026

A later Manual FOV edit removed the active search bound in the private capture.
The current bundle instead reaches 200 iterations, with the last relative
objective improvement 7.835e-8 versus the 1e-9 stopping check. Point RMSE changes
from 1.67207 to 1.82004 px while the combined objective drops from 12,280,325.106
to 593.026. Startup fits points without the supplied line/plane/mirror relations;
the joint endpoint passes the physical and per-camera checks. This distinction
exposed a bug: the new candidate route required point-only RMSE to decrease.

Eligibility now requires improvement in the same weighted objective minimized
by fitting, with a numerical relative-gain guard. It retains both objective
values and both point scores. The UI explicitly announces candidate availability
and the point-error tradeoff. Nonconvergence messages state iteration exhaustion
or a stalled step; diagnostic endpoints retain the last ten accepted gains.
The optimizer and automatic calibration acceptance are unchanged.

`test_focal_candidate_tradeoff.py` reproduces the eligibility failure using
independent exact-pixel geometry in a wrong world frame. It checks physical
plane/mirror repair and withheld projection after fitting; the final statistical
refusal is mocked only to select the provisional-result path. The test fails
before the change and passes afterward. The focused 88-test group passes.
`verify_focal_candidate_blender.py --reload` checks a controlled point-error
increase through modal publication/application. A zero-solve real-candidate
`verify_focal_candidate_apply.py --operator` run checks publication through the
registered blocking Refine operator and Use Best Fit application: native RMSE
1.82006 px and maximum projection difference 0.0020 px. No source blend was saved.

These results establish eligibility and application fidelity, not real-world
calibration. The strict convergence tolerance is a separate measured opportunity;
any relaxation needs independent shape, weak-evidence and parameter-stability
checks before changing automatic acceptance.
