# Independent no-VP focals — bounded prototype

## Frozen noisy-pick continuation (before numerical calls)

The next experiment keeps the same four three-view geometries: mixed and shared
focals, weak translation, and pure rotation. It adds independent Gaussian
0.5 px standard-deviation noise to every image coordinate, with fixed seeds
20260912 through 20260915 in that order. Exact noisy requests and the manifest
are saved in `cases/independent-focal-noise/` before any numerical call. The
truth and withheld points remain unchanged and are used only after an input-only
candidate is fixed. Each noisy request gets a **fresh ordinary Sync** result as
its initializer; the original exact-pick result must not certify noisy startup.
Each fresh call is reserved in one `ExperimentBudget` ledger with at most six
calls, 120 seconds each and 600 cumulative active seconds. A 660-second outer
process deadline applies.

From each fresh result, start the independent-focal optimizer at three common
scales of the stored focal estimates: 0.7, 1.0 and 1.3. These are different
initial guesses, not a shared-lens constraint; all three focals move freely.
The same 0.6–1.6 stored-focal bounds, anchor pose and unit-baseline gauge apply.
Use at most 300 optimizer function evaluations and at most 300 Jacobian and
10,000 residual evaluations in 30 active seconds per run. Across the twelve
predeclared runs, allow at most 3,600 Jacobian and 120,000 residual evaluations,
300 active optimizer seconds and a 360-second outer process deadline. Every
run is reserved with its complete noisy request, fresh initializer, source and
runtime identity before optimization. Use `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `~/venvs/my/bin/python`.

Input-only selection will first require convergence, all three cameras and all
points, no focal bound and fitted residuals consistent with the stated noise.
Cross-start focal disagreement and local linearized sensitivity will diagnose
stability, without thresholds chosen from truth. Report all candidates and
withheld geometry independently; a low residual or optimizer convergence alone
does not certify the FOV. If the fixed budget ends with unresolved uncertainty,
record that design boundary instead of calling the iteration cap an ambiguity
detector. No production promotion is planned before positive and weak-motion
controls agree on an input-only acceptance policy.

## Frozen dense convergence control (before numerical calls)

The twelve sparse noisy-pick fits all reached the declared 300-evaluation cap.
That cap does not establish ambiguity. A single alternate-backend control now
uses the **same four exact noisy requests and fresh ordinary-Sync initializers**
above, each at start scale 1. The focal/pose/point parameterization, bounds,
residual, finite-difference step, loss and tolerance are unchanged. Only the
Jacobian storage and TRF linear solver change: dense finite differences and
the exact linear solve replace sparse finite differences and LSMR. No new
inner Sync call is allowed. The four fits are reserved with complete request,
initializer and source/runtime identity before each run, in a separate ledger.

Each dense fit is capped at 300 Jacobian, 30,000 residual evaluations and 30
active seconds; all four together at 1,200 Jacobian, 120,000 residual
evaluations and 120 active seconds, with a 180-second outer process timeout.
`max_nfev=300` remains. Report convergence, fitted residual, support, focal
estimates, independent withheld geometry and comparison to the matching sparse
start. This isolates optimizer behavior rather than selecting a backend or
threshold by synthetic truth. Use the same one-thread Python environment.

## Frozen calibrated noisy-start controls (before numerical calls)

Fresh ordinary Sync initialized the noisy shared case with very poor withheld
geometry, despite having all three cameras and 23 points. To isolate focal
guess error from noisy registration, run two additional controls: mixed and
shared cases with **the identical previously frozen noisy 2D picks**, the same
private camera poses and points, but intrinsics copied from the saved true-K
input fixtures. Those calibrated intrinsics are an explicit experimental
input, not an optimizer selection or recovered truth. Freeze complete requests
and hashes before solving. Use a separate `ExperimentBudget` ledger with two
calls maximum, 120 seconds each, 240 cumulative active seconds and a
270-second outer process timeout. No prior noisy initializer is reused and no
other Sync call is authorized. Compare support, fitted error and independent
withheld geometry with the guessed-K noisy starts. A true-K collapse would
identify a separate registration/BA defect worth fixing before focal UI work.

## Frozen calibrated-startup defect investigation (before probes)

Both fresh true-K noisy controls accepted severe misreconstructions while
retaining three cameras and 23 points. Preserve those original results. Trace
the existing numerical Sync stages on the **same complete mixed true-K noisy
request** first, then inspect which pair/graph choice or later stage creates
the wrong basin. At most eight new exploratory Sync invocations, including
trace and candidate variants, are reserved in one ledger, with 120 seconds per
call, 600 cumulative active seconds and a 720-second outer process limit.
Candidate selection uses only the 2D inputs, calibration and fitted state;
independent truth remains post-result evaluation. Focused regression runs after
an identified fix are tracked as verification rather than exploratory calls.
Require the corrected path to preserve camera/point support, improve fitted
error, and pass the same global-similarity withheld checks on both calibrated
noisy cases plus the existing exact/noisy controls. A product fix needs a
focused before/after regression and a user-facing changelog line.

## Noisy-pick and calibrated-startup results

Four fresh ordinary-Sync starts on the frozen 0.5 px noisy requests used four
of six allowed inner calls (16.58 active seconds). All returned three cameras
and 23 points, but their independent withheld geometry varied sharply. Twelve
sparse three-focal trials used 3,510/3,600 Jacobian and 43,941/120,000 residual
evaluations. Every fit hit `max_nfev=300`; that cap was a numerical termination,
not an ambiguity detector. Four dense-Jacobian/TRF-exact controls reused the
same fresh starts, made no new Sync call, consumed 607/1,200 Jacobian and
51,721/120,000 residual evaluations, and all converged. Thus sparse LSMR
convergence caused the first refusals, while convergence alone did not certify
geometry.

| Dense noisy fit | Fitted RMSE | Max focal error | Max withheld RMS | Outcome |
| --- | ---: | ---: | ---: | --- |
| Mixed focals | 0.338 px | 8.4% | 1.112 px | Useful improvement over fresh guessed-K Sync's 2.690 px; strict accuracy unmet |
| Shared focals | 0.249 px | 9.4% | 1.458 px | Large improvement over fresh guessed-K Sync's 203.4 px; strict accuracy unmet |
| Weak translation | 0.403 px | 100% | Behind-camera holdout | Converged at a focal bound with unsupported geometry |
| Pure rotation | 0.380 px | 8.4% | 11.109 px | Converged with unsupported finite depth |

At an explicitly assumed 0.5 px independent coordinate noise, the local
linearized 95% upper focal excursions are 8.7–22.2% for mixed/shared,
effectively unbounded for weak translation, and above 100% for the
pure-rotation **free-depth BA model**. This does not rule out focal information
from a separate rotation-only model. Read-only homography fits to raw picks
give symmetric transfer RMS of 24–296 px in mixed/shared but 0.76–1.07 px
in weak/rotation. Homography compatibility means depth evidence is ambiguous:
pure rotation is one explanation, and a translated planar scene is another.
The pure-rotation BA nevertheless fabricates 4.85° median pair parallax with
all fitted points in front, so fitted cheirality/parallax alone cannot certify
translation. These measurements are not product cutoffs.

Two separately frozen **calibrated** noisy controls used the identical picks
and the saved true-K input intrinsics, in two of two inner calls (6.29 active
seconds). Unmodified Sync accepted mixed/shared at 4.87/4.35 px while putting
withheld geometry behind the cameras. A ledgered trace found 14- and 15-pick
pair baselines shrunk to 0.000039 and 0.00176 scene units despite subpixel
pair fits. The graph then accepted a 9.29 px candidate under its 10 px limit.

The narrow production correction holds the transformed camera-center baseline
at its seed length during two-view ray-distance refinement and normalizes the
residual by that arbitrary free-scale gauge. It does not supply metric scale or
change pose locks. Three of eight exploratory Sync probes were spent on the
original trace and corrected mixed/shared controls (16.64 active seconds).
The corrected solves retain three cameras and 23 points with 0.376/0.319 px
fitted RMSE, all withheld objects in front, maximum 1.974/0.900 px withheld
RMS and aligned center errors 0.068/0.032 object diagonals. The noisy mixed
case still misses the strict 1 px / 2% geometry oracle; this is a substantial
recovery, not proof of metric accuracy. A focused regression fails on the old
checkout for both full solves and the direct gauge invariant (seed length
5.477 shrank to 0.0000224); it passes after the fix and checks nonzero private
centers, a tenfold world rescaling and exact pose locks.

The direct pair-collapse fix is supported. A future independent no-VP focal
mode needs input-only ambiguity handling and an explicit uncertainty display.
Full support, converged fitting, no focal-bound contact and positive depth are
necessary but insufficient. Raw homography compatibility must flag ambiguous
depth evidence without labelling the motion as pure rotation. These four
scenes and one noisy seed do not establish a reliable product threshold or a
2% FOV promise. Exact requests, ledgers, source/runtime keys, candidate states
and independent assessments are in `cases/independent-focal-noise/`.
The ledger hashes distinguish each historical wrapper/runtime revision, but
the sparse and dense wrapper source files were amended during this work and
their exact intermediate bytes were not snapshotted. Thus the saved trials are
fully inspectable as input/result evidence, while exact wrapper replay from
the hash alone is unavailable. The original product code is retrievable at
`3045dd4`; the final code and regression are retained together in the fix.

## Frozen plan before optimizer calls

The four exact, previously saved guessed-intrinsics cases stay fixed: mixed
focals, shared focals, weak translation, and pure rotation. The saved ordinary
Sync result supplies only an initial pose and point cloud. The optimizer sees
the request's 2D correspondences, stored intrinsics and initial result; it does
not receive the independent camera, point or withheld-object truth. The oracle
is applied after the candidate and input-side diagnostics have been written.
No new inner Sync call is planned; if one is needed, reserve it in a separate
24-call, 180-second-per-call, 1200-active-second ledger with an outer process
timeout before executing it.

The experiment jointly varies three log focals, two secondary camera poses and
all free points using sparse finite-difference bundle adjustment. The anchor
camera has identity pose, and the first baseline is fixed at unit length. This
removes one global similarity gauge for translated scenes. The pure-rotation
case has no physical baseline: finite-depth output from this parameterization
cannot prove translation or depth. Search bounds are 0.6–1.6 times each stored
focal, fixed before seeing the oracle. Point reprojection and full
camera/point support are the only optimization/selection evidence. Report the
gauge-removed Jacobian singular spectrum and bound hits as diagnostics, without
tuning an empirical acceptance threshold to these four examples.

The **separate optimizer budget**, frozen before execution, is at most 3,000
residual evaluations, 80 Jacobian evaluations and 90 active seconds per case;
12,000 residual evaluations, 320 Jacobian evaluations and 360 active seconds
across all four cases. Finite-difference residual calls count. An external
process timeout of 420 seconds bounds code outside the internal meter. The
ledger reserves each full exact input and source/runtime identity before work;
the result stores the optimizer counters, candidate and post-selection oracle.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and
`~/venvs/my/bin/python` for this numerical trial.

Success evidence requires every camera and point, each focal within 2% of
independent truth, one proper global similarity aligning the free scene, and
withheld object projection within 1 px. Low training RMSE alone is insufficient.
The exact weak-translation true-K control from the prior corpus is known to be
accurate; do not select an uncertainty cutoff merely to reject its guessed-K
counterpart. Pure rotation must be presented as depth unobservable even if its
training residual is tiny. Stop at a cap or conflicting evidence, and report
failed optimization honestly.

## Frozen sensitivity follow-up before diagnostic evaluations

The first optimization ended within its declared caps, but the mixed candidate
hit `max_nfev=80` very near an exact fit. To inspect observability without
selecting by synthetic truth, a separate read-only diagnostic will compute one
sparse finite-difference Jacobian at each of the four already frozen candidate
states. It will report focal standard errors from the local linearized fit,
assuming **0.5 px independent standard deviation per image coordinate**. This
is a stated uncertainty model, not an estimate from these exact picks and not
a product threshold. Singular directions with numerical rank loss report
unbounded uncertainty; finite errors remain only local approximations. No
candidate will be rerun or changed by this diagnostic.

The diagnostic cap is 4 Jacobian evaluations, 120 residual evaluations and 20
seconds total, including finite differences, with a 30-second outer process
timeout. Its own ledger records the exact saved candidates and source/runtime
identity before each evaluation. It will not spend any inner Sync call.

## Observed results

The four optimizer calls consumed 3,749 residual and 298 Jacobian evaluations,
under the 12,000/320 limits; optimizer active time was 1.39 seconds. The
sensitivity follow-up used 50 residual and four Jacobian evaluations. No new
inner Sync solve was made. All candidates retained three cameras and 23 points,
and no focal touched the search bounds. Exact inputs, initialized solver
results, candidate records, source/runtime hashes, counters and post-selection
assessments are saved in
[`cases/independent-focal-prototype/`](cases/independent-focal-prototype/).

| Frozen guessed-K case | Fitted RMSE | Maximum focal error | Maximum withheld RMS | Outcome |
| --- | ---: | ---: | ---: | --- |
| Mixed focals | 0.000142 px | 0.018% | 0.00114 px | Candidate geometrically accurate; optimizer hit 80-evaluation cap, so **refused** |
| Shared focals | `3.73e-10` px | `<1e-7`% | `<1e-6` px | Converged; accurate on this exact case |
| Weak translation | 0.000626 px | 13.2% | 1.454 px | Cap reached; inaccurate |
| Pure rotation | 0.001066 px | 24.8% | 11.895 px | Cap reached; depth unobservable |

The mixed candidate's independent camera centers differ by at most 0.00036
object diagonals after one proper global similarity; its focal estimates are
700.12, 849.95 and 999.99 px, compared with the oracle's 700, 850 and 1000 px.
That is a promising numerical candidate, **not a successful algorithm result**:
the declared iteration cap stopped before convergence. None of the candidate
choices used those truth values. The weak and pure-rotation candidates show why
the similarly tiny fitted errors cannot authorize geometry or focal accuracy.
The weak and rotation refusals in this prototype are iteration-cap stops, **not
an implemented ambiguity detector**. Their independent oracle failures reveal
the hazard; a future input-only decision rule needs new validation.

With the declared **0.5 px independent coordinate-noise assumption**, the
local linearized 95% focal upper relative excursions are roughly 9–20% for both the mixed and
shared fits, vastly wider for the weak-baseline fit, and effectively unbounded
for pure rotation. These intervals describe sensitivity near the saved
candidates; they neither prove global uniqueness nor account for biased picks,
distortion, crops, or correspondence error. The saved raw Jacobian singular
ratios also differ strongly across these scenes, but they depend on parameter
units and are **not** an acceptance threshold. The exact no-noise recovery is
therefore insufficient evidence for a 2% real-image FOV promise. No production
Sync, lens UI or result acceptance behavior changed.

The prototype is runnable with SciPy in the numerical environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venvs/my/bin/python \
  tools/synthetic_sync/independent_focal.py --out /tmp/independent-focal-new-run
```

Use a new directory because the runner will not overwrite its ledger. The
saved source identity identifies the exact code used for this run at commit
`45ac27f`; later input validation or reporting edits do not retroactively
change its result. The historical sensitivity JSON field named `half_width`
stores the upper relative excursion `exp(1.96σ)−1` from a log-focal interval;
the lower excursion is `1−exp(−1.96σ)`. Future runs use the corrected key.
