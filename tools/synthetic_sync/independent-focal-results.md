# Independent no-VP focals — bounded prototype

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
