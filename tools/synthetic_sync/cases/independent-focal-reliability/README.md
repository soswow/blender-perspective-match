# Incomplete and imperfect point-FOV evidence

The frozen matrix tests 8/12/16/23 points, spread versus clustered picks,
partial camera overlap, 0.5 px generated coordinate noise, one 25 px wrong
correspondence, and a 15% alternative stored FOV. The fitter sees only picks,
stored cameras and initial Sync state. Independent truth and withheld-object
checks are read after each acceptance decision is recorded. The solver's
`pick_sigma_px` is 1 px for numerical trials; 0.5 and 1 px are also covered by
focused raw-pick screen tests. This is a local independent-Gaussian coordinate
noise model, not a claim about real pick errors.

`run-02` and `run-03` contain 14 and 10 **saved-start** bundle attempts. They
isolate focal fitting, because the initial poses and points came from prior
23-point Sync solves. `run-01` stopped before numerical work on a saved-start
lookup error. Source tarballs and exact request/initial/result ledgers preserve
each trial. The unchanged imported source outside these tarballs is at base
commit `6f77f4f`; the saved-start archives include the then-current focal
implementation and runner. The follow-up archive includes all imported
runner sources. No scene truth enters solver selection.

The initial 24 bundle attempts used no new inner Sync call and well under the
30 s per-attempt, 300 s per-run caps. The 8-point and naive 12-point subsets
were ineligible because a peripheral camera had fewer than eight picks. An
eligible 12-point subset has six three-view points plus three more on each
anchor link. With the prior full-scene start, eligible exact 12/16/23 and
four-view 16 controls were accurate. Noisy 12/16/23, central-cluster 16 and
15%-shifted initial-FOV controls were accepted with all true focals inside the
reported local 95% intervals. Several failed the deliberately strict 2% focal
and 1 px withheld-object oracle. The old `false_precise_acceptance` label in
the JSON means that strict oracle failed; it does **not** mean the displayed
focal interval falsely excluded truth. Three 25 px wrong-pick controls reached
0.75–0.81 px fitted RMSE yet did not converge. A fitted-residual pick hint was
tested and discarded because the wrong two-view pick could be absorbed into
free 3D without a uniquely large residual.

`epipolar-01` and `epipolar-02` preserve the initial and corrected raw-pick
probe. Each ran eight diagnostics with 331 rank-two model fits total, at most
69 fits per diagnostic. Active times were 0.025 and 0.029 s total; the declared
caps were 1,000 fits and 10 s per diagnostic. Leave-one-out fundamental fits
showed a strong conflict on the deliberately wrong view pair. In the shared
case, the lowest-cost omitted correspondence was **not** the truly wrong one.
The product therefore reports only the camera pair. `fresh-bundle-01` records
an intermediate pre-fit refusal gate. Review found that leave-one-out error
omits model uncertainty and can have high leverage; the final implementation
uses the probe **only after a fit already refuses** for nonconvergence or
noise inconsistency. Healthy acceptance never pays for the probe. The hint
requires at least 12 shared picks, a noise-consistent retained fit, and an
eight-assumed-sigma withheld Sampson excursion. It has caps of 1,000 model
fits and 10 s, polls cancellation, and silently omits a hint at those caps.
This is a heuristic diagnostic, not a calibrated p-value or proof of a bad
pick. Thin overlaps and other model ambiguities remain for the fit checks.

`fresh-sync-01` and `fresh-sync-02` reserved eight inner Sync calls under the
authorized 12-call follow-up cap. Two calls in the first ledger were duplicate
inputs caused by a runner selection bug; they remain recorded and were not
silently erased. The six distinct eligible exact/noisy 12/16, bad-pick 23 and
alternative-FOV 23 inputs all registered every camera and point. Total active
Sync time was 51.53 s, maximum 7.72 s, within the 120 s/call limits.
`fresh-bundle-01` and final `fresh-bundle-02` each ran six public lens-route
bundle attempts using these exact fresh starts, with no extra Sync solve. Total
active bundle time was 0.456 s across both; all trials ended under 30 s.
Exact 12 points passed the independent geometry
oracle. Noisy mixed 12/16 and shared 12 were accepted, with true focal coverage
in every reported interval; maximum focal errors were 9.59%, 7.83% and 9.99%.
The noisy mixed withheld maximums were 2.44 and 1.33 px, while shared 12 had
0.99 px maximum. The alternative-FOV 23 case was accepted with interval
coverage. The 25 px wrong pick was refused after the fit did not converge, with
a tentative camera-pair hint. The strict geometry oracle limits expose
uncertainty, but no 2% or 1 px real-pick guarantee is claimed.

Reproduce the saved-start matrix with `--out NEW_DIR` and optionally
`--followup`; use `--fresh-sync` and `--remaining` for the documented fresh
inputs, then `--fresh-bundle` after placing those ledgers at the expected
paths. Use `gtimeout 420` for Sync, `gtimeout 240` for fresh bundle, and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venvs/my/bin/python`. Every runner
requires a new output directory and refuses to overwrite its ledger.
