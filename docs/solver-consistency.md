# Consistent solving and reports

Implementation record for the shared Sync and lens-fit workflow. This document
tracks the behavior contract and verification evidence; `sync.md` remains the
user guide.

The four requested workflow changes are implemented and verified. On the
supplied project, Refine → Solve → Solve preserves overlays within 0.001px,
retains all supported evidence, and needs no repeat registration. The project
was opened read-only and never saved. The full suite passes 587 tests with two
optional-sample skips. Remaining validation limitations are recorded below.

## Behavior contract

- Solve Sync holds lens calibration fixed while fitting camera poses, points,
  and lines. Refine Lenses additionally frees the permitted focal parameters.
  Both use the same final objective and constraint semantics.
- A usable current solution supplies the starting poses and geometry. Camera
  registration remains available for missing cameras and unusable starts.
- With unchanged evidence, a replacement must preserve supported cameras and
  landmarks, satisfy physical and geometric checks, and not worsen the common
  weighted objective. Point RMSE alone cannot select a worse line fit.
- Certified applied fits persist their effective point weights so crossing an
  outlier threshold cannot silently change the next solve's objective. Changed
  evidence recomputes weights; old or incompatible weight records reinitialize.
- Known 3D, Ground, mirror, shared plane, parallel, camera-role, and lock
  semantics remain active in joint fitting. Resource limits and insufficient
  information must be reported, never handled by silently dropping evidence.
- Fit Only observations determine their camera against shared geometry; they
  cannot move that geometry or the other cameras through joint coupling.
- Locks retain their existing root-transform meaning. In particular, Lock
  Translation fixes root translation rather than optical center; changing root
  rotation can still move a camera whose private center is nonzero. Lock
  Rotation retains the permitted discrete world-axis orientations.
- Applied results own their camera calibrations, root transforms, point/line
  geometry, residuals, and reports together. A preview or failed attempt cannot
  replace the displayed errors of the applied solution.
- Solve and Refine produce a report without another solve or opening a browser.
  Open Last Report shows the saved result, including whether it was applied and
  whether later input edits have made it outdated. Optional investigation is
  separate from applying a solution.
- Reports distinguish point error, line error, coverage, and constraints.
  Provisional focal results retain their refusal and explicit application step.

## Work ownership

| Work | Owner | Status |
| --- | --- | --- |
| Shared joint optimizer and feature parity | Sol numerical worker | Integrated, including hard-plane line publication correction |
| Scene seed capture, request migration, continuation and acceptance | Sol scene worker | Integrated; generated and private continuation checks passed |
| Automatic reports, report provenance, and investigation UI | Sol report worker | Integrated; focused and generated Blender checks passed |
| Architecture review, integration, documentation and combined verification | Parent | Complete; conditioning and visual-review limitations recorded below |

Workers used isolated worktrees from `722d665`. Integration used reviewed diffs;
the installed development link was not changed.

## Verification requirements

- Generic synthetic Refine → Solve → Solve sequences check point and line
  projections, camera/landmark coverage, constraints, and withheld geometry.
  A converged solution must remain stable within numerical tolerances.
- Feature controls cover hard/soft Known 3D, Ground intersections, Known 3D
  lines, pose/rotation/translation locks, Fit Only, mirror references and slack,
  plane/parallel relations, distortion, and permitted focal grouping.
- Changed picks, constraints, roles, camera transforms, and save/reopen exercise
  seed provenance and stale-result handling.
- Generated Blender checks exercise blocking/background publication, final
  fitted calibration ownership, errors during application, file reload, and
  reports that require no additional numerical solve.
- Focused worker checks precede one integrated numerical suite and appropriate
  native Blender checks. Numerical investigations use bounded attempt ledgers;
  exact private evidence stays outside the repository.

## Evidence

Report work: 17 focused report tests and a generated Blender probe passed. The
probe exercises seven reports without numerical solves, including final focal
ownership, an origin adjustment, refusal/provisional labels, stale reports,
and report-write failure after valid application. Common-fit removal comparisons,
partial coverage and retained-result refusal warnings have focused tests. Browser visual inspection
was unavailable because the browser tool blocked the local report URL.

Continuation work: request migration tests and a generated zero-solve native
apply/save/reopen probe passed in the worker checkout. A bounded cold → seeded
→ edited-pick control preserved independent point, line, and withheld-camera
checks. The integrated request, seed, investigation and report subset passed 38 tests.
The first full integrated suite ran 561 tests in 426 seconds, with seven
failures, two errors and two absent-sample skips. Focused corrections addressed
obsolete fixture locks, refusal/metric assertions, a test import, and a fake
state missing the new seed fields. One actual weak-axis startup regression was
fixed by retaining a point-only initializer for eligible free-point graphs;
the final fit still receives all constraints. Its independent geometry checks
and the exact four-camera focal recovery checks pass. The final integrated
rerun passes: **578 tests in 425.081 seconds, two optional-sample skips**.
Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/Users/sasha/venvs/my/bin/python scripts/run_unittests.py`. The complete log is
`/tmp/pm-checkpoint-unittests-20260914.log`.

Integrated zero-solve Blender checks pass for report publication, continuation
selection, cancellation, save/reopen, progress, provisional-fit reload, focal
application and mirror-origin placement.
After the final corrections, the native point-FOV application and continuation
probes pass again with zero numerical solves. Archives:
`/tmp/pm-checkpoint-focal-apply-20260914` and
`/tmp/pm-checkpoint-continuation-20260914`. Compilation and `git diff --check`
also pass. No private project names or match identifiers were added to code,
tests, documentation or changelog.

The first generated Refine → Solve → Solve sequence stopped during Refine:
the bundle accepted its line chart, but publication moved the finite helper
midpoint along a slightly tilted line, violating its hard supporting plane.
The public scorer correctly refused that endpoint. The corrected model projects
the line direction into its hard plane inside image residuals, geometric priors,
derivatives and publication. A saved initialized replay passes with a maximum
published line-plane gap of 1.94e-8 world units. The final correction used three
bundles and no registration calls; 18 focused integrated checks pass, including
plane support/frame derivatives and the independent sequence assessor.

A subsequent native Refine succeeds with 0.213px point RMSE and valid geometric
relations. Its continuation provenance was not certified, so the sequence
stopped before Solve. A zero-solve replay traced that to stale evaluated camera
transforms during application. Updating Blender's view layer before stamping
certifies the exact saved endpoint; the generated application regression also
passes. Independent withheld geometry remains 2.76px RMS against
the fixture's 1px target; the target has not been relaxed. Those are separate
questions from whether the fitted picks and lines agree.

## Bounded continuation and current status

The user authorized at most 30 additional minutes, starting
2026-09-14 13:41:01 UTC, with a hard stop at 14:11:01 UTC. That timebox ended
without a commit, live add-on reload, or save of the supplied project. The user
subsequently authorized completion and a commit.

The resumed checks found and corrected three further issues:

- A valid camera rotation rounded by Blender to float32 failed the common
  scorer's matrix-validity check, unnecessarily restarting registration. The
  allowance now derives from float32 precision; shear/reflection still refuse.
- Seeded registration exposed a missing `dataclasses.fields` import. A focused
  cancellation-before-registration regression covers that entry path.
- Mirrored-line priors used arbitrary finite helper midpoints in the public
  scorer, while fitting used a point near the anchor. With slightly imperfect
  mirrored directions, sliding the same infinite line changed its score. Both
  paths now use the nearest point to the anchor for this prior. Known 3D line
  midpoints retain their supplied role in supporting planes. Fixed-focal
  acceptance checks the represented geometry's complete, frozen-weight public
  objective, as independent focal fitting does, instead of requiring numerical
  equality with its internal chart cost.

Fresh fixed-mirror, free-scale and live-mirror Refine → Solve → Solve sequences
passed seed certification, effective-weight identity, complete support,
constraints and full applied calibration/root/geometry ownership checks. Both
Solve stages skipped camera registration. Before the final mirror-prior
correction, maximum overlay movement was below 0.00006px. A fresh live-mirror
sequence after that correction also passes, with maximum point/line movement
below 0.00009px and no endpoint refusal.

The private project comparison completed twice, including after the final
mirror-prior correction. All support and geometric checks passed; both seeded
solves skipped registration. The final maximum point/line overlay movement was
below 0.0008px, and both solves completed without the former internal/public
score-mismatch refusal. File hashes before and after match; the supplied file
was never saved. These are application/continuation checks, not proof that the
real reconstruction matches unknown ground truth.

Strict monotonicity flags on **rescored applied** objectives remain false on
some sequences because Blender's float32 serialization changes the transforms
slightly between applications. Raw numbers and flags are preserved. Each
accepted numerical candidate passes the same-current-evidence non-worsening
check before application; all applied-state discrepancies stay within the
probe's float32 bounds. This does not promise bit-identical repeated solves.

The final full-suite rerun passes: **587 tests in 411.862 seconds, two
optional-sample skips**. Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/Users/sasha/venvs/my/bin/python scripts/run_unittests.py`; complete log:
`/tmp/pm-bounded-final-unittests-20260914.log`. Compilation and
`git diff --check` pass. Focused tests for the new input gates, endpoint
comparison, mirrored-line slide invariance and Known 3D plane support pass.
Work stopped at approximately **14:08 UTC**, about 27 minutes into the approved
30-minute timebox, with no numerical jobs left running.

## Known limitations and follow-ups

1. **Noisy synthetic accuracy is sensitive to weak geometric constraints.** The composite
   fixtures retain their original 1px target and still fail it:

   | Fixture | Maximum focal error | Withheld RMS |
   | --- | ---: | ---: |
   | Fixed mirror | 0.264% | 2.757px |
   | Free scale | 0.732% | 2.937px |
   | Live mirror | 0.371% | 3.140px |

   The free-scale case has one unobservable world rotation. Aligning only its
   permitted rotation and scale from training points reduces withheld RMS to
   2.683px, still above target. Fixed/live cases have three observable rotation
   coordinates but known normals only about 14 degrees apart. A paired fit
   with **exact true focal lengths**, identical noisy picks/relations and frozen
   weights still has 2.797px withheld RMS, with valid support/relations. This
   separates the accuracy failure from focal estimation; it does not establish
   accurate camera poses or 3D. Its inner endpoint was captured explicitly even
   though the then-current wrapper refused the representation-cost difference.
   Two additional fixed-mirror controls isolate the effect of noise. With the
   same noisy picks and frozen weights, a fit started at exact true poses,
   geometry and focal lengths moves from effectively zero withheld error to
   2.7973px while reducing the fitting objective from 8.8432 to 5.01281. It
   reaches the same endpoint as the earlier biased start. Conversely, exact
   oracle picks and strokes let the biased start recover to 0.000051px
   withheld error at the true focal lengths. Both controls retain full support
   and pass endpoint acceptance. This rules out a deterministic model offset
   or the tested local-minimum explanation for the fixed-mirror case; it does
   not certify the noisy reconstruction's accuracy. The free/live variants
   have not received these separate noise-free controls.

   The original 1px assessments remain unchanged and failing. Future accuracy
   work should assess better-separated geometric references or additional
   views with paired noise controls, rather than tuning that target to these
   results. Low fitted error and stable repeat solves do not prove accurate
   withheld geometry. This limitation is separate from the corrected
   Refine/Solve continuation and report ownership.
   The exact controls, frame diagnostic and replay commands are in
   [the accuracy evidence record](../tools/synthetic_sync/continuation-accuracy-results.md).
2. **Visual report inspection remains unavailable.** Programmatic and native
   report checks pass. The browser policy blocked local HTML; do not bypass it
   with another serving/browser path. Inspect visually when an allowed path is
   available.

The requested solver/report behavior is complete and continuation is verified
on generated and supplied inputs. The fixed-case noisy accuracy finding is
characterized, not hidden by a changed threshold. Further reconstruction
accuracy research and visual report review remain follow-ups; neither requires
another solve to open the report or finish applying a lens fit.

## Retained evidence and budgets

- Final accuracy controls:
  `/tmp/pm-continuation-accuracy-controls-20260915` and
  `/tmp/pm-continuation-accuracy-clean-20260915`. Two bundles, no registration,
  0.888803 active seconds; no further numerical calls needed. Six focused
  assessment tests pass, including the new zero-solve truth-start and
  noise-removal check. No production code or accuracy threshold changed.
- Final live-mirror sequence:
  `/tmp/pm-continuation-live-mirror-canonical-20260914`.
- Final private read-only sequence:
  `/tmp/pm-private-continuation-canonical-20260914`; prior comparison:
  `/tmp/pm-private-continuation-bounded-20260914`.
- Three preceding source-frozen variants:
  `/tmp/pm-continuation-joint-mirror-certified-20260914`,
  `/tmp/pm-continuation-free-gauge-final-20260914`,
  `/tmp/pm-continuation-live-mirror-final-20260914`.
- Paired truth-focal raw endpoint and wrapper decision:
  `/tmp/pm-continuation-known-focal-captured-20260914`; its predecessor
  `/tmp/pm-continuation-known-focal-control-20260914` consumed a call but omitted
  the raw endpoint. The replacement call remains charged separately.
- Gauge audits: `/tmp/pm-gauge-{free,fixed,live}-20260914.json`.
- Earlier hard-plane correction: `/tmp/pm-hard-line-20260914-final`.

The original generated-sequence allowance was 6 registration calls, 12 bundles
and 900 active seconds. Before the final mirror correction, cumulative use was
7 registration calls, 14 bundles and 70.956712 active seconds, including
explicitly reviewed extensions for the third variant and the lost-endpoint
control replacement. The fresh live-mirror verification after that correction
added 1 registration and 3 bundles (13.094855 seconds), for **8 registration
calls, 17 bundles and 84.051567 active seconds**. These are retained experiments,
not an open allowance for more calls.

The separate private check used 1 registration and 3 bundles (3.399684 seconds).
After the representation fix, a reviewed repeat used another 1 registration and
3 bundles (3.401393 seconds). Total: **2 registration calls, 6 bundles and
6.801078 active seconds**. Each process enforced its own outer deadline and
verified the source file hash. All private captures stay outside the repository.
Full-suite tests are separate from these instrumented investigation ledgers.

The implementation and evidence were integrated in the main checkout.
Isolated `pm-*` worktrees remain as source snapshots, including
`/tmp/pm-endpoint-acceptance-20260914`; remove them only after checking their
remaining contents. The live add-on was not reloaded. Restart Blender once
before trying this update so the new workspace RNA fields register reliably;
the add-on's Reload button is suitable for later code-only edits.

Solver-call ledgers measure numerical experiments, not model usage. Model token
totals are not exposed by the current tool interface; no claim of measured token
or cost savings is made.
