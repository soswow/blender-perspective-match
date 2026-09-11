# Constraint contribution and weak line geometry

Investigation: 10–11 September 2026, following `3f6b331`. Python/NumPy numerical
replay and Blender 5.1.0/macOS; exact requests and runtime metadata are recorded
by the runners. No user scenes or images were used.

## What the experiment isolates

The first nine families mostly supplied enough point evidence to recover the
cameras even if lines or symmetry contributed little. `constraints.py` adds
three positive/removal pairs. Measurements stay identical when the constraint
is removed:

| Evidence | With constraint | Without constraint |
| --- | --- | --- |
| Visible strokes in one non-anchor camera, no point picks | Known 3D lines recover its pose | Refuse unsupported single-view free lines |
| One observation per member of a mirror point pair | Recover both 3D points | Leave them unreconstructed; retain supported cameras |
| One stroke per member of a mirror line pair | Recover the infinite lines | Leave them unreconstructed; retain supported cameras |

Exact inputs pass all six contracts. A seventh case adds another mirror-line
stroke. Point position and line direction/offset are checked against independent
truth. Stroke endpoints need not identify the same physical positions across
images, so the oracle compares infinite lines, allowing different helper lengths.
Camera checks still use withheld object geometry. Known 3D line endpoints are
excluded from withheld observations because those coordinates were evidence.

Blender checks also solve, remove the constraint in RNA, and solve again.
Unsupported reconstructed geometry and helpers must disappear. A refusal must
preserve the previous valid cameras. Fresh-process reopening repeats the sequence.

## Finding: a good fit can conceal a weak 3D line

Three layouts with 0.3 px noise each flagged the two-stroke mirror-line geometry:

| Layout seed | Supporting-plane angle | Line direction error | Line offset / object diagonal |
| --- | ---: | ---: | ---: |
| 0 | 2.94° | 8.20° | 4.25% |
| 1 | 1.10° | 23.45° | 12.19% |
| 2 | 2.55° | 2.48° | 3.97% |

In seed 1, fitted point RMSE was only 0.21 px. Camera withheld RMS was 0.00,
0.22 and 0.39 px, so camera alignment alone would also miss the line problem.
The two image strokes back-project to almost the same plane after reflecting
one across the mirror. Small stroke errors can therefore move their intersection
substantially.

An independent implementation uses a null-space calculation for each image
plane, then a least-squares intersection. With **true** cameras, it still gives
21.32° direction error for seed 1; with solved cameras it matches the solver's
23.45°. That supports an evidence-conditioning diagnosis, not an arithmetic fix
or a new optimizer. It does not establish that every line failure has this cause.

Adding one visible stroke from the third camera changes seed 1 as follows:

| Quantity | Original | Extra view |
| --- | ---: | ---: |
| Best supporting-plane separation | 1.10° | 35.35° |
| Line direction error | 23.45° | 0.79° |
| Line offset / object diagonal | 12.19% | 0.34% |

This is a remedy verified for the captured setup, not a universal selection rule.

## Product response and regression contract

Sync now reports weak support for free 3D lines whose inlier interpretation
planes have no pair separated by the existing `LINE_PLANE_MIN_SINE` threshold
(about 6.9°). Mirror partners contribute reflected planes. Contradicting strokes
are excluded using the existing line reconstruction residual cutoff. Known 3D
lines and reflections of Known 3D lines are exempt because geometry is supplied.
The message and HTML report name affected lines; geometry and pose acceptance
are preserved. No new numerical threshold or solver objective was introduced.

The frozen `cases/mirror-lines-weak.json` expects an honest warning instead of
precise line geometry. Its required cameras must still pass. Line errors are
reported and only explicitly expected weak lines may exceed their accuracy
limits. The extra-view test requires the ordinary accuracy contract and that
the warning clears. Tests also cover ordinary free lines, known-geometry
exemptions, outliers, report escaping and incorrect geometry hidden behind
otherwise accurate cameras. The weak-line regression failed before the change.

## Limits and next work

The angle is a cheap geometric warning, **not a confidence interval**. It does
not quantify pick noise, short-stroke uncertainty, camera-pose uncertainty or
uncertainty in the mirror plane. A larger separation does not guarantee accurate
geometry; a narrow one can be accurate with exact evidence. Fixed directions
may help direction while depth remains weak. The existing threshold is a
conservative reuse, not a calibrated probability of error. False alarms and
missed cases need a broader bounded study before expanding quality claims.

Local verification passed: seven exact numerical variants, positive/removal
Blender cases, three live-removal/fresh-reopen sequences, a rendered frozen weak
case and the existing smoke. The full suite passed 241 tests (9 skipped); 30
focused tests passed after final report and warning-contract adjustments.
The generated product HTML passed content tests but was not visually inspected
because no browser was available. Hosted CI has not run yet.

The next follow-up implemented [product/probe request parity and capture/replay](../debug-sync/README.md),
using generated scenes with locks, slack and automatic origin preparation.
A debugging command must solve the same problem as the user's operation before
its diagnosis can be trusted; parity alone does not establish geometric accuracy.
