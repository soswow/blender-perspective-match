**Perspective Match: reliability and AI development proposal**
Investigation baseline: commit `5d876f6` / extension 0.5.0, 10 September 2026. Latest product checkpoint: live landmark mirror-plane positioning, 12 September 2026, described under **Current frontier**. Read that section first; the dated checkpoints preserve history, and the original proposal is retained for rationale rather than as an implementation checklist.

**My recommendation is to make every solver decision reproducible from a complete, versioned input, and judge it with checks independent of the implementation.** Build that foundation around the existing fixtures, Blender smoke test, and diagnostics. Then use it to improve calibration ownership, quality reporting, and selected solver decisions. This would turn a debugging session into an addition to a reusable capability.

Your clarification is that both metric accuracy and visual alignment matter, depending on the project, and most image sets depict one rigid object, with occasional mixed versions. I therefore keep rigid reconstruction as the normal model and recommend distinguishing the quality claims in the result. I would not add separate solver modes until examples show that they need different behavior.

The deeper product issue is that three different promises are currently close together: “these measurements fit,” “this reconstruction is well determined,” and “the Blender viewport represents that reconstruction.” The code has made substantial progress on each, but failures still occur at their boundaries. Another recurring issue is that a geometric model can be wrong for the evidence: uncertain calibration, approximate CAD, imperfect symmetry, or images of different versions of an object can make precise simultaneous agreement impossible.

## Current frontier — 12 September 2026

**Latest outcome:** a shared mirror plane can now follow a reconstructed point
landmark in ordinary Sync and point-FOV fitting, with supplied world-axis or
object orientation. The landmark is fitted jointly, not copied into a fixed
Empty. The live-reference checkpoint below records scope and verification;
unknown plane orientation remains a separate proposal.

**Dropdown follow-up:** real use exposed a UI boundary missed by direct RNA
assignment: Blender's menu stores enum values as float32, rounding the previous
31-bit hashed IDs so the setter could silently clear a selection. Both mirror
selectors now use exactly representable 24-bit values; persistent landmark IDs
remain strings. The native live-reference verifier reproduces the menu conversion
(failed before the fix), checks both selectors and the None sentinel, and forces
a hash collision at the numeric limit. This is a menu-boundary regression, not an
automated mouse-click test. See
`tools/synthetic_sync/verify_landmark_mirror_blender.py` and Blender's
[enum-menu implementation](https://github.com/blender/blender/blob/blender-v5.1-release/source/blender/editors/interface/interface.cc).

**Initial implementation scope (completed):** investigate practical point-FOV reliability
with incomplete/imperfect picks and alternative initial FOVs, carrying findings
through to actionable diagnostics; then support Is in Plane and supplied point
symmetry in focal fitting. The two sequential phases below retain bounded
experiments and coherent outcome commits. Remaining accuracy limitations are
explicit; this does not close the wider reliability plan.

**Imperfect-pick phase:** the [frozen reliability follow-up](../tools/synthetic_sync/cases/independent-focal-reliability/README.md)
separates 24 saved-start optimizer trials from eight fresh Sync calls (six
distinct inputs and two recorded harness duplicates), then 12 fresh-start
bundle trials. Selected eligible exact/noisy 12-point, noisy 16-point and
alternative-initial-FOV cases fit successfully. Noisy maximum focal errors
remain about 7.8–10%, within the broad local intervals; neither an eight-point
recipe nor calibrated 95% coverage on real picks is established. Naive 8/12
subsets of this partially visible object were ineligible on per-camera counts.
The product now names an under-supported camera and its pick count before
registration. A failed fit may suggest checking a pair of views using a bounded
raw-correspondence diagnostic. It deliberately does not accuse one landmark or
gate acceptance: withheld-model uncertainty and leverage make that stronger
claim unjustified. Success incurs no diagnostic work. The 25 focused numerical
tests pass after integration; combined verification is recorded with the
plane/symmetry phase below. Initial guesses and noisy geometry remain
recorded limitations, not closed findings. This phase is a diagnostic outcome,
not a new accuracy guarantee.

**Plane and symmetry phase:** the point-FOV path now supports X/Y/Z and Free
Is in Plane buckets and point mirror pairs with a supplied Mirror Empty,
including Plane Slack and normal-only Mirror Slack. It keeps the anchor's
stored orientation and position fixed; external axis/mirror references must
be meaningful in that frame. Every point still needs two picked views, so
one-view constraint seeding remains a separate extension. Known 3D, Ground,
lines, locks, missing mirror planes and unsupported camera roles are refused.
The raw-pick depth screen remains conservative even when constraints could
otherwise supply depth. Conditional focal uncertainty propagates click noise
through the constrained fit without counting geometric springs as new picks.

The [constraint evidence](../tools/synthetic_sync/cases/focal-constraints/RESULTS.md)
uses 11 of 12 reserved Sync calls and 15 of 18 bundle fits, with source archives
and exact inputs. Four exact hard positives and matched removal controls fit
accurately; final-source replays pass hard-gap and metric-scale-bound guards.
An off-anchor supplied mirror needs a free baseline parameter because anchor
placement makes scale conditional on that plane. Through-anchor mirrors keep
the unobservable scale gauge. Translation-invariance and input-immutability
regressions protect that decision and its normal conversion.

Noisy Free-plane enforcement gave essentially the same withheld error as
removal (2.026 vs 2.022 px); no general accuracy gain is claimed. A noisy soft
axis case reached 0.858 px withheld RMS with 2.19% maximum focal error. A biased
mirror offset produced 13.59 px raw-world withheld error but 0.458 px after a
separate training-derived scale alignment: that is conditional metric bias,
not a newly discovered mismatch that pixels can resolve. The old baseline-ratio
checker exaggerated another noisy error; immutable evidence retains it and
links the corrected training-point-only scale assessment. Never align cameras
individually or use withheld points to choose that scale.

Integrated Blender 5.1.0 checks passed four exact hard cases and two noisy soft
cases through preparation, numerical isolation, stale-constraint rejection,
direct joint application and fresh-process read-only reopening. The exact
native-camera withheld RMS values are below 0.001 px. The soft cases retain
their accuracy flags; application/reopening success does not relabel them as
accurate reference-world geometry. Seven native initial Sync/bundle calls were
used: six distinct cases and one explicitly retained retry after correcting
the Blender checker's old scale convention. Logs, source archives and generated
scenes are under `.local/focal-constraints-validation/` (the failed first soft-axis
attempt is retained). The verifier now saves applied generated state before
the native check so future checker corrections can reopen it without solving.
Four exact cases and their read-only reopens are wired into CI; hosted execution
remains unverified. The combined numerical run executed 418 tests in 463.923 s
with two optional local-YAML skips. Its only failures were 21 subcases of the
new frozen-fixture comparison, which demanded bit-identical generated floats
across NumPy environments. That test now checks structure exactly and floats
at 1e-14 relative/absolute tolerance, consistent with existing fixture tests;
the frozen inputs and geometry accuracy thresholds are unchanged. The affected
constraint and focal modules then passed all 32 tests in 1.574 s. The complete
suite was not repeated after this test-only correction. The Blender 5.1.0
`validate_addon.py` smoke test passed. No private project was modified.

**Weak-evidence constraint follow-up:** the [paired pilot](../tools/synthetic_sync/cases/focal-constraint-reliability/RESULTS.md)
compares correct, removed and slightly wrong plane/mirror relations with partial
point overlap and cameras on the same side of the object. It used 12 initial
Sync calls and 18 bundle fits, retaining exact inputs, source snapshots and
all results. Each camera has 11–12 picks and each point two or three views;
visibility is an unobstructed scaffold, not an opaque-object simulation.

A saved-start cross isolated a startup defect. Enforcing a correct hard axis
plane during Sync at guessed intrinsics produced a poor start (2.894 px) from
which even the unconstrained focal bundle failed. The same picks initialized
without the plane let the constrained joint bundle converge. The public
point-FOV route now uses one point-only initial registration, with all original
relations enforced in the final joint fit. It does not add a second registration
or weaken final constraint checks. The fixed public case fits at 0.232 px and
satisfies its hard plane to 8.5e-9 world RMS.

Withheld shape RMS in this one noise draw was 1.04 vs 1.42 px for axis plane
versus removal, 2.26 vs 2.46 px for Free plane, and 1.66 vs 2.35 px for mirror
pairs. These modest improvements are examples, not a measured general failure
rate. Shape assessment permits one positive global similarity fitted only to
training points; fixed-frame errors remain separately visible. In particular,
the accepted axis case has 3.32 px withheld error with only anchor-fixed scale
alignment. Correct mirror fitting was stable from both initializations; the
wrong axis member and tilted mirror still refused after the startup change.

A 1.43-degree wrong stored anchor orientation with a supplied mirror was
accepted: 15.12 px raw-frame withheld error but 1.24 px after the single shared
alignment. This preserves the distinction between external-frame accuracy and
visual shape; it is not proof of a shape defect or correct supplied references.
Every accepted case's four conditional focal intervals contained truth, but
one noise draw cannot establish statistical coverage. The raw-pick depth
screen remains conservative; this work does not enable constraint-only depth
recovery, unknown mirror orientation, one-view FOV points, distortion or crops.

Integration verification passed: **423 numerical tests in 428.718 seconds,
with two optional local-YAML skips**, including the genuine public weak-axis
regression. Two fresh Blender 5.1.0 exact positives (axis plane and off-anchor
mirror) passed preparation, application, stale-input rejection and read-only
reopening; native withheld RMS was 0.000280 and 0.000201 px respectively.
This separately declared verification used one new public regression solve
plus two Blender Sync/bundle pairs, beyond the 12/18 experimental budget;
existing suite calls are the normal regression workload. Reopens used no solves.
Generated scenes and logs are under `.local/focal-reliability-validation/`.
The existing CI constraint checks exercise the changed startup path too; hosted
CI was not run in this session. No private project was modified.

**Open question for later — infer the symmetry plane orientation:** the user
may know several corresponding mirror pairs and that all share a plane,
without knowing its orientation or having an object to represent it. A landmark
can now supply the plane's position as described below, but joint inference of
its normal from pairs/cameras remains unimplemented. Establish when the
observations determine it, which freedoms remain, and how to report ambiguity,
including partial visibility and approximate symmetry.

**Live landmark mirror position — implemented:** the user's intermediate idea
is now a product option. Choose Mirror Position → Landmark, select an on-plane
point, and supply a world plane or optional orientation object. Selection uses
landmark identity, so rename/reorder/reopen do not substitute another point.
The selected point need not be the object's center and remains a reconstructed
landmark; it is not promoted to exact Known 3D. Its cached position and the
orientation object's translation are excluded from the numerical plane input.
Deleting/excluding the point or changing it to a line produces a clear refusal,
not a fallback to a static Empty.

The numerical input `mirror_landmark_id` reaches ordinary Sync, Diagnose,
ordinary lens searches and independent point-FOV fitting. Sync snapshot v4
stores it and accepts v1–3 with a None default. Reflection residuals use the
current point with analytic derivatives; positive Mirror Slack permits a normal
offset relative to that point. Ordinary Sync carries the accepted offset into
later reconstruction/recovery and restores it when an attempted BA update is
rejected. The new reference supplies no metric scale; point-FOV retains its
free baseline gauge. The normal remains fixed in the anchor frame.

Ordinary Sync requires two location-enabled picked views of the reference or
an explicit Known 3D reference. Point-FOV retains its two-view point rule and
other eligibility limits. The reference cannot itself be a mirror-pair member.
One-view reference support and estimating a missing normal are separate work.
Diagnose excludes the model-defining point from leave-one-out removal, while
retaining its picks and ordinary residual reporting.

The independent [live-reference fixture](../tools/synthetic_sync/focal_live_reference.py)
has 19 points and 76 exact picks, including a separately picked on-plane point;
the unused supplied plane position is deliberately wrong. Focal tests compare
clean/stale initial reference coordinates and verify the reference Jacobian,
normal offset and free-scale behavior. Sync regressions cover a noisy reference,
a stale fixed-plane control, soft offset, missing/invalid sources, point/line
Jacobian checks and dependent one-view geometry using an updated point/offset.
Worker verification used eight full Sync regression calls in the ordinary lane
and three public Sync attempts (two successful) plus seven bundle fits in the
focal lane; these were test runs, not an exploratory benchmark.

Integrated Blender 5.1.0 runs passed ordinary Sync and point-FOV with **no Mirror
Empty**, source/identity validation, stale-job rejection, direct application,
and read-only save/reopen. Exact native withheld RMS was 0.000100 px and
0.000175 px respectively. Preparation and reopening used zero solves; fresh
native verification used two Sync calls and one bundle. Source captures,
generated scenes and logs are in `.local/landmark-mirror-validation/`. The
addon smoke test passed. The combined numerical suite passed **440 tests in
456.626 seconds with two skips for absent optional local YAML files**.

The new CI step repeats both native modes and their reopens; hosted CI is not
yet verified. Remaining coverage limits: the new live-reference one-view line
logic has deterministic stage/Jacobian checks but no new full recovered-camera
line scenario, and outer BA rollback branches were reviewed rather than forced
through a new end-to-end rejection case. Exact synthetic success does not prove
reference accuracy or orientation correctness in a real project. Blender must
be restarted after this update because workspace RNA properties were added.
No private project was opened or saved.

### Finding disposition — keep fixes and unresolved evidence distinct

Confirmed reproducible defects should proceed to a focused regression and fix
within the reliability work, not accumulate as reports. An experimental stop
limits that experiment; it does not close the finding. Close a defect only with
the fixing commit and verification. Keep accuracy flags with uncertain causes
visible until diagnosed; do not relabel conflicting evidence as fixed geometry
or relax its oracle to make a status green. Update this register when a finding's
disposition changes; the dated reports below retain the original measurements.

| Finding | Current disposition | Evidence / next action |
| --- | --- | --- |
| Exact 2D-only startup accepts inaccurate geometry despite correct intrinsics | **Fixed and verified in `cc7d14d`** | [Frozen case and follow-up](../tools/synthetic_sync/no-vp-bootstrap-results.md): robust anchor connection plus direct/bridge competition restore shared- and mixed-focal true-K geometry. Actual old-solver regression fails; integrated 374-test suite and Blender solve/apply/reopen pass. |
| Noisy calibrated 2D-only startup collapses camera baselines | **Fixed and verified in `2eb7088`; residual accuracy findings remain** | [Noisy continuation](../tools/synthetic_sync/independent-focal-results.md): ray-distance refinement could reduce its objective by shrinking the camera baseline. Holding its unobservable scale gauge prevents that collapse. The old regression fails; fixed noisy mixed/shared fitted RMSE is 0.376/0.319 px with all withheld features in front. Mixed withheld RMS 1.974 px and center error 0.068 object diagonals remain accuracy findings, not closed by this fix. |
| Guessed intrinsics can produce low-error but inaccurate free 3D | **Open quality/uncertainty gap** | [Unknown-focal continuation](../tools/synthetic_sync/unknown-focal-results.md): shared and mixed guessed-K Sync runs fit at ~0.49 px but fail withheld geometry. The actual Same Lens route succeeds on shared K with an explicit ±25% range; a general warning or acceptance rule needs stronger evidence. |
| Correct hard plane can prevent independent point-FOV startup at guessed intrinsics | **Fixed and verified at this checkpoint** | [Weak constraint pilot](../tools/synthetic_sync/cases/focal-constraint-reliability/RESULTS.md): crossing saved starts isolates premature constraint enforcement. One point-only registration followed by the constrained joint fit changes the preserved refusal to an accepted 0.232 px fit; wrong-reference controls still refuse. Weak-view and fixed-anchor accuracy limits remain explicit. |
| Pure rotation is accepted as reconstructed free 3D | **Ordinary Sync remains open; new point-FOV mode refuses preserved controls** | [Frozen true/guessed-K rotation controls](../tools/synthetic_sync/unknown-focal-results.md) expose ordinary Sync's depth ambiguity. The optional point-FOV mode screens raw picks for a connected graph of non-homographic support, checks the stated noise model and rejects deficient local geometry. Its tested refusals do not establish a general Sync ambiguity detector or a universal false-positive rate. |
| Mild noisy free-scale/overhead cases and five of ten noisy graph cases exceed provisional accuracy limits | **Open accuracy findings; cause not established** | [Initial pilot](../tools/synthetic_sync/pilot-results.md), [evidence placement](../tools/synthetic_sync/evidence-results.md), [graph sweep](../tools/synthetic_sync/graph-results.md). Reassess after the exact startup fix; compare input uncertainty with reconstruction sensitivity before claiming a solver defect. |
| Weak mirror-line geometry under noisy or biased strokes | **Geometry limit still open; weak-support warning fixed** | [Constraint evidence](../tools/synthetic_sync/constraint-results.md). Depth is poorly determined by nearly coincident supporting planes. More copies of the same weak evidence do not establish accuracy; test better evidence or an honest uncertainty response. |
| Contradictory recovered camera remains above the withheld limit | **Open inconsistent-evidence outcome; healthy-camera corruption fixed** | [Recovery evidence](../tools/synthetic_sync/recovery-results.md), [line-bearing continuation](../tools/synthetic_sync/recovery-acceptance-results.md). Keep the remaining ~2.43 px flag; do not expect exact geometry from intentionally inconsistent picks. |
| Biased hard/soft Known 3D references displace geometry | **Intentional model conflict, not a demonstrated arithmetic defect** | [Bias controls](../tools/synthetic_sync/biased-reference-results.md). Softening helps but does not certify truth. Current prior-release pilot adds no detection beyond the existing anchor warning; test hidden bias before adding a product warning. |
| Promoted reconstruction points treated as trusted Known 3D | **Open workflow/representation issue** | User clarification below: these are working estimates with shared uncertainty. Preserve this distinction in future fitting/diagnostics; no automatic promotion to independent evidence. |
| Recovered-update acceptance does not protect the full line/constraint objective | **Open coverage and acceptance-contract gap** | [Accepted-stage control](../tools/synthetic_sync/accepted-recovery-results.md). No remaining natural accepted-update damage is established by that control; find a paired damaging case before changing acceptance policy. |
| Independent unknown focal lengths without VPs | **Implemented for bounded point graphs, including plane and supplied-mirror relations** | [Production controls](../tools/synthetic_sync/cases/independent-focal-production/README.md), [imperfect-pick evidence](../tools/synthetic_sync/cases/independent-focal-reliability/README.md), [constraint controls](../tools/synthetic_sync/cases/focal-constraints/RESULTS.md), and [user workflow](sync.md#independent-fov-estimates-from-shared-points-no-vp-lines). Exact controls and native fit/apply/reopen pass; weak depth is refused and selected failed fits now identify a pair to check. Local uncertainty remains conditional and noisy accuracy flags remain. One-view relation members, missing-plane inference, other constraint types, distortion and unknown crop offsets are not handled by this mode. |
| Native modal scheduling, undo/redo, platform/package/hosted CI | **Unverified boundaries** | Generated preparation/job/reopen checks cover specific paths, not these broader claims. Select one bounded check when it reaches priority. |
| Line Jacobian reparameterization / performance prototypes | **Not promoted; no demonstrated overall improvement** | [Line-position experiments](../tools/synthetic_sync/line-position-gauge-results.md). The actual projection defect was fixed; a prototype with worse geometry is not a pending fix to ship. |

**Product fixes already landed (representative grouped audit):** Fit Only mirror
ownership (`b902124`); sparse-graph ground propagation (`76e79b1`); preservation
of established geometry during recovered updates (`5a4a34d`); hard-ground/shared
plane initialization (`19373fe`); supported one-view plane recovery (`3808b7b`);
mirror/plane/parallel compatibility (`3aa0dcc`, `6d4ff2f`, `3997bd3`); stale
Diagnose/lens results and lens rollback (`6c5712a`, `f6a7673`, `fd0633f`);
background job retirement (`de057d2`); lens support/initialization (`2ab1165`,
`4cd7a19`); line-bearing recovered rebuild crash (`5ce36a0`); and infinite-line
projection (`c304fbe`). Their focused regressions and the integrated suite retain
the fixes. This list is a disposition index, not a substitute for the linked
case-specific contracts or a claim that every geometry is now accurate.

### Current scope and checkpoint

**Objective and scope:** improve Sync reliability through reproducible evidence and independent checks of the rest of the object, beyond fitted picks. Both visual alignment and metric accuracy matter. Synthetic scenes are the primary corpus; private projects are optional evidence. Images are optional. **Latest user steering:** reliable FOV estimation from shared landmarks without dependable VP lines is now a priority; both shared-lens sets and mixed lenses/zooms/crops are common. Image transformations and distortion are now in scope when they affect that workflow. Automatic AprilTag/VP detection remains optional, not a prerequisite. Earlier deferrals below are historical.

**Newest result:** **Estimate FOV from Points** is an opt-in product mode under
Refine Lenses when Same Lens is off. It jointly fits independent focal lengths,
camera poses and free 3D points without VP lines or Known 3D. The implemented
scope is 3–8 cameras and 8–80 points, at least eight picks per camera and
two views per point, fixed principal points and zero distortion. The current
plane/symmetry continuation adds Is in Plane and supplied point-mirror
relations; Known 3D, Ground, lines and unsupported camera roles still cause an
explicit refusal rather than being omitted. Manual FOV supplies the starting values, with a separate
default ±40% focal window. Existing lens searches keep their ±18% default.
The full workflow is in [the Sync guide](sync.md#independent-fov-estimates-from-shared-points-no-vp-lines).

The mode uses fresh Sync initialization followed by a NumPy-only joint fit,
so Blender needs no new SciPy dependency. Acceptance requires raw-pick depth
evidence, convergence, complete support, positive observed depths, an interior
focal solution, acceptable per-camera fit and a full-rank local model. The
explicit Pick Error setting controls a noise-consistency check and local 95%
focal intervals. Boosting landmark weights does not pretend the original
clicks became more precise. These are local conditional sensitivity estimates,
not certified calibration accuracy or a search for every alternative solution.
A homography-compatible scene may be planar or rotating; the refusal asks for
translated views and depth-spread points without claiming to identify motion.

Accepted fits apply focal lengths, poses and points together through the
existing rollback machinery, without another Solve Sync replacing the fitted
geometry. Refusals leave scene geometry, cameras and plates unchanged. Stale
inputs and switched modes are rejected. Restart Blender for the new RNA fields.
Ordinary Solve Sync still lacks a general ambiguity guard; unknown crops,
distortion, biased/correlated picks and promoted-reference uncertainty remain
open boundaries.

**Point-FOV integration checkpoint:** 393 numerical tests passed in 419.423
seconds with the OpenCV environment (two absent optional sample-YAML skips).
Blender 5.1.0 passed actual mixed/shared guessed-K numerical fit, independent
withheld-object projection checks through evaluated native cameras, joint
application and fresh-process read-only reopening of both generated scenes.
Weak-baseline and pure-rotation cases returned useful refusals without scene
changes. A separate controlled-result Blender check passed direct application,
stale-input/mode rejection, rollback and image ownership without numerical
solves. The existing broad Blender add-on smoke test also passed. The same
fixed four-case and reopen checks are now wired into Blender
CI with a three-minute step limit and artifact retention; hosted execution has
not been verified. Native modal scheduling, undo/redo and platform packaging
remain unverified; these checks do not claim them.

Verification used four reserved inner Sync calls and four point-bundle calls;
reopening added zero solves. Logs, requests and generated files are under
`.local/point-focal-validation/`. The separate numerical exploration used ten
of twelve reserved Sync calls and ten of twelve bundle attempts; its frozen
requests, failed attempts and exact per-run source archives are preserved in
[the production corpus](../tools/synthetic_sync/cases/independent-focal-production/README.md).
Older prototype results are still historical; the new archives do not repair
their provenance retroactively. No private project was modified. Sol workers
implemented the numerical mode and Blender integration independently; main
review corrected the numerical guards, covariance interpretation and native
input/application contract before running integrated verification. This is
delegation in practice, not a measured token-savings claim. Code, tests, tools,
changelog and documentation belong to one coherent feature commit.

**Next priorities:** challenge this mode with held-out noisy and biased picks,
alternative initial FOVs, partial overlap and crop/principal-point errors;
measure erroneous acceptance as well as useful refusal and withheld geometry.
Use those results to improve the product's evidence guidance beyond the bounded
plane/point-mirror support. Separately extend honest ambiguity handling to
ordinary Sync and preserve the distinction between promoted working references
and independent Known 3D. Do not close the noisy mixed accuracy flag solely
because the focal fit converges or its local interval is finite. Each bounded
experiment must feed a fix, a usable diagnostic or an explicit unresolved
finding; exhausting its solve budget is not by itself a reason to stop the
authorized development task.

**Previous product result — noisy pair-baseline fix:** the
[noisy focal continuation](../tools/synthetic_sync/independent-focal-results.md)
led to `2eb7088`. Correctly calibrated noisy cases
previously accepted distorted geometry with withheld features behind cameras;
two-view ray refinement now holds its seed baseline length instead of shrinking
it to improve the residual. This removes an invalid optimization direction;
it supplies no metric scale and preserves pose locks.

The independent-focal experiment also separated numerical stalls from evidence
ambiguity: four dense fits converged where sparse fits exhausted their budget,
but weak-motion fits still had wrong depth. Mixed/shared noisy fits improved
withheld alignment while retaining broad local focal uncertainty (8.7–22.2%
upper excursions under an explicit 0.5 px noise assumption). Pure-rotation
fits fabricated parallax despite raw picks remaining homography-compatible.
At that checkpoint no independent no-VP lens mode or general ambiguity detector
was shipped; the point-FOV mode above supersedes the former limitation only.
Historical guessed-K focal fits used the pre-fix initializer; their results
must not be presented as post-fix performance. The preceding
[unknown-focal continuation](../tools/synthetic_sync/unknown-focal-results.md)
established a working actual Same Lens search when its explicit range included
the correction, and `cc7d14d`'s true-K startup fix remains verified.

**Noisy pair-baseline integration checkpoint:** the full numerical suite passed
386 tests (two absent optional sample-YAML skips) in 380.523 seconds with the
OpenCV environment. Blender 5.1.0 passed generated exact shared-true-K scene
creation, numerical solve, camera application and fresh-process reopening.
The noisy numerical regression fails on the pre-fix `3045dd4` checkout and
passes after the fix; focused corpus checks also pass on both Python
environments. Validation logs and generated files are under
`.local/noisy-pair-validation/`; no private project file was modified. The
worker branch `experiment/focal-noise` retains its implementation/evidence
commit; its clean worktree is removed after integration. Nothing was pushed.
Implementation, tests, changelog, user docs and this review checkpoint form one
coherent fix commit. The earlier three unpublished prototype commits were
combined in `3045dd4`; published/release history was left intact.

**Historical next decision (now addressed by the point-FOV implementation):**
evaluate independent no-VP focal recovery from the corrected
startup, with model comparison and explicit uncertainty. Optimizer convergence,
positive fitted depths and fitted parallax all failed to identify ambiguous
motion in the preserved controls. A homography-compatible scene can be planar
or rotating; do not turn this small corpus into a cutoff that claims to identify
the motion. The surviving noisy mixed accuracy flag also needs stronger evidence
before being called another arithmetic defect or closed as expected noise.

**Earlier independent-focal prototype checkpoint:** prototype/evidence, guards and
corpus checks are combined in `3045dd4` (the three unpublished integration
commits were squashed; their original history remains on
`archive/independent-focal-before-squash`). Shared-lens user instructions are
in `036dbd5` and [the Sync guide](sync.md#shared-point-matches-with-an-approximate-shared-fov-no-vp-lines).
Seven focused evidence tests pass on both Python environments without new
numerical solves. In the OpenCV environment, use `PYTHONPATH=tests:.` with
unqualified test module names to avoid an installed `tests` package shadowing
this repository. No production code changed; the existing 374-test/Blender
checkpoint was not repeated. Sol implemented the prototype; main reviewed its
gauge, supported-input guards, uncertainty interpretation and preserved limits.
This establishes a workable delegation pattern, not measured token savings.
Next, test realistic pick noise and alternative starting guesses under a fresh
declared budget before deciding on a production independent-focal mode or an
uncertainty warning. Iteration-limit refusals are not an ambiguity detector.

**Earlier unknown-focal integration checkpoint:** evidence is committed as `fc41142`,
read-only corpus checks as `9a35f2b`, and their cross-NumPy angle-roundoff
correction as `70060e2`. All three new evidence tests pass on both the OpenCV
environment and default interpreter without numerical solves; the worker also
passed fourteen focused bootstrap/budget tests (including one existing regression
solve). The experiment itself used fifteen inner Sync calls, 46.43 active seconds.
No production code changed, so the prior 374-test/Blender product checkpoint was
not repeated. Sol owned the implementation, artifacts and documentation; main
review caught the missing corpus checks, stale next-step wording and derived-angle
portability issue, then integrated and reran only the new read-only checks.
This is operational use of the Sol-led workflow, not a controlled token-savings
comparison. Private partial usage counters are in
`.local/unknown-focal-validation/task-usage.json`. The clean worker worktree is
removed, its branch retained, older stashes untouched, and nothing pushed.

**Reference provenance clarified by the user:** trustworthy external Known 3D is
rare. Most Known 3D points are promoted from agreement among existing matches to
help fit later matches; they are working estimates, not independent measurements.
The normal startup may contain only corresponding 2D picks, with no VP lines or
ground points. Do not require Known 3D to bootstrap that workflow or use promoted
points to certify its accuracy. Free reconstruction has an arbitrary global
frame and scale; absolute dimensions require additional metric evidence. Existing
Known 3D Slack can soften working references, but does not make them independent
or recover their lost uncertainty/correlation. Provenance-aware reference handling
is a future product question, not an implemented feature or a prerequisite for
this pilot.

**Earlier checkpoint (before the no-VP focal work):** the truth-free reference-sensitivity pilot is committed as `d302390`; analytic pinhole line projection is fixed in `c304fbe`. The projection regressions fail on old code and pass after the fix, including arbitrary line directions, combined transforms and unchanged distorted-camera behavior. Combined validation passed **349 tests, 9 skipped, 260.811 seconds**, plus Blender 5.1.0 smoke. Two additional diagnostic comparison tests and independent no-solve projection/cached-pair checks passed after integration. Evidence tooling is committed as `8abac35`. Both workers have finished; their worktrees are archived and removed. No task is running. Hosted CI remains unverified; nothing has been pushed. Earlier reference-bias and plane/parallel work remains in `67f8877` and `3997bd3`.

| Area | Established capability | Remaining boundary |
| --- | --- | --- |
| Reproduction and independent truth | Complete versioned `SyncSolveRequest`; arbitrary saved-scene capture/replay; product/probe parity; synthetic camera, point, line and withheld-object checks; order/cache replay | Capture has no independent real-world truth. General search and arbitrary camera/operation reduction are not implemented. |
| Constraint and graph coverage | Ground/known/free scale, sparse graphs, roles, recovery, Is in Plane, mirror/parallel combinations; paired removal controls; forbidden-geometry and named line-accuracy reduction; recovered rebuild keeps line IDs out of point triangulation; free hard-plane fits preserve compatible axis/CAD parallel directions; pinhole line projection is invariant to sliding its representative point | Four paired local CAD-bias controls now quantify hard/soft compromise; no broad guarantee for biased planes, imperfect symmetry, conflicting priors or noisy/weak geometry. Recorded accuracy flags remain flags. |
| Blender state and jobs | Generated create/apply/reopen and selected role/removal/origin sequences; shared lens inputs; stale Diagnose/lens rejection; lens rollback; job retirement across load/cancel; joint point-FOV application and no-op refusal | Native modal scheduling, undo/redo, other operators and general transactional application remain unverified. |
| Honest quality comparisons | Weak-line support warning, plane-derived depth labels, removal of unsupported scale wording; lens scoring includes recovered point picks and retains successful-incumbent support; refused lens trials restart registration rather than reuse placeholder poses; optional point-FOV depth/noise guards and conditional local intervals | Truth-free prior release measures model dependence; its first pilot added no detection over the existing anchor warning. General Sync sensitivity/ambiguity reporting, actual registration provenance, full constraint/line-aware acceptance and Iterate Known 3D comparison remain open. Point-FOV intervals do not cover unknown crops, biased picks or alternative solutions. |
| Development workflow | Focused numerical runner; PR/main numerical and Blender CI configuration; isolated-worktree workflow with sequential integration | Hosted/package/platform validation is incomplete. Parallel work does not itself reduce model usage. |

**Completed continuation from `16a6cb6`:** two Sol/high workers investigated separate numerical boundaries. Reference sensitivity landed as `d302390`; no product warning was promoted. The line-position trial led to the narrow projection fix `c304fbe` and guarded evidence tooling `8abac35`; BA/Jacobian and acceptance stayed unchanged. The main thread reviewed independent checks, corrected ownership/comparison/reporting gaps, validated and committed each increment. The line worker used 17 exploratory solves versus its assigned initial cap of 12; that overrun and its cause are recorded below. Both task worktrees were archived and removed. Existing stashes and detached baseline worktrees remain untouched.

**Open directions after the third bounded parallel follow-up:**

The no-VP lens and experiment-budgeting investigation below is the newest
priority. These four earlier directions remain open, rather than four concurrent
assignments.

1. **Acceptance across constraint types.** The recovered-camera guard protects established point fits; line-only support and the full constrained objective are not protected by that contract. The [first line-bearing experiment](../tools/synthetic_sync/recovery-acceptance-results.md) found a crash, now fixed; afterward its candidate was rejected and production matched the freeze control. The stage-only positive control now establishes an accepted candidate, but does not show recovery-caused final damage; it exposed a separate plane/parallel defect. Natural accepted-routing coverage and line-aware acceptance remain open.
2. **Weak evidence and biased references.** The reference-release pilot (`d302390`) retained support and quantified reference/camera movement on eight exact/noisy controls, but the existing anchor warning already detected both biased references. Do not add a new warning from that result. A worthwhile next test puts reference error along the anchor viewing ray, where its anchor projection stays unchanged, or removes an unavailable anchor pick; retain accurate noisy controls and compare against existing checks. Biased planes, imperfect symmetry and the recorded noisy graph cases remain open.
3. **Remaining Blender lifecycle boundaries.** Choose one bounded undo/redo or native modal sequence, building on `verify_jobs.py` and `verify_job_reload.py`. Do not repeat the already-covered stale-input and old-callback cases as new investigations.
4. **Measured structural or performance experiments.** The observed along-line incident led to a projection correction, not a BA representation change. Orthogonalizing the line Jacobian reduced optimizer calls and internal drift but slightly worsened independent geometry on the deliberately biased case; no BA-only optimization was promoted. Existing instrumented timings include diagnostic probes and do not establish a clean speed improvement. Broader representation, alternative seeds and residual-only evaluation still need corpus evidence and uncontaminated stage timings.

**Completed continuation from `80a6d82`:** two Sol/high workers used separate worktrees. The reference-bias quartet landed as `67f8877`. The accepted-update investigation led to plane/parallel fix `3997bd3` after review identified an exact relation missed by the broad accuracy oracle. Each worker began with production read-only and bounded solves; only the confirmed defect received implementation authorization. Main reviewed and integrated the changes. Existing stashes/baseline worktrees remain untouched.

**Lens initialization, now fixed:** the first Sol/high investigation proved the warm-placeholder failure through identical true-focal requests. The subsequent worker fixed the common lens evaluation path: reuse only successful poses; keep explicit locks and refusal scoring separate. The frozen outer-search regression fails on old code and passes with all independent geometry checks on the fix (0.000000033 px point RMS). All 15 focused lens tests pass, including successful warm-start controls, explicit locks and useful partial refusals. The underlying direct-Sync warm/cold reproducer intentionally retains its old result; the lens search now avoids that invalid input. See [the complete evidence](../tools/synthetic_sync/lens-initialization-results.md).

**Selective escalation procedure:** use Sol/high with fresh context for a bounded numerical experiment; the main thread owns hypothesis review, independent correctness criteria and the decision record. Give the worker only relevant paths, owned files, controls, a solve budget and a stopping condition (the first trial allowed at most 20 solves). Return artifact paths and concise measurements rather than full logs. Escalate unresolved geometry or conflicting evidence; do not rerun the worker's entire investigation in the main thread. Evaluate the workflow by review/rework needed and recorded usage where available, not by parallelism alone. No measured token-savings claim is available from this first handoff.

### Latest investigation: 2D-only startup pilot

**Fix continuation:** `cc7d14d` repairs the pair-registration failure with no
truth poses or Known 3D inputs. The strongest non-anchor pair was connected
through a weaker six-pick anchor overlap because image displacement outranked
spread/support; later a direct anchor pose bypassed better registered-view
bridges. Correcting only one choice did not restore accuracy. Both changes now
apply to unconstrained free-point startup; constrained/locked routing retains
its previous policy. The final shared and mixed true-K controls pass, and the
input-order control passes. Eleven of twelve exploratory follow-up calls used
41.36 cumulative active seconds; focused regression and integration test calls
are separate verification, not unrecorded exploratory attempts. Main also ran
the actual new regression against the unmodified old solver: it failed with
the original withheld errors in 3.179 seconds. No focal search or uncertainty
claim was added. Exact before/after stage records remain in the linked report.

**Integration evidence:** the generated Blender solve/apply and fresh-process
reopen both pass; maximum withheld projection RMS is 0.000088 px with RNA and
Blender camera precision included. Shared and mixed true-K numerical traces pass
below 0.000001 px. Review also tightened the policy for the core API's location
participation field and added complete source/runtime identity to diagnostic
cache keys, with mocked checks. Historical ledger records were not rewritten.
The final OpenCV-enabled integrated suite passed **374 tests in 274.117 seconds,
with two absent-sample-YAML skips**. Validation logs and generated Blender results
are retained in `.local/no-vp-fix-validation/`.
The worker's clean worktree is removed; its two commits are combined in the
single logical product-fix commit `cc7d14d`, with changelog, user docs, regression
and diagnostic evidence together.

**Cost practice, not a savings claim:** this continuation kept diagnosis,
regression, implementation and focused controls with one Sol worker. Main did
not repeat its exploratory solves; it reviewed and ran direct before-fix and
combined integration verification. A pre-final snapshot records main 5.34 million
input tokens (5.30 million cached) / 12.2 thousand output, and worker 10.58 million
input (10.39 million cached) / 39.0 thousand output. Main uncached input and
output were lower than the preceding tooling-building task, but total processing
was higher and the tasks differ; do not claim net savings. Repeated coordination
and corrections still cost tokens. Continue reducing main activity and use
Sol-led implementation with a single strong-model review where feasible, rather
than adding more parallel workers. Private boundaries and counters are in
`.local/no-vp-fix-validation/task-usage.json`; final reporting is outside that
snapshot. The executable tests now retain the repeated cache-identity lesson.

**Executed:** one of sixteen permitted Sync calls, 3.51 of 720 cumulative active
seconds, with 180-second per-call and 720-second outer process limits. The input
has three views, 23 noncoplanar landmarks, correct focal lengths, arbitrary
stored poses including the anchor, and no ground/Known 3D/VP/pose locks. Sync
reported success at 1.375 px fitted RMSE, but withheld projections missed by
5.28–17.28 px after one proper global similarity. It downweighted five exact
picks. The stop rule prevented guessed-focal and lens-search runs. See the
[frozen evidence and oracle controls](../tools/synthetic_sync/no-vp-bootstrap-results.md).
This establishes one reproducible startup accuracy failure, not its cause or
a claim that correspondence-only calibration is impossible.

**Validation and integration:** the OpenCV-enabled full numerical suite passed
371 tests in 264.746 seconds, with two absent-sample-YAML skips. The final eleven
focused budget/no-VP tests passed under both Python environments, including the
portable comparisons in `d1ad8be` and cache-result isolation in `e073887`.
Both generated Blender preparation/reopen controls passed. No production solver
behavior changed. The worker's commits are integrated, its clean worktree was
removed, and its experiment branch is retained. Older stashes and baseline
worktrees remain untouched. Nothing has been pushed.

**Independent review:** true-camera projections match all input picks within
1.2e-13 px. The two stronger view pairs have 15 and 14 shared noncoplanar points;
the remaining pair has six. Correct geometry passes the oracle after an arbitrary
global similarity; wrong focal/pose/points cannot hide behind zero reported RMSE.
Blender 5.1 generated preparation checks passed for shared true-K and mixed
guessed-K inputs, including actual Manual FOV application, missing origins,
Sync/lens preparation and fresh-process reopening. Numerical calls were blocked.
They establish unchanged prepared inputs and the current focal eligibility rules,
not solved-camera application or the native interactive workflow.

**Next numerical question after the unknown-focal continuation:** the remaining
guessed-K and weak/pure-rotation controls and a small actual Same Lens search are
now recorded in the [bounded continuation](../tools/synthetic_sync/unknown-focal-results.md).
Investigate an input-only independent-focal strategy for mixed lenses and a
geometry-only ambiguity response for pure rotation, with noisy and near-critical
controls before adopting a cutoff. Keep truth out of candidate selection,
preserve withheld checks, and do not promote triangulated working references to
independent evidence. Distortion optimization remains unimplemented.

**Workflow trial:** a fresh Sol/high worker owned the numerical cases/oracle and
returned one evidence packet at the planned review gate. Main independently
implemented the ledger and Blender preparation controls. Review found strict
float comparisons repeating a recently fixed portability mistake and incomplete
source hashing; the worker received a no-solve correction task. Record this as
review/rework, not a flawless handoff. The ledger reserves before execution,
retains failures and interrupted reservations, enforces call/active-time limits,
and supports exact serialized-result reuse with matching metadata. Its scope and
native-code timeout limits are in [development guidance](development.md#budgeting-experiments-and-model-usage).
Private usage snapshots are in `.local/no-vp-workflow/`. The initial live snapshot
was late; `task-usage.json` recovers the main baseline from the cumulative counter
before this task's start event, including initial planning. Through its pre-final
snapshot, main processed 3.32 million input / 17.6 thousand output tokens; the
fresh worker processed 2.48 million input / 21.3 thousand output tokens including
review corrections. Cached input was about 98% and 96%, respectively. These are
processing counts, not quota or billing measurements; they exclude subsequent
final reporting. Main also built tooling, so these counters cannot isolate
coordination overhead or establish token savings. Repeat measurement on the next
bounded task using the now-existing tooling before judging the model mix.

### Preparatory research: no-VP lenses and measurable agent work

**Status:** source audit and usage research complete; no new lens behavior has
been implemented. The OpenCV environment (`~/venvs/my`) ran 360 tests in 262.014 s:
two skipped because optional sample YAML files are absent; four subtest failures
were confined to one frozen-generator comparison using exact float serialization.
The observed differences were rounding precision, not geometry changes. Test-only
commit `0fcfe57` preserves exact structure and permits tiny floating-point
differences; all eight tests in the affected module then passed under both the
OpenCV environment (Python 3.12.6 / NumPy 2.0.1 / OpenCV 4.13.0) and the default
interpreter. The full suite was not repeated after that test-only correction.
The earlier 349-test result above remains the previous combined product checkpoint.

**Observed capabilities and gaps:**

| Route | Current behavior | Boundary |
| --- | --- | --- |
| Manual FOV / imported calibration | Supplies starting intrinsics without VP lines | Manual FOV is a supplied estimate, not a measurement inferred from landmarks. |
| Refine Lenses, Same Lens on | Can search without VP lines by scaling every starting focal length with one multiplier, keeping principal point and distortion fixed | A local correction, not independent recovery of unrelated focal guesses; it retains their starting ratios rather than enforcing one physical lens. |
| Refine Lenses, Same Lens off | Searches independently where VP orientation is usable | No-VP and one-point matches have their focal frozen. |
| Iterate Known 3D | Alternates VP-based camera refinement using Known 3D pins with Sync | Requires enough VP lines and at least four Known 3D picks. Shared 2D landmarks alone do not satisfy this contract. |
| Core Known 3D pin fitter | Can fit pose/focal/principal point without VPs when supplied fixed 3D coordinates | This capability is not exposed by Iterate's eligibility path; tentative triangulated points are not independent Known 3D evidence. |

Evidence: [lens input collection](../scene/__init__.py), especially
`collect_lens_refine_inputs`, `known_3d_iterate_roots` and
`ensure_ground_frame_from_landmarks`; [focal scaling/search](../core/lens_refine.py);
[pin fitter](../core/pin_refine.py); [current user workflow](sync.md).
Solve Sync prepares a calibrated ground frame, whereas Refine Lenses prepares
origins without that same ground-frame initialization. A guessed focal can
therefore influence the anchor frame before refinement. The anchor is fixed
during Sync, and no-VP focal candidates keep their private orientation. This is
a supported reason to test the full preparation path, not proof that every
such setup fails.

**Original pilot proposal (superseded by the executed 2D-only case above):** use the existing synthetic harness
for a no-VP, pinhole, centered-image pilot. Start with three translated views and
well-spread points at several depths. Compare a shared-focal case with a mixed
focal case, using known truth but perturbed starting calibration. Include a
ground-supported Blender preparation path and separate direct solver inputs.
First evaluate the existing methods and a true-calibration control; if the
control fails, investigate preparation/pose recovery before adding focal search.
Only then test a probe that permits independent focal changes and revisits the
anchor initialization. Keep this out of production until evidence warrants it.

Freeze the cases and oracle limits before searching. Judge focal error, camera
and reconstructed-point error, and withheld object projections using only one
legitimate global frame/scale alignment. Do not align each camera independently
or select trials using synthetic truth. Add low-parallax and pure-rotation
controls: a low fitted residual with divergent focal/withheld geometry must be
reported as ambiguity, not accurate calibration. Planar controls need careful
interpretation; planarity alone does not make every multiview calibration
impossible. Stop or change course if the favorable exact cases cannot recover
from modest focal errors, results depend strongly on starting guesses, or the
oracle cannot distinguish wrong geometry. Record every inner solve against a
predeclared budget; do not start an unbounded grid search.

Known crop/resize transformations should follow this pinhole pilot, then
controlled distortion. Do not initially optimize focal, principal point,
distortion and free geometry together: one can absorb errors in another.
Independent [COLMAP guidance](https://colmap.github.io/faq.html#fix-intrinsics)
likewise treats principal-point refinement cautiously and notes that shared
intrinsics across views can improve constraints. That supports the experimental
ordering, not replacing this project's solver with COLMAP.

**Agent-cost finding:** local metadata from two recent completed worker batches
shows about 18.17 million worker input tokens (roughly 96% cached) and 13.45
million main-thread input tokens (roughly 99% cached) during their corresponding
time windows. These are cumulative processing counts, not unique context sizes,
bills or subscription-quota measurements. Main activity cannot all be attributed
to the workers, and there is no matched Astra-only baseline. The supported
conclusion is that coordination overhead deserves measurement; savings are not
established. Private aggregate evidence is retained locally in
`.local/agent-usage-research/report.md`, outside version control.

The next operational trial should keep the already-used fresh Sol briefs, add
planning/review gates for the expensive main model, and measure the whole task
including review. Put enforceable solve limits and resumable, fingerprinted
artifacts in the harness; put task ownership and escalation rules in development
instructions. A new skill is not needed. See [the budgeting protocol](development.md#budgeting-experiments-and-model-usage).

**Commit workflow authorized by the user:** commit completed, verified increments as this reliability plan proceeds. Separate independently reviewable changes; keep each product fix with its regression, required changelog/user docs and reusable reproduction. Do not ask again for routine commits within this work. This authorization does not include pushing, releases or tags. Update this record with the resulting hashes so interruptions do not leave the next agent guessing which work landed.

**Read only what the next task needs:** [AGENTS.md](../AGENTS.md), [harness commands and contracts](../tools/synthetic_sync/README.md), [capture/replay](../tools/debug-sync/README.md), [parallel workflow](development.md#parallel-agent-work), then the relevant result record linked above. Preserve the oracle/schema, focused checks and exact failure evidence in any handoff. Use code and executable contracts to resolve discrepancies with historical prose.

**Do not repeat without new evidence:** the evidence-placement experiment missed its promotion threshold; automatic non-anchor origin preparation passed its controls; request/role omissions, stale-job application, lens score coverage and the demonstrated plane/mirror/parallel defects are fixed. Their checkpoints below give the controls and limits. No general solver rewrite or new skill is justified merely by the length of the history.

**Local recovery inventory at this checkpoint:** two retained stashes are based on `78f2322` and `0889ab0`; earlier inspection found their useful work represented by the shared-request and reducer commits. Three detached diagnostic worktrees remain under `.local/`: `mirror-plane-baseline-bfd122d`, `recovery-baseline-f117f4c` and `reducer-baseline-569da33`. The completed parallel worktrees were removed. The latest task branches `investigate/accepted-recovery-update` and `investigate/biased-reference-controls` remain at `80a6d82`; main owns their integrated commits. Raw paired records and original worker artifacts are archived under `.local/worker-handoffs/accepted-update/` and `biased-references/`; its validation is under `.local/accepted-update-integration/`. The latest `investigate/reference-sensitivity` and `investigate/line-position-gauge` branches remain at `16a6cb6`; their task worktrees were removed after main integration. Their artifacts are under `.local/worker-handoffs/reference-sensitivity/` and `line-position-gauge/`; current full validation is under `.local/line-projection-integration/`. Recheck `git status`, `git stash list` and `git worktree list` before future implementation; do not pop or delete these automatically. This inventory is machine-local, not a prerequisite for replaying checked-in cases.

When work lands, update this frontier and append its evidence to the checkpoint history. Keep old measurements dated; do not turn an earlier limitation into a new task when a later checkpoint already closed it.

The selective-escalation trial's worker worktree was removed after its script, exact case and findings were copied into this checkout. Branch `investigate/lens-initialization` remains at `be69b73` with no worker commits. Detailed local requests/control outputs and the original worker script are preserved under `.local/lens-initialization-trial/`; the script/case/results committed in `4cd7a19` are sufficient for reproduction.

**Parallel follow-up handoff:** two Sol/high workers started from `be69b73`: `fix/lens-refused-warm-start` owned the lens module/tests; `investigate/recovery-line-acceptance` first investigated with production read-only, then received narrow ownership of the point-ID crash fix and its regression. The main thread reviewed both diffs, preserved exact cases/results, and owns combined checks and documentation. Neither worker committed. Both task worktrees were archived under `.local/worker-handoffs/` and removed without force after integration; their branches remain at `be69b73`. Detailed numerical/Blender logs are in `.local/lens-fix-integration/` and `.local/recovery-line-integration/`. Existing stashes and baseline worktrees remain untouched.

## Implementation history

**Independent-focal numerical prototype — 12 September 2026:** the bounded
[three-focal experiment](../tools/synthetic_sync/independent-focal-results.md)
used saved exact 2D-only Sync starts, free camera/point bundle adjustment,
input-only candidate selection, and a separate local sensitivity check. Its
shared case converged and passed the withheld oracle; its highly accurate
mixed candidate stopped at the declared iteration cap. Weak translation and
pure rotation stayed inaccurate at low training RMSE. Commit `45ac27f`
preserves the exact run source and ledgers. This is numerical tooling only;
production calibration and acceptance remain unchanged.

**Third bounded parallel follow-up — 12 September 2026:** [reference-release controls](../tools/synthetic_sync/reference-sensitivity-results.md) compare original and released priors using only request/result data. All 16 solves succeeded; all eight pairs retained cameras, landmarks and picks. Biased-reference release gaps were about **0.178–0.180 scene units**, versus at most **0.009** for accurate noisy controls. The existing stored-anchor check already warned on both biased references and no accurate ones. The reusable diagnostic landed as `d302390`; no new warning or automatic reference selection was added. `d81d957` clarifies that the earlier numerical-only report omitted Blender's existing preparation warning.

The [line-position investigation](../tools/synthetic_sync/line-position-gauge-results.md) exposed a genuine inconsistency behind the large intermediate motion: finite sample spans could reject an otherwise visible infinite line after its representative point slid along that same line. Fix `c304fbe` projects the viewing plane analytically for pinhole cameras, preserving the original sampled approximation for distorted calibration. Independent sample projection, along-line shifts, transform composition, input immutability and degenerate/behind-camera controls verify the primitive. All **349 tests, 9 skipped**, and Blender smoke passed. The intentionally biased accepted-stage case still passes its unchanged independent limits; the prototype's lower fitted objective did not imply better withheld truth, so this is a correctness fix with a recorded tradeoff, not a general optimization claim.

**Further agent-work lessons:** compare new diagnostics against existing product checks before promoting them. Make prototype inputs immutable, use an independent oracle, and check cached comparisons for identical requests and lost support. Count every draft/repeated solve: the line worker used **17 exploratory solves**, exceeding its assigned initial cap of 12. Capture needed instrumentation on the first run and reuse artifacts instead of repeating a sweep to add reporting. Diagnostic residual-shift probes must be timed separately from optimizer work; the retained exploratory timings are instrumented and cannot support a clean speed claim.

**Second bounded parallel follow-up — 12 September 2026:** the accepted-update worker constructed a stage-only positive control: an already posed camera is explicitly marked recovered, and its proposed update actually changes the state. Independent checks did not show damaging acceptance. Review did find a separate exact-constraint defect: generic free-line fitting inside a hard plane could undo a compatible world-axis parallel direction, even though its broad one-degree line-accuracy check passed. The freeze control already has the error, so this is not evidence that accepting the recovered update caused it. The fix landed as `3997bd3`; see the before/after controls in [the accepted-update record](../tools/synthetic_sync/accepted-recovery-results.md).

The independent [biased-reference controls](../tools/synthetic_sync/biased-reference-results.md) use identical exact picks and truth, two locally wrong Known 3D positions, two truthful Known 3D positions, and a fixed calibrated anchor plus hard ground to retain the world frame. With hard priors the worst withheld-camera RMS is **6.978 px**; softening Known 3D reduces it to **1.359 px**, while biased-point truth RMS falls **0.178 → 0.025 scene units**. Softening also moves the truthful priors slightly. Both unbiased controls pass; both biased controls retain their original accuracy flags. This demonstrates useful compromise under a wrong model, not a new solver defect or a method for choosing slack automatically.

**Historical representation motivation at `3997bd3`:** the accepted candidate's line endpoints slid thousands of scene units along the same infinite line, while final rebuilding restored their extent. The subsequent experiment above traced a projection inconsistency and produced fix `c304fbe`. It did not justify the separate Jacobian/representation change. Do not repeat this original experiment as an untouched open task; use the retained case and report to investigate a newly demonstrated boundary.

**Workflow lesson from review:** broad accuracy tolerances and exact declared relations are different contracts. Check both. Likewise, hard coplanarity needs signed member positions: equal absolute distances on opposite sides of a reference plane do not prove that the members share a plane. Keep these checks executable, and have generated diagnostic prose derive its claims from actual results rather than assuming the original run will repeat. The stronger coordinator's review identified these gaps in otherwise plausible worker evidence; there is still no measured token-savings claim.

**Selective-escalation investigation — 12 September 2026:** after the integrated checkpoint, the user adopted the coordinator/cheaper-worker workflow. The first bounded delegation confirmed the refused-result initialization defect described in Current frontier and [the result record](../tools/synthetic_sync/lens-initialization-results.md). That first checkpoint added replayable evidence only; the next authorized follow-up below implemented the fix from this reproduction.

**Parallel implementation follow-up:** the next authorized delegation fixed lens initialization and independently found a recovered-line crash. Lens trials now reuse successful estimated poses only, preserving explicit locks and useful partial-refusal improvements. Rebuilding recovered geometry triangulates point-observation IDs rather than the mixed point/line list. Both frozen regressions fail on the old code and pass with the fixes. After the crash fix, the acceptance case's production and frozen-stage geometry are identical: the existing guard rejects the proposed update; the healthy camera has 0.541 px withheld RMS and the free line has zero direction error. The intentionally contradictory recovered camera retains its 2.431 px accuracy flag. Broader line/plane acceptance remains an unconfirmed hypothesis. No thresholds were relaxed.

Validation of that integration: **329 numerical tests, 9 skipped; passed in 241.602 seconds**, plus final Blender smoke. All 15 focused lens tests and 14 Blender lens-ownership controls passed before the point-ID integration. The production patches affect the lens evaluation boundary and one point-triangulation argument; no UI, RNA or acceptance thresholds changed. User-facing behavior is recorded in Unreleased and the Sync guide. Workers returned bounded evidence and were stopped. The user then requested incremental commits: lens work landed as `4cd7a19`, recovered-line work as `5ce36a0`, followed by the separate decision-record cleanup. No additional investigation was started at that checkpoint; the active continuation above followed it.

**First implementation checkpoint — 10 September 2026 (historical)**

The user chose **synthetic Sync evidence and withheld object alignment first**. The initial pilot added nine seeded scene families, independent projection/occlusion, camera and withheld-object metrics, order/cache replay, portable reports and generated Blender scene/save-reopen checks. It also fixed focused-test bootstrapping and added PR/main numerical and Blender CI configuration. At that checkpoint only, there was no shared production request, arbitrary-project capture, reducer or background lifecycle coverage. Subsequent checkpoints added them; the Current frontier table is the current status.

| Initial pilot result | Evidence at that checkpoint |
| --- | --- |
| Numerical corpus | Nine exact cases and 36 order/cache variants passed; a 27-case noisy sweep produced four borderline accuracy flags. |
| Blender comparison | Field-by-field comparison with `prepare_diagnose_sync`, solve/apply and evaluated cameras; zero slack, preset origins and stored pinhole calibration. |
| Product behavior | No production solver/UI change in the initial pilot; the evidence-placement experiment followed and is recorded next. |

Validation at this checkpoint: **225 unit tests run, 9 skipped; suite passed**, existing Blender smoke passed, and all nine generated families passed creation/application and fresh-process replay on Blender 5.1.0/macOS. The four noisy flags are **not confirmed solver defects**: two exceed the provisional center-error limit after free-scale alignment, and two exceed the overhead withheld-pixel limit. Do not weaken limits just to make them green.

For future sessions, start with the [harness commands and limitations](../tools/synthetic_sync/README.md), [pilot measurements](../tools/synthetic_sync/pilot-results.md), and [oracle/regression tests](../tests/test_synthetic_sync.py). The executable case schema and evaluator define current behavior; do not treat every original proposal below as implemented or still required. Local reports and generated `.blend` files are disposable artifacts, not repository dependencies.

**Evidence-placement follow-up (`3f6b331`):** the pilot checkpoint was committed as `55b1487`. The [evidence-placement experiment](../tools/synthetic_sync/evidence-results.md) then completed 94 solves using the two frozen overhead cases, paired support-only controls and fresh noise. A peripheral landmark gave a median 9.8% paired improvement, below the predeclared 20% promotion threshold; no new picking advice or weighting change is justified yet. Exact picks solve accurately and halving the original noise approximately halves withheld error. Read these results before repeating the single-added-landmark experiment.

**Constraint-contribution follow-up — 11 September 2026:** [the new corpus and results](../tools/synthetic_sync/constraint-results.md) test cases where Known 3D lines or one-sided mirror features supply necessary information. Paired controls remove only the constraint, keeping the picks. Blender tests also remove it after solving and verify that unsupported helpers disappear, or that a refused solve preserves the previous camera poses. Required reconstructed points and infinite lines now have independent geometry checks as well as withheld camera checks.

This produced the first product change from the laboratory. With 0.3 px pick noise, a mirrored line had **23.4° direction error despite 0.21 px fitted point RMSE**. An independent plane-intersection calculation reproduced the sensitivity even with true cameras. The evidence was weak: its two supporting planes differed by only 1.1°. Sync now reports weak 3D line support in its message and HTML report, using the existing line-plane separation threshold; it retains the geometry. An additional stroke from a distinct third view reduced direction error to 0.79° and cleared the warning. The frozen regression explicitly expects a warning; ordinary accuracy contracts remain unchanged. This is a limited geometric diagnostic, not a confidence interval or a general ambiguity classifier.

Validation for that change: the full suite passed **241 tests, 9 skipped**, followed by **30 focused tests** after the final warning-contract assertion and report adjustments. Seven exact numerical constraint variants, generated Blender create/apply/reopen cases, three live constraint-removal sequences, the rendered weak-case replay/removal, and the existing Blender smoke passed locally. Browser availability prevented visual inspection of the HTML; content/escaping tests passed. Hosted CI remains unverified until a push is authorized.

The constraint checkpoint was committed as `0889ab0`.

**Request-parity follow-up — 11 September 2026:** `core/sync/request.py` now defines the complete numerical input and its checksummed JSON representation. Solve Sync and Diagnose share preparation and forwarding. The three existing solving probes use that same request, including locks and slack. [Capture/replay](../tools/debug-sync/README.md) works on an arbitrary saved project without saving it, and numerical replay needs only Python/NumPy. The format includes full calibration and weights; it contains no independent truth and is separate from synthetic expectations.

The generated `verify_requests.py` check reproduced the old probe omission, then passed **15 request/result comparisons** across imperfect constraints with nonzero slack and a live lock, global locks, and automatic origin preparation. Input `.blend` checksums stayed unchanged. Capture and ordinary-Python replay also agreed on the request fingerprint. Case processes are isolated: attempting to reuse scene generation in one process exposed leftover RNA state. A second deliberate failure showed that the Blender oracle could accept an unrelated `ValueError` as an expected refusal; an explicit solver-rejection type now preserves diagnostics while unexpected errors fail. Numerical accuracy under imperfect constraints is still a separate question.

**Integration with camera roles:** the intervening `0319933` commit added Solve / Lock Pose / Fit Only roles and centralized their scene collection. The shared request retains that collector. Its solver-signature check caught the two new participation fields during stash-conflict resolution, so capture now preserves them in schema version 2; older version 1 snapshots explicitly retain their pre-role semantics. A fourth generated state adds Fit Only to the comparison, bringing it to **20 passing request/result comparisons**. Synthetic numerical cases now forward the same explicit role sets as generated Blender scenes, so recovered-camera stages are exercised consistently.

Merged-checkpoint validation: **255 unit tests, 9 skipped; passed**, Blender smoke passed, and ordinary-Python replay of the Fit Only snapshot passed. No source project was saved by capture or probes. The next question at that checkpoint was whether excluded observations could create or improve mirror geometry or suppress a weak-support warning; the role-boundary follow-up below answered it.

The shared-request checkpoint, including stash-conflict resolution, was committed as `569da33`.

**Role-boundary follow-up:** [the new role corpus](../tools/synthetic_sync/roles.py) reproduced two leaks in mirror reconstruction: one-sided Fit Only observations supplied missing mirror points/lines, and a Fit Only stroke reshaped a weak mirror line and suppressed its warning. Mirror seeding/enforcement and line-support diagnostics now use only views permitted to contribute 3D; pose fitting still retains Fit Only observations. Line refresh applies the same filter consistently to parallel/mirror constraint helpers. A Known 3D partner still supplies reflected geometry when appropriate.

Four regression tests and six generated role cases now pass. The mirror tests failed before the fix; the parallel-line control already preserved geometry and is not claimed as a reproduced defect. Both live Solve→Fit Only transitions passed through Blender and fresh-process reopening, including removal of stale point/line helpers. The full suite passed **259 tests, 9 skipped**. The larger 4 px stroke-bias experiment remains a recorded camera-accuracy flag, not a reason to relax limits or change the solver. See the [role experiment commands and limitations](../tools/synthetic_sync/README.md).

The role-boundary fix was committed as `b902124`.

**Origin-preparation follow-up:** [six Blender controls](../tools/synthetic_sync/preparation.py) now cover preset, missing and locked non-anchor origins in ordinary and overhead layouts. They preserve the same independent truth while checking the allowed private-center change, unchanged evidence/intrinsics/orientation, the applied camera, and replay from raw saved state. All six and their fresh-process replays passed, with maximum withheld RMS below 0.00007 px. This pilot supports the existing behavior; it does not justify an origin rewrite. Anchor-origin/world-frame changes, calibrated ground initialization and undo remain outside this check.

The origin checkpoint was committed as `78f2322`.

**Reduction follow-up:** [the bounded reducer](../tools/synthetic_sync/reduce.py) now removes unrelated free landmarks while preserving all cameras, ground/known references, lines, mirror features, roles and independent truth. Every accepted deletion must retain the same forbidden output on the old code, pass every other accuracy check there, and pass the full oracle on the fixed code. Each candidate runs in separate cold interpreters with source fingerprints, exact inputs, results and logs; the final candidate is replayed again. The two confirmed mirror ownership failures shrank from **90 to 21** and **84 to 15** point picks in six deletions each, about 22 seconds per reduction locally. Both reduced inputs also passed Blender creation/application and fresh-process reopening. Minimality applies only to permitted free-point deletions; this is not a general ambiguity proof or scene reducer.

The reducer checkpoint was committed as `34efe80`.

**Sparse-graph follow-up:** [five-camera chain/loop controls](../tools/synthetic_sync/graph-results.md) exposed another confirmed product defect. A chain fitted every point essentially exactly while its later cameras missed withheld object geometry by **446–593 px RMS**, and its later ground landmarks floated above the floor. Single-view ground positions were seeded only from the anchor; already-registered cameras did not supply their additional ground evidence. Registration and reconstruction now use shared-world ground rays from cameras allowed to contribute 3D, preserving existing metric references and ground-agreement rules. This also places a supported Fit Only middle camera while correctly leaving its downstream branch unregistered.

Both focused regressions failed before and passed after the change. All **20 exact arrangements** (chain, loop, broken link, Fit Only bridge and locked bridge × four seeds) passed the independent camera checks, with maximum withheld RMS below **0.003 px**. Stronger seed-zero contracts check reconstructed/missing points too. Blender create/apply/reopen and the live role transition passed, including stale-helper cleanup; input reversal/cache replay and Blender smoke passed. The full unit suite passed **268 tests, 9 skipped**. The separate 0.3 px noise sweep flagged **five of ten cases**, with maximum withheld RMS 3.37 px; those remain accuracy/sensitivity findings under provisional limits, not confirmed new bugs. No threshold was relaxed.

The sparse-graph checkpoint was committed as `76e79b1`.

**Recovery and shared-plane integration:** [the recovered-camera experiment](../tools/synthetic_sync/recovery-results.md) confirmed that a late 3D update could spoil a previously accurate camera when a recovered camera supplied contradictory elevated picks. A paired control skips only that update. The intervening `f117f4c` **Is in Plane** work already restored constraint forwarding and kept geometric springs outside robust pick downweighting; the earlier omission identified here is no longer outstanding. Nevertheless, that revision still displaced the healthy camera by **36.89 px withheld RMS**. A candidate update now preserves the full set of constraints and is accepted only within the established fit allowance for each previously solved camera. Otherwise Sync keeps the earlier geometry and says so. Soft Known 3D can still improve, and a positive control combines it with an exact plane. The contradictory camera retains its **2.432 px** accuracy flag; this fix does not relabel inconsistent evidence as accurate.

Validation: **294 unit tests, 9 skipped; passed**, followed by all three recovery tests including a newly added real background-cancellation callback. Blender smoke and the positive soft-Known/plane creation/application/reopen case passed. The reducer now protects plane membership, and all 20 generated request/result comparisons passed with nonzero plane slack included. The latest interrupted stash was integrated with the new plane code; the two older stashes were inspected and their work is already represented by the shared-request and reducer commits, so they remain unapplied.

The recovery checkpoint was committed as `5a4a34d`.

**Independent shared-plane follow-up:** [four plane families and paired controls](../tools/synthetic_sync/plane-results.md) check separate axis buckets, tilted Free planes, floor intersections and line geometry against known construction and withheld object projections. They support the existing behavior, including Fit Only stroke exclusion. One new deterministic defect appeared when a Free plane included hard floor points: initialization moved those points about **±0.00047 scene units** before they became fixed metric references. Camera checks alone passed. The initializer now preserves the ground seeds; an explicit floor-distance contract fails before and passes after the fix. Nonzero Ground Slack still permits refinement. This is a small constraint violation, not a claim of a large observed camera error.

All **10 exact/control and 10 noisy/control cases** passed, with maximum withheld RMS **0.145 and 1.469 px** respectively (the Fit Only pair uses its fixed 0.3 px stroke bias in both batches). The full suite passed **300 tests, 9 skipped**. Both floor and line cases passed Blender creation/application, live plane removal and reopening; the frozen floor regression also passed reversed evidence and cold/warm replay. CI includes these plane checks; hosted execution remains unverified without a push.

The hard-ground plane checkpoint was committed as `19373fe`.

**Single-view plane follow-up — 12 September 2026:** the [contribution experiment](../tools/synthetic_sync/plane-results.md#single-view-contribution-follow-up--12-september-2026) establishes a useful capability beyond preserving existing results. A supported X or tilted Free plane plus one permitted camera pick determines the missing point's depth, verified by an independent linear calculation. The previous solver omitted it. Sync now seeds these points for hard planes, requiring independently located supporting members and a forward, well-separated ray. Removing the plane or changing the only contributing view to Fit Only removes the unsupported point. Soft planes and single-view lines retain their prior behavior.

Twelve exact cases across two seeds and six 0.3 px noise cases pass the independent camera and point checks. Noisy point errors are **0.158% / 0.225%** of the object diagonal, with maximum camera RMS **1.469 px**. Blender removal and role-change sequences pass creation/application/reopening and clear the unsupported helper. Diagnose labels the depth source **Plane + one view**; low pick error does not independently establish depth. The full suite passes **304 tests, 9 skipped**; the source test and CI workflows retain these controls. This is a bounded capability, not a guarantee for wrong physical planes or poorly conditioned plane support.

The single-view plane checkpoint was committed as `3808b7b`.

**Scale wording follow-up:** a synthetic case with only an unknown axis plane produced “scale from 1 plane.” An independent transformation scales the entire reconstruction about the anchor camera by 0.5 or 2 while preserving every image projection and the shared-coordinate constraint, proving that absolute size remains undetermined. The result now lists **constraints** without claiming that those counts establish scale. The heuristic-only claim was also removed because counts do not identify the initialization route used. Geometry is unchanged. Both new scale tests and the existing plane/report checks pass (18 focused tests); the previous full suite passed 304 tests with 9 skips.

The scale wording checkpoint was committed as `bfd122d`.

**Mirrored-line plane follow-up:** the [paired plane experiment](../tools/synthetic_sync/plane-results.md#mirrored-line-contribution-follow-up--12-september-2026) confirmed a second boundary defect. Even with exact camera poses and an independently established hard plane, final mirrored lines missed their true direction by **21.32°** and left the plane. Independent stroke/plane intersection gave less than **0.32°** error. Repeating the previous plane projection alone still left about **9.5°** error. Fitting the line within the plane, then preserving that plane during mirror enforcement, reduces error to **0.075°** and keeps the reflection. The unlocked-camera control also passes. Removing only plane membership retains the same CAD evidence and restores the expected weak-support warning. The diagnostic now counts actual independent plane support, excluding lines themselves and plane-derived one-view points. Blender live removal and reopening pass, including an explicit physical-plane oracle; camera alignment alone would have missed the failure. Validation: **309 unit tests, 9 skipped; passed**, all four numerical controls, reversed-input/cache replay, and Blender smoke.

The mirrored-line plane fix was committed as `3aa0dcc`.

**Line-accuracy reduction follow-up:** the [existing reducer](../tools/synthetic_sync/README.md#reducing-a-confirmed-regression) now handles explicitly named line direction, offset and physical-plane failures as well as forbidden geometry. Every deletion must retain the same failed numerical checks on the same lines, pass all other old-code checks, and pass the full fixed-code oracle. Missing/nonfinite geometry, camera drift and weak-line exceptions cannot masquerade as the intended failure. The plane/mirror case shrank **88 → 19 point picks** in six deletions, about **4.5 seconds** locally, with all ground/CAD references, strokes, locks, roles, truth and thresholds unchanged. Its final cold replay and Blender live-removal/reopen checks pass. The smaller input is frozen alongside the original; general ambiguity or arbitrary accuracy reduction remains out of scope. A historical replay also caught current plane defaults being sent to code that predates planes; the worker now omits only inactive, unsupported plane defaults and records that adaptation. Active settings remain an error. Validation: 19 reducer/oracle tests and the subsequent 20 reducer/plane tests passed; both historical predicates passed cold replay. The previous full suite passed 309 tests with 9 skips.

The line reducer checkpoint was committed as `fd1890c`.

**Background input follow-up:** inspecting job ownership exposed a concrete input omission before the stale-result experiment. The blocking lens search forwarded Is in Plane groups and Plane Slack; the sidebar worker sent neither. The new [generated job check](../tools/synthetic_sync/README.md#background-job-input-parity) captures both real entry paths, compares every prepared numerical field, and exercises both Same Lens and per-match modes. It fails on the omitted plane fields before the fix. Both paths now use `LensRefinePrep.solver_kwargs()` and `run_lens_refine()`, removing the duplicated forwarding list. This is a confirmed routing defect; no claim is made yet about the size of a camera error caused by it. The check uses real RNA and the worker callback with synchronous execution and simulated window-manager plumbing, so it does not establish live thread safety. All four captured calls, the eight existing focal-refinement unit tests and Blender smoke pass.

The background input checkpoint was committed as `a58a164`.

**Diagnose ownership follow-up:** the prior stale-result hypothesis is now a deterministic reproduction through the actual operator worker and finish paths. After a plane membership, camera-role or pick edit, Diagnose published its old result and overwrote current landmark errors. All old numerical results still passed independent camera/geometry checks; the defect was result ownership. Diagnose now captures the prepared request hash and scene identity, then compares a read-only collection before applying diagnostics. Changed inputs, a missing anchor or another active scene discard the result. Unchanged and unrelated-object controls still apply. The seven generated controls pass, and apply is forbidden from running automatic origin/ground preparation. This tests deterministic interleavings with real RNA and numerical solving; it does not claim live UI scheduling, file-load cancellation or generic thread safety. The 20 existing product/probe/request comparisons still pass. The four lens-entry parity checks and Blender smoke also pass after the collector change.

The Diagnose ownership checkpoint was committed as `6c5712a`.

**Lens ownership follow-up:** a real synthetic Same Lens search reduces fitted error **10.74 → 0.00002 px** and restores withheld object alignment within **0.00027 px**. Replaying that valid result after controlled edits confirmed that the previous apply path overwrote current state after plane, role, pick, focal, search-range and origin changes. Apply now checks the lens job's complete numerical inputs, scene identity and camera targets before writing. Fourteen controls also cover VP edits, live camera/root changes, camera deletion and another scene; unchanged, unrelated-object and active-match controls still apply accurately. Rejections preserve calibration, actual camera data, root transforms and landmark diagnostics and must be the explicit `StaleSyncResult`, not an arbitrary exception. The experiment replays one real search result from identical prepared inputs, so it tests ownership without relying on timing. It is not a general thread-safety guarantee. Validation: all 14 lens ownership controls, seven Diagnose controls, four lens-entry captures, 17 focused numerical/request tests and Blender smoke pass. Hosted CI remains unverified without a push.

The lens ownership checkpoint was committed as `f6a7673`.

**Application-failure follow-up:** controlled errors confirmed that camera writes could stop halfway through, and that an internal Sync error was reported as an ordinary numerical refusal. Refine Lenses now reuses `PinSyncSnapshot` to restore stored and live cameras, root transforms, origins, diagnostics, landmark state, scene camera/render size and cached plates after an application error. The snapshot retains actual camera values: recomputing the old Blender lens from rounded RNA produced a small but measurable change. Cached images are retained during application so invalidation cannot delete the rollback source. Restoration bypasses camera diagnostics, allowing a persistent diagnostic error to be reported without breaking restoration itself. `SyncSolveRejected` remains a distinct intentional partial outcome that keeps refined lenses and its original numerical diagnostics.

Validation: six generated application controls pass, covering success, errors during camera diagnostics, before/after Sync writes, after plate rebuilding, and numerical refusal. Successful application still aligns withheld object points within **0.00027 px**; all four internal-error controls restore captured state exactly and propagate the injected error. All 14 prior lens-ownership controls, 21 pin/lens numerical tests and Blender smoke also pass. The cached plate is a generated ownership check, not a distortion-quality test. This is bounded recovery for the lens apply path, not a claim that arbitrary Blender deletion, rollback failure, undo or every operator is transactional.

The application-failure checkpoint was committed as `fd0633f`.

**File-load lifecycle follow-up:** the actual load handler reset Refine Lenses but left Diagnose marked running and its cancellation event unset. A generated `.blend` reload reproduced the blocked new job. A second controlled interleaving showed that an old lens operator's finish/cancel callback could clear a new job's running state or cancel its event. The old wait path also referenced an unimported `time` module. Background Sync operators now own their result boxes and cancellation events; load/unregister retires both jobs, and a retired operator cannot consume results, progress or cancellation from a new one. Cancelling a modal retires it without blocking on the numerical worker. This does not change the solver's cooperative cancellation checks.

The new [reload harness](../tools/synthetic_sync/verify_job_reload.py) passes **12 controls** across both operators: unchanged completion, load before/after completion, old cancel/timer callbacks, and cancellation before starting another job in the same scene. A new lens job still aligns withheld geometry within **0.00027 px**. The result is computed numerically once and replayed only from identical inputs; workers intentionally return the completed result even after cancellation to test ownership independently of timing. Real file-load handlers run, but modal window plumbing and report launching are substituted. The seven Diagnose and 14 lens edit controls, six application-failure controls, four lens-input captures and Blender smoke also pass. VP detection, Iterate Known 3D and native UI scheduling remain separate lifecycle gaps.

The file-load checkpoint was committed as `de057d2`.

**Parallel work checkpoint — integrated:** two agents worked in isolated worktrees from `fd0633f` while the main thread completed file-load lifecycle checks. The lens branch (`838fba5`) was reviewed and integrated as `2ab1165`; the constraint branch (`37c278c`) was integrated as `6d4ff2f`. Only adjacent changelog bullets conflicted; all were retained. The [development workflow](development.md#parallel-agent-work) records distinct ownership, focused checks, evidence handoff and one-at-a-time integration. No new skill or development-install change was needed. Both agents have stopped, their clean worktrees were removed, and local measurements were copied to `.local/parallel-evidence/lens-support/` and `.local/parallel-evidence/constraint-interactions/`. The two older stashes and pre-existing diagnostic worktrees remain untouched.

**Lens-scoring result:** [the frozen experiment](../tools/synthetic_sync/lens-results.md) reproduced headline error falling **19.989 → 12.904 px** while all-supported-pick error worsened **19.989 → 22.883 px** and a clean camera's withheld error reached **20.479 px**. No camera actually disappeared: a recovered camera's residual left the headline score. Lens selection now directly reprojects supported point picks, including recovered cameras, and retains a successful incumbent's camera/point/line support. It preserves the accurate lenses in that case. Separate controls still recover an ordinary 12.5% focal error and preserve useful numerical-refusal improvements. The latter controls use true locked poses to isolate scoring; an unlocked doubled-focal warm-start problem remains documented and unaddressed. Per-match/coupled controls test acceptance routing, not independent VP accuracy.

**Combined-constraint result:** [eight paired cases](../tools/synthetic_sync/constraint-interaction-results.md) reproduced mirrored lines ignoring a compatible Known 3D parallel direction. Fitting the reflected pair at an independently fixed direction now preserves both its hard plane and parallel constraint. Direction error falls **0.0746° → 0°** with a hard plane and **21.318° → 0°** with soft/absent plane support. Weak depth remains a warning; it is not relabeled accurate. Float32 request controls exposed a precision rejection during review, so compatibility permits one float32 epsilon while output direction/reflection/plane limits remain unchanged. Invalid or conflicting exact priors do not supply an exact direction.

Native Blender creation/application and fresh-process reopening both pass for the combined case, including strict post-checks of parallel and mirror relations; maximum withheld error is **0.0000442 px**. This exposed a harness assumption, not a product failure: a Known 3D line uses two linked endpoint objects, whereas the old geometry check expected every line to have a generated mesh. The shared Blender runner now verifies the actual representation in both cases. The new frozen case protects that path in CI alongside explicit relation post-checks.

**Integrated validation:** the full numerical suite passed **327 tests, 9 skipped**, in 230.5 seconds. The combined Blender checkout passed smoke, 14 lens-ownership controls, six application-failure controls, 12 file-load/job controls, and the new combined-constraint create/apply/reopen checks with strict relation post-checks. The earlier seven Diagnose controls and four entry-parity captures passed before numerical integration. Hosted CI remains unverified because nothing was pushed.

**Stopping checkpoint:** the user requested that the current tasks be finished, committed and then stopped. No new investigations were started after that request. The next session can choose among the documented gaps; there is no active parallel task to resume.


Keep the larger objective: each debugging session should improve a reusable input, independent check, reducer or lifecycle test. The recovered-camera guard still assesses point fits rather than the full constrained objective or line-only support. Image transforms, undo, biased references and calibrated uncertainty remain separate gaps; no general solver rewrite is justified by the evidence so far. Hosted CI remains unverified without a push.


## Original investigation and revised recommendations

The investigation observations below refer to `5d876f6` unless explicitly marked as updated. The opportunity sections retain the original strategic ranking, with implementation status reconciled through the checkpoints above. Use Current frontier to choose new work; the original effort estimates and pilot matrix are historical.

**1. What I inspected and what the evidence supports**

I read the repository instructions, README, user/development/sync documentation, both planning documents, representative paths through VP solving, calibration import, image transforms, pin/lens refinement, sync registration and adjustment, scene application, properties, background operators, overlays, reporting, fixtures, tests, debugging tools, and CI/release scripts. I reviewed recent history and representative fixes, including calibration copying, camera drift, undo, GPU memory, recovered cameras, and line anchoring.

I ran the existing unit runner: **213 tests in 63.942 seconds, OK with 9 skips**. The skips include OpenCV-dependent detection and the independent Brown–Conrady comparison, plus optional local calibration files. I also ran `scripts/validate_addon.py` in factory-startup Blender: **passed on installed Blender 5.1.0**. That is a source-checkout smoke test, not validation of all four distributable packages or an interactive GPU endurance test. These are historical baseline measurements; the newer pilot checkpoint is above.

Using the existing `probe_graph.py --no-solve`, I inspected four private projects read-only. They included dense point overlap, larger graphs, Known 3D points, lines, parallel/mirror constraints, pose locks and recovered cameras. These were stored-state observations, not new accuracy measurements or independent truth. Older saved flags can predate newer properties and were not treated as new defects. No user `.blend` was saved. File identities and local dumps remain outside this repository, as required by `AGENTS.md`.

**The current architecture has useful foundations.** A match owns a private camera and calibration through an Origin Empty; Sync maps those private frames into the anchor frame. Most numerical work is separated from Blender RNA. The staged solver already handles planar cases, graph bridges, partial registration, joint bundle adjustment (adjusting cameras and landmarks together), soft geometric constraints, resection against reconstructed points, and recovered-view refinement. The Blender adapter handles source/undistorted/display coordinates and applies the results. See [scene collection](../scene/__init__.py), [sync entry](../core/sync/solve.py), and [types](../core/sync/types.py).

Several recommendations that would sound reasonable in a generic review are already implemented: spatial/radial observation balancing, N-view triangulation with a reprojection polish, thawing 3D after camera adjustment, graph-aware candidate selection, pose caching, parallel pair evaluation, analytic blocks in the adjustment Jacobian, cancellation for several jobs, camera ownership modes, per-match errors, and portable HTML reports. Existing tests include Jacobian comparison, cheirality, pure-rotation refusal, collinearity, cache invalidation, outlier influence, and line degeneracy. These deserve preservation. See [BA tests](../tests/test_sync_ba.py), [pose tests](../tests/test_sync_pose.py), [ground tests](../tests/test_sync_ground.py), and [line tests](../tests/test_sync_lines.py).

**Baseline observation, addressed for source tests:** useful tests were not consistently positioned to prevent a release regression. At the baseline, the only checked-in workflow ran on version tags. It installs NumPy and runs unit tests, then validates/builds the extension; it never runs the Blender smoke test. The local release script also runs unit tests. Thus important integration coverage exists but is a manual step, and OpenCV behavior is not deliberately exercised by that unit-test environment. See [release workflow](../.github/workflows/release.yml), [release script](../scripts/release.sh), and [development checks](../docs/development.md).

**Baseline observation, fixed by the pilot runner:** targeted testing had an import-order dependency. Running only `test_sync_solve.py` in ordinary Python failed with `ModuleNotFoundError: bpy`, although full discovery passed. Full discovery first imports a test that constructs a substitute `match_perspective` package; several numerical suites depend on that side effect. This makes the agent's natural “run the relevant tests” loop unreliable. Centralize the test bootstrap or make the numerical package importable without Blender, then support selecting a test module through the existing runner. This is a narrow import-contract fix, not a reason to reorganize the entire repository. See [bootstrap side effect](../tests/test_apriltag_detect.py), [focused suite import](../tests/test_sync_solve.py), and [runner](../scripts/run-unittests.sh).

**Baseline observation, partially addressed by the new independent oracle:** important synthetic truth shares code with the implementation under test. The pair-coverage document explicitly asks for true versus stored cameras. That is the right intention. However, `build_views` obtains the true intrinsics through production `remap_intrinsics_to_size`, then normally reuses them as stored intrinsics. Its projection helper calls production `sync.project_private_point`. These tests can expose wrong pose seeds and interactions, but cannot independently prove the remapping or projection itself. Some remap tests assert the current policy rather than a physical image transformation. The separate OpenCV distortion comparison is a valuable exception. See [coverage ledger](../tests/edge_pairs.md), [pair fixtures](../tests/pair_fixtures.py), [shared projection](../tests/sync_fixtures.py), and [OpenCV comparison](../tests/test_core.py).

**Baseline observation, request omission now fixed:** the diagnostic tools could solve a different problem from the product. `probe_graph.py` and `probe_resected.py` forwarded mirrors but omitted `fixed_similarities`, workspace rotation/translation locks, and workspace ground/Known 3D slack. The shared-request follow-up above closes that omission in all three solving probes. The separate stored-pose comparison still needs care: `probe_graph.py` computes some residuals from session similarity metadata, whereas the current per-match overlay helper reads the live root transform. Neither comparison is inherently useless; label the state being compared. See [probe solver call](../tools/debug-sync/probe_graph.py), [resect probe](../tools/debug-sync/probe_resected.py), [production application](../scene/__init__.py), and [current-match scoring](../scene/__init__.py).

**Observed: calibration remapping encodes assumptions that image dimensions cannot establish.** An exact width/height swap returns swapped focal lengths and principal-point coordinates. A real clockwise raster rotation requires a coordinate reversal as well. With the fixture calibration, current code returns principal point `(2021, 1497)`; a clockwise 90° rotation gives `(1978, 1497)` under integer pixel-center coordinates. Counterclockwise gives `(2021, 1502)`. The precise one-pixel convention is secondary; simple transposition cannot represent both rotations. Likewise, an arbitrary crop needs its offset, not just its output size. This demonstrates a limitation for actual image transformations; it does not prove which transformation any supplied photograph underwent. See [remap policy](../core/ros_camera_info.py). The standard camera equations and distinction between intrinsics and distortion are documented by [OpenCV](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

**Supported hypothesis: several fixes are correcting the objective or state model after real use exposes the mismatch.** Recent history repeatedly changes which evidence can move cameras or landmarks, when a camera is accepted, how isolated picks retain influence, and which stored/live error is shown. The latest line fix touches reconstruction, mirrors, pose selection, recovered refinement, and acceptance, with extensive new regressions. That is evidence of coupled responsibilities, not evidence that the tests or fix are poor. The pixel residual is doing several jobs that should have distinct contracts. Relevant history: `52bd46b`, `c5aeab7`, `7d648af`, `4b834e0`, `575a44d`; [changelog](../CHANGELOG.md).

**Supported hypothesis: some apparent solver failures may be evidence/model disagreement.** You confirmed occasional mixed versions, while most sets are one rigid object. Apparent source groupings in private references do not prove a geometry conflict. Physical variants, CGI, retouching, or approximate CAD can make one precise rigid reconstruction impossible. Raising weights and adding slack can redistribute that conflict indefinitely. I would treat this as a specific diagnostic possibility, not assume it explains most failures. Since both metric and visual outcomes matter to you, a useful visual fit should not automatically receive a claim of metric accuracy.

**Baseline planning audit:** these comparisons describe the investigation revision; recheck any affected code before using them to schedule work.

| Planning claim | Assessment at the investigation baseline |
| --- | --- |
| Items 1–5 are implemented | The main capabilities exist. Do not schedule them again as missing features. |
| Same Lens means one shared focal | Current code applies one multiplicative correction to each starting focal; existing differences remain. That is useful, but does not establish one physical lens group. |
| Start reconstruction at the strongest pair and compose into the anchor | The current function scores up to four promising edges, then chooses a camera to solve against the anchor. It guides an anchor-based start; it does not reconstruct the winning non-anchor pair as an independent seed and compose that reconstruction. |
| Show which pair seeded each camera | The HTML report shows an available overlap route, chosen by shared-point count. `SyncSolveResult` does not preserve the actual winning registration route and branch. |
| “Diagnose holdout” scores a subset after fitting all points | That is a useful stratified residual report, but it is not held-out validation. Those observations have already influenced the fit. |
| Ground slack is a bounded tolerance | It is a soft spring and can be exceeded; the current solver reports excesses. Do not write tests that assume every result must stay within ±slack. |
| Pose graph is automatically next | It remains a plausible experiment, but the evidence above justifies changing the order. A better seed cannot repair a wrong plate transform or stale scene application. |

Sources: [plan](../docs/sync-accuracy-plan.md), [shared focal search](../core/lens_refine.py), [seed selection](../core/sync/pose.py), [report routes](../ui/sync_report.py), [result type](../core/sync/types.py).

**2. Ranked opportunities**

The table preserves the original strategic ranking and rough effort estimates, not remaining work estimates or delivery commitments. The pilot in part 3 has already led to the implementation history above. The status paragraphs below distinguish completed work from remaining proposals; Current frontier suggests the next bounded choices.

| Rank | Opportunity | Expected benefit | Initial effort | Confidence and success measure |
| --- | --- | --- | --- | --- |
| 1 | Complete solve snapshots, replayable cases, independent checks, automatic gates | Finds failures earlier and makes every future agent session start from the same evidence | 3–5 days for a narrow pilot; several further days to generalize | High confidence in workflow benefit; measure exact replay, known-fault detection, failure-to-reproduction time, runtime and false alarms |
| 2 | Explicit calibration/image transforms and transactional scene state | Prevents whole classes of copied-K, plate, stale-result and camera-ownership bugs | 1–2 weeks incrementally after the pilot | High confidence in the boundary problems; migration design is the largest uncertainty |
| 3 | Separate fit, stability, completeness and constraint agreement | Reduces confidently wrong results and directs useful intervention | 3–5 days for diagnostics first | High value, medium uncertainty in reliable thresholds; measure wrong-but-accepted results versus unnecessary warnings |
| 4 | Collect better evidence and retain its origin | Reduces manual tuning and prevents circular “validation” | 2–4 days for provenance/quality hints; more for tag geometry | Benefit depends on capture versus archival-reference workload; measure useful picks and time to an acceptable match |
| 5 | Test bounded solver simplifications against the corpus | Can remove recurring heuristic interactions and improve difficult graphs | Separate 3–5 day experiments; production adoption longer | Medium/low until measured; require geometry gains at equal evidence coverage and acceptable runtime |

**Rank 1: make the same case executable through the numerical and Blender paths.**

**Status:** the independent synthetic oracle, complete arbitrary-scene capture/replay, probe migration, camera-role transitions, non-anchor origin preparation, selected job lifecycle sequences and bounded forbidden-geometry/line-accuracy reduction are implemented. Nonzero-slack collection/replay and selected soft-constraint cases are covered; broad accuracy under biased references is not. Image transforms, broader operation sequences and general reduction/search remain proposals.

**Implemented foundation:** `core/sync/request.py` defines the complete versioned numerical request, including calibration, observations, roles, ground/Known 3D/line/mirror/plane constraints, locks, slack and initial/fixed transforms. Capture and synthetic records preserve actual numeric inputs and runtime metadata. Keep raw-scene operations and image-mapping provenance distinct from this prepared numerical input; neither is fully represented by a solver request. A seed alone becomes insufficient when the generator changes.

The sidebar, diagnostics and solving probes now consume that common request. `tools/sync_snapshot.py` records preparation notes separately from checksummed numerical evidence. `prepare_diagnose_sync` still performs automatic ground/origin initialization in memory; capture leaves the source file unchanged but is not a no-mutation scene collector. `collect_sync_request` supplies the read-only collection used for stale-result checks. Preserve this distinction when extending [preparation and collection](../scene/__init__.py).

Continue extending three economical layers, using the existing numerical and Blender runners:

- **Fast numerical cases:** independently generated cameras and observations; true K separate from stored K; varied baseline, depth, pose seeds, visibility, noise, incorrect correspondences and conflicting constraints. Target existing decision boundaries such as five shared points, the 40-landmark freeze/thaw transition, nearly parallel rays, and protected weights.
- **Blender operation cases:** reuse the smoke-test machinery for import/copy, bind/replace, source/undistorted display, principal-point edits, matched/adjusted mode, rename, delete, undo/redo, temporary save/reopen, and cancellation. Check invariants after each operation and actual evaluated camera projection. Existing smoke coverage already includes many fixed sequences; generalize those rather than replacing them.
- **A small interactive performance check:** repeat orbit, draw/edit, match switching and sidebar visibility in a real viewport; measure frame time and GPU memory growth after warm-up. Headless tests cannot cover the Metal buffer failure. Reuse [debug-memory](../tools/debug-memory/README.md); keep platform endurance runs periodic rather than making every unit run expensive.

The deterministic numerical subset, Blender smoke and selected state/constraint sequences are configured for pushes/PRs in `.github/workflows/tests.yml`; hosted execution is still unverified. An explicit OpenCV-enabled environment and actual packaged-extension enablement remain future validation work outside the Sync-first pilot. Test dependencies belong outside Blender's installed Python. Source imports and `extension validate` do not prove wheel loading. Broader seeded exploration can run nightly or on demand with a wall-clock budget. Publish skipped-capability counts so a green result has a clear meaning.

Automated exploration should prioritize disagreement: low training error but large ground-truth pose error; sensitivity to tiny pick changes; different results after harmless renaming; large current-viewport error despite low stored error; or improvement obtained by losing cameras. Retain diverse failures, not thousands of nearby copies. A reducer can remove observations, cameras or operations while preserving the failure. It must also preserve the case's solvability class; deleting the last depth cue is not a legitimate simplification of a well-constrained failure.

Hypothesis is an optional development dependency for generating and shortening operation sequences. It can compare actions against a small independent state model, but the first useful experiment does not require it. Its [stateful testing documentation](https://hypothesis.readthedocs.io/en/latest/stateful.html) describes the action/invariant mechanism. Keep minimized failures as explicit repository fixtures, not only in a tool's private example database.

This still misses real image formation, unusual Blender contexts, and human misunderstanding. Private projects are optional integration evidence; they must not become “ground truth” merely because their saved cameras once looked acceptable. Public regressions should contain generic synthetic geometry, consistent with AGENTS.md.

**Rank 2: represent what happened to an image, and make state changes explicit.**

**Status:** Diagnose/lens stale-input checks, shared lens forwarding, bounded lens rollback and file-load/cancel job retirement are implemented and reproduced through generated Blender controls. General undo/native scheduling, other operators, explicit image transforms and calibration migration remain open.

Preserve original calibration and its origin. Represent resize, crop, raster rotation and reflection as explicit transformations from calibrated pixels to the current plate. If dimensions alone cannot distinguish operations, mark the mapping as assumed and offer a simple preview/choice. Keep the canonical distortion model in its own coordinate system and compose the image mapping around it; this avoids scattered rules for swapping tangential coefficients. Lens groups should be explicit if shared physical calibration is intended. This builds on existing source-pixel storage and source/display mapping functions. See [image mapping](../scene/__init__.py) and [calibration copy](../scene/__init__.py).

Move legacy calibration repair to a visible, versioned preparation step. `solve_landmark_sync` currently modifies stretched input intrinsics in place despite a “frozen” input contract. Retain compatibility for old files, but record the repair and preserve the original values; a solver should not silently redefine its evidence. See [input repair](../core/sync/solve.py).

Keep Matched and Adjusted Camera modes. Extend that ownership rule to a complete validity check: which evidence revision produced the camera, helper positions, errors and report? Show “needs solve” when those inputs change. Preserve the last accepted result separately from a trial. Use stable match IDs for replay/jobs, with names only as labels, as landmarks already do.

The original stale-result hypothesis was reproduced and fixed: Diagnose validates the captured request and scene; Refine Lenses validates complete numerical inputs, scene and camera targets. Lens application reuses `PinSyncSnapshot` to restore prior state after internal application errors. Both background jobs retire old callbacks across load/cancel. Preserve these contracts when extending prepare → solve → validate → apply. Undo/native scheduling and transactional coverage beyond the tested lens path still need evidence; the existing controls do not prove them. See [background Diagnose](../ui/operators.py), [lens apply and snapshots](../scene/__init__.py).

There is also a worthwhile isolated-worker experiment. Long-lived Python threads overlap Blender activity today. The code correctly copies solver inputs, but Blender's own documentation warns that Python threads can remain unsafe even beyond direct `bpy` calls and recommends process isolation. Test a child worker using the serialized request, cancellation, and main-thread application; compare startup cost and responsiveness before adopting it. See the [Blender 5.1 threading documentation](https://docs.blender.org/api/5.1/info_gotchas_threading.html). Packaging across four platforms is the main cost. This is not an explanation of the already-fixed GPU leak.

Success means fresh versus warm runs agree, cancel/undo/load cannot apply an old job, camera/plate projections agree, and old files migrate predictably. This work will not make an underconstrained scene solvable.

**Rank 3: report what is trustworthy, and keep comparisons honest.**

**Status:** weak-line support warnings, plane-derived depth labels, corrected scale wording and lens scoring/support retention are implemented. General sensitivity reporting, registration provenance, broader acceptance audits and evidence-aware picking advice remain proposals.

Give the user separate answers: how much evidence participated; how well points and lines fit; how strongly geometric priors were violated; how sensitive the reconstruction is to reasonable input changes; and whether the current viewport still matches that result. Keep the existing HTML report and per-match overlay errors. Add evidence to them instead of introducing another dashboard.

The lens-scoring audit is complete. A synthetic case reproduced a lower headline RMSE caused by recovered-camera residuals leaving the score, while withheld geometry worsened. Cameras themselves did not disappear. Lens refinement now directly scores supported point projections and preserves a successful incumbent's camera/point/line support; see [lens results](../tools/synthetic_sync/lens-results.md). Sync's headline and acceptance were not changed by that fix. Iterate Known 3D comparison and broader constraint/line-aware acceptance remain audit targets. Keep point acceptance and line fit separate when needed, but do not label their scalar as total reconstruction quality. See [final sync scoring](../core/sync/solve.py), [lens cost](../core/lens_refine.py), and [iteration acceptance](../scene/__init__.py).

For real data, begin with interpretable diagnostics: triangulation angles, depth spread, image coverage, bridge fragility, center/edge residuals, signed residual patterns, and actual registration provenance. Then add small perturbation trials in Diagnose: move picks within plausible uncertainty and report how much camera position, focal length and useful model geometry move. Describe this as sensitivity, not a calibrated probability of correctness. A local covariance estimate cannot detect all alternative solutions and must account for unconstrained freedoms; [Ceres' covariance guidance](https://ceres-solver.readthedocs.io/latest/nnls_covariance.html) explains the scale/coordinate ambiguity in reconstruction.

True held-out validation requires excluding the check observation from every fitting stage, including calibration refinement and triangulation. A landmark observed in three or more views can be reconstructed from training views and projected into a withheld view, provided that camera is constrained by other training evidence. Holding out an entire free landmark with no independent 3D location gives nothing to project. Preserve depth/coverage when selecting training data, and reserve a final set that is not repeatedly used to choose parameters. On synthetic scenes, independent known 3D test vertices are simpler and stronger.

Replace categorical conclusions like “one huge residual means the pick is mismatched” with ranked explanations. A wrong pick, wrong calibration, erroneous CAD or an alternate pose can generate similar residuals. A good recommendation would say “depth is unstable; one elevated landmark in these two views would distinguish the alternatives.” Validate the advice by simulating that added measurement and checking that it actually reduces ambiguity.

Measure false reassurance and false alarms separately. Do not aim for maximum warning count. This will still miss shared systematic errors and disagreements outside the observed regions.

**Rank 4: retain measurement provenance and help acquire informative evidence.**

Record whether a pick was manually observed, snapped, detected, or projected from the current camera. `Landmarks from Selected` produces camera-projected picks; the guide correctly warns these are not independent evidence for Known 3D polish. RNA observations currently store coordinates/confidence but not that origin. Retaining provenance would let diagnostics avoid presenting self-generated agreement as validation and ask the user to verify a projected pick on the image. Do not silently discard established workflows. See [observation schema](../properties/__init__.py), [auto-projection](../scene/__init__.py), and [user guide](../docs/user-guide.md).

For self-captured photographs, offer guidance before another shoot: broader baseline, one off-plane point, overlap between weakly connected groups, coverage around the modeled region, and a measured length when scale matters. For archival images, the useful advice may instead be “calibration mapping is assumed” or “these sources may not share one rigid geometry.” Those are different workflows and should not force the same automatic lens grouping.

**Deferred by the Sync-first scope:** a possible later feature experiment is optional measured geometry from tags or a simple distance constraint. Detection already obtains four corners, but assignment primarily keeps the center; printing tools already exist. Known marker dimensions could contribute metric information, and corners can supply more geometry. Do not treat four corners from one small blurry tag as four independent, equally reliable observations. Retain their common tag identity, reject poor localization, and keep the current center workflow for small tags. Compare ground-truth scale/pose and manual picking time on a small controlled capture before expanding the model. See [tag detection/assignment](../detect/apriltags.py) and [printing tools](../tools/print-apriltags/README.md).

Avoid adding more tuning controls until the evidence says which user decision they represent. Pick confidence concerns measurement reliability; modeling importance concerns what the user wants fitted. Their meanings should remain distinct even if both eventually influence optimization. Neither should turn unobserved detail into certainty.

**Rank 5: use the corpus to choose a specific structural solver improvement.**

Three experiments look plausible; only promote the one matching the measured failure distribution.

- **A small set of alternative graph seeds:** retain a few discrete pose/plane alternatives and use a third view or cycle consistency to distinguish them before joint adjustment. Current code already considers graph-aware candidates, so reuse that machinery. Start with three-/four-camera ambiguous fixtures and a fixed candidate budget. A full pose graph is justified only if seed failures dominate. Cycles cannot validate absolute scale or repair shared wrong calibration.
- **Less redundant pose/line representation:** the renderer ultimately needs camera rotation and center. Investigate solving those directly in a small numerical path, with the existing adapter deriving rig transforms. Per-camera similarity scale and translation can express redundant projected-camera states; the collapse repair and conditional scale unlocking show that this representation deserves scrutiny. Keep the meaningful global scale freedom explicit, and preserve locked roots. For lines, independently test fitting direction as well as position: current BA moves midpoints while holding seeded directions fixed, followed by enforcement/reconstruction passes. Use an infinite line with a canonical position representation; estimate visible helper length separately. These experiments could remove recurrent state and line-depth interactions, but are not yet a recommendation to replace the solver. See [similarity model](../core/sync/types.py), [BA contract](../core/sync/ba.py), and [collapse regression](../tests/test_sync_pose.py).
- **A measured numerical-performance improvement:** existing BA already has analytic blocks, but residual-only evaluations currently also build the Jacobian, and the solve uses dense normal equations. First measure the unnecessary work and stage costs on representative snapshots. Only then compare a residual-only path, block elimination, or an independent optimizer. A new native solver dependency has real four-platform maintenance costs. See [residual evaluation](../core/sync/ba.py) and [iteration](../core/sync/ba.py).

Evaluate geometry, accepted evidence, stability and runtime together. An alternative optimizer is another check, not an oracle: it can share the same wrong camera model.

**3. Original pilot proposal — historical scope and acceptance rationale**

The original proposal was a **3–5 day pilot** around calibration-to-Blender agreement and complete solve replay. Following the user's clarification, implementation began with the narrower Sync-only pilot recorded above and has progressed well beyond that first checkpoint. Everything from Scope through Stop or change course below preserves the original proposal, not a new task list. Shared requests, replay, selected operations and bounded reduction are already implemented; the image/distortion matrix remains deferred. Current acceptance limits are in `tools/synthetic_sync/scenarios.py` and explained in its README; the original suggested limits below were not adopted unchanged.

**Scope.** Make a shared complete request from the existing preparation/collection boundary, adapt one existing probe to consume it, and establish production/probe parity for locks and slack. Give the unit runner a reliable focused-test entry. Add an independent projection/image-transformation reference and a small operation-case runner built from the existing Blender smoke machinery.

Use 24 named combinations: three scene layouts (healthy depth/baseline, overhead with depth support, deliberately weak depth/baseline), four image paths (unchanged, uniform resize, known 90° rotation, known crop), and two lens conditions (pinhole and modest Brown–Conrady distortion). The initial geometry can be a generic asymmetric arrangement of points and edges. Keep noisy picks as a small second sweep after the exact cases; do not render photorealistic scenes yet. Add a handful of fixed transition cases: switch/back, source/undistorted toggle, principal-point edit, undo/redo, and matched/adjusted ownership. Temporary synthetic `.blend` round-trips are allowed in the future harness; user files remain read-only.

A known rotation/crop case should state its transform explicitly. Where today's UI cannot represent that information, classify the result as a demonstrated contract gap. Do not invent a hidden assumption to make the test pass.

**Correctness checks.** Construct truth without importing production projection/remap helpers. Cross-check that independent pinhole reference against a natively configured Blender camera and, where available, OpenCV projection/distortion. Apply raster transforms to points/images directly, then compare source picks, display positions and evaluated camera projection. Keep pixel-center conventions explicit. Use truth known from construction for pose/3D checks; fix the anchor and metric references, or align only the genuinely unobservable global similarity. Per-camera alignment would conceal mistakes.

Classify every case before scoring:

| Case class | Acceptable outcome |
| --- | --- |
| Well constrained, within supported image/model assumptions | Correct pose/geometry and source-to-viewport agreement |
| Solvable only up to scale or another known freedom | Correct observable quantities and explicit unmeasured freedom |
| Ambiguous or weak | Warning/alternative/refusal appropriate to the evidence; no claim of precise unique depth |
| Invalid or contradictory | Useful explanation and preservation of the previous valid scene |

An overhead view is not automatically ambiguous; its supporting geometry matters. Wrong calibration is not automatically an invalid input either—it may be recoverable if the data constrain calibration.

**Artifacts.** A versioned case schema; deterministic cases; complete request/result records; a Blender adapter runner; a minimal failure reducer; and a report containing geometry error, fit by evidence type, participating cameras/observations, timings and state mismatches. The fast fixed cases should be runnable in CI. A future command could expose `capture`, `replay`, `compare`, and `minimize`; its exact name is unimportant. Keep images optional for numerical replay and local originals outside the public regression corpus.

**Success criteria.**

- The UI preparation and diagnostic replay produce equivalent requests, including all locks and slack. Replaying a frozen request yields the same participants and numerically equivalent result under a recorded runtime; warm-cache behavior is also checked.
- The independent oracle detects at least two deliberately introduced fault classes, such as a wrong principal point after raster rotation and a dropped pose lock. It must not rely on the existing implementation's expected answer.
- Establish explicit tolerances before evaluating improvements: proposed initial noise-free limits are 0.1 px for pure projection agreement, 0.5° rotation and 1% of scene extent for center/3D recovery on the well-conditioned metric cases. Raster/detection cases get a separate localization budget. These are pilot acceptance targets, not claims about achievable real-world accuracy.
- A failing case replays from one command without the user's file, and the reducer retains its constraint class. Target a replay under a minute and the fixed pilot suite under five minutes on the recorded machine.
- No false failure is created by legitimate scale freedom, excluded evidence, or a deliberate ownership transition. Repeated runs must not produce unexplained category changes.
- The report can name which boundary failed: image mapping, state collection, numerical solve, scene application, or display. That attribution is the experiment's main product.

**Stop or change course if** most failures are generator mistakes or mislabeled ambiguity; if “independent” truth still requires production remap code; if operation replay is flaky under controlled startup; or if the harness cannot reproduce real workflow inputs without substantial bespoke scene handling. Fix the oracle/schema before expanding. If the pilot only rediscovers isolated old bugs and does not expose boundary gaps, keep its useful regressions and shift effort to state/provenance. If most accurately replayed cases show coherent input-model conflict, invest in calibration/evidence guidance rather than a more elaborate pose graph. If numerical seed ambiguity dominates instead, move the bounded multi-candidate experiment forward.

**How AI development would compound from this.** A future bug session should deliver a complete captured request or operation trace, the independent expected behavior, a minimized generic regression, a fix at the first failing boundary, and a before/after corpus comparison. The report should distinguish reproduced facts from hypotheses and record why any real-data claim lacks independent truth. A human should mainly decide whether the modeled promise matches the work they need to do.

Executable tooling should own input collection, environment capture, replay, minimization, metrics, selected test execution and pass/fail behavior. Tests should own invariants and regressions. A short architecture document should own coordinate conventions, state ownership and migration contracts. AGENTS.md should point to those mechanisms and retain policy, rather than duplicate evolving algorithms across several instruction files. The existing stage map remains useful; consolidate duplicated maps only where divergence is demonstrated.

**Skill decision, updated:** the original prerequisite of successful reuse in two debugging sessions has been exceeded. That does not itself justify a skill: checked-in harness commands and the parallel workflow currently carry the reusable procedure. Reconsider a thin routing skill only if sessions repeatedly fail to find or correctly use those tools. Its trigger would be a Perspective Match camera/sync/plate discrepancy; inputs would be a case file or read-only `.blend` plus expected behavior; workflow would be capture → classify → replay both layers → minimize → verify; outputs would be a diagnosis, regression and evidence-backed handoff. Verification belongs in the executable harness and relevant Blender checks. Maintenance should cover command routing, without duplicating solver thresholds or algorithms.

**Ideas I would not pursue now.** A repository-wide rewrite or file split without a contract change; photorealistic random rendering before a trustworthy geometric oracle; default hard outlier deletion that can hide wrong calibration; automatically raising peripheral weights whenever the fit looks bad; optimizing focal/principal point/distortion/geometry simultaneously without observability checks; exhaustive spanning-tree searches; a dependency-heavy solver replacement without an equal-input benchmark; using current `.blend` results as golden geometry; or a general autonomous agent swarm searching for bugs without a defined pass/fail oracle. Each could consume maintenance time while preserving the underlying uncertainty.

**4. Questions that would materially change the ranking**

1. You confirmed both metric accuracy and visual alignment matter, depending on the project. What error is noticeable or costly for each, at the dimensions and viewing distances you use? This would set meaningful acceptance tolerances.
2. You confirmed mixed versions are occasional and most sets depict one rigid object. Are the worst unresolved cases concentrated in those mixed sets, or also common in controlled single-object captures? That would distinguish evidence conflict from reconstruction failure.
3. What fraction of recent debugging time is spent on wrong geometry, stale Blender behavior, waiting for solves, or figuring out which tool to use? Even five rough incident summaries would help weight the corpus.
4. When you take photographs yourself, can you control cropping/orientation and include a measured length or a small non-coplanar marker arrangement? That determines the value of acquisition guidance and metric tag support.
5. **Decision made:** use synthetic scenes as the primary development corpus; private projects are optional evidence when a missing behavior is discovered. The remaining question is what unattended runtime is acceptable as exploration grows.

The implemented laboratory is useful across these answers. They primarily determine which remaining product/solver experiment to choose next.
