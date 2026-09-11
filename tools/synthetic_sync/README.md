# Synthetic Sync laboratory

Generate Sync evidence from a known object and cameras, solve it, and check how
the recovered cameras project **object geometry never supplied as picks**. No
private projects, AprilTags, VP detection, OpenCV, or images are needed for the
numerical run. Blender can create packed reference renders for inspection.

This supplements `tests/sync_fixtures.py`, `tests/pair_fixtures.py` and
`tests/edge_pairs.md`. Those tests cover specific solver branches and calibration
pairs; this harness checks camera accuracy against an independent reference and
compares the numerical request with evidence collected from real Blender state.

## Run and inspect

From the repository root, using Python with NumPy:

```sh
python3 tools/synthetic_sync/run.py --family all --out /tmp/pm-synthetic
python3 tools/synthetic_sync/run.py --family all --permute --warm --out /tmp/pm-invariance
python3 tools/synthetic_sync/run.py --family all --count 3 --noise-px 0.3 --out /tmp/pm-noisy
```

Open `report.html`. Green rings are true projections of withheld vertices and
surface samples; red dots are their projections through recovered cameras.
Each run writes exact JSON cases and result/assessment JSON with request hash,
Git revision, dirty-tree flag, Python/NumPy/platform versions and elapsed time.
Keep the case **and** the code revision when preserving a failure; the dirty-tree
flag alone cannot reconstruct uncommitted code. Seeds make exploration repeatable;
saved JSON is the regression artifact if the generator later changes.

See [initial pilot observations](pilot-results.md) for the first noisy sweep and
the distinction between a flagged experiment and a confirmed solver defect.

Exit status is nonzero if any accuracy contract fails or the solver raises.
Exceptions are recorded as failures, never accepted as a valid refusal. Reports
include solver messages. A missing required camera fails even when the solver's
remaining cameras fit well. Reversing inputs reruns the accuracy contract;
`--warm` also requires successive cached solves' withheld projections to agree
within 0.01 px. It does not claim identical internal optimization trajectories.

Replay a saved case without regenerating it:

```sh
python3 tools/synthetic_sync/run.py --case /tmp/pm-noisy/overhead-0.json --out /tmp/pm-replay
```

## Real Blender path

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup --disable-autoexec -b --python-exit-code 1 \
  --python tools/synthetic_sync/blender_case.py -- \
  --case /tmp/pm-synthetic/partial_symmetry-0.json \
  --out /tmp/pm-blender --roundtrip --render
```

Use a new output directory. The command creates `input.blend`, `solved.blend`,
metrics, the complete collected numerical `request.json`, a portable oracle
report, and the extension's own `product-report.html`. `--roundtrip` opens **input.blend** in another
factory-startup Blender process and repeats the solve in `reopened/`.
`--render` adds PNG reference images and packs them in the generated files;
without it, packed blank images supply the actual image datablocks required by
the extension. The reference mesh and truth cameras are in `Synthetic truth`.
Enable the extension before inspecting the generated file in the UI. Select a
match to compare its camera against the reference object and packed image.

The Blender runner:

1. Constructs truth cameras directly using Blender's camera data API. Checks
   native Blender projections against the independent NumPy reference (0.002 px).
2. Creates actual match hierarchies, stored calibrations, image datablocks,
   landmark picks, Known 3D Empties, mirror links, parallel links and pose locks.
3. Compares **every supplied solver field** with `prepare_diagnose_sync`'s
   collected request, allowing only float32 storage rounding and ordering.
4. Calls `scene.solve_and_apply_sync`, then switches matches and checks evaluated
   Blender cameras against the recovered projections (0.01 px).
5. Saves only generated files and optionally replays saved input in a fresh process.

`--load /path/to/input.blend` can replay a generated file into a *different*
output directory. It verifies a request fingerprint. Never pass a user project;
this is a generator/replayer, not a general scene repair tool.

## Current corpus and its contracts

| Family | Evidence/state | Required outcome |
| --- | --- | --- |
| `ground` | Visible surface picks plus floor picks | All three cameras in the anchor frame |
| `known_3d` | Known nonplanar points, no ground | All three cameras in the anchor frame |
| `free_scale` | Free points, no ground or metric pins | Cameras correct up to one shared similarity |
| `mixed_lines` | Ground/points, free and Known 3D vertical lines, world-axis parallel links; unequal stroke extents | All cameras accurate |
| `partial_symmetry` | Selected mirrored point pairs on a box with an asymmetric raised detail | All cameras accurate without imposing symmetry on the whole object |
| `locked_bridge` | Fixed live pose for the middle view; fewer picks in the anchor | Lock preserved and all cameras accurate |
| `overhead` | Third view almost vertically above the object | All cameras accurate |
| `disconnected` | Fourth camera with no observations | Connected three accurate; fourth excluded |
| `collinear` | Only collinear free points in two views | Refusal with a message |

Each camera has its own known focal length and off-center principal point. Except
for the anchor and explicit pose lock, stored private poses are deliberately
unrelated to truth. Frustum and mesh ray intersections determine visibility;
the generator does not give a camera picks on hidden surfaces. Noise is independent
Gaussian pixel error in each coordinate, not a realistic model of every picking
mistake. Seeds currently jitter camera positions and focal lengths around these
nine arrangements; this is a bounded pilot, not broad random scene search.

The default exact-data contract is withheld RMS ≤1 px per required camera,
rotation error ≤1 degree and center error ≤2% of object diagonal. The noisy
contract uses withheld RMS ≤max(1, 6×noise) px, with the same pose limits.
These are **provisional experiment limits**, not established product guarantees
or statistical confidence intervals. Maximum pixel error is reported separately.
No threshold is changed automatically to make a run pass.

For free-scale cases, one proper global rotation/translation/scale is estimated
using reconstructed **training landmarks only**. That same transform applies to
every camera; no camera gets its own alignment and withheld points never help
fit it. Other families use the anchor frame directly. This allows legitimate
scale ambiguity while exposing inconsistent camera geometry.

The oracle in `geometry.py` does not import Perspective Match. Its pinhole math
has analytical tests and a native Blender cross-check. A deliberate wrong-camera
test produces zero error on planar fitted picks but fails on withheld 3D points.
This guards against making fitted RMSE the answer by accident.

## What this still misses

- Unknown/wrong intrinsics, distorted/cropped/resized images and undistortion
  state; all current lenses are ideal pinhole cameras with square pixels.
- Wrong landmark identities, gross outliers, soft constraint slack and free-line
  initialization. The separate constraint-contribution cases below cover Known
  3D line-only pose and one-sided mirror points/lines, but do not exhaust sparse
  graphs or interactions among constraints.
- Mouse picking, drawing, undo/redo, deletion/recreation, hot reload and long edit
  sequences. Constraint removal after a solve is covered. The Blender path calls the shared scene service and actual RNA;
  it does not simulate sidebar clicks or the async operator lifecycle.
- Large scenes and performance scaling. Timings aid diagnosis but are not a
  benchmark gate across machines.
- A general ambiguity classifier: the pilot labels one known degenerate case.
  A useful failure message is required, but its explanation is not yet evaluated.
- Photographic/nonrigid/model mismatch and aesthetically preferred compromises.
  Real projects remain valuable when they reveal evidence this model omits.

The most useful next step is to inspect failures before expanding the generator.
Distinguish a harness mistake, an unfair contract, noise sensitivity and a solver
defect. Preserve the exact case, reduce its cameras/picks/constraints while
retaining the same independently measured failure, and add a focused regression
when the expected behavior is established. The bounded named-geometry reducer
below is implemented; general reduction/search remains future work. Add new scene actions or evidence types to the shared
request format and Blender collection comparison together; do not add truth as
a solver shortcut.

## Maintenance and CI

### Product/probe request parity

`verify_requests.py` runs inside factory-startup Blender and compares Solve Sync,
Diagnose and all three existing solving probes against restored `SyncSolveRequest`
snapshots. It covers nonzero slack, imperfect metric references, a live pose lock,
global locks, Fit Only participation, confidence versus outlier protection, and automatic origin setup.
Each state runs in a fresh process. This validates collection and replay, not a
general accuracy contract for biased references. A deliberately injected
`ValueError` must fail the harness rather than masquerade as an expected refusal.

See [capture/replay commands and limitations](../debug-sync/README.md). The
production snapshot format is distinct from the synthetic case format: it can
represent real collected calibration and weights, but has no truth or accuracy
expectation. Generated Blender runs now retain both formats.

Synthetic cameras default to the UI's **Solve** role. Both numerical and Blender
runners forward explicit `location_match_ids` and `readonly_match_ids`; optional
case fields can specify Fit Only participation. This matters because recovered
cameras allowed to move 3D have an additional thaw stage. An empty membership
list is distinct from an omitted field. The Blender runner verifies the collected
sets and rejects combinations that its three UI roles cannot represent.

### Constraint contribution checks

```sh
python3 tools/synthetic_sync/constraints.py --out /tmp/pm-constraints
```

This adds three families with paired removal controls, separate from `run.py`'s
nine base families:

- **Known 3D lines:** a second camera has only visible strokes, with no point
  picks or observations in other cameras. Known line geometry determines its
  pose. Removing Known 3D leaves unconstrained single-view lines and must refuse.
- **Mirror points:** each member of a pair has only one observation. With the
  mirror plane, reconstruct both positions accurately. Without the link, leave
  those points unreconstructed while retaining the supported cameras.
- **Mirror lines:** the analogous one-sided strokes have different extents.
  Check the recovered infinite line's direction and distance, allowing helper
  endpoints and their order to differ. Removing the relation must not invent 3D.

The reference object has an asymmetric detail; only explicitly paired features
are mirrored. Required point/line geometry is evaluated independently, so an
accurate camera cannot hide incorrect reconstructed landmarks. CAD endpoints
are excluded from withheld checks in the Known 3D line case.

For a live Blender edit sequence, pass `--drop-constraint` with one of these
positive cases to `blender_case.py`. It solves, removes the relation in RNA,
solves again, and verifies that unsupported geometry/helper objects disappear.
A refused solve must preserve the previously valid camera matrices.
`--roundtrip` repeats that sequence in another process. Required point Empties
and line meshes are also checked against the solver result in world space.

`tests/test_synthetic_constraints.py` covers the oracle and six exact positive/removal
cases, plus the frozen weak-line case and its additional-view remedy. The command
also emits a seventh exact variant with that extra view. CI runs the numerical
checks plus the three Blender edit sequences. Noisy versions are optional
exploration; `--noise-px 0.3 --seed 1` changes measurements without changing
the intended constraint relationships.

The [constraint investigation results](constraint-results.md) explain the weak
mirror-line discovery and the product diagnostic it motivated. The frozen
`cases/mirror-lines-weak.json` expects `warn`: required cameras must still pass,
and the solver must identify both weak lines. Their direction/offset errors
remain visible in the report, but are accepted only for those explicitly named
weak lines. An unrequested warning cannot excuse an ordinary accuracy failure.
Generated noisy cases keep their strict accuracy contracts for exploration.

### Fit Only reconstruction boundaries

```sh
python3 tools/synthetic_sync/roles.py --out /tmp/pm-roles
```

Six cases check that Fit Only observations cannot supply missing mirror geometry,
repair a weak mirrored line, or move a parallel line. Supporting cameras are
locked in paired stroke controls so their permitted pose changes cannot explain
3D drift. Required cameras and withheld geometry still have accuracy checks.
The unit suite also verifies that a Known 3D partner can legitimately supply a
reflected feature seen only by Fit Only; excluded observations do not erase known
geometry.

Pass `--role-case /tmp/pm-roles/fit-only-mirror_points.json` to `blender_case.py`
while starting with the corresponding positive constraint case. The runner
solves, changes the role in RNA, and solves again. Unsupported helpers must
disappear. `--roundtrip` repeats this live transition after reopening the original
input. The two cases must differ only in participation and expectations.

The two mirror exclusions and weak-line control failed before the role filtering
fix. The parallel-line control already preserved geometry; it guards the same
boundary without being presented as a reproduced defect. An exploratory
`--stroke-offset-px 4` draw preserved line geometry but moved the Fit Only camera
enough to cross the provisional 1 px withheld limit (about 1.92 px). That is a
measurement-conflict flag, not proof of a solver bug. The default 0.3 px stroke
bias passes the unchanged camera limits. Role ownership and geometric accuracy
are different checks; the tool reports both.

### Origin preparation and camera application

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup --disable-autoexec -b --python-exit-code 1 \
  --python tools/synthetic_sync/preparation.py -- \
  --family overhead --state missing_origin --out /tmp/pm-origin --roundtrip
```

Run `ground` or `overhead` with `preset`, `missing_origin` or `locked_origin`.
The changed origin belongs to a non-anchor camera; the independent anchor-frame
truth remains unchanged. Preparation may change that private camera center but
must preserve its calibration/orientation and all evidence. A pose-locked camera
must keep its missing origin and its pose. The prepared request is then solved
and checked through evaluated Blender cameras against withheld object geometry.

Artifacts retain the original case/raw `input.blend`, `prepared-case.json`,
`prepared.blend`, collected `request.json`, `solved.blend`, metrics and reports.
Fresh-process replay starts from the **raw input**, so it repeats preparation.
An exact prepared case can also be replayed numerically through `run.py`.

All six exact controls passed creation/application and reopening locally; maximum
withheld RMS was below 0.00007 px. No product change was justified by this pilot.
These checks do not cover missing anchor origins (which can redefine the world
frame), calibrated-ground initialization, image-coordinate changes, or undo/redo.

### Background job input parity

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup --disable-autoexec -b --python-exit-code 1 \
  --python tools/synthetic_sync/verify_jobs.py -- --out /tmp/pm-job-inputs
```

The generated state combines nonzero slack, plane and mirror constraints, Known
3D, a locked camera and Fit Only. It compares every prepared numerical field at
the actual Refine Lenses entry for blocking and sidebar worker calls, in both
Same Lens and per-match modes. It also checks cancellation/progress forwarding.
All four captured inputs are saved as JSON. The check substitutes the numerical
search and modal window-manager plumbing; it executes the real worker callback
synchronously, without opening a report or relying on timing.

This reproduced background omission of `plane_groups` and `plane_slack` while
the blocking path retained both. `LensRefinePrep.solver_kwargs()` and
`scene.run_lens_refine()` now own forwarding for both paths. Comparing only the
shared preparation had missed this duplicate call site. This is a routing
regression, not a benchmark of lens accuracy, live UI scheduling or cancellation
latency. Scene edits during a job remain a separate lifecycle question.

### Reducing a confirmed regression

```sh
git worktree add --detach /tmp/pm-before-role-fix 569da33
python3 tools/synthetic_sync/roles.py --out /tmp/pm-role-inputs
python3 tools/synthetic_sync/reduce.py \
  --case /tmp/pm-role-inputs/fit-only-mirror_points.json \
  --baseline-root /tmp/pm-before-role-fix --max-attempts 30 --out /tmp/pm-reduced
```

`--fixed-root` defaults to the current checkout. Neither checkout is edited.
The reducer removes only unrelated free landmarks and their picks. It preserves
every camera, ground/Known 3D point, mirror pair, plane member, line, role, lock, expectation
and truth record. Each accepted deletion must retain the **same named forbidden
geometry** on the old code, pass every other accuracy check there, and pass the
full independent oracle on the fixed code. The default predicate requires an
anchor-frame `solve` contract with `excluded_points` or `excluded_lines`.

For a confirmed line-accuracy defect, name every affected required line:

```sh
git worktree add --detach /tmp/pm-before-plane-line-fix bfd122d
python3 tools/synthetic_sync/reduce.py \
  --case tools/synthetic_sync/cases/mirror-lines-with-plane.json \
  --baseline-root /tmp/pm-before-plane-line-fix \
  --line-accuracy side_edge_left --line-accuracy side_edge_right \
  --max-attempts 30 --out /tmp/pm-line-reduced
```

This alternative requires the **same failed checks on each named line**:
direction, offset and/or distance to the same physical plane. Every other check
must pass, including cameras, required supporting points, finite nondegenerate
lines and all unselected geometry. Weak-line accuracy exceptions are forbidden
for the selected lines. Classified oracle failures preserve this signature
without parsing printed error text or relaxing thresholds. It still requires an
anchor-frame `solve` contract; removing a camera or changing gauge is out of scope.

Each solve runs in a fresh Python process with pose caching off. `protocol.json`
records the budget and source fingerprints before execution; every candidate
retains its exact input, both results, oracle assessments and child logs. Source changes during the
run abort it. `--solve-timeout` defaults to 60 seconds per solve; a process error
or timeout aborts reduction. A solver exception cannot satisfy the predicate.
The final reduced input is replayed cold on both revisions before being accepted. Historical
revisions without plane parameters may omit only inactive plane defaults; the
result records these omissions. Active plane membership or nonzero slack aborts
such a replay, and unknown arguments are never silently discarded. This adapter
is enabled only by the historical worker, not ordinary case solves.

The two initial mirror ownership cases shrank from **90 to 21** and **84 to 15**
point picks in six accepted deletions each (about 22 seconds per reduction on the
recorded local runtime). Every optional free point was removed. This is minimal
only within the permitted deletions: no camera, ground reference, mirror feature
or line was eligible. Passing known-truth checks plus retaining those references
is a safeguard, not a general proof of uniqueness. The line-plane case shrank from **88 to 19** point picks, retaining six ground
references, three Known 3D plane references and both strokes. All six deletions
and the final cold replay preserve both lines' angle, offset and plane failures
on `bfd122d`, while the fixed code passes all checks. Runtime was about **4.5 s**
locally; both poses are locked to isolate the geometry. The reduced case also
passes Blender creation/application, live plane removal and reopening.

The tool does not minimize arbitrary accuracy failures, ambiguous inputs,
operation sequences or timing problems. A small case with the same failing
metrics is reproducible evidence, not proof that every candidate followed an
identical internal solver path.

The exact [reduced cases](cases/README.md) have unit regressions and were also
checked through Blender creation/application and fresh-process reopening. Old
checkout replay is an on-demand investigation, not a dependency of normal CI.
Future cases should expand the predicate only after establishing what evidence
must remain to preserve their meaning.

### Sparse five-camera graphs

```sh
python3 tools/synthetic_sync/graphs.py --out /tmp/pm-graphs
python3 tools/synthetic_sync/graphs.py --seed 1 --count 3 --out /tmp/pm-graph-sweep
```

Five cameras circle the asymmetric object with different known lenses. Each
neighbor pair has eight visible surface points and four visible ground points;
those landmarks have no picks in other cameras. The ground plane ties successive
pair scales to the anchor frame. `chain`, `loop`, `broken_link`, `fit_only_bridge`
and `locked_bridge` specify the intended participants before solving. The broken
link leaves two cameras disconnected. The Fit Only middle view can fit ground
from the preceding Solve view, but must not extend reconstruction into the tail.
The locked middle view can supply that ground while keeping its exact pose.

Every result checks withheld object projections, pose, required/excluded cameras
and reconstructed point positions. A point needs two permitted posed views, or
one permitted posed ground view meeting the shared plane; unsupported points
must stay absent. `--role-case` on the Blender runner checks the chain → Fit Only
transition, including disappearance of unsupported helpers, and can replay it
after reopening. CI runs the five exact controls and this Blender transition.

The initial chain exposed a confirmed defect: zero fitted-pick error accompanied
hundreds of pixels of withheld error and ground landmarks above the plane.
Ground was seeded only from the anchor, leaving later pair scales to a heuristic.
Ground rays from registered cameras allowed to supply 3D now seed subsequent
poses and reconstruction. See [the measurements and remaining gaps](graph-results.md).
The frozen `cases/ground-chain.json` protects the original failing evidence.

`--noise-px 0.3 --count 2` is a small exploratory sweep, with the same provisional
accuracy limits as the base corpus. A flag is a measurement for investigation,
not automatically a solver defect. This graph experiment does not test arbitrary
camera counts, anchorless components, graphs lacking a metric connection,
incorrect correspondences or biased soft constraints.

### Recovered-camera reconstruction

```sh
python3 tools/synthetic_sync/recovery.py \
  --case tools/synthetic_sync/cases/recovered-ground-conflict.json \
  --compare-freeze --out /tmp/pm-recovery
```

This observes the production stage that lets recovered Solve cameras update 3D.
The frozen case deliberately shifts one camera's elevated picks while preserving
its floor evidence. The trace records ground, Known 3D, mirror and shared-plane
gaps before and after the stage. The paired freeze control skips only that stage;
it is an experiment, not an alternative product mode. Exact input, environment,
results, traces and independent withheld-camera checks are retained.

The ordinary accuracy contract stays strict: the contradictory camera still
crosses its withheld-error limit. The regression requires that previously good
cameras and hard ground survive the update. `recovery.py` exits nonzero for an
execution error; accuracy flags are recorded observations. `run.py` remains the
strict whole-case accuracy gate. See [measurements and limitations](recovery-results.md).

`--stage-control-recovered view_2` explicitly marks an already posed camera as
recovered. Use this only to isolate the update, and report it as a stage control,
not evidence that registration naturally selected that route. A positive test
combines imperfect soft Known 3D with an exact Y plane: refinement must remain
useful, honor the plane, and pass the independent camera checks.

### Shared-plane controls

```sh
python3 tools/synthetic_sync/planes.py --out /tmp/pm-planes
python3 tools/synthetic_sync/planes.py --noise-px 0.3 --out /tmp/pm-planes-noisy
```

Ten cases cover separate axis buckets, a tilted Free plane, a Free plane meeting
hard ground, a plane-constrained line, four paired plane-removal controls, and
a Fit Only stroke pair with fixed supporting cameras. Construction planes and
their surface points are known independently; required point/line checks and
withheld object projections test accuracy beyond coplanarity. These base cases
are already solvable without planes, so the removal controls test preservation,
not indispensable information gain. The Fit Only pair always uses its fixed
0.3 px stroke bias, independent of the sweep's point-noise setting.

The frozen `cases/free-plane-hard-ground.json` reproduces a confirmed defect:
Free-plane initialization moved ground seeds off Z=0 before fixing them in BA.
An explicit floor-distance contract catches this even when camera checks pass.
Its positive Ground Slack control still allows floor points to refine. The
Blender runner's `--drop-constraint` option now supports these plane cases; it
clears plane membership in live RNA and repeats the unchanged camera/geometry
checks. `--roundtrip` repeats from the saved raw input in a fresh process.

See [the plane measurements and limitations](plane-results.md). These checks do
not establish general uncertainty, prove a plane's physical correctness from
images alone, or cover arbitrary mirror/parallel/plane combinations.

`--contribution` instead runs X and Free-plane single-view point cases, each
with a no-plane and Fit Only control. The permitted point must be recovered
accurately; the controls must omit it. A separate linear ray/plane calculation
checks that its depth is determined without using the production projector.
`--noise-px 0.3` adds noise to every camera's picks while preserving the truth.
The Blender runner can replay these cases with `--drop-constraint` or the paired
`--role-case`, checking that the unsupported helper disappears after a live
change and after reopening. Results retain `plane_seeded_landmark_ids`; the
product report explains that their depth depends on the plane and one view.

`--mirrored-line` runs the frozen noisy mirrored strokes with and without a
hard Free plane established by three separate Known 3D points, both with true
camera locks and with cameras free to solve. The removal controls retain those
same points, picks, mirror pairs and camera settings. They must still report
weak line support; the positive cases must meet ordinary line-accuracy limits.
The independent reference intersects each stroke plane with the constructed
physical plane, using true cameras rather than production projection code.
`cases/mirror-lines-with-plane.json` freezes the locked positive case and adds an
explicit `plane_max_distance` contract. `--drop-constraint --roundtrip` exercises
live plane removal and fresh-process replay in Blender.

### Evidence-placement follow-up

```sh
python3 tools/synthetic_sync/evidence.py --trials 8 --out /tmp/pm-evidence
```

This bounded experiment keeps the two overhead camera arrangements fixed and
resamples only pick noise. It compares two predeclared new landmarks: an outer
surface point and a central raised point. Both require three picks. A two-pick
support-only control isolates the effect of adding the overhead observation.
Supporting observations are identical in each paired comparison; Known 3D truth
is never supplied. New candidates are disjoint from the unchanged withheld
checks. The original flagged draws are reported separately from fresh draws.

Use repeated `--case path.json` arguments to start from exact saved overhead
cases. The default reads the two frozen JSON files in `cases/`, preserving the
original inputs if the generator changes. Fresh noise seeds start at 100 and
are reused across the two layouts, so the 16 layout/draw comparisons contain
eight independent noise patterns rather than 16 independent scene trials.
Output must be an empty directory. `protocol.json`
records the candidates and decision rule before any solve; `summary.md` and
`summary.json` retain paired improvements, failures and per-height errors.
Every variant also has an exact replayable case/result and a withheld overlay
in `report.html`. Baseline zero/half-noise controls scale the same error vector.

The default is 94 solves (16 fresh draws × five variants, plus two original
draws × seven variants), around several minutes locally. It is an on-demand
experiment, not a mandatory CI sweep. Its exit status reports execution errors;
accuracy flags remain measured outcomes in the report. Replay an individual case
with `run.py --case ...` to enforce its accuracy contract as an exit status.

The candidate policy is specified using the synthetic object's known structure;
it is not yet advice inferred from uncertain real-world picks. The two layouts
are small variations of one object, so they cannot establish a universal picking
rule. See `tests/test_synthetic_evidence.py` for the paired-evidence safeguards.
The [initial evidence-placement results](evidence-results.md) record why the
first run did not justify a new product rule.

`tests/test_synthetic_sync.py` checks the oracle and nine exact seed-zero cases.
Run just that module with `./scripts/run-unittests.sh test_synthetic_sync`.
The test runner explicitly loads the numerical package without Blender's entry
point, so focused core tests no longer depend on an earlier detection test's
import setup.

`.github/workflows/tests.yml` runs normal tests, input-order/cache variants and
Blender smoke/save-reopen cases on PRs and pushes to main. Reports and generated
files are uploaded for 14 days. CI does not render reference images, install
OpenCV wheels, change the release workflow or publish anything. Noisy sweeps are
exploratory and intentionally separate from the passing deterministic CI corpus.
