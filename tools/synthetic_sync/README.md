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
when the expected behavior is established. Automatic reduction/search is not
implemented in this pilot. Add new scene actions or evidence types to the shared
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
