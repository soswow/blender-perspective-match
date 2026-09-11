# Shared-plane investigation

11 September 2026, after integrating the user's `f117f4c` plane feature.
The recovery guard was committed as `5a4a34d` before this experiment.

The first controls support the existing implementation: separate axis buckets,
an unknown tilted plane and a plane-constrained line preserve accurate cameras
and reconstructed geometry. A Fit Only stroke does not reshape the line. The
new failure was at initialization: a Free plane could move hard ground seeds
before bundle adjustment fixed their positions.

## Independent controls

`planes.py` constructs physically visible points on the existing asymmetric
object. The tilted plane follows an explicit equation, rather than a plane
fitted by production code. Camera checks use object vertices and face samples
that were never supplied to Sync. Required points and infinite lines also have
truth checks, so a coplanar but misplaced reconstruction cannot pass solely
because it is flat.

Each of four families has a control removing only plane membership. These cases
already have enough multi-view evidence to solve; removal establishes that the
extra constraint does not damage a consistent setup. It does **not** establish
how much indispensable information the plane supplies. The final two cases add
an imperfect Fit Only stroke with the supporting camera poses fixed, checking
that excluded evidence cannot change the line. That pair always uses a 0.3 px
stroke bias, including in the otherwise exact batch.

## Ground failure and fix

The frozen `cases/free-plane-hard-ground.json` places two floor landmarks at
`y = -2, z = 0` and two front-face landmarks at `y = -0.9, z = 1.1`. They share
the true plane `z = y + 2`. The picks have 0.3 px noise. All camera and ordinary
point-accuracy checks pass before the fix, yet the floor points end at
**+0.000467 and -0.000467 scene units**. Removing only plane membership leaves
them at zero.

`apply_plane_seed` projected every free member into its provisional fitted
plane. Ground landmarks were subsequently fixed as metric references, so their
displaced heights survived refinement. The initializer now preserves those
ground seeds. The focused regression fails before and passes after this small
change. Nonzero Ground Slack still allows movement in the adjustment stage;
the positive control passes camera checks with ground movement up to about
0.00125 scene units.

The evaluator has an explicit optional `ground_max_distance` contract for cases
that require hard floor preservation. This fixture sets it to 0.000001 scene
units. It is not silently imposed on all noisy or soft-ground cases. A deliberate
floor displacement with otherwise true cameras fails that check.

## Validation and practical limits

The ten exact/control cases pass, with maximum withheld RMS **0.1444 px**
(including the deliberately biased Fit Only stroke), about 31 seconds of
recorded solve time locally while other validation ran. The frozen noisy floor
case passes ordinary/reversed input and cold/warm replay, with maximum withheld
RMS **1.393 px** and ground heights at floating-point zero. The provisional
camera limits were not relaxed.

The ten 0.3 px noise/control cases also pass, with maximum withheld RMS
**1.469 px**. The full suite passes **300 tests, 9 skipped**.

Both the noisy floor case and the plane-line case pass generated Blender
creation, collection, application, live removal of plane membership, and
fresh-process reopening. Those commands are included in CI; hosted execution
still needs an authorized push. No user `.blend` was modified.

This is a small interaction check, not a broad stress test. It does not certify
arbitrary mirror/parallel/plane combinations, single-view reconstruction from
a supported plane, Free-plane conditioning, or correct interpretation of a
physically wrong plane constraint. Plane distance alone does not establish
camera accuracy; both remain in the recorded results. The next useful question
is whether a plane supported by other views can recover a feature picked only
once, with an explicit no-plane and Fit Only control.

## Single-view contribution follow-up — 12 September 2026

The preservation checkpoint was committed as `19373fe`. The next experiment
adds a fourth, independently posed camera and one extra surface point seen
only in that camera. X and tilted Free variants have other reconstructed
members that establish the plane. A separate linear ray/plane calculation
recovers the known point without calling the production projector. The
baseline omitted that point in both variants; this was a capability gap in
the previous joint-BA-only behavior, not a broken accuracy promise.

Sync now seeds missing points from a supported **hard** plane and exactly one
posed camera allowed to contribute 3D. An axis bucket needs one other located
member; a Free bucket needs three non-collinear members. The new point supplies
no evidence for its own seed plane. Fit Only observations, unsupported planes,
backward intersections and grazing rays are excluded. Grazing uses the existing
plane-geometry angular budget (`LINE_PLANE_MIN_SINE = 0.12`, about 6.9 degrees
from the plane); it is not a calibrated confidence bound. Soft planes and
single-view lines retain their prior behavior.

This runs during initial reconstruction and after cameras are recovered, with
the existing constrained joint adjustment still refining the result. Capture
and replay retain the new `plane_seeded_landmark_ids` result metadata. Diagnose
labels the affected points **Plane + one view** and explains that fitting the
one pick cannot independently verify depth.

Both axes, no-plane controls and Fit Only controls pass across two exact seeds:
**12 cases**, maximum withheld camera RMS below **0.000003 px**. The one-point
positions agree with construction to numerical precision. Six corresponding
0.3 px noise cases pass unchanged accuracy contracts, with maximum camera RMS
**1.469 px** and target point errors of **0.158% / 0.225%** of the object diagonal
for X / Free. Each six-case batch took about 36–41 seconds locally while other
checks were running. Those samples establish a useful bounded capability, not
a general noise guarantee.

Blender X-plane removal and Free-plane Solve→Fit Only transitions both pass
creation, application and fresh-process reopening, including disappearance of
the unsupported point helper. The final report metadata also passes that Free
transition. Maximum withheld RMS is below 0.00004 px in Blender. Reversed-input
and cold/warm replay pass; HTML content and escaping are checked, but a browser
visual review was unavailable in this session.

The full suite passes **304 tests, 9 skipped**, after correcting a missing
argument in the new report test fixture. The report-only rerun also passes.

The route does not initialize an unposed camera, establish absolute scale from
an unknown plane, resolve failed multi-view triangulation, or validate the
user's physical plane assumption. Minimal Free support can be sensitive to
noise even when it is mathematically non-collinear. The next diagnostic question
is whether current scale/quality wording distinguishes such constraint-derived
geometry from independent metric evidence.
