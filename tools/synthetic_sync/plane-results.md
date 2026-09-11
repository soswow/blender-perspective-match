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
