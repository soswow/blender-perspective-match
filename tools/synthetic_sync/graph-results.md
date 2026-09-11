# Ground scale through sparse camera graphs

Investigated 11 September 2026, following the camera-role and reduction work.
This is a Sync experiment with ideal known intrinsics and exact observations;
no image analysis, distortion or private project is involved.

## Intended information

Five cameras circle the asymmetric object with distinct focal lengths and
off-center principal points. Each adjacent pair shares eight surface and four
ground landmarks, picked only in those two cameras. The five arrangements are
a chain, a closing loop, a broken link, a Fit Only middle camera, and a locked
middle camera. Frustum/occlusion tests verify physical visibility. All private
non-anchor poses are unrelated to truth unless explicitly locked.

The ground is one known plane in the anchor world. Once a camera is posed, one
of its ground rays can locate a new floor point in that world. Four non-collinear
floor points support the next camera. Thus this particular chain has metric
support; mere graph connectivity without that information would not establish
the same claim. Required cameras and geometry are declared before solving.

## Confirmed failure and fix

At `34efe80`, the seed-zero chain reported about **8e-13 px** fitted point RMSE.
Its last three cameras had withheld RMS of **446, 486 and 593 px**. Ground
landmarks on later links lay approximately **1.18–1.20 scene units above Z=0**.
The loop, broken-link and locked-middle controls passed. Changing the middle
camera to Fit Only left it unregistered despite its supported ground picks.

The numerical path formed single-view ground positions only from anchor picks.
Later views therefore lacked available metric correspondences during graph
registration and relied on relative-pose scale guesses. The dense-scene
freeze/thaw path did not rescue this case; a tiny fitted residual did not prove
that the ground constraints or geometry were right.

The fix supplies ground hypotheses from registered cameras permitted to move
3D, using their transformed shared-world camera poses. Registration fills gaps
in its existing cloud with these hypotheses. Reconstruction keeps Known 3D and
anchor references authoritative and applies the existing triangulation-agreement
check before pinning ground at zero slack. Fit Only observations remain excluded
from this source of geometry. No threshold, camera coverage requirement or
withheld-error limit was relaxed.

Both focused tests fail before and pass after the change. The original exact
case is frozen in `cases/ground-chain.json`; a second expectation on the same
evidence checks the Fit Only role boundary. The generator's oracle additionally
requires supported landmark positions and excludes unsupported geometry.

## Measurements and limits

All **20 exact arrangements** (five variants × seeds 0–3) passed camera coverage
and independent withheld checks. Maximum required-camera withheld RMS across
them was **0.003 px**. Input reversal and warm-cache replay also passed for the
frozen chain; the first numerical solve took about 2.5 seconds and subsequent
cached solves about 0.14 seconds locally. These are observations on one runtime,
not performance guarantees.

The strengthened seed-zero contracts also pass required/excluded point checks.
Blender creation, application and fresh-process reopening test the live
chain → Fit Only role transition, including geometry helper removal.

The full unit suite passed **268 tests with 9 skips** in 162 seconds, and the
existing Blender smoke test passed on Blender 5.1.0. Hosted CI is configured
but has not been run in this continuation; no push was performed.

The separate **0.3 px noise sweep flagged five of ten cases** (seeds 0–1), with
maximum required-camera withheld RMS **3.37 px**. All cameras required by those
cases remained registered; some exceeded the provisional 1.8 px/2% center
limits. These modest noisy deviations are not the exact chain's hundreds-of-pixels
failure. Their classification remains noise sensitivity or an unresolved accuracy
issue, rather than a new confirmed solver bug. Limits remain unchanged, and the
noisy sweep is not an expected-to-pass CI gate.

This does not make every connected graph well constrained. With no metric/depth
connection, pairwise links can retain independent scale freedoms. Nor does it
establish behavior under incorrect ground labels, wrong K, biased CAD, gross
outliers, long edit sequences or arbitrarily large graphs. Existing regression
tests still cover mislabeled ground fallback and ordinary constraint behavior.

Two code paths merit a separate follow-up: acceptance of a thaw currently compares
point reprojection error rather than the whole constrained objective, and
`_thaw_recovered_location` does not forward ground/mirror/Known 3D slack to its BA
call. These are observed contract gaps, not independently reproduced new defects
in this experiment. Do not silently broaden this fix to redesign BA; construct
paired constraint controls first.
