# Independent point-FOV constraint trials

The frozen cases come from `tools/synthetic_sync/focal_constraints.py`. Their
truth projections use its standalone pinhole calculation; training picks and
withheld landmarks are disjoint. The saved anchor rotation and center equal
truth in every case, while other saved private poses are deliberately wrong.
The scaffold assumes unobstructed landmarks and checks camera frusta, not
occlusion. Each landmark has at least two views and each camera at least eight
picks. One-view constraint seeding is outside this fit's current scope.

`run-01` through `run-03` retain the exact request, result, source archive,
runtime metadata, and one reservation per inner Sync and bundle call. `run-04`
replays four hard cases with the final bundle source from `run-02`'s archived
successful Sync world states; it reconstructs scale-one private-to-world
similarities that reproduce those saved camera centers and rotations exactly.
Across all runs, 11 of 12 authorized inner Sync calls (119.0 s total) and 15
of 18 bundle fits (<0.6 s total) completed. Every batch had a 720 s outer
timeout. No numerical attempt was repeated after a solver failure.

| Case | Pick noise | Max focal error | Withheld RMS | Constraint evidence |
| --- | ---: | ---: | ---: | --- |
| Free hard, exact | 0 px | <0.001% | 0.000013 px | Plane RMS <2e-14 world |
| Axis hard, exact | 0 px | 0.00035% | 0.000446 px | Plane RMS <2e-13 world |
| Mirror off anchor, exact | 0 px | <0.001% | 0.000014 px | Mirror gap <3e-10 world; scale ratio 0.99999993 |
| Mirror through anchor, exact | 0 px | <0.001% | 0.000003 px | Mirror gap <2e-11 world; free scale |
| Exact Free/axis/mirror removals | 0 px | <0.001% | <0.000015 px | Identical picks and truth; relation absent |
| Free hard, noisy | 0.35 px stated | 2.37% | 2.026 px | Plane RMS <2e-8 world |
| Free hard removed, same noisy picks | 0.35 px stated | 2.23% | 2.022 px | No independent accuracy gain claimed |
| Axis soft, noisy | 0.35 px stated | 2.19% | 0.858 px | Plane RMS 0.00363 world (<0.08 slack) |
| Mirror offset biased and soft, noisy | 0.35 px stated | 2.32% | 13.590 px raw world | Fitted mirror gap 0.000311 world |

The four final-source hard replays reproduced the `run-02` accepted results
to printed precision and passed the final hard-gap and scale-bound guards.
The four exact hard cases and their same-pick removal controls establish that
the public path fits valid relations and does not lose unconstrained point
support. The noisy Free hard control enforces coplanarity but gives essentially
the same independent geometry as its removal; it is not an accuracy-improvement
claim. The noisy Free hard `run-01` summary used an obsolete camera-baseline
alignment and reported 15.956 px withheld RMS. Its immutable numerical ledger
is accompanied by `run-01/free-hard-reassessment.json`, which applies one
positive scale fitted from training points only and yields the 2.026 px value
above. The `run-03` removal used this corrected assessment from the outset.

The biased mirror plane offset is a conditional metric control. Moving the
plane along its normal can be exactly accommodated by scaling cameras and
points about the fixed anchor without changing image picks. Its raw world
13.590 px withheld error therefore remains visible and must not be called a
correct metric reconstruction. As a separate visual-shape diagnostic,
`run-03/mirror-biased-soft-shape-reassessment.json` permits one scale derived
only from training points and obtains 0.458 px withheld RMS. That alignment
does not satisfy the supplied metric plane and does not replace the raw result.
Soft offset thaw does not recover the true plane from these picks alone.

The local fit conditions its focal intervals and pixel-noise check on the
supplied relations; geometric spring rows are not extra independent clicks.
The fixed anchor frame remains an input condition. A supplied axis or mirror
orientation inconsistent with that frame is not repaired by this fit. The
`mirror-tilted-hard` frozen case records that orientation-mismatch control for
later explicit evaluation; it was not numerically attempted in this budget.
