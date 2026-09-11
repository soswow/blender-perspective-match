# Frozen investigation cases

`overhead-0.json` and `overhead-2.json` preserve the exact original noisy inputs,
expectations and independent truth from the first pilot. They were generated
with geometry seeds 0 and 2 and 0.3 px noise, then captured before changing the
generator. They contain only synthetic geometry.

These cases cross provisional withheld-error limits on the current solver.
They are not yet confirmed solver bugs and are not expected-to-fail unit tests.
The evidence-placement experiment uses them as fixed inputs and reports both
the originally flagged draws and fresh noise draws. Keep these JSON files fixed;
add a new case if the intended evidence changes.

Replay either case independently:

```sh
python3 tools/synthetic_sync/run.py --case tools/synthetic_sync/cases/overhead-0.json --out /tmp/pm-overhead
```

A nonzero exit means its accuracy contract failed. Inspect the result and
withheld-object report before concluding that a production fix is needed.

`mirror-lines-weak.json` freezes the seed-one mirror-line contribution case with
0.3 px noise. Independent reconstruction shows nearly coincident supporting
planes and a large line-direction error even with true cameras. Its contract
therefore expects `warn` for the two specified lines while still requiring
accurate cameras and finite reconstructed geometry. The measured line errors
remain in the report; the warning does not claim precise reconstruction. See
[the investigation](../constraint-results.md). Adding a stroke from a distinct
view restores the ordinary accuracy contract in the corresponding test.

`fit-only-mirror-points-reduced.json` and `fit-only-mirror-lines-reduced.json`
preserve the confirmed camera-role regressions after removing unrelated free
landmarks. The former has 21 point picks, the latter 15 plus two strokes. All
cameras, ground references, mirror features, roles and independent truth remain.
Both reproduce forbidden geometry on `569da33`, while passing the full accuracy
and ownership contracts after `b902124`. They also pass generated Blender
creation/application and fresh-process save/reopen checks. These are passing
regressions in `tests/test_synthetic_reduce.py`, unlike the exploratory overhead
flags above. See [the reducer protocol](../README.md#reducing-a-confirmed-regression).

`ground-chain.json` freezes the exact five-camera chain that exposed missing
propagation of ground scale beyond anchor-visible landmarks. On `34efe80`, the
solver fits every pick to numerical precision but misplaces later cameras and
raises their ground points above Z=0. The independent withheld check fails by
up to about 593 px RMS. A Fit Only middle-camera variant of this same input also
fails to register despite four ground observations supported by the preceding
posed view. `tests/test_synthetic_graphs.py` checks both regressions; the separate
graph generator adds loop, broken-link and locked-bridge controls.

`recovered-ground-conflict.json` preserves a deliberately inconsistent camera:
its elevated picks shift by 180 px while its floor picks retain their original
0.3 px noise. Before the recovery guard, the final reconstruction update spoils
a previously accurate camera. `tests/test_synthetic_recovery.py` protects that
healthy camera and ground while retaining the recovered pose. The contradictory
camera still fails the ordinary withheld-accuracy contract; this is **not** a
fully passing accuracy fixture. Use `recovery.py --compare-freeze` to isolate the
stage, or `run.py` to see the remaining strict accuracy flag. See
[the recovery investigation](../recovery-results.md).

`free-plane-hard-ground.json` has two hard floor landmarks and two elevated
surface points sharing a tilted Free plane, with 0.3 px pick noise. All points
have physically visible picks and independent truth. On `5a4a34d`, initialization
moves the floor points about ±0.00047 scene units from Z=0. Ordinary camera and
point-accuracy checks pass; the explicit `ground_max_distance` contract fails.
The fixed solver preserves those floor seeds. `tests/test_synthetic_planes.py`
also checks a positive nonzero Ground Slack control, so the fix cannot pass by
disabling all ground refinement. Both the full case and plane-removal control
are expected to pass the complete camera, geometry and floor contracts.
