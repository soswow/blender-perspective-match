# Accepted recovered-camera update and plane/parallel interaction

12 September 2026. Synthetic numerical evidence only; no private project was
modified. The acceptance policy was not changed.

## Result and exact paired boundary

One deliberately labelled **stage-only** case reached an actually committed
recovered-camera 3D candidate. The already posed `view_2` was explicitly marked
recovered at the stage boundary, so this does not establish natural
peel/resection routing. The paired freeze control used the identical request and
bypassed only `_thaw_recovered_location`; both retained final free-line rebuild.
The candidate and committed snapshots were equal, `kept_joint_geometry=false`,
and landmarks, a camera pose and line geometry changed. The two runs had identical
request fingerprints. No rejected/no-op candidate is counted as accepted.

The checked-in [exact case](cases/accepted-recovery.json) extends `mixed_lines`,
seed 0, 0.3 px noise, with unchanged line strokes. Three front-face Known 3D
references are biased +0.03 scene units in Z; Known 3D slack is 0.08, ground
slack 0.02. Four front-face points and the free vertical line share a hard `Y#1`
plane. Both lines declare hard `WORLD_AXIS_Z`. Case JSON SHA-256:
`d58710a23ce9e12d7636adbe34ec0d33f985a4d2140e5c09aa349a2480901d2b`.

The accepted update improves all three biased soft Known 3D references. In the
baseline solver, camera RMS, line offset and physical-plane position made small
tradeoffs within their independent accuracy limits. It did **not** establish
recovery-caused final geometry damage. A stricter independent hard-direction
check did reveal a **pre-existing generic plane/parallel defect**, present
before the stage and in the freeze control. Hard plane membership held while the
vertical line deviated 0.263920 degrees (direction sine 0.00460625) before the
stage and 0.220712 degrees (sine 0.00385214) after the accepted update and final
rebuild. The existing hard parallel interaction limit is sine `1e-8`. Ordinary
line accuracy allows 1 degree and did not detect the violated relation.

| Independent baseline measure | Before / freeze final | Accepted candidate | After final line rebuild | Limit |
| --- | ---: | ---: | ---: | ---: |
| soft Known 3D truth errors | 0.016163 / 0.014337 / 0.012777 | 0.010939 / 0.007262 / 0.006044 | unchanged | improve |
| free-line Z angle | 0.263920° | 0.263920° | 0.220712° | sine <= 1e-8 |
| free-line offset / object diagonal | 0.00116274 | 0.00175300 | 0.00147327 | <= 0.02 |
| physical-plane offset | 0.00206921 | 0.00265038 | 0.00265038 | <= 0.005 |
| signed point/line-endpoint coplanarity spread | 4.99e-9 | 1.44e-8 | 7.53e-9 | < 1e-5 |
| healthy `view_1` withheld RMS | 0.407929 px | 0.473260 px | 0.473260 px | <= 1.8 px |
| stage `view_2` withheld RMS | 0.469770 px | 0.565268 px | 0.565268 px | <= 1.8 px |

The candidate's maximum changes were 0.0181795 scene units in landmarks,
0.0181974 in line endpoints and 0.00644146 in camera centers. Final line
reconstruction subsequently moved an endpoint 0.0189344, without moving a
camera. These trace boundaries distinguish accepted-candidate contribution
from final rebuilding.

Physical-plane offset compares output with independent truth; coplanarity
compares **signed** offsets of the four grouped points and both free-line
endpoints. A negative oracle control puts members on opposite sides of the plane
with equal absolute truth offsets and detects their 0.02-unit separation. This
prevents absolute-distance cancellation from masquerading as coplanarity.

## Bounded production fix and fixed replay

`_rebuild_free_line_segments` first enforces parallel directions, then calls
`enforce_plane_line_segments`. Before the fix, independently supported plane
fitting replaced the free line direction from noisy strokes, overlooking its
compatible world-axis/known-line fixed prior. The mirror pass handled the
special mirrored interaction, but the generic free line had no mirror pair.

The fix forwards parallel links into the hard-plane fit, uses the existing
compatible axis/CAD resolver, and fits line *position* along the supported
plane at the fixed direction. Low-information strokes retain the fixed direction
through fallback. An incompatible fixed direction is ignored by that joint
plane fit; the plane retains its prior behavior. This avoids a later independent
parallel pass that could break plane membership.

| Independent fixed-replay measure | Before / freeze final | Accepted candidate | Final rebuild |
| --- | ---: | ---: | ---: |
| soft Known 3D truth errors | 0.015620 / 0.013817 / 0.012325 | 0.009726 / 0.005911 / 0.004710 | unchanged |
| free-line Z direction sine | 0 | 0 | 0 |
| free-line offset / object diagonal | 0.00066674 | 0.00020694 | 0.00053276 |
| physical-plane offset | 0.00000319 | 0.00000483 | 0.00000483 |
| signed coplanarity spread | 6.13e-9 | 1.37e-8 | 9.46e-9 |
| healthy `view_1` withheld RMS | 0.280399 px | 0.230284 px | 0.230284 px |
| stage `view_2` withheld RMS | 0.404078 px | 0.258813 px | 0.258813 px |
| free-line endpoint Z extent | 0.101–1.303 | −8925.946–−8924.744 | 0.101–1.303 |

The fixed candidate temporarily slides approximately 8926 units **along** its
infinite free line during recovered BA. Infinite-line reprojection, hard
direction and plane membership are invariant to that movement. The final common
line rebuild restores the finite endpoints to z 0.101–1.303 (truth 0.1–1.3),
and the returned result exactly matches that final snapshot. This is a measured
conditioning concern for a future bounded representation experiment, not a
demonstrated final-output defect or a reason to broaden this fix. The driver
checks final finite extent separately.

## Replay, validation and scope

Use fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/synthetic_sync/accepted_recovery.py \
  --case tools/synthetic_sync/cases/accepted-recovery.json \
  --out /tmp/accepted-recovery
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_unittests.py \
  test_sync_planes test_synthetic_accepted_recovery test_sync_lines \
  test_sync_mirrors test_synthetic_planes test_synthetic_constraint_interactions
```

The old solver emits its exact traces and exits 1 for the hard axis violation;
the fixed solver exits 0. The driver also rejects no-op/rejected candidates,
unequal paired request hashes, failed independent line/plane/camera accuracy,
lost signed coplanarity, failed Known 3D improvement, bad final finite extent,
or a mismatch between returned state and final trace. Baseline records are in
`/tmp/accepted-recovery-final-v2/`, fixed records in
`/tmp/accepted-recovery-fixed-replay/` in this workspace. Copies are retained
under `.local/worker-handoffs/accepted-update/`; integration logs are under
`.local/accepted-update-integration/`. These local files are not needed to replay
the checked-in case. The stage-path regression was rerun against the original production files: it
fails on the measured direction sine `0.00460626 > 1e-8`, establishing the
geometric failure through the unchanged solver entry. After the fix, the
regression replays the frozen JSON and passes. Combined validation passed
**340 tests, 9 skipped, 265.508 seconds**, plus Blender 5.1.0 smoke. The final
frozen-case hookup passed its two focused tests separately; the worker also
passed 50 related tests and the exact paired replay.

The investigation used one parameter variant and no sweep. It did not assess
every hard relation or establish naturally recovered routing. The accepted
candidate's transient along-line drift remains separately visible in the trace.
