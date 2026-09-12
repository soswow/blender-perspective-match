# Plane, mirror and parallel interaction results

Baseline: `fd0633f`, 12 September 2026. The bounded experiment confirmed that
mirrored lines could lose an independently fixed **Is Parallel To** direction,
even when cameras and a compatible hard plane were accurate. The fix preserves
that direction while fitting the reflected pair jointly, inside its independently
supported hard plane when available.

## Construction and independent contracts

`cases/plane-mirror-parallel.json` extends the existing reduced mirrored-plane
fixture with one separate Known 3D reference edge. Its direction is `(0, 1, .8)`,
the direction of both mirrored edges. The plane is `z = .72 + .8*y`, the mirror
is `x = 0`, and all three relationships are physically compatible. Three
non-collinear Known 3D points establish the plane independently of the free
lines. The reference edge and its exact stroke are physically visible; the
original two weak mirror strokes retain their frozen 0.3 px noise. CAD endpoints
and supporting points are separate from withheld object checks.

The eight controls use identical truth, points, strokes and Known 3D references:

- Hard, soft (`Plane Slack = .02`) and absent-plane cases with locked poses.
- A hard-plane case with unlocked poses.
- Each of those four cases paired with removal of **only** the parallel links.

The existing independent camera, withheld-object, line-direction, line-offset
and physical-plane checks remain. Additional checks require parallel direction
sine at most `1e-8` and reflected-line distance at most `1e-6` scene units. A
weak-line warning cannot waive either relation. Deliberately perturbed geometry
proves these new checks fail even when camera checks and the existing ordinary
line-accuracy budget pass.

Soft/absent-plane controls explicitly retain the prior weak-line expectation.
An independent fixed-direction solution of their reflected stroke planes has
condition number **101.3** and **0.08311 scene-unit** position error. Knowing a
direction does not remove that weak depth evidence. A hard plane reduces the
independent reference error to **0.002620 scene units**. These calculations use
constructed cameras and separate projection/linear algebra, never Sync's fitted
line or camera helpers.

## Observations

Values below are for one mirrored line; its partner has the same direction and
offset errors. Offset is a percentage of the object diagonal.

| Control | Baseline direction | Fixed direction | Fixed offset | Maximum fixed withheld RMS |
| --- | ---: | ---: | ---: | ---: |
| Hard plane, locked, parallel | 0.07461° | 0° | 0.07267% | <1e-12 px |
| Hard plane, locked, parallel removed | 0.07461° | 0.07461° | 0.08649% | <1e-12 px |
| Soft plane, locked, parallel | 21.3181° | 0° | 2.30495% | <1e-12 px |
| Soft plane, locked, parallel removed | 21.3181° | 21.3181° | 10.1362% | <1e-12 px |
| No plane, locked, parallel | 21.3181° | 0° | 2.30495% | <1e-12 px |
| No plane, locked, parallel removed | 21.3181° | 21.3181° | 10.1362% | <1e-12 px |
| Hard plane, unlocked, parallel | 0.06574° | 0° | 0.06868% | 0.27001 px |
| Hard plane, unlocked, parallel removed | 0.06574° | 0.06574° | 0.08070% | 0.27027 px |

Baseline parallel-on and parallel-off geometry was identical. Four active
parallel contracts failed; all four removal controls passed. The final code
passes all eight. Hard-plane parallel sine is below `3e-16`; reflected-line
distance is below `3e-16` scene units. The weak soft/absent-plane results remain
warnings, with their position errors visible; they are not relabeled accurate.

The line rebuild applied parallel enforcement before one-stroke mirrored edges
were reattached. Later plane/mirror fitting could also replace a fixed parallel
direction. A later independent parallel pass would break plane or reflection
membership. The fix resolves compatible directions from world-axis/Known 3D
parallel groups and fits the pair at that direction during mirror enforcement.
It preserves compatible independent hard planes and uses both reflected stroke
sets when no hard plane supplies depth. Known geometry remains unchanged and
only location-enabled observations contribute to reconstruction.

## Storage precision and validation

A float32 roundtrip casts **request values only**, retaining double-precision
truth and every output limit. The initial `1e-8` compatibility tolerance rejected
the independently rounded direction/plane normals: their sine discrepancy was
`2.90754e-8`. Compatibility now permits one float32 epsilon (`1.19209e-7`). This
is an input-consistency tolerance, not a relaxed output parallel or plane check.
All eight float32 controls pass. The hard locked case stays within
`3.36e-8` scene units of the true plane and `0.0000530 px` withheld RMS.

Focused tests cover incompatible world-axis/CAD groups, zero/nonfinite Known
3D directions, unchanged Known geometry, and paired Fit Only stroke edits for
hard and soft plane cases. Eight new tests pass; the main reconstruction test
failed in four subcases before the fix. The 43 existing mirror, line, plane,
synthetic-plane and role tests pass. No full numerical suite or native Blender
run was performed in this isolated task; integration owns those checks.

Commands (use fresh output directories):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/synthetic_sync/constraint_interactions.py --out .local/constraint-interactions/fixed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/synthetic_sync/constraint_interactions.py --float32 --out .local/constraint-interactions/float32-fixed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_unittests.py test_synthetic_constraint_interactions
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_unittests.py test_sync_mirrors test_sync_lines test_sync_planes test_synthetic_planes test_synthetic_roles
```

The first command against the original solver produces the four failures above.
Every runner invocation preserves exact case JSON, request fingerprints, complete
results, independent assessments, environment information and an HTML report.
Local baseline/fixed/float32 evidence lives under `.local/constraint-interactions/`.

To apply the additional invariants to a saved numerical or Blender result
(without running another solve):

```sh
python3 tools/synthetic_sync/constraint_interactions.py \
  --case tools/synthetic_sync/cases/plane-mirror-parallel.json \
  --result /path/to/result.json --out /tmp/pm-interaction-postcheck
```

Both a direct result record and a JSON object wrapping it in `result` are
accepted. This entry was verified against a saved numerical result. For a
native Blender check, use the frozen active case in `blender_case.py`, then
post-check the evaluated camera result using this command.

## Limits

This is one physically coherent family with paired ablations, not a general
search over interacting constraints. Free-only parallel families, mutually
inconsistent priors, biased CAD, unknown distortion, soft-plane depth quality,
recovered line-only camera acceptance and larger graphs remain separate work.
The float32 experiment tests stored precision, not RNA collection or live UI
scheduling. No private scene was read or changed, no package was installed, and
no threshold was loosened to suppress an independent accuracy failure.
