# Infinite-line midpoint gauge: bounded numerical experiment

12 September 2026. The experiment below used diagnostic monkeypatches only
against base `16a6cb6`; a separate, narrow production pinhole-projector fix was
added afterward in this worktree. No experiment outcomes were rewritten.
The exact accepted-stage case is `cases/accepted-recovery.json` (SHA-256
`d58710a23ce9e12d7636adbe34ec0d33f985a4d2140e5c09aa349a2480901d2b`).
All comparisons used the same case request, one OpenBLAS/OMP/MKL thread, Python
3.14.7, NumPy 2.4.1, macOS arm64, solver base `16a6cb6`. The stage-only
recovered-camera route and final line rebuild are unchanged. Final replay output
is in `/tmp/line-position-gauge-v3/` and `/tmp/line-position-gauge-analytic-v1/`.

## Finding

In the accepted case, the free line has fixed Z direction and belongs to a hard
Y plane. The Y-plane spring and ideal infinite-line image projection are
invariant when its midpoint slides along Z. BA nevertheless has three midpoint
parameters, and the line-image Jacobian estimates all three by forward finite
differences. Small false along-line derivatives enter a diagonally damped normal
equation. The baseline's largest *evaluated* midpoint norm was 33,696 scene
units; its accepted recovered-stage midpoint ended around Z = -8,925.3. The
normal final rebuild restored the finite line to Z = 0.101–1.303, so this case
still establishes no final-output defect from the excursion.

This gauge cannot be removed for every line. Axis-plane springs constrain the
component along an axis when line direction has a component on that axis.
FREE-plane residuals refit their normal using member midpoints; shifting one
midpoint can change both the fitted plane and every member residual. For a
mirrored line pair, shifting line A by direction `dA` changes the pair gap by
`cross(H dA, dB)`, where `H` reflects across the mirror plane. It vanishes only
when the reflected direction aligns with line B. Known-line and anchor logic
can also fix the entire midpoint. A global two-parameter midpoint would alter
the implemented objective for these cases.

The sampled projector has a separate deterministic inconsistency: the very
same visible, unbounded line returned a valid projected image line at its
original midpoint and `None` after a +100 or +9000 unit shift along itself.
Its finite ±0.25…32 sampling window lay behind the camera. The weighted BA
residual then jumped about 77 px (squared-cost jump about 36,000). At a +1
shift the complete weighted residual vector changed by at most 6e-13, while a
-9000 shift changed it by about 1–2e-6 from floating-point error. This is an
implementation effect, not a physical along-line constraint. The result should
be tested separately from BA parameterization.

## Prototypes and paired evidence

The first prototype projects **only line-image Jacobian midpoint rows** onto
the plane perpendicular to that line's fixed direction. It leaves all residual
values and point, plane, mirror, and camera derivatives intact. The second
prototype patches only the pinhole line projector: it forms the camera viewing
plane normal from the line and camera center, applies the camera rotation and
`K^-T`, and normalizes the homogeneous image line. Distorted calibrations keep
the sampled path. Neither prototype changes production code.

Before any solve, the analytic projector's image line passed through points
from the independent synthetic pinhole projection within 1.2e-13 px under identity and a
scaled/rotated/translated similarity. Shifts of ±100 and ±9000 yielded the
same line exactly in both frames. Camera-crossing and wholly-behind,
camera-parallel lines were refused. A non-unit caller direction was unchanged
after projection. The initial prototype normalized an `np.asarray` view in
place; the checked-in diagnostic copies it first. BA's supplied direction is
already normalized, so this did not explain the large recovered-stage starting
cost in the exploratory replay, but mutating it was invalid and would affect
other callers. The analytic full-solve numbers below were captured before that
copy correction; no extra full solve was run after it. The no-solve independent
projection and immutability checks were rerun on the corrected function.

| Accepted-stage measure | Baseline | Jacobian projection | Analytic projection |
| --- | ---: | ---: | ---: |
| Initial BA residual / Jacobian evaluations | 54 / 16 | 25 / 12 | 25 / 12 |
| Recovered BA residual / Jacobian evaluations | 21 / 8 | 16 / 8 | 16 / 8 |
| Sum of measured BA seconds | 0.384 | 0.218 | 0.231 |
| Whole solve seconds, first run | 3.995 | 3.773 | 4.174 |
| Whole solve seconds, repeat | 3.999 | 3.797 | 4.010 |
| Maximum evaluated free-line midpoint norm | 33,696 | 1.658 | 1.658 |
| Returned recovered candidate Z extent | -8925.946…-8924.744 | 0.102…1.304 | 0.102…1.304 |
| Final finite Z extent | 0.101…1.303 | 0.103…1.304 | 0.103…1.304 |
| Minimum fitted cost, initial / recovered BA | 12.506 / 18.164 | 12.315 / 17.682 | 12.315 / 17.682 |
| Final withheld RMS, views 1 / 2 (px) | 0.230 / 0.259 | 0.454 / 0.550 | 0.454 / 0.550 |
| Final independent physical-plane offset | 0.0000048 | 0.002564 | 0.002564 |

The two prototypes found essentially the same candidate and final geometry.
Their fitted objectives improved, but their withheld cameras and independent
plane position worsened on this deliberately biased-reference case. Both
remain within the case's accuracy limits and both accepted-stage guards commit
the candidate. This is a real independent-geometry tradeoff, so neither
prototype establishes a general quality improvement from this one case. The
BA-only timing reduction is clear, but whole-solve timing is small and
unstable. Evaluation counts are deterministic.

The prototype's recovered BA started with a high objective (about 30,432
versus 24.69 baseline), then converged. Cached stage inputs show this is an
earlier-path/seed difference, not a disagreement between projection functions
at the same starting geometry. Both recovered-stage line midpoints were near
Z = 0.703. At the prototype's pre-stage state, grouped point Y coordinates
were near -0.902055, while the rebuilt line seed reset its midpoint Y to
-0.9. The resulting 0.002055 gap and hard-plane spring 60,000 account for at
least 15,204 squared cost. The baseline points were near -0.900003, for a
comparable contribution of only 0.037. Reprojecting the cached starting
midpoint through each cached world camera with **both** primitives gave image
lines equal within 5.4e-13 and stroke-endpoint residuals equal within 5e-13
px. Full packed starting parameter vectors were not retained; this comparison
isolates the line-pixel contribution using the exact saved midpoint and pose.

At final output, both prototypes preserve exact Z parallel direction (sine
zero), hard-plane signed membership spread below 8e-9, and finite line length
about 1.202. The baseline/prototype final infinite-line offset differs by
0.00323 scene units and camera centers by at most 0.00939. The separate
non-axis mirrored FREE-plane control passed all independent camera, line,
mirror, and parallel checks in baseline and both prototypes. It did not change
geometry, iterations, or evaluation counts (9 residual and 4 Jacobian calls).

The ABBA accepted-case baseline and Jacobian-prototype repeats were bitwise
identical in reported geometry; the analytic-prototype repeat was likewise
identical. The final reproducible protocol uses six solves for the first
prototype and three for the analytic follow-up. During driver development, a
failed draft used two solves and an earlier successful six-solve run preceded
the cost-instrumented repeat, making 17 exploratory solves total. No further
solve sweep was run.

## Replay

Use fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 tools/synthetic_sync/line_position_gauge.py --out /tmp/line-position-gauge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 tools/synthetic_sync/line_position_gauge.py --analytic \
  --baseline-dir /tmp/line-position-gauge --out /tmp/line-position-gauge-analytic
```

The JSON output contains exact input hash, objective minima, per-BA evaluation
counts and timing, candidate and final snapshots, independent assessments,
all-residual shift checks, and baseline/prototype/repeat comparisons. The
analytic-projection sample and degeneracy checks run before its solve schedule.
The script retains the historical finite-span sampler as an explicit baseline
even after the production pinhole projector is fixed; a fresh `baseline` run
does not silently become the new production algorithm. The script does not
load Blender or any private project.

After the production fix, a no-solve routing check confirmed that the current
pinhole projector returns a line for the +100 shift, the diagnostic historical
baseline still returns `None`, and the analytic diagnostic returns a line.
`tests/test_sync_projection.py` first failed three targeted cases on the old
implementation, then passed four cases after the fix, including tilted camera
and similarity rotations together and a frozen nonzero-distortion sample.
Focused line, plane, mirror, and projection tests passed 33/33. No additional
full synthetic benchmark solve was run after the production change.


## Integration validation and replay safeguards

The production correction landed as `c304fbe`. Main validation passed **349 tests,
9 skipped, 260.811 seconds**, plus Blender 5.1.0 smoke. The benchmark tables above
remain observations from the original prototypes, not a newly rerun production
benchmark.

The recorded stage and whole-solve timings include diagnostic residual-shift
probes. Those probes bypass optimizer-facing evaluation counters but consume
time, with different cost under sampled and analytic projection. These are
instrumented timings, not a clean runtime comparison. Future driver runs record
probe time separately; even the subtracted time retains ordinary tracing overhead.
Optimizer-facing residual/Jacobian call counts exclude those diagnostic probes.

Replay rejects mismatched or absent request fingerprints, reports lost/gained
camera, point and line support, and exits nonzero when a scheduled run fails its
independent accuracy assessment. A negative optimization result remains valid
when those unchanged contracts pass; no speed or improvement gate is imposed.
