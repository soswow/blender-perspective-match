# Focal-bound optimization and crop controls

Checkpoint: 12 September 2026, following `7a2cc6c`.

## Observed optimizer defect

The joint LM step previously solved for every focal, pose and landmark and then
clipped out-of-range focals. The remaining coordinates still followed a step
computed for an unavailable focal change. Repeated damping could stall at a
substantially worse objective than a bounded optimization method.

The independent linear regression minimizes `(x+y-3)^2 + (x-2)^2` with `x<=1`.
The unconstrained solution is `(2,1)`; clipping gives `(1,1)`, whereas resolving
the other coordinate at the bound gives the constrained solution `(1,2)`.
`tests/test_focal_optimizer.py` also covers upper-bound freezing, lower-bound
release, ordinary unconstrained LM agreement and cancellation between re-solves.

`core/focal_optimizer.py` returns feasible active-set damped steps. The outer
loop still accepts only objective-reducing steps. This is not a general exact
bounded quadratic-program solver or a guarantee of the global camera solution.
No focal range, geometry, noise or uncertainty acceptance gate was relaxed.

## Saved-objective comparison

One read-only private capture supplied inputs and partial initial registration.
Subsequent trials reused that startup, including the existing provisional-pose
completion; no repeated full Sync or camera application was needed. These are
same-input optimizer comparisons, not independent geometry certification.

| Method / bounds | Weighted squared residual | Point RMSE | Line RMSE |
| --- | ---: | ---: | ---: |
| Previous clipped LM / ±75% focal | 33488.84 | 14.01 px | 9.42 px |
| SciPy TRF / same bounds | 17566.58 | 9.81 px | 7.96 px |
| New NumPy step / same bounds | 17566.58 | 9.81 px | 7.96 px |
| SciPy TRF / absolute 5°–130° FOV | 7449.40 | 6.46 px | 4.55 px |

The integrated NumPy path completed 20 iterations in about 1.5 seconds versus
100 iterations / about 7.1 seconds for the old bundle replay. Startup is excluded;
these single-run timings are not a general performance benchmark. Every bounded
endpoint still hit focal limits. Broad-range fitting approached very narrow
FOVs and did not establish correct geometry. A same-span constraint-removal
control reduced point RMSE to 8.28 px but still hit focal bounds; it does not
prove that any particular constraint or pick is wrong.

`compare_focal_optimizers.py` preserves endpoint camera/point data, separate
point/line/prior residuals, depth signs, bound hits and source/input hashes.
Its one-shot local-variable trace is intentionally research tooling: it must
be maintained when the production parameterization changes. SciPy is required
only by the comparison tool. Private inputs and outputs are not repository
fixtures. `probe_focal_startup.py --joint` also retains the production endpoint
on bound/non-convergence refusal, without making it an applicable result.

## Independent crop control

`focal_crop.py` crops the existing live-reference oracle by a known asymmetric
pixel rectangle. Physical cameras and 3D stay fixed; picks, withheld pixels and
principal points translate by exactly the crop offset. All checked observations
remain inside the crop. The paired wrong-calibration case changes only the
input principal points back to image centers.

Correct shifted intrinsics and guessed focals recover withheld projections
within 0.1 px (observed about 0.000021 px before the bounded-step integration).
The centered-principal-point control refuses at a focal bound. This isolates
a plausible failure mechanism; it does not diagnose an arbitrary real crop.
The test uses oracle pose/point startup, not fresh registration, and does not
estimate unknown crop offsets. It also does not cover distortion or deforming
objects.

## Verification and next decisions

The integrated change passes 80 focused numerical tests, including point,
line, plane, live-mirror, weak-evidence and crop controls, plus the two existing
16/32-camera oracle-start capacity checks. A fresh-registration
Blender line fit/apply check reaches about 0.000414 px maximum withheld error.
Native operator callback/RNA checks cover textual point-FOV activity, ordinary
progress normalization and the existing apply/refusal lifecycle. They substitute
window-manager scheduling rather than automating mouse interaction.

Next establish whether original frames or known crop metadata can supply the
fixed principal points. Unknown-offset fitting would need separate weak-case
and withheld-observation experiments. Keep rejected candidates diagnostic until
there is an explicit reversible preview workflow; a lower residual is not a
claim that the cameras are closer to reality.
