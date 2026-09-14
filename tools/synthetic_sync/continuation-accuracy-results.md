# Generated continuation accuracy controls

The noisy composite Refine → Solve → Solve fixtures preserve camera, point,
line, relation and support consistency, but their original independent 1px
withheld-point target fails: fixed mirror 2.757px, free scale 2.937px and live
mirror 3.140px. The flag remains in the assessor; low fitted pick error is not
accepted as a substitute for independent accuracy.

The fixed-mirror fixture has four identical oracle cameras across its composed
generators, 108 point picks, 12 line strokes and eight disjoint withheld points.
Its point-pick radial noise is 0.266px RMS. The unperturbed 3D truth satisfies
the separately checked X/Free planes, parallelism and mirror exactly. The
fixed off-anchor mirror plane supplies metric scale. The assessor evaluates the
fixed world frame without an alignment rotation or scale.

Two fixed-focal controls used that exact fixture and source-frozen archives.
The oracle-start control began at true K, true camera poses, points and lines,
with the original noisy evidence and its certified effective weights. The
noise-free control began at the previously fitted, biased geometry and true K,
replaced each pick/stroke with its oracle pixel, and recomputed weights because
the evidence changed. Both wrapper and inner bundles accepted, retained every
camera/point/line and relation, and made no Sync registration call.

| Control | Start objective | End objective | Start withheld RMS | End withheld RMS |
| --- | ---: | ---: | ---: | ---: |
| True geometry, original noisy picks | 8.843206 | 5.012814 | 0.000015px | **2.797296px** |
| Biased geometry, exact oracle picks | 9.982591 | 0.000000143 | 2.757018px | **0.000051px** |

The noisy oracle-start endpoint agrees with the earlier truth-K fit from the
biased endpoint within 8.1e-6 in recorded camera/geometry coordinates and
0.00018px withheld RMS. That paired result separates this failure from focal
estimation and from a local-minimum trap. The exact-pixel control recovers the
oracle from the biased endpoint, excluding a fixed model/oracle offset for this
case. The noisy estimator legitimately lowers its training objective while
moving away from truth. It ends at 0.213px point RMSE and 2.797px withheld RMS.

A read-only training-point rigid-rotation diagnostic measures 0.294° frame
drift. It reduces training 3D RMS from 0.02558 to 0.00983 world units and
withheld RMS from 2.797 to 1.243px. That rotation moves the supplied X and
mirror normals by only 0.150° and 0.087°; the normals are about 13.8° apart.
This quantifies sensitivity of the weakly oriented fixture to small pixel
noise. **The rotation is forbidden by the fixed world constraints and is not
used to pass the accuracy check.** Residual camera/geometry error remains even
after this diagnostic alignment.

The defensible fixture checks are exact-oracle validation, full current-evidence
support and relation checks, the original raw noisy withheld flag, convergence
from true geometry to the same noisy objective, and recovery of subpixel
withheld accuracy from a biased start when the picks are noise-free. The 1px
target remains a useful noise-free check, but this single noisy composite does
not establish it as a guaranteed bound. Noise-free free-scale and live-mirror
variants were not run, so this causal conclusion is specific to the fixed case.

Archives:

- `/tmp/pm-continuation-accuracy-controls-20260915`: noisy oracle start,
  ledger, exact input/output/source, and read-only frame audit. Source SHA-256
  `12978f4021bec5e9be5e81f7105ec7fd99650968f5a005dfdb6edf73143ef309`.
- `/tmp/pm-continuation-accuracy-clean-20260915`: exact-pixel control,
  ledger and exact input/output/source. Source SHA-256
  `2d88e82fac804963a7bd84034379174f6314f0448f2e438df7e20f89d3a3370d`.
- `/tmp/pm-continuation-known-focal-captured-20260914`: independent prior
  truth-K noisy control from the fitted start.

Commands, from the repository root with `OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`:

```sh
/Users/sasha/venvs/my/bin/python tools/synthetic_sync/continuation_accuracy_control.py --source /tmp/pm-continuation-joint-mirror-certified-20260914 --out /tmp/pm-accuracy-replay-noisy --mode oracle_noisy
/Users/sasha/venvs/my/bin/python tools/synthetic_sync/continuation_accuracy_control.py --source /tmp/pm-continuation-joint-mirror-certified-20260914 --out /tmp/pm-accuracy-replay-clean --mode endpoint_clean
/Users/sasha/venvs/my/bin/python tools/synthetic_sync/continuation_accuracy_control.py --source /tmp/pm-continuation-joint-mirror-certified-20260914 --out /tmp/pm-accuracy-replay-noisy --mode frame_audit
/Users/sasha/venvs/my/bin/python scripts/run_unittests.py test_sync_continuation_assessment
```

Each numerical call was reserved and captured by `ExperimentBudget` before
entry, capped at 120 seconds per bundle and 130 seconds per process. The two
completed bundles used 0.431 and 0.458 active seconds; zero Sync calls and two
of four authorized bundles were consumed. A pre-call attempt to reuse certified
noisy weights with changed exact picks was rejected by the evidence-identity
gate; it used no bundle call. The revised exact-pixel control correctly
recomputes its weights. The read-only frame audit used no solve. Six focused
assessment tests pass; no product solver code or acceptance threshold changed.
