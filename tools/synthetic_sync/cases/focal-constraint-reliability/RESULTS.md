# Point-FOV constraint reliability on weak evidence

This frozen pilot asks whether point-plane and supplied point-mirror relations
remain usable with shorter baselines, shallow depth, and incomplete overlap.
`focal_constraint_reliability.py` generates ten four-camera controls with
identical same-family picks: cameras span 4 world units laterally at about 7
units depth; every point has two or three picked views and every camera has
11–12 picks. The independent pinhole oracle checks frusta, not occlusion.
Training landmarks and eight withheld landmarks are disjoint. Pick coordinates
have deterministic 0.35 px Gaussian perturbations; 0.35 px is also the stated
per-coordinate noise in fitting. Non-anchor stored poses and focals are wrong.
The original `mirror-tilted-hard.json` is copied byte-for-byte as an eleventh,
stronger-baseline orientation-conflict control.

All calls used Python 3.12.6, NumPy 2.0.1 and OpenCV 4.13.0, with
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1` and an outer 720 s timeout per batch.
The POSIX ledgers reserved every attempt before execution. Across `run-01`
through `run-04`, **12/12 fresh inner Sync calls and 18/18 bundle calls** were
used; all completed. Inner Sync took 203.18 s total and bundles under 1.4 s.
Source/runtime manifests, source archives, exact requests and numerical
results are retained in each run directory. `run-01` is pre-fix; `run-02` and
`run-03` cross archived starts without new Sync; `run-04` is the fixed public
path. No trial parameter or acceptance threshold was tuned between cases.

| Frozen case | Pre-fix public result | After removal-start bundle | Withheld shape RMS if accepted |
| --- | --- | --- | ---: |
| Weak axis hard | Refused: bundle did not converge (2.894 px start) | Accepted 0.232 px | 1.04 px |
| Weak axis removed | Accepted 0.218 px | — | 1.42 px |
| Weak axis with 0.04-world wrong member | Refused: did not converge | Refused: camera fit deteriorated | — |
| Weak Free hard / removed | Both accepted 0.234 / 0.230 px | — | 2.26 / 2.46 px |
| Weak mirror hard / removed | Both accepted 0.231 / 0.213 px | Hard accepted at 0.231 px | 1.66 / 2.35 px |
| Weak mirror tilted 2.29° | Refused: pick-noise inconsistency | Same refusal | — |
| Weak mirror with 1.43° stored-anchor error / removed | Both accepted 0.273 / 0.213 px | Constrained result reproduced | 1.24 / 2.35 px |
| Original stronger mirror tilted ~10° | Refused: camera fit deteriorated | — | — |

The hard-axis case isolates a start problem. Its pre-fix constrained Sync
reports success but yields 2.894 px start error; the same-pick removal starts
at 0.490 px. Crossing those saved world states shows that the removed start
with the **hard-axis final bundle** accepts and satisfies the plane to
8.5e-9 world RMS. The hard-axis start still fails with the relation removed
from the final bundle. The narrow product fix therefore registers cameras
from point picks alone in the independent-FOV route, then applies the original
plane/mirror relations in the single final joint fit. `run-04` verifies that
exact public route: 0.232 px fit, 1.04 px withheld shape RMS, all four focal
intervals containing their independent truth and the same hard plane gap.
It is the only fresh post-fix call in this pilot.

The correct mirror reached the same final fit from its relation-free start to
about 1e-5 px. The slightly tilted mirror and wrong-axis member still refuse
from the new start. The wrong stored-anchor control reaches the same accepted
fit from its relation-free start: 15.12 px raw-frame withheld RMS versus 1.24
px after **one** positive global similarity fitted solely on training points.
That gap illustrates the fixed-anchor frame assumption; it is not evidence of
unobserved shape distortion. `run-01` and `run-04` reassessments save the full
training-only alignment rotation, translation and scale for exact reuse. A
unit oracle control proves a single rigid/scale transformation gives near-zero
shape holdout error while changing one camera's focal and depth remains visible.

Every accepted case's four *conditional* 95% focal intervals contained truth
in this one noise draw. These rows do not establish interval calibration or
reliability across many scenes. The relation-free plane cases have an arbitrary
global scale, so raw world-frame error is not a fair accuracy comparison with
a supplied offset mirror plane. Plane and mirror priors can be wrong; a
low pixel error or reported interval cannot certify them. The weak wrong
references here were refused, while the wrong anchor frame was accepted as
an explicit input-condition limitation. The 0.04-world bad plane member and
2.29° mirror tilt are only two examples of erroneous references.

The focused automated regression uses the frozen weak-axis request through
the real public path. It fails before this change and accepts afterward;
the frozen `run-01` and `run-04` ledgers provide the before/after numerical
records. Running that test again is a new solve and belongs to the final
integration verification budget, outside this exhausted pilot.

## Integrated verification

The main checkout passed 423 numerical tests in 428.718 seconds with two skips
for absent optional local YAML files. This includes the real public weak-axis
regression. Existing Blender verification passed a fresh `axis-hard-exact` and
`mirror-hard-offcenter-exact` fit/application and read-only reopen. Native
withheld RMS was 0.000280 and 0.000201 px, with no accuracy flags. These three
new public Sync/bundle pairs are post-integration verification, separate from
the 12/18 pilot cap; normal existing-suite solves are separate too. Reopening
uses zero solves. Logs and generated scenes are retained locally under
`.local/focal-reliability-validation/`; no private scene was opened or saved.
