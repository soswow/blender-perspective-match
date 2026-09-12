# Unknown focal lengths from 2D-only matches — bounded continuation

## Frozen plan (before numerical calls)

The existing eight exact three-view cases are fixed inputs. Their cameras have arbitrary stored private poses, no VP lines, no ground/Known 3D, no pose locks, square pixels, centered principal points and zero distortion. The saved shared- and mixed-focal true-K runs are positive registration controls and will not be repeated. First run the shared guessed-K (all focals 25% high) and mixed guessed-K (independent 25% high, 20% low and 15% high) cases through ordinary Sync. Then run weak-baseline and pure-rotation true/guessed-K cases to check whether low fitted residuals conceal ambiguity. If useful, trial only a few common focal scales through the same Sync path; any choice must use input-side support and reprojection, never synthetic truth. An independent-focal prototype needs evidence that the baseline failure is a recoverable focal error rather than a registration failure or unobservable geometry.

Every exploratory inner Sync call, including failed or repeated variants, is reserved in one persistent ledger with a 16-call maximum, 180 seconds per call and 900 cumulative active seconds. An outer process timeout also applies. Each candidate key includes complete numerical core and relevant harness hashes, runtime versions and thread settings. Focused regression solves, if a confirmed defect warrants them, are recorded separately.

Success requires all three cameras and 3D point support, focal relative error within 2% for each camera, accurate cameras and points under one proper global similarity, and withheld object reprojection below 1 px. Report low-RMSE failures as failures. Pure rotation is a depth-unobservable control; an accepted finite-depth reconstruction cannot be certified by exact 2D picks. Weak-baseline results may be sensitive, so report actual errors and refusals without claiming physically impossible perfect recovery. Stop after a timeout, depleted ledger, or evidence that a proposed candidate cannot be chosen without truth. Do not alter the frozen geometry or oracle thresholds.

## Results — 15 of 16 exploratory calls

All 15 attempts completed under one ledger, consuming 46.43 of 900 allowed active seconds. Each used `~/venvs/my/bin/python` with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` and an outer process timeout. The numerical core was unchanged throughout. The harness gained the actual API wrapper after the first ten calls, so each attempt's key records its own complete source/runtime identity. The frozen shared/mixed true-K controls from the preceding bootstrap repair were read, not rerun; both had withheld RMS below `1e-6` px.

The unmodified Sync baselines recovered all three cameras and 23 landmarks in every case:

| Frozen case | Fitted RMSE | Maximum withheld RMS | Maximum center error / object diagonal | Outcome |
| --- | ---: | ---: | ---: | --- |
| Shared guessed K (all 25% high) | 0.489 px | 2.815 px | 0.513 | Accepted; inaccurate |
| Mixed guessed K (+25%, −20%, +15%) | 0.487 px | 2.688 px | 0.507 | Accepted; inaccurate |
| Weak translation, true K | `1.14e-11` px | `<1e-6` px | `<1e-6` | Accurate |
| Weak translation, guessed K | 0.010 px | 1.574 px | 0.538 | Accepted; inaccurate |
| Pure rotation, true K | `6.59e-6` px | 17.465 px | 0.776 | Accepted; depth unobservable |
| Pure rotation, guessed K | 0.0011 px | 11.895 px | 0.867 | Accepted; depth unobservable |

The exact weak-translation true-K control is recoverable, but its 25%-high focal counterpart is not accurate. Pure rotation cannot triangulate 3D at any focal: the near-zero residuals there are not evidence of object depth. In the accepted output, maximum reconstructed point-ray separation was about `0.00008°` for true-K rotation and `0.73°` for guessed-K rotation, versus `1.52°` for the weak-baseline true-K case. These observations do not establish a safe product cutoff between weak and degenerate scenes; a threshold chosen between those particular numbers would overfit the controls. No production acceptance rule was changed.

Four separately ledgered manual common-scale probes on the shared guessed-K case tested `0.75`, `0.85`, `1.15` and then `0.80`. The first three were a small input-side bracketing check; `0.80` was selected as the midpoint of the two lower-residual candidates (`0.75` and `0.85`), without using focal truth. It happened to equal the synthetic true scale and passed every independent check with fitted RMSE `1.53e-7` px. That single midpoint is a control, not evidence of general focal-search convergence.

The final five calls exercised **the actual `refine_lenses_from_landmarks(..., share_lens=True)` implementation**, with the generated no-VP camera inputs and a deliberately small search of three coarse and three fine samples over an explicit ±25% window. It evaluated scales `1`, `0.75`, `1.25`, `0.703125` and `0.796875`; the product score selected `0.796875`. The selected focal error was 0.39% in each view, maximum withheld RMS was 0.056 px, maximum aligned center error was 0.008 object diagonals, and all required points and cameras passed. It retained all three cameras and 23 landmarks. The default sidebar window is ±18%; its coarse range would not include `0.80` from this 25%-high starting guess. This is evidence that the existing shared-scale route can work **when configured to search the needed range and the starting focal ratios are correct**.

The mixed case needs independently changing ratios; Same Lens preserves the three biased starting ratios, while the current independent mode freezes no-VP focals. No independent focal prototype was justified by one remaining exploratory call. The five-call actual shared route and frozen matrix separate the supported local correction from this open product capability. Focal truth was never given to the solver or search, and candidate selection did not use the independent oracle. The present fitted-RMSE status can still certify inaccurate geometry for guessed K or ambiguous camera motion; deciding how to communicate uncertainty and how to search three unrelated focals needs a separate bounded design and validation step.

Raw complete requests, numerical records, assessments, source/runtime keys and call accounting are in [`cases/unknown-focal-continuation/ledger.jsonl`](cases/unknown-focal-continuation/ledger.jsonl). Individual Sync assessments and the complete [`actual-shared-search.json`](cases/unknown-focal-continuation/actual-shared-search.json) preserve the selected trial and all alternatives. The local runner is [`unknown_focal.py`](unknown_focal.py). No numerical regression test or product code change was made, because this phase established an unsupported route and an observability boundary rather than a narrow solver defect with a justified fix.

To inspect saved results without spending a numerical call:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venvs/my/bin/python \
  scripts/run_unittests.py test_synthetic_no_vp_bootstrap test_synthetic_budget
```
