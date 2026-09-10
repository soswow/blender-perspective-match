# Evidence-placement follow-up — 10 September 2026

**Decision: do not add automatic picking advice or change solver weights on this
evidence.** An outer point helped modestly; it did not meet the experiment's
predeclared threshold for promoting a placement rule.

The two overhead fixtures already contain every visible training pick. A new
free landmark therefore costs three observations: two supporting views and the
overhead view. We compared two predeclared positions (outer surface versus
central raised surface), each with a two-pick support-only control. The control
isolates the marginal overhead observation without pretending that supporting
measurements are free. All measurements carry independent 0.3 px Gaussian noise;
none receives true Known 3D coordinates. Withheld geometry stays unchanged and
is disjoint from both new candidates.

Across 16 layout/noise comparisons:

| Variant | Median withheld RMS | 90th percentile | Improved versus paired baseline | Median paired error ratio |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.062 px | 1.435 px | — | — |
| Outer surface, three picks | 0.937 px | 1.032 px | 12/16 | 0.902 |
| Central raised surface, three picks | 1.066 px | 1.433 px | 9/16 | 0.988 |

All fresh comparisons retained every camera and passed the original provisional
accuracy limits. The outer point reduced paired error by a median 9.8%, below
the predeclared 20% requirement. It met the other requirements (at least 70%
improving, no lost cameras, no worse 90th percentile). The central point's median
paired improvement was only 1.2%. This is a small descriptive experiment: eight
noise seeds are reused across two nearby camera layouts, so these are not 16
independent scene trials.

The two originally flagged draws are separate from that comparison. Their
withheld errors changed from 1.909 → 1.786 px and 1.843 → 1.276 px with the outer
point. Those selected successes overstate its average benefit. Halving each
original pick-error vector produced 0.954 and 0.923 px; removing noise recovered
the geometry to numerical precision. This is consistent with ordinary noise
sensitivity, not evidence of a persistent wrong solution at perfect input.
It does not exclude optimizer issues in other regimes.

The study made 94 numerical solves. Five focused experiment tests passed, and
the first augmented noisy case passed the actual Blender path and save/reopen.
The reusable outputs are `evidence.py`, frozen JSON inputs in `cases/`, paired
controls, noise resampling/scaling, per-height errors and exact replay artifacts.

```sh
python3 tools/synthetic_sync/evidence.py --trials 8 --out /tmp/pm-evidence
```

The next useful expansion is **constraint-dependent cases**. Current mixed
cases contain enough point evidence to solve even if a line or mirror relation
contributes little. Build cases where removing the intended constraint causes
an independently explainable loss of recoverable information, and compare
reconstruction as well as camera projection. Preserve the existing visibility
and withheld checks. This tests more consequential boundaries than tuning the
current mild noise flags or adding more nearly identical random seeds.
