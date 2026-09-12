# Lens search coverage experiment

Run the real numerical search and retain exact inputs, every trial, ordinary
Sync output, per-match diagnostics and independent withheld geometry:

```sh
python3 tools/synthetic_sync/lens_support.py \
  --case tools/synthetic_sync/cases/lens-recovered-score.json --out /tmp/pm-lens-score
python3 tools/synthetic_sync/lens_support.py --kind recovery --out /tmp/pm-lens-recovery
python3 tools/synthetic_sync/lens_support.py --kind refused_start --out /tmp/pm-lens-refused
python3 tools/synthetic_sync/lens_support.py --kind refused_improving --out /tmp/pm-lens-partial
```

The output directory must be empty. The command records numerical outcomes;
accuracy flags remain observations rather than an exit-status gate. The noisy
view deliberately exceeds the ordinary held-out accuracy contract. Its two
clean cameras must remain accurate; no thresholds are relaxed for the full case.
`tests/test_lens_refine.py` enforces these narrower regression requirements and
checks all successful recovery controls against the ordinary independent oracle.

## Confirmed defect

The frozen case has three initially exact lenses, independently constructed
Known 3D points, and exact picks on two cameras. Only the third camera has
seeded 30 px pick noise. The Same Lens search uses the normal ±18% window with
three coarse samples and no fine samples, enough to expose the comparison.
No numerical solver or camera result is substituted.

On `fd0633f`, a scale of 0.82 became an apparent improvement when the noisy
camera changed from joint registration to recovery. Its residual remained in
per-match diagnostics, but disappeared from the ordinary Sync headline.

| Measurement | Initial | Old selected | Fixed selected |
| --- | ---: | ---: | ---: |
| Shared focal multiplier | 1 | 0.82 | 1 |
| Sync headline RMS, px | 19.9892 | 12.9042 | 19.9892 |
| All supported point picks RMS, px | 19.9892 | 22.8831 | 19.9892 |
| Clean anchor withheld RMS, px | <0.00001 | 20.4788 | <0.00001 |
| Other clean camera withheld RMS, px | <0.00001 | 1.8549 | <0.00001 |
| Noisy camera withheld RMS, px | 18.5065 | 18.7662 | 18.5065 |

All three cameras and their point picks stayed present. This demonstrates loss
from the scored evidence set, **not actual camera deletion**. A bounded local
sweep of noise and initial focal bias also found initially skipped cameras,
but did not demonstrate an accepted trial that deleted a previously posed
camera. These are synthetic contradictory picks, not a new accuracy claim for
high-noise measurements.

The lens-only fix scores every supported point pick with the existing production
projector, including recovered cameras. This agrees with the independent
projector on the fixed evidence above. It does not depend on per-match display
diagnostics, which can mix point and line errors when the joint point set is
empty. It does not alter Sync's joint acceptance or headline, introduce line error into mixed point/line pose
acceptance, or change confidence weights. A separate comparison guard requires
an incumbent successful solve's posed cameras and reconstructed points/lines to
remain present and the candidate to remain successful. Initially disconnected
cameras impose no new requirement; recovery can add support. Newly accepted
support is retained in later comparisons. Nonfinite or behind-camera supported
point projections cannot win a search. Missing or nonfinite display diagnostics
do not affect a well-defined geometric score.

## Positive controls and limits

- A shared 12.5% correction recovers biased starting lenses: supported-pick RMS
  falls from 8.37245 px to below 0.000001 px, and all cameras pass independent
  withheld geometry checks.
- With true camera poses explicitly locked and starting focals doubled, an
  initially refused solve recovers at scale 0.5: reported RMS falls from
  118.18876 px to effectively zero, with all independent checks passing.
- Restricting that locked case to ±12.5% preserves the intentional partial
  outcome: RMS improves from 118.18876 to 88.64157 px, lenses change, and Sync
  still refuses. The change does not blanket-reject unsuccessful candidates.

The pose locks in the refused-start controls isolate lens scoring. An exploratory
unlocked doubled-focal case did not recover even at the true focal candidate;
the refused result's identity transforms were passed back as warm starts. The
[initialization follow-up](lens-initialization-results.md) now confirms that
reusing these placeholders causes a true-focal refusal while fresh registration
succeeds on identical evidence. The subsequent lens-only fix uses fresh
registration after refusals and reuses successful poses; this is separate
from the scoring fix. These controls do not establish general
failed-start recovery.

Fourteen focused tests pass locally. The real regression fails by assertion with
the old lens module. Wiring controls exercise shared/coarse/fine, per-match and
coupled acceptance, including camera/point/line loss and success-to-refusal.
Those wiring controls use fake results and establish routing, not geometry
failure size. The real searches use imported-style calibrations without VP
lines; per-match VP accuracy, distorted imagery, live Blender application and
large/noisy sweeps remain outside this experiment. Existing VP guardrails and
line-only scoring remain in place.
