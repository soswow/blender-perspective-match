# Initial pilot observations — 2026-09-10

The nine exact-data families passed numerical accuracy checks. Their 36 ordered,
reversed and cached variants passed too. All nine families also passed actual
Blender scene collection/application and fresh-process save/reopen on Blender
5.1.0 on macOS. The PR workflow targets the existing release pin, Blender 5.1.2
on Linux; it still needs its first hosted run.

With independent 0.3 px Gaussian noise per pick coordinate, seeds 0–2 produced
23 passing cases and four flags against the provisional limits:

| Case | Fitted RMS | Flag |
| --- | --- | --- |
| `free_scale-0` | 0.247 px | Anchor center after global cloud alignment: 2.38% of object diagonal, limit 2% |
| `free_scale-1` | 0.182 px | Anchor center after global cloud alignment: 2.10%, limit 2% |
| `overhead-0` | 0.537 px | Overhead camera's withheld RMS: 1.909 px, limit 1.8 px |
| `overhead-2` | 0.571 px | Overhead camera's withheld RMS: 1.843 px, limit 1.8 px |

These are **investigation cases, not confirmed solver defects**. The free-scale
flags depend on fitting one global similarity to a noisy reconstructed cloud;
they do not mean the locked anchor moved during solving. Their withheld pixel
checks pass. The overhead flags expose a small difference between fitted-pick
error and predictions elsewhere on the object. None establishes a catastrophic
failure or a statistically justified product accuracy limit.

Reproduce and retain exact case JSON:

```sh
python3 tools/synthetic_sync/run.py --family all --count 3 --noise-px 0.3 --out /tmp/pm-noisy
python3 tools/synthetic_sync/run.py --case /tmp/pm-noisy/overhead-0.json --out /tmp/pm-overhead-replay
```

The first investigation should vary small amounts of pick noise around the
saved overhead case, inspect where the withheld error occurs and measure how
much one well-placed additional observation helps. If different reasonable
solutions predict withheld geometry equally well, examine uncertainty and the
accuracy contract before changing the optimizer. Keep this exploratory sweep
outside required CI until the expected behavior is established; do not loosen
the limits merely to turn these four cases green.

Harness validation also caught two mistakes in the new tooling itself: retained
RNA element references became stale while growing a Blender collection, and
JSON constraint lists needed conversion to the solver's tuple contract for cache
keys. Both were corrected in the harness. The scene-input comparison and cached
replay now cover those paths. No production solver changes were made.
