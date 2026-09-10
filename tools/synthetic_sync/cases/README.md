# Frozen investigation cases

`overhead-0.json` and `overhead-2.json` preserve the exact original noisy inputs,
expectations and independent truth from the first pilot. They were generated
with geometry seeds 0 and 2 and 0.3 px noise, then captured before changing the
generator. They contain only synthetic geometry.

These cases cross provisional withheld-error limits on the current solver.
They are not yet confirmed solver bugs and are not expected-to-fail unit tests.
The evidence-placement experiment uses them as fixed inputs and reports both
the originally flagged draws and fresh noise draws. Keep these JSON files fixed;
add a new case if the intended evidence changes.

Replay either case independently:

```sh
python3 tools/synthetic_sync/run.py --case tools/synthetic_sync/cases/overhead-0.json --out /tmp/pm-overhead
```

A nonzero exit means its accuracy contract failed. Inspect the result and
withheld-object report before concluding that a production fix is needed.
