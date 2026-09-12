# Lens initialization trial

Production revision: `be69b73dd83e883ba28dd3fee31281b256b2ca92`. Investigation
date: 12 September 2026. The reproduction below describes the baseline;
the subsequent lens-only fix is recorded at the end.

Replay the preserved input from the repository root:

```sh
python3 tools/synthetic_sync/lens_initialization.py \
  --case tools/synthetic_sync/cases/lens-refused-initialization.json \
  --out /tmp/pm-lens-initialization
```

Use an empty output directory. The script writes the exact case, three complete
request snapshots and result/assessment JSON. Exit zero means the investigation
ran, not that every solve was accurate: the warm refusal is the observed defect.
The paired requests are checked for equal inputs except their initial transforms;
the candidate focal values are checked against independent truth. This is a
bounded doubled-focal experiment, not a general lens-case runner.

The seed-0 paired trial confirms a causal initialization failure. Starting from
`lens_case("refused_start")`, the only scenario edit was removal of the explicit
pose locks. The saved true-focal warm and cold `SyncSolveRequest` records have
identical evidence and numerical settings except for `initial_similarities`.
The candidate focal values exactly equal the independently generated truth.

The unlocked doubled-focal baseline refused at 58.508617 px and returned an
exact identity similarity for all three cameras. Reusing those identities at
the true focal refused at 127.427223 px. A cold solve of the same true-focal
request succeeded at 0.000000033 px, with all supported picks at 0.000000033 px.

Independent withheld-camera RMS was 0, 146.824249, and 277.294935 px for the
warm refusal, versus 0, 0.000000092, and 0.000000008 px for the cold success.
The warm values describe the solver's refusal placeholder identity transforms;
they are not accepted bad camera geometry.

Commands:

```sh
python3 -m py_compile tools/synthetic_sync/lens_initialization.py
git diff --check
python3 tools/synthetic_sync/lens_initialization.py \
  --out .local/lens-initialization/paired
python3 tools/synthetic_sync/lens_support.py --kind recovery \
  --out .local/lens-initialization/control-recovery
python3 tools/synthetic_sync/lens_support.py --kind refused_start \
  --out .local/lens-initialization/control-refused-start-locked
python3 tools/synthetic_sync/lens_support.py --kind refused_improving \
  --out .local/lens-initialization/control-refused-improving
```

Controls behaved as documented. Recovery selected scale 0.875 and reduced all
supported-pick RMS from 8.372447 px to 0.000000045 px. Locked refused-start
selected scale 0.5 and changed refusal at 118.188764 px to success at numerical
zero. Refused-improving selected scale 0.875 and remained refused while reducing
the reported RMS from 118.188764 px to 88.641573 px.

This is one deterministic seed and 12 numerical solves total: three paired
solves plus three three-trial controls. It covers imported-style calibration,
Known 3D points, exact synthetic picks, and a shared-focal true candidate. It
does not establish behavior for VP reorientation, distortion, noisy picks,
per-match focal search, Blender application, or other graph topologies.

The coordinating agent reviewed the script and registration contract, added the
explicit true-focal input check, then replayed the saved case in a fresh process
through the final CLI. All three outcomes and metrics reproduced (15 solves
including that integration replay); the cold result passed the independent
oracle. The existing full numerical/Blender suite was not repeated for this
diagnostic-only change.

Mechanism: `core/lens_refine.py` forwards the incumbent result's similarities
to trial solves. `core/sync/solve.py` can return identity placeholders when a
solve is refused. `core/sync/pose.py` treats supplied initial similarities as
already registered, skipping fresh pairwise registration for those cameras.
The paired experiment demonstrates the consequence at the actual true candidate;
it does not test a changed end-to-end focal search.

The bounded fix needs to separate usable estimated poses from refusal
placeholders before reuse while preserving explicit pose locks, successful warm
starts and useful partially improving refusals. This is a result-contract
decision: do not infer that every unsuccessful numerical API returns unusable poses.

This was the first selective-escalation trial: one Sol/high worker with fresh
context, followed by main-thread contract/diff review and one exact-case replay.
No worker production edits were needed. Model usage totals were not exposed in
the handoff, so this establishes a reviewable workflow, not measured token savings.

## Lens-only fix follow-up

The current Sync refusal constructors all return identity placeholders and no
reconstructed landmarks. The lens search's common evaluation path now reuses
poses only from a successful incumbent. Explicit fixed poses are forwarded
separately on every trial. Refusal scoring and selection are unchanged, so
useful focal improvements can still be retained even if Sync continues to refuse.
If Sync later exposes useful poses in unsuccessful results, introduce an explicit
pose-usability contract rather than relying on the present success flag.

The frozen-case outer-search regression fails on `be69b73` (no improvement,
selected refusal at 58.508617 px). With the fix it selects a successful result at
0.000000033 px, and the independent withheld-object oracle passes. All 15 focused
lens tests pass, including ordinary focal recovery, locked refused-start recovery,
useful improving refusal and support-retention routing. The change covers shared,
per-match and coupled search call sites through the same evaluation function;
the new independent numerical regression exercises shared search.

The paired warm/cold diagnostic still demonstrates the underlying Sync seed
semantics after this fix: callers supplying placeholder poses directly can still
get the warm refusal. The fix prevents the lens search from supplying them.
It adds no extra trial evaluations, but refused-start trials now perform fresh
registration, so those searches can cost more than reusing placeholder poses.

Combined validation with the recovered-line rebuild fix: 329 numerical tests
run, 9 skipped, passed in 241.602 seconds; final Blender smoke passed on 5.1.0.
The 14 Blender lens-ownership controls passed after lens integration and before
the point-ID fix. No user file was saved.
