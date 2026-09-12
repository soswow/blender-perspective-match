# Development

## Development install (edit → reload)

Point Blender's installed extension at this git checkout with a symlink (no ZIP copy):

```sh
./scripts/link-dev.sh
```

That replaces `~/Library/Application Support/Blender/5.1/extensions/user_default/match_perspective` with a link to this repo (and downloads OpenCV wheels if missing). Enable **Perspective Match** once if it is not already on — **disable then re-enable** after the first link so Blender extracts the OpenCV wheel. The add-on still enables without OpenCV; **Detect VP Lines** / **Find AprilTags** / **Snap to AprilTag** stay hidden and Info logs that the wheel is missing.

Daily loop:

1. Edit and save in the editor.
2. In Blender: click **Reload Perspective Match** at the bottom of the sidebar (or **F3 → Reload Perspective Match**). That control is hidden for zip installs; it only shows when this checkout is linked. Prefer it over System → Reload Scripts—that often leaves panels and PropertyGroups on stale class objects.
3. Test. Watch the system console for `Perspective Match: reloaded from disk` (the button only queues the reload; the tear-down runs a moment later so Blender does not crash).

Exit any running Draw / Pick Origin modal before reloading. After adding, renaming, or removing RNA properties on `PMSession` / `PMWorkspace`, **restart Blender**—property schema changes are not reliably hot-reloadable. The same applies the first time after a package-layout change (modules moved into subpackages): **fully quit Blender**, then reopen — disable/re-enable alone can leave a stale `scene` / `core` module object in memory. Enable is idempotent (skips RNA types that are already live), so a failed unregister should not block the next enable.

Re-run `./scripts/link-dev.sh` if you later **Install from Disk** and Blender overwrites the symlink with a ZIP extract.

ZIP builds (`./scripts/build-extension.sh`) are only needed for a local packaging check. Published zips come from GitHub Actions on version tags.

User-visible changes go in `CHANGELOG.md` under `## [Unreleased]` in the same commit (see `AGENTS.md`). Do not bump `blender_manifest.toml` until a release.

## Parallel agent work

Use separate worktrees for independent problems, with one main thread reviewing
and integrating them. A worktree isolates files and a branch; it does not isolate
CPU, Blender's installed development link, or the underlying Git repository.
Do not run `link-dev.sh`, stash/pop, switch the main branch, or change another
worktree from an agent's task.

1. Start from a committed, verified checkpoint. Inspect `git status`,
   `git stash list` and `git worktree list` first so interrupted work is accounted
   for. Give each agent a new branch and a directory outside the main checkout
   (avoids recursively compiling nested worktrees).
2. Give each task a concrete question, owned files, expected evidence, test budget
   and stopping condition. For geometry, require exact input plus an independent
   camera/point/line check and a useful positive or removal control. A suspected
   defect is not permission to retune thresholds until the test passes.
3. Keep production ownership distinct, for example lens selection versus line
   reconstruction versus Blender job lifecycle. Agents can share findings; if a
   fix crosses another lane, agree on one owner before either edits that file.
   The main thread owns the decision record, shared harness contracts and CI.
4. Each agent runs focused checks and makes a reviewable commit, including the
   required changelog/user docs for product changes. Return the commit hash,
   exact test commands/results, artifacts and remaining uncertainty. Do not run
   several full suites or large Blender sweeps at once: Sync already uses worker
   threads and numerical libraries may use additional cores.
5. The main thread reads each diff and its evidence, integrates one commit at a
   time (for example `git cherry-pick <commit>`), resolves shared prose carefully,
   and checks the affected behavior. Run the full combined numerical suite and
   relevant Blender checks after integration. Passing separate branches does not
   establish that their combination works.
6. Preserve useful exact cases and measurements in the repository before cleaning
   up a worktree. Confirm its changes were integrated and it has no uncommitted
   work; remove it without force. Record any remaining branch/worktree and the
   next action in the continuing decision record when a session is interrupted.

Parallel work is useful when the questions can be answered independently. A
single dependent debugging chain is usually faster with one agent. Keep this
workflow in the repository; executable reproducers and tests carry the lasting
knowledge rather than an additional skill duplicating these commands.

### Budgeting experiments and model usage

Fresh, narrowly briefed Sol/high workers are already the established workflow;
switching to fresh context is not a new saving. Use the stronger main model at
defined gates: approve the question and oracle, then review a complete evidence
packet. Escalate earlier only for a blocked decision or contradictory evidence.
Avoid repeatedly reading partial logs or duplicating the worker's exploration.

Before a numerical experiment, specify the permitted cases, total solver-call
budget (including drafts, failed attempts and retries), wall-time limit and stop
condition. An outer lens search can contain many Sync calls: define and count
both, rather than calling an entire search one solve. The next harness improvement
uses [ExperimentBudget](../tools/synthetic_sync/budget.py) to enforce this budget
and save an attempt ledger before each invocation; written limits alone have
already been exceeded. Instrument every inner Sync call, not just the outer
search. Reuse results only when exact
input, source, options and relevant environment fingerprints match. Preserve
failures and timeouts as evidence too. The ledger is POSIX/main-thread only and
uses Python signal deadlines; pair it with an outer process timeout if native
code could block signal delivery. Resuming retains call counts and charges
interrupted attempts their reserved time. The caller must supply source-content,
environment and option fingerprints; a Git revision plus a dirty flag is not
sufficient. Cached values are serialized evidence, not reusable live solver objects.

Measure a task's main-thread usage delta plus its children, with input, cached
input and output separate. Cached input is part of input; reasoning output is
part of output. Do not sum cumulative counter snapshots or count inherited
history twice. Record review/rework, accepted evidence and elapsed time alongside
usage. Parent activity during a worker's time window is an association, not exact
task attribution. Local token counters are not a measurement of subscription
allowance or an invoice. See the official [subagent guidance](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [pricing explanation](https://learn.chatgpt.com/docs/pricing).

For the next small task, freeze its brief and artifact requirements before
starting; capture the complete planning-through-review interval. Compare with a
similar completed task, reporting differences in difficulty and rework. Treat
this as an operational pilot, not proof that one model combination is cheaper.

## Wheels

Wheels are **not** stored in git (~50–65 MB each). Before building or linking:

```sh
./scripts/fetch-wheels.sh
```

`./scripts/build-extension.sh` and `./scripts/link-dev.sh` call that for you. Release zips are built with `--split-platforms` so each OS package only embeds its own OpenCV binary.

## Checks and build

Compile-check Python:

```sh
python3 -m compileall -q .
```

Run unit tests (works even if the checkout directory is not named `match_perspective`):

```sh
./scripts/run-unittests.sh
```

Use an ordinary Python environment with OpenCV to exercise optional numerical
coverage. On the maintainer's current machine that environment is available as:

```sh
source ~/venvs/my/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_unittests.py
```

Record Python, NumPy and OpenCV versions with experiment results. OpenCV does
not remove skips for absent private sample files; Blender smoke tests remain a
separate check. Do not install packages into Blender's Python for this purpose.

Run a focused test without relying on test discovery order:

```sh
./scripts/run-unittests.sh test_sync_solve
./scripts/run-unittests.sh test_core.CoreGeometryTests.test_vanishing_point_intersection
```

Generate synthetic Sync scenes and verify camera alignment on object points that
were never used as picks:

```sh
python3 tools/synthetic_sync/run.py --family all --out /tmp/pm-synthetic
```

See [Synthetic Sync laboratory](../tools/synthetic_sync/README.md) for noisy sweeps,
exact JSON replay, portable reports, and generated `.blend` files with optional
reference renders. The PR/main test workflow runs numerical checks and real Blender
save/reopen checks; artifacts are retained for 14 days.

Run the Blender smoke test:

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup -b --python scripts/validate_addon.py
```

Validate and build distributable ZIPs (one per platform, each with its OpenCV wheel):

```sh
./scripts/build-extension.sh
```

Override Blender's location when necessary:

```sh
BLENDER_BIN="/path/to/blender" ./scripts/build-extension.sh
```

The smoke test covers registration, multi-match create/switch/unload/prune, VP solve, camera projection, origin placement, project import, undistorted plates, and cleanup.

## Release

On a clean `main`, after Unreleased bullets exist:

```sh
./scripts/release.sh 0.3.7
```

That bumps `blender_manifest.toml`, moves `## [Unreleased]` into a dated section, commits, tags `v0.3.7`, and pushes. The **Release** GitHub Action (tag `v*`) fetches OpenCV wheels, installs a pinned Blender 5.1 Linux tarball, runs `extension validate` / `extension build --split-platforms`, and publishes the four zips on the GitHub Release. Do not attach zips by hand unless Actions failed.

## Project layout

```text
match_perspective/
  blender_manifest.toml   # Blender extension metadata
  __init__.py             # Registration / reload entry
  core/                   # geometry.py; sync/ package; lens_refine; ROS camera_info
                          # Sync map: AGENTS.md (keep it current when stages move)
  detect/                 # AprilTags, auto VP lines, edge snap (OpenCV optional)
  properties/             # RNA PropertyGroups
  scene/                  # Camera/background integration + distortion plates
  ui/                     # Operators, panel, overlay, icon registration
                          # Overlay sizes/hit radii use preferences.system.ui_scale (Retina)
  icons/                  # PNG icon assets
  wheels/                 # OpenCV wheels (gitignored; ./scripts/fetch-wheels.sh)
  scripts/                # build / link-dev / fetch-wheels / release / tests
  .github/workflows/      # PR/main tests; tag-only zip build + GitHub Release
  tests/                  # Pure geometry / sync / detect regressions
  tools/                  # Standalone helpers (AprilTag sheets, FOV plotter, sync dump)
  docs/                   # User guide, sync, development, TODOs
  CHANGELOG.md            # Keep a Changelog; Unreleased → version at release
  AGENTS.md               # Changelog / docs conventions for agents
```
