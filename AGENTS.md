# Agent notes

Conventions for humans and coding agents working in this repo.

## Changelog

User-visible work must land with a bullet under `## [Unreleased]` in `CHANGELOG.md` **in the same commit** as the code (Keep a Changelog: Added / Changed / Fixed / Removed).

Do this when the change affects matching, sync, UI, install, OpenCV extras, or documented behavior. Skip it for refactors, tests, comments, and internal-only edits.

Write one short user-facing line, not a commit subject. Do not invent a version heading or bump `blender_manifest.toml` — that happens at release.

```markdown
## [Unreleased]

### Added
- Optional OpenCV: hide Detect VP Lines when the wheel is missing.
```

If `[Unreleased]` has no matching subsection yet, add it. Leave dated `## [x.y.z]` sections untouched.

To ship: `./scripts/release.sh 0.3.7` on a clean `main`. That cuts Unreleased, bumps `blender_manifest.toml`, tags `v0.3.7`, and pushes. GitHub Actions builds the four platform zips and creates the GitHub Release.

## Regression tests

When fixing a bug, add a focused automated regression test whenever the failure can be reproduced deterministically and testing it is practical. The test should fail before the fix and pass afterward. If a regression test is not practical, state why in the handoff.

## Docs

If the sidebar workflow, sync rules, or install story changed, update `docs/user-guide.md`, `docs/sync.md`, and/or `README.md` in that same change. Do not leave the changelog as the only record.

## Debugging tools

Headless helpers under `tools/` (and `scripts/validate_addon.py`) for investigating a `.blend` without clicking the sidebar. If you build a new dump, probe, or reproduction script while solving a problem, **check it in** and add a bullet here so the next agent can find it.

- `tools/debug-sync/` — sync graph, stored vs *recovered* optical-axis tilt vs world Z, homography vs mixed RMSE, `solve_landmark_sync` on a saved scene. See `tools/debug-sync/README.md`.
- `tools/explore-vp-intrinsics/` — VP-line residual vs FOV / principal-point / λ (does not touch landmarks).
- `scripts/validate_addon.py` — Blender smoke test (register, match CRUD, VP solve, origin, import).

## Do not

- Commit or push unless the user explicitly asks to (e.g. “commit this”, “create a commit”)
- Commit `*.zip` or `wheels/*.whl`
- `pip install` into Blender’s Python
- Create GitHub releases or tags unless asked (use `./scripts/release.sh` when asked to release)
