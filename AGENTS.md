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

## Docs

If the sidebar workflow, sync rules, or install story changed, update `docs/user-guide.md`, `docs/sync.md`, and/or `README.md` in that same change. Do not leave the changelog as the only record.

## Do not

- Commit or push unless the user explicitly asks to (e.g. “commit this”, “create a commit”)
- Commit `*.zip` or `wheels/*.whl`
- `pip install` into Blender’s Python
- Create GitHub releases or tags unless asked (use `./scripts/release.sh` when asked to release)
