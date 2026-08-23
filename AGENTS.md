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

## Where code lives

Keep this map accurate when you add a module, move a stage, or change a named constant.

| Area | Path |
| --- | --- |
| VP / single-camera geometry | `core/geometry.py` |
| Landmark-graph sync | `core/sync/` (package; import as `match_perspective.core.sync`) |
| Focal search | `core/lens_refine.py` |
| Blender cameras, stills, Solve Sync apply | `scene/__init__.py` |
| Image analysis (AprilTags, VP detect, edge/tag snap) | `detect/` |
| Operators / panel / overlay | `ui/` |
| RNA | `properties/__init__.py` |

Do not special-case a `.blend` filename in solver code or UI copy.

### Sync package (`core/sync/`)

`from match_perspective.core import sync` still works. Submodules:

| Module | Role |
| --- | --- |
| `constants.py` | `ACCEPT_RMSE_PX`, `STRETCHED_PIXEL_RATIO`, `GROUND_PLANE_Z_FRACTION` |
| `types.py` | `SimilarityTransform`, observations, `SyncSolveResult` |
| `projection.py` | Project, rays, triangulate, image-line geometry |
| `pose.py` | Essential / PnP / IPPE / pairwise register |
| `ground.py` | Calibrated On Ground plane init (`estimate_anchor_ground_plane`) |
| `lines.py` | Free / Known 3D lines, Is-Parallel-To |
| `ba.py` | Joint BA, residuals, leave-one-out Diagnose |
| `solve.py` | `solve_landmark_sync` stages |

**Solve stages** (in order): pairwise register → peel cameras above `ACCEPT_RMSE_PX` → joint BA → peel again → resect skipped stills against frozen 3D (On Ground / near-Z=0 if off-plane picks disagree) → report. Recovered cameras must not fail the joint RMSE. Copying locked K onto a different aspect uses one scale for fx and fy unless the sizes are an exact portrait/landscape swap (same pixels, axes swapped). Solve Sync sets fy=fx when they already differ by more than `STRETCHED_PIXEL_RATIO`.

**When you change sync:** update this map if stages or files moved; put a new threshold in `constants.py` instead of a raw `40.0`; keep function docstrings to a short contract (what / what not), not algorithm history. Tests: `tests/test_sync_pose.py`, `test_sync_ground.py`, `test_sync_ba.py`, `test_sync_lines.py`, `test_sync_solve.py` (helpers in `tests/sync_fixtures.py`). Pairwise covering (true camera vs stored K/pose): `tests/edge_pairs.md`, `tests/pair_fixtures.py`, `tests/test_edge_pairs.py`.

## Debugging tools

Headless helpers under `tools/` (and `scripts/validate_addon.py`) for investigating a `.blend` without clicking the sidebar. If you build a new dump, probe, or reproduction script while solving a problem, **check it in** and add a bullet here so the next agent can find it.

- `tools/debug-sync/` — sync graph, stored vs *recovered* optical-axis tilt vs world Z, homography vs mixed RMSE, `solve_landmark_sync` on a saved scene; `probe_resected.py` splits ground vs off-plane RMSE on a recovered still. See `tools/debug-sync/README.md`.
- `tools/explore-vp-intrinsics/` — VP-line residual vs FOV / principal-point / λ (does not touch landmarks).
- `scripts/validate_addon.py` — Blender smoke test (register, match CRUD, VP solve, origin, import).

## Do not

- Commit or push unless the user explicitly asks to (e.g. “commit this”, “create a commit”)
- Commit `*.zip` or `wheels/*.whl`
- `pip install` into Blender’s Python
- Create GitHub releases or tags unless asked (use `./scripts/release.sh` when asked to release)
