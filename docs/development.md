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
  .github/workflows/      # Tag-only zip build + GitHub Release
  tests/                  # Pure geometry / sync / detect regressions
  tools/                  # Standalone helpers (AprilTag sheets, FOV plotter, sync dump)
  docs/                   # User guide, sync, development, TODOs
  CHANGELOG.md            # Keep a Changelog; Unreleased → version at release
  AGENTS.md               # Changelog / docs conventions for agents
```
