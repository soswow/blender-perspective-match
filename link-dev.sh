#!/bin/sh
# Symlink this git checkout into Blender 5.1 so edits apply without rebuilding a ZIP.
# Safe to re-run after Install from Disk overwrites the link.

set -eu

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"
BLENDER_VERSION="${BLENDER_VERSION:-5.1}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXT_ROOT="${HOME}/Library/Application Support/Blender/${BLENDER_VERSION}/extensions/user_default"
TARGET="${EXT_ROOT}/match_perspective"

if [ ! -x "$BLENDER_BIN" ]; then
    printf 'Blender binary not found: %s\n' "$BLENDER_BIN" >&2
    printf 'Override with BLENDER_BIN=/path/to/blender\n' >&2
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/blender_manifest.toml" ] || [ ! -f "${SCRIPT_DIR}/__init__.py" ]; then
    printf 'This does not look like the Perspective Match extension root: %s\n' "$SCRIPT_DIR" >&2
    exit 1
fi

mkdir -p "$EXT_ROOT"

# Replace a prior ZIP install or stale link with a live checkout.
if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
    if [ -L "$TARGET" ]; then
        printf 'Removing existing symlink: %s\n' "$TARGET"
    else
        printf 'Removing installed copy: %s\n' "$TARGET"
    fi
    rm -rf "$TARGET"
fi

ln -s "$SCRIPT_DIR" "$TARGET"

printf 'Linked:\n  %s\n→ %s\n\n' "$TARGET" "$SCRIPT_DIR"
printf 'In Blender: enable Perspective Match once if needed, then after edits use\n'
printf '  F3 → Reload Scripts\n'
printf '(or Preferences → Extensions → Refresh Local). Restart Blender if a modal\n'
printf 'draw tool leaves stale handlers.\n'
