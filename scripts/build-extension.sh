#!/bin/sh

set -eu

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

if [ ! -x "$BLENDER_BIN" ]; then
    printf 'Blender binary not found or not executable: %s\n' "$BLENDER_BIN" >&2
    exit 1
fi

cd "$REPO_ROOT"

# OpenCV wheels are not in git — fetch before validate/build.
"${SCRIPT_DIR}/fetch-wheels.sh"

"$BLENDER_BIN" --factory-startup --command extension validate
# One zip per platform so each package only embeds its own OpenCV binary.
"$BLENDER_BIN" --factory-startup --command extension build --split-platforms

printf '\nExtension packages built in: %s\n' "$REPO_ROOT"
ls -lh "$REPO_ROOT"/match_perspective-*.zip 2>/dev/null || true
