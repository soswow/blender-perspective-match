#!/bin/sh

set -eu

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ ! -x "$BLENDER_BIN" ]; then
    printf 'Blender binary not found or not executable: %s\n' "$BLENDER_BIN" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

# OpenCV wheels are not in git — fetch before validate/build.
./fetch-wheels.sh

"$BLENDER_BIN" --factory-startup --command extension validate
# One zip per platform so each package only embeds its OpenCV binary.
"$BLENDER_BIN" --factory-startup --command extension build --split-platforms

printf '\nExtension packages built in: %s\n' "$SCRIPT_DIR"
ls -lh "$SCRIPT_DIR"/match_perspective-*.zip 2>/dev/null || true
