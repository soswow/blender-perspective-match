#!/bin/sh

set -eu

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ ! -x "$BLENDER_BIN" ]; then
    printf 'Blender binary not found or not executable: %s\n' "$BLENDER_BIN" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

"$BLENDER_BIN" --factory-startup --command extension validate
"$BLENDER_BIN" --factory-startup --command extension build

printf '\nExtension package built in: %s\n' "$SCRIPT_DIR"
