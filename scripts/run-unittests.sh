#!/bin/sh
# Compile-check and run unit tests regardless of the checkout directory name.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

python3 -m compileall -q "$REPO_ROOT"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
ln -s "$REPO_ROOT" "$TMP_DIR/match_perspective"

cd "$TMP_DIR"
python3 -m unittest discover -s match_perspective/tests -v
