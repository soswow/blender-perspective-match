#!/bin/sh
# Compile-check and run unit tests regardless of the checkout directory name.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

python3 -m compileall -q "$REPO_ROOT"

python3 "$SCRIPT_DIR/run_unittests.py" "$@"
