#!/bin/sh
# Download OpenCV wheels for Blender 5.1 (Python 3.13) into ./wheels/.
# Wheels are not committed — run this before build / link-dev.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WHEELS_DIR="${SCRIPT_DIR}/wheels"
# Pin so every platform ships the same OpenCV build (aruco / AprilTag 25h9).
OPENCV_VERSION="${OPENCV_VERSION:-5.0.0.93}"
PACKAGE="opencv-contrib-python-headless==${OPENCV_VERSION}"

# pip --platform tags that match Blender extension platforms:
#   macos-arm64, macos-x64, linux-x64, windows-x64
PLATFORMS="
macosx_13_0_arm64
macosx_14_0_x86_64
manylinux2014_x86_64
win_amd64
"

mkdir -p "$WHEELS_DIR"
cd "$WHEELS_DIR"

missing=0
for platform in $PLATFORMS; do
  # Already have a matching wheel for this pin + platform?
  if ls -1 "opencv_contrib_python_headless-${OPENCV_VERSION}"*"${platform}"*.whl >/dev/null 2>&1; then
    continue
  fi
  missing=1
  break
done

if [ "$missing" -eq 0 ]; then
  printf 'OpenCV wheels already present for %s\n' "$OPENCV_VERSION"
  ls -lh "$WHEELS_DIR"/*.whl
  exit 0
fi

printf 'Downloading %s (binary wheels, no deps — Blender already ships NumPy)…\n' "$PACKAGE"
for platform in $PLATFORMS; do
  python3 -m pip download "$PACKAGE" \
    --dest "$WHEELS_DIR" \
    --only-binary=:all: \
    --no-deps \
    --python-version=3.13 \
    --platform="$platform"
done

printf '\nWheels ready:\n'
ls -lh "$WHEELS_DIR"/*.whl
