#!/bin/sh
# Cut a version, tag it, and push so GitHub Actions can build and publish zips.
# Usage: ./scripts/release.sh 0.3.7

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
MANIFEST="${REPO_ROOT}/blender_manifest.toml"

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s <major.minor.patch>\n' "$0" >&2
  exit 1
fi

VERSION=$1
case "$VERSION" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *)
    printf 'Version must look like 0.3.7, got: %s\n' "$VERSION" >&2
    exit 1
    ;;
esac

TAG="v${VERSION}"

cd "$REPO_ROOT"

if [ -n "$(git status --porcelain)" ]; then
  printf 'Working tree is not clean. Commit or stash first.\n' >&2
  git status --porcelain >&2
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
  printf 'Releases must be cut from main (current branch: %s)\n' "$BRANCH" >&2
  exit 1
fi

if git rev-parse --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
  printf 'Tag already exists: %s\n' "$TAG" >&2
  exit 1
fi

CURRENT=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$MANIFEST" | head -n 1)
if [ -z "$CURRENT" ]; then
  printf 'Could not read version from blender_manifest.toml\n' >&2
  exit 1
fi
if [ "$CURRENT" = "$VERSION" ]; then
  printf 'blender_manifest.toml is already %s\n' "$VERSION" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/changelog.py" check-unreleased
"${SCRIPT_DIR}/run-unittests.sh"

# BSD and GNU sed both accept -i.bak; remove the backup after.
sed -i.bak 's/^version = ".*"/version = "'"${VERSION}"'"/' "$MANIFEST"
rm -f "${MANIFEST}.bak"

python3 "${SCRIPT_DIR}/changelog.py" cut "$VERSION"

git add "$MANIFEST" "${REPO_ROOT}/CHANGELOG.md"
git commit -m "$(cat <<EOF
Release ${VERSION}

EOF
)"

git tag -a "$TAG" -m "Perspective Match ${VERSION}"
git push origin HEAD
git push origin "$TAG"

printf '\nTagged %s and pushed. GitHub Actions will build platform zips and create the Release.\n' "$TAG"
