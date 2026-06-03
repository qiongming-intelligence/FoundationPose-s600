#!/usr/bin/env bash
# Fetch the pinned upstream NVlabs/FoundationPose source into third_party/.
#
# Upstream is NVIDIA-proprietary and NOT committed to this repo (see
# docs/upstream_pin.md). This script reproduces the local vendored checkout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/FoundationPose"
REMOTE="${FOUNDATIONPOSE_REMOTE:-https://github.com/NVlabs/FoundationPose}"
SHA="${FOUNDATIONPOSE_SHA:-a1b694b83e633c2cb6115b9063d940a687759392}"

if [[ -d "$DEST/.git" ]]; then
  echo "upstream already present at $DEST"
  echo "current SHA: $(git -C "$DEST" rev-parse HEAD)"
  echo "to re-pin: rm -rf '$DEST' && FOUNDATIONPOSE_SHA=<sha> $0"
  exit 0
fi

echo "cloning $REMOTE @ $SHA -> $DEST"
mkdir -p "$(dirname "$DEST")"
git clone "$REMOTE" "$DEST"
git -C "$DEST" checkout --detach "$SHA"

echo "done. pinned SHA: $(git -C "$DEST" rev-parse HEAD)"
echo "NOTE: this checkout is git-ignored and must not be committed."
