#!/bin/bash
# Create a persistent ext3 overlay image for Apptainer/Singularity
# This lets you increase writable space inside the container.
# Edit variables below as needed.

set -euo pipefail

OVERLAY_SIZE_MB=${OVERLAY_SIZE_MB:-8192}  # 8GB by default
OVERLAY_IMG=${OVERLAY_IMG:-/data/home/zdhs0054/overlay-${OVERLAY_SIZE_MB}M.img}

echo "Creating overlay image: $OVERLAY_IMG (${OVERLAY_SIZE_MB} MB)"
if [[ -f "$OVERLAY_IMG" ]]; then
  echo "File already exists: $OVERLAY_IMG"
  exit 0
fi

# Create sparse file
dd if=/dev/zero of="$OVERLAY_IMG" bs=1M count=0 seek="$OVERLAY_SIZE_MB"

# Format as ext3 (needs mkfs.ext3 available on host)
if ! command -v mkfs.ext3 >/dev/null 2>&1; then
  echo "mkfs.ext3 not found. Please install e2fsprogs (or ask admin)."
  exit 1
fi
mkfs.ext3 -F "$OVERLAY_IMG"

# Show result
ls -lh "$OVERLAY_IMG"
echo "Done. Use with: --overlay $OVERLAY_IMG"
