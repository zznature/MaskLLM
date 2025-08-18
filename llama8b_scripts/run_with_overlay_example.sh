#!/bin/bash
# Example: run MaskLLM container with custom overlay image
# Adjust OVERLAY_IMG and container command as needed.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OVERLAY_IMG=${OVERLAY_IMG:-/data/home/zdhs0054/overlay-8192M.img}

if [[ ! -f "$OVERLAY_IMG" ]]; then
  echo "Overlay image not found: $OVERLAY_IMG"
  echo "Create one with: bash llama8b_scripts/create_overlay_img.sh"
  exit 1
fi

# Example command: using apptainer (or singularity) with overlay image
# Replace image path and binds according to your environment
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/path/to/your.sif}

if [[ ! -f "$APPTAINER_IMAGE" ]]; then
  echo "Please set APPTAINER_IMAGE to your container .sif file."
  exit 1
fi

# Inside the container, we can install Python packages to /usr/local or /opt using the overlay
# Also wire project-local external packages dir if used
EXT_PKGS_DIR="$PROJECT_DIR/.ext_pkgs"

CMD=(apptainer exec --nv --overlay "$OVERLAY_IMG" "$APPTAINER_IMAGE" bash -lc "\
  export PYTHONPATH='$EXT_PKGS_DIR':\$PYTHONPATH; \
  python --version; \
  which python; \
  bash llama8b_scripts/run_llama8b_infer.sh \
")

printf "[Run] %s\n" "${CMD[*]}"
"${CMD[@]}"
