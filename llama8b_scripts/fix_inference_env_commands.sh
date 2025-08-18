#!/bin/bash
# Fix HF inference env without upgrading torch
# - Support pinning transformers version (default 4.40.0)
# - Avoid accelerate / device_mesh by uninstalling accelerate and setting env
# - Install to user-writable dir to avoid Apptainer overlay space issues
# HPC Server Command Execution Protocol: batch commands in one script

set -euo pipefail

# Config
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.40.0}
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INSTALL_TARGET="$PROJECT_DIR/.ext_pkgs"
PIP_CACHE_DIR="/tmp/pip-cache"

mkdir -p "$INSTALL_TARGET" "$PIP_CACHE_DIR"

# Uninstall accelerate to avoid transformers importing hooks that need torch.device_mesh
python -m pip uninstall -y accelerate || true

# Install packages into INSTALL_TARGET to avoid overlay space; disable cache
python -m pip install -i "$PIP_MIRROR" --no-cache-dir \
  --target "$INSTALL_TARGET" \
  "transformers==${TRANSFORMERS_VERSION}" \
  "safetensors>=0.4.0" \
  "sentencepiece>=0.1.99" \
  einops tqdm

# Print env exports for the user shell
cat <<EOF

Add these to your shell for this session:

export PYTHONPATH="$INSTALL_TARGET:":\$PYTHONPATH
export TRANSFORMERS_NO_ACCELERATE=1

Then rerun:
  bash llama8b_scripts/run_llama8b_infer.sh

EOF

# Quick import check using the installed target
PYTHONPATH="$INSTALL_TARGET":$PYTHONPATH TRANSFORMERS_NO_ACCELERATE=1 \
python - <<'PY'
import transformers, torch, os
print('transformers', transformers.__version__)
print('torch       ', torch.__version__)
print('TRANSFORMERS_NO_ACCELERATE=', os.environ.get('TRANSFORMERS_NO_ACCELERATE'))
print('OK')
PY

echo "Done."
