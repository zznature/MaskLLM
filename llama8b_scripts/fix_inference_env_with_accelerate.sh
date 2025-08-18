#!/bin/bash
# Setup HF inference env with Accelerate (Solution B)
# - numpy==1.26.x to match older torch builds
# - transformers==4.40.0 (or set via TRANSFORMERS_VERSION)
# - tokenizers==0.19.1 (required by transformers>=4.40)
# - accelerate==0.23.0 (compatible; no device_mesh requirement)
# All installed into project-local .ext_pkgs to avoid system conflicts and overlay limits

set -euo pipefail

TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.40.0}
ACCELERATE_VERSION=${ACCELERATE_VERSION:-0.23.0}
TOKENIZERS_VERSION=${TOKENIZERS_VERSION:-0.19.1}
NUMPY_VERSION=${NUMPY_VERSION:-1.26.4}
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INSTALL_TARGET="$PROJECT_DIR/.ext_pkgs"
mkdir -p "$INSTALL_TARGET"

# Install pinned deps into project dir
python -m pip install -i "$PIP_MIRROR" --no-cache-dir --target "$INSTALL_TARGET" \
  "numpy==${NUMPY_VERSION}" \
  "transformers==${TRANSFORMERS_VERSION}" \
  "tokenizers==${TOKENIZERS_VERSION}" \
  "accelerate==${ACCELERATE_VERSION}" \
  "safetensors>=0.4.0" \
  "sentencepiece>=0.1.99" \
  packaging einops tqdm

# Print exports
cat <<EOF

Add these to your shell (current session):

export PYTHONPATH="$INSTALL_TARGET":\$PYTHONPATH
unset TRANSFORMERS_NO_ACCELERATE
export TOKENIZERS_PARALLELISM=false

Then run:
  bash llama8b_scripts/run_llama8b_infer.sh

EOF

# Quick check
PYTHONPATH="$INSTALL_TARGET":$PYTHONPATH python - <<'PY'
import os, torch
import transformers, tokenizers
try:
    import accelerate
    acc = accelerate.__version__
except Exception:
    acc = 'MISSING'
print('numpy        =', __import__('numpy').__version__)
print('torch        =', torch.__version__)
print('transformers =', transformers.__version__)
print('tokenizers   =', tokenizers.__version__)
print('accelerate   =', acc)
print('TRANSFORMERS_NO_ACCELERATE =', os.environ.get('TRANSFORMERS_NO_ACCELERATE'))
print('OK')
PY

echo "Done."
