#!/bin/bash
# Wrapper script to run Llama8b inference inside the MaskLLM container environment
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Mixed environment: container PyTorch + external transformers
# Ensure container PyTorch is prioritized in PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:${PROJECT_DIR}/.ext_pkgs"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ACCELERATE=1   # avoid accelerate requirement; use explicit device placement
export TRANSFORMERS_OFFLINE=1         # force offline mode for local models
export HF_DATASETS_OFFLINE=1          # disable datasets online features
export DISABLE_MLFLOW_INTEGRATION=TRUE # disable mlflow integration

# Additional environment variables from run_maskllm_native.sh for UCC/UCX conflict prevention
export TORCH_UCC_DISABLE=1
export UCC_DISABLE=1
export NCCL_UCX_DISABLE=1
export UCX_DISABLE=1
export TORCH_DISTRIBUTED_BACKEND=gloo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
# Ensure UTF-8 locale for Chinese prompts
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "[Info] Mixed environment configuration:"
echo "  - Container PyTorch: /usr/local/lib/python3.10/dist-packages"
echo "  - External transformers: ${PROJECT_DIR}/.ext_pkgs"
echo "  - PYTHONPATH: $PYTHONPATH"

MODEL_DIR="assets/checkpoints/Llama8b"   # model config and tokenizer files location
TOKENIZER_DIR="assets/checkpoints/Llama8b"  # local tokenizer module under this dir
VOCAB_FILE="assets/checkpoints/Llama8b/vocab.txt"  # explicit vocab path
PROMPT_FILE=""                             # optional prompt file path
PROMPT_TEXT="<用户> 山东最高的山是什么山？<AI>"
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=50
DO_SAMPLE=1            # 1: sampling on, 0: greedy
DTYPE="float16"        # float32 | float16 | bfloat16
USE_GPU=1              # 1: use GPU if available, 0: CPU
TRUST_REMOTE_CODE=0

echo "[Info] Script directory: $SCRIPT_DIR"
echo "[Info] Project root directory: $PROJECT_DIR"
echo "[Info] Changing to project root: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Verify the model directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[Error] Model directory not found: $MODEL_DIR"
    echo "[Error] Current working directory: $(pwd)"
    echo "[Error] Contents of current directory:"
    ls -la
    exit 1
fi

echo "[Info] Model directory found: $MODEL_DIR"
CMD=("python" "$SCRIPT_DIR/llama8b_infer.py" --model "$MODEL_DIR")

if [[ -n "$TOKENIZER_DIR" ]]; then
  CMD+=(--tokenizer "$TOKENIZER_DIR")
fi

if [[ -n "$VOCAB_FILE" ]]; then
  CMD+=(--vocab "$VOCAB_FILE")
fi

if [[ -n "$PROMPT_FILE" ]]; then
  CMD+=(--prompt-file "$PROMPT_FILE")
else
  CMD+=(--prompt "$PROMPT_TEXT")
fi

CMD+=(--max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
[[ "$DO_SAMPLE" == "1" ]] && CMD+=(--do-sample)
CMD+=(--seed 42 --dtype "$DTYPE")
[[ "$USE_GPU" == "1" ]] && CMD+=(--use-gpu)
[[ "$TRUST_REMOTE_CODE" == "1" ]] && CMD+=(--trust-remote-code)

printf "[Run] %s\n" "${CMD[*]}"
"${CMD[@]}"
