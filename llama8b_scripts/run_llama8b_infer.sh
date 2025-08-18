#!/bin/bash
# Wrapper script to run Llama8b inference inside the MaskLLM container environment
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Use container-native environment only (no project-local overrides)
unset PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ACCELERATE=1   # avoid accelerate requirement; use explicit device placement
# Ensure UTF-8 locale for Chinese prompts
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

MODEL_DIR="assets/checkpoints/Llama8b"   # change if needed
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

cd "$PROJECT_DIR"

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
