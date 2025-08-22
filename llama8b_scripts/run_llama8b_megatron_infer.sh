#!/bin/bash
# Megatron format Llama8b inference script
# Corresponding to the HF format run_llama8b_infer.sh for comparison testing
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🚀 Megatron Llama8b 推理脚本"
echo "==============================================="

# Megatron模型配置
CHECKPOINT_DIR="output/checkpoints/llama8b_megatron_tp8"
TOKENIZER_TYPE="NullTokenizer"  # 使用NullTokenizer处理119696词汇表
VOCAB_SIZE=119696
MAKE_VOCAB_DIVISIBLE_BY=128

# 模型架构参数 (与转换时一致)
NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32
MAX_POSITION_EMBEDDINGS=4096
SEQ_LENGTH=512

# 并行配置
TENSOR_MODEL_PARALLEL_SIZE=1  # 推理时使用单卡，避免复杂性
PIPELINE_MODEL_PARALLEL_SIZE=1

# 推理参数 (与HF脚本保持一致)
PROMPT_TEXT="<用户> 山东最高的山是什么山？<AI>"
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=50
USE_SAMPLING=true

# 环境配置
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ACCELERATE=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd "$PROJECT_DIR"

# 检查checkpoint是否存在
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "❌ Checkpoint目录不存在: $CHECKPOINT_DIR"
    echo "请先完成模型转换！"
    exit 1
fi

echo "📋 配置信息:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Tokenizer: $TOKENIZER_TYPE"
echo "  词汇表大小: $VOCAB_SIZE"
echo "  并行配置: TP=$TENSOR_MODEL_PARALLEL_SIZE, PP=$PIPELINE_MODEL_PARALLEL_SIZE"
echo "  测试prompt: $PROMPT_TEXT"
echo "  推理参数: max_tokens=$MAX_NEW_TOKENS, temp=$TEMPERATURE, top_p=$TOP_P, top_k=$TOP_K"
echo ""

# 注：现在使用Python脚本进行推理，无需构建复杂的CMD参数

echo "🔄 执行Megatron推理..."

# 使用Python推理脚本
PYTHON_SCRIPT="$SCRIPT_DIR/llama8b_megatron_infer.py"
PYTHON_CMD=(
    "python3.10" "$PYTHON_SCRIPT"
    "--checkpoint" "$CHECKPOINT_DIR"
    "--prompt" "$PROMPT_TEXT"
    "--max-tokens" "$MAX_NEW_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--top-k" "$TOP_K"
    "--tensor-parallel-size" "$TENSOR_MODEL_PARALLEL_SIZE"
    "--pipeline-parallel-size" "$PIPELINE_MODEL_PARALLEL_SIZE"
)

echo "命令: ${PYTHON_CMD[*]}"
echo ""

# 执行推理
if bash run_maskllm_native.sh "${PYTHON_CMD[@]}"; then
    echo ""
    echo "✅ 推理完成！"
else
    echo "❌ 推理失败！"
    echo "请检查错误信息和配置"
    exit 1
fi

echo ""
echo "🎉 Megatron格式Llama8b推理测试完成！"
echo "💡 可以与HF格式输出对比验证一致性"
