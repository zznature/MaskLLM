#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <checkpoint_path> [model_size] [num_gpus] [sparsity_type]"
    echo "示例: $0 output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000 7b 4 sparse"
    exit 1
fi

CHECKPOINT_PATH=$1
MODEL_SIZE=${2:-"7b"}
NUM_GPUS=${3:-"4"}
SPARSITY_TYPE=${4:-"sparse"}

echo "=== 运行MaskLLM评估脚本 ==="
echo "检查点路径: $CHECKPOINT_PATH"
echo "模型大小: $MODEL_SIZE"
echo "GPU数量: $NUM_GPUS"
echo "稀疏类型: $SPARSITY_TYPE"
echo ""

# 设置环境变量
export PYTHONPATH="/data/home/zdhs0054/.local/lib/python3.10/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

echo "1. 设置环境变量..."
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "2. 解决cuDNN库问题..."
if [ -f "$CONDA_PREFIX/lib/libcudnn.so.8.9.7" ]; then
    if [ ! -f "$CONDA_PREFIX/lib/libcudnn.so.8" ]; then
        ln -sf "$CONDA_PREFIX/lib/libcudnn.so.8.9.7" "$CONDA_PREFIX/lib/libcudnn.so.8"
        echo "创建cuDNN符号链接: libcudnn.so.8 -> libcudnn.so.8.9.7"
    fi
else
    echo "警告: 未找到cuDNN库"
fi
echo ""

echo "3. 验证transformers安装..."
python3 -c "import transformers; print('transformers版本:', transformers.__version__)" || {
    echo "transformers导入失败，请先运行: bash test_transformers.sh"
    exit 1
}
echo ""

echo "4. 检查评估脚本是否存在..."
EVAL_SCRIPT="scripts/ppl/evaluate_llama2_wikitext2.sh"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "错误: 评估脚本不存在: $EVAL_SCRIPT"
    exit 1
fi
echo "找到评估脚本: $EVAL_SCRIPT"
echo ""

echo "5. 运行评估..."
echo "执行命令: bash $EVAL_SCRIPT $CHECKPOINT_PATH $MODEL_SIZE $NUM_GPUS $SPARSITY_TYPE"
echo ""

# 运行评估脚本
bash "$EVAL_SCRIPT" "$CHECKPOINT_PATH" "$MODEL_SIZE" "$NUM_GPUS" "$SPARSITY_TYPE"

echo ""
echo "=== 评估完成 ===" 