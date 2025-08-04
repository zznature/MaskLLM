#!/bin/bash

# 生成预计算的mask文件用于HF模型评测
# 基于export_hf.md文档的流程

set -e  # 遇到错误时退出

PROJECT_DIR=$(pwd)
echo "当前工作目录: $PROJECT_DIR"

# 路径配置
DENSE_HF="assets/checkpoints/llama2_7b_hf"
SPARSE_HF="output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format"
MASK_OUTPUT="output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format_mask.pt"
COMPRESSED_MASK_DIR="output/precomputed_masks"
COMPRESSED_MASK_FILE="$COMPRESSED_MASK_DIR/llama2_7b_hf_maskllm_wikitext103_c4format_mask_compressed.npz"

echo "=== 生成预计算mask文件 ==="
echo "稠密HF模型: $DENSE_HF"
echo "稀疏HF模型: $SPARSE_HF"
echo "输出mask文件: $MASK_OUTPUT"

# 检查输入文件
if [ ! -d "$DENSE_HF" ]; then
    echo "错误: 稠密HF模型不存在: $DENSE_HF"
    echo "请先下载LLaMA-2 7B HF模型"
    exit 1
fi

if [ ! -d "$SPARSE_HF" ]; then
    echo "错误: 稀疏HF模型不存在: $SPARSE_HF"
    echo "请先运行convert_checkpoint_to_hf.sh生成HF模型"
    exit 1
fi

echo ""
echo "=== 第1步: 计算mask文件 ==="
python tool_compute_mask_hf.py \
    --dense $DENSE_HF \
    --sparse $SPARSE_HF \
    --save $MASK_OUTPUT

echo "mask文件生成完成: $MASK_OUTPUT"

echo ""
echo "=== 第2步: 压缩mask文件 ==="
mkdir -p $COMPRESSED_MASK_DIR

python tool_compress_mask.py \
    --mask_ckpt $MASK_OUTPUT \
    --output $COMPRESSED_MASK_FILE

echo "压缩mask文件生成完成: $COMPRESSED_MASK_FILE"

echo ""
echo "=== 预计算mask生成完成! ==="
echo "压缩mask文件保存在: $COMPRESSED_MASK_FILE"
echo ""
echo "可以使用以下命令结合官方HF模型进行评测:"
echo "python eval_llama_ppl.py --model meta-llama/Llama-2-7b-hf --mask $COMPRESSED_MASK_FILE"
echo ""
echo "或者使用lm-eval-harness结合mask进行评测:"
echo "# 注意: 需要确认lm-eval-harness是否支持mask参数" 