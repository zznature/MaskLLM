#!/bin/bash

# 完整的checkpoint到HuggingFace转换流程
# 基于export_hf.md文档的流程

set -e  # 遇到错误时退出

PROJECT_DIR=$(pwd)
echo "当前工作目录: $PROJECT_DIR"

# 检查点路径配置
SPARSE_CKPT="output/checkpoints/llama2-7b-tp4-mask-only-wikitext103-c4format/train_iters_800/ckpt/iter_0000800"
TP1_CKPT="output/checkpoints/llama2-7b-tp1-mask-only-wikitext103-c4format/train_iters_800/ckpt/iter_0000800"
HF_REFERENCE="assets/checkpoints/llama2_7b_hf"
HF_OUTPUT_TEMP="output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format_fp32"
HF_OUTPUT="output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format"

echo "=== 第1步: 合并稀疏mask到模型参数 ==="
echo "输入checkpoint: $SPARSE_CKPT"
if [ ! -d "$SPARSE_CKPT" ]; then
    echo "错误: 源checkpoint不存在: $SPARSE_CKPT"
    exit 1
fi

python tool_apply_sparsity.py --ckpt_dir $SPARSE_CKPT
echo "稀疏性应用完成"

echo ""
echo "=== 第2步: 转换Tensor Parallelism从TP=4到TP=1 ==="
echo "运行TP转换脚本..."
bash scripts/tools/convert_llama2_7b_tp4_to_tp1.sh
echo "TP转换完成"

echo ""
echo "=== 第3步: 导出到HuggingFace格式 (FP32) ==="
echo "输入TP=1 checkpoint: $TP1_CKPT"
echo "HF参考模型: $HF_REFERENCE"
echo "输出HF模型 (临时FP32): $HF_OUTPUT_TEMP"

if [ ! -d "$HF_REFERENCE" ]; then
    echo "错误: HuggingFace参考模型不存在: $HF_REFERENCE"
    echo "请先下载LLaMA-2 7B HF模型到 $HF_REFERENCE"
    exit 1
fi

python tool_export_to_hf.py \
    --hf_ckpt $HF_REFERENCE \
    --megatron_ckpt $TP1_CKPT/mp_rank_00/model_optim_rng.pt \
    --save_ckpt $HF_OUTPUT_TEMP

echo "HuggingFace模型 (FP32) 导出完成"

echo ""
echo "=== 第4步: 复制tokenizer文件到临时目录 ==="
echo "复制tokenizer文件到临时输出目录..."
cp $HF_REFERENCE/tokenizer* $HF_OUTPUT_TEMP/
echo "tokenizer文件复制完成"

echo ""
echo "=== 第5步: 转换为Float16格式 ==="
echo "输入模型 (FP32): $HF_OUTPUT_TEMP"
echo "输出模型 (FP16): $HF_OUTPUT"

python convert_hf_to_float16.py \
    --input $HF_OUTPUT_TEMP \
    --output $HF_OUTPUT

echo "Float16转换完成"

echo ""
echo "=== 第6步: 清理临时文件 ==="
echo "删除临时FP32模型文件..."
rm -rf $HF_OUTPUT_TEMP
echo "临时文件清理完成"

echo ""
echo "=== 转换完成! ==="
echo "HuggingFace格式模型 (Float16) 保存在: $HF_OUTPUT"
echo ""
echo "Float16格式优势："
echo "  ✅ 模型大小减半 (约7GB → 3.5GB)"
echo "  ✅ 推理速度提升"
echo "  ✅ 显存占用减少"
echo ""
echo "可以使用以下命令进行评测:"
echo "python eval_llama_ppl.py --model $HF_OUTPUT"
echo ""
echo "或者使用lm-eval-harness进行评测:"
echo "lm_eval --model hf --model_args pretrained=$HF_OUTPUT --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge" 