#!/bin/bash

# 快速转换checkpoint到HuggingFace格式
# 根据HPC规则，集中所有命令到一个脚本

echo "=== MaskLLM Checkpoint to HuggingFace 转换脚本 ==="
echo "当前工作目录: $(pwd)"
echo ""

echo "第1步: 设置脚本权限"
chmod +x convert_checkpoint_to_hf.sh generate_precomputed_mask.sh scripts/tools/convert_llama2_7b_tp4_to_tp1.sh
echo "脚本权限设置完成"
echo ""

echo "第2步: 检查前置条件"
CHECKPOINT_DIR="output/checkpoints/llama2-7b-tp4-mask-only-wikitext103-c4format/train_iters_800/ckpt/iter_0000800"
HF_REFERENCE="assets/checkpoints/llama2_7b_hf"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 错误: 源checkpoint不存在: $CHECKPOINT_DIR"
    exit 1
else
    echo "✅ 源checkpoint存在: $CHECKPOINT_DIR"
fi

if [ ! -d "$HF_REFERENCE" ]; then
    echo "❌ 错误: HF参考模型不存在: $HF_REFERENCE"
    echo "请先下载LLaMA-2 7B HF模型或运行: python scripts/tools/download_llama2_7b_hf.py"
    exit 1
else
    echo "✅ HF参考模型存在: $HF_REFERENCE"
fi
echo ""

echo "第3步: 执行checkpoint到HuggingFace转换"
echo "注意: 请确保在MaskLLM容器环境内运行此脚本"
echo ""

# 执行转换
bash convert_checkpoint_to_hf.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 转换成功完成!"
    echo ""
    echo "HuggingFace模型 (Float16格式) 已保存到:"
    echo "  output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format/"
    echo ""
    echo "Float16格式优势:"
    echo "  ✅ 模型大小减半 (约7GB → 3.5GB)"
    echo "  ✅ 推理速度提升"
    echo "  ✅ 显存占用减少"
    echo ""
    echo "可以使用以下命令进行评测:"
    echo "  # 使用项目内评测脚本"
    echo "  python eval_llama_ppl.py --model output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format/"
    echo ""
    echo "  # 使用lm-eval-harness"
    echo "  lm_eval --model hf --model_args pretrained=output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format --tasks hellaswag,piqa,winogrande"
    echo ""
    echo "如需生成预计算mask文件，请运行:"
    echo "  bash generate_precomputed_mask.sh"
else
    echo ""
    echo "❌ 转换过程中出现错误，请检查日志输出"
    exit 1
fi 