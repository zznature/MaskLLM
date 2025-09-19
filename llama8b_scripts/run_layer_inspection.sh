
#!/bin/bash
#
# 检查Llama8b-MaskLLM特定层的权重分布
#

echo "=== Llama8b-MaskLLM 层权重检查 ==="
echo "检查指定层的参数分布情况"
echo

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 设置Python路径
export PYTHONPATH=/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH

# 检查模型文件
FP16_FILE="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"

if [ ! -f "$FP16_FILE" ]; then
    echo "❌ FP16模型文件不存在: $FP16_FILE"
    exit 1
fi

echo "✓ 找到FP16文件: $FP16_FILE"
echo "文件大小: $(du -h "$FP16_FILE" | cut -f1)"
echo

echo "首先探索模型结构..."
echo

# 先探索结构
python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/inspect_layer_weights.py \
    --model "$FP16_FILE" \
    --explore-only \
    --explore-depth 5

echo
echo "=========================================="
echo "现在尝试检查第1层的注意力权重..."
echo

# 尝试查找第1层的注意力权重
python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/inspect_layer_weights.py \
    --model "$FP16_FILE" \
    --layer "language_model.encoder.layers.1.self_attention.query_key_value.weight"

echo
echo "如果上面的路径不正确，让我们尝试其他可能的路径..."
echo

# 备用路径选项
BACKUP_PATHS=(
    "language_model.encoder.layers.0.self_attention.query_key_value.weight"
    "language_model.encoder.layers.1.self_attention.dense.weight"
    "language_model.embedding.word_embeddings.weight"
    "language_model.encoder.layers.0.mlp.dense_h_to_4h.weight"
)

for path in "${BACKUP_PATHS[@]}"; do
    echo "----------------------------------------"
    echo "尝试路径: $path"
    python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/inspect_layer_weights.py \
        --model "$FP16_FILE" \
        --layer "$path" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功找到权重: $path"
        break
    else
        echo "❌ 路径无效: $path"
    fi
done

echo
echo "层权重检查完成！"
