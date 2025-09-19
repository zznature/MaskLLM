#!/usr/bin/env python3
"""
专门分析 layer0.self_attention.query_key_value.weight 的参数分布
"""
import torch

# 加载模型
model_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"
checkpoint = torch.load(model_path, map_location='cpu')

print("🔍 访问Layer0的Query-Key-Value权重...")

# 根据实际模型结构，使用扁平化的键名
target_key = 'layers.0.self_attention.query_key_value.weight'
encoder = checkpoint['model']['language_model']['encoder']

if target_key in encoder:
    weight = encoder[target_key]
    print(f"✅ 成功找到权重: {target_key}")
    print(f"   形状: {weight.shape}")
    print(f"   数据类型: {weight.dtype}")
else:
    print(f"❌ 未找到权重键: {target_key}")
    print("可用的相关键:")
    layer0_keys = [k for k in encoder.keys() if k.startswith('layers.0.')]
    for key in layer0_keys[:10]:  # 只显示前10个
        print(f"  {key}")
    exit(1)

print("🎯 Layer0 Self-Attention Query-Key-Value Weight Analysis")
print("=" * 60)

# 基本信息
print(f"张量形状: {weight.shape}")
print(f"数据类型: {weight.dtype}")
print(f"参数总数: {weight.numel():,}")

# 扁平化权重
flat_weights = weight.flatten()

# 基本统计
print(f"\n📊 统计信息:")
print(f"最小值: {flat_weights.min():.8f}")
print(f"最大值: {flat_weights.max():.8f}")
print(f"均值:   {flat_weights.mean():.8f}")
print(f"标准差: {flat_weights.std():.8f}")

# 稀疏性分析
threshold = 1e-6
near_zero = (flat_weights.abs() < threshold).sum()
total = len(flat_weights)
sparsity = float(near_zero) / total

print(f"\n🎯 稀疏性分析:")
print(f"阈值 (|w| < {threshold}): {near_zero:,} / {total:,}")
print(f"稀疏率: {sparsity:.6f} ({sparsity*100:.4f}%)")

# 权重分布统计
very_small = (flat_weights.abs() < 1e-4).sum()
small = ((flat_weights.abs() >= 1e-4) & (flat_weights.abs() < 1e-3)).sum()
medium = ((flat_weights.abs() >= 1e-3) & (flat_weights.abs() < 1e-2)).sum()
large = (flat_weights.abs() >= 1e-2).sum()

print(f"\n📈 权重分布:")
print(f"|w| < 1e-4:        {very_small:,} ({float(very_small)/total*100:.2f}%)")
print(f"1e-4 ≤ |w| < 1e-3: {small:,} ({float(small)/total*100:.2f}%)")
print(f"1e-3 ≤ |w| < 1e-2: {medium:,} ({float(medium)/total*100:.2f}%)")
print(f"|w| ≥ 1e-2:        {large:,} ({float(large)/total*100:.2f}%)")

# 权重样本
print(f"\n🔍 权重样本 (前30个):")
for i in range(min(30, len(flat_weights))):
    print(f"[{i:2d}] {flat_weights[i]:.8f}")

# 极值样本
print(f"\n🔥 极值样本:")
sorted_values, sorted_indices = torch.sort(flat_weights.abs(), descending=True)
print("绝对值最大的10个权重:")
for i in range(min(10, len(sorted_values))):
    original_value = flat_weights[sorted_indices[i]]
    print(f"  第{i+1}大: {original_value:.8f} (|{sorted_values[i]:.8f}|)")

print(f"\n✅ 分析完成!")
