#!/usr/bin/env python3
"""
最简单的权重检查脚本
"""
import torch
import os

# 模型路径
model_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"

print("🔍 快速权重检查")
print(f"文件: {os.path.basename(model_path)}")
print(f"大小: {os.path.getsize(model_path) / (1024**3):.1f} GB")

# 加载模型
checkpoint = torch.load(model_path, map_location='cpu')
model = checkpoint['model']['language_model']

print(f"\n📂 language_model结构:")
print(f"- 顶层键: {list(model.keys())}")

# 探索encoder layers
encoder = model['encoder']
layers = encoder['layers']
print(f"- 层数: {len(layers)}")

# 检查第1层
layer1 = layers['1']
print(f"\n🎯 第1层结构:")
print(f"- 键: {list(layer1.keys())}")

# 检查self_attention
sa = layer1['self_attention']
print(f"- self_attention键: {list(sa.keys())}")

# 找到query_key_value权重
if 'query_key_value' in sa and 'weight' in sa['query_key_value']:
    weight = sa['query_key_value']['weight']
    
    print(f"\n📊 query_key_value.weight 分析:")
    print(f"- 形状: {weight.shape}")
    print(f"- 数据类型: {weight.dtype}")
    print(f"- 参数数量: {weight.numel():,}")
    
    # 基本统计
    flat = weight.flatten()
    print(f"\n📈 统计信息:")
    print(f"- 最小值: {flat.min():.8f}")
    print(f"- 最大值: {flat.max():.8f}")
    print(f"- 均值: {flat.mean():.8f}")
    print(f"- 标准差: {flat.std():.8f}")
    
    # 稀疏性
    near_zero = (flat.abs() < 1e-6).sum()
    total = len(flat)
    sparsity = float(near_zero) / total
    print(f"- 稀疏率: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"- 接近零的参数: {near_zero:,} / {total:,}")
    
    # 权重分布
    small = (flat.abs() < 0.001).sum()
    medium = ((flat.abs() >= 0.001) & (flat.abs() < 0.01)).sum()
    large = (flat.abs() >= 0.01).sum()
    
    print(f"\n📊 权重分布:")
    print(f"- |w| < 0.001:     {small:,} ({float(small)/total*100:.1f}%)")
    print(f"- 0.001 ≤ |w| < 0.01: {medium:,} ({float(medium)/total*100:.1f}%)")
    print(f"- |w| ≥ 0.01:      {large:,} ({float(large)/total*100:.1f}%)")
    
    # 前20个权重样本
    print(f"\n🎯 前20个权重值:")
    for i in range(min(20, len(flat))):
        print(f"  [{i:2d}] {flat[i]:.8f}")
    
    print(f"\n✅ 分析完成!")
else:
    print("❌ 未找到query_key_value.weight")
    if 'query_key_value' in sa:
        print(f"query_key_value类型: {type(sa['query_key_value'])}")
        if isinstance(sa['query_key_value'], dict):
            print(f"query_key_value键: {list(sa['query_key_value'].keys())}")