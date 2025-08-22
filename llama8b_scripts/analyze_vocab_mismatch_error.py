#!/usr/bin/env python3
"""
词汇表分片不匹配错误分析脚本
分析checkpoint中的embedding权重形状与训练脚本期望形状的差异
"""

import math

def analyze_vocab_mismatch():
    print("🔍 词汇表分片不匹配错误分析")
    print("=" * 60)
    
    # 错误信息分析
    print("📋 错误信息分析:")
    print("错误类型: RuntimeError in VocabParallelEmbedding")
    print("Checkpoint权重形状: torch.Size([14962, 4096])")
    print("当前模型期望形状: torch.Size([14976, 4096])")
    print("差异: 14976 - 14962 = 14 个参数")
    print("")
    
    # 原始配置分析
    print("📊 原始Llama8b配置:")
    original_vocab_size = 119696
    print(f"原始词汇表大小: {original_vocab_size}")
    
    # Megatron并行分片计算
    print("")
    print("🔧 Megatron张量并行分片计算:")
    tensor_parallel_size = 8
    
    # 计算分片大小
    vocab_size_per_partition = original_vocab_size // tensor_parallel_size
    remainder = original_vocab_size % tensor_parallel_size
    
    print(f"张量并行大小: {tensor_parallel_size}")
    print(f"基础分片大小: {vocab_size_per_partition}")
    print(f"余数: {remainder}")
    print("")
    
    # Megatron的分片策略
    print("📋 Megatron分片策略分析:")
    print("分片分配:")
    for i in range(tensor_parallel_size):
        start_idx = i * vocab_size_per_partition
        if i < remainder:
            end_idx = start_idx + vocab_size_per_partition + 1
        else:
            end_idx = start_idx + vocab_size_per_partition
        shard_size = end_idx - start_idx
        print(f"  分片 {i}: [{start_idx}:{end_idx}] = {shard_size} 个词汇")
    
    print("")
    
    # make-vocab-size-divisible-by 分析
    print("🔧 make-vocab-size-divisible-by 分析:")
    divisible_by = 128  # 从脚本中看到的值
    
    # 计算填充后的词汇表大小
    padded_vocab_size = math.ceil(original_vocab_size / divisible_by) * divisible_by
    print(f"填充标准: {divisible_by}")
    print(f"填充后词汇表大小: {padded_vocab_size}")
    print(f"填充增加: {padded_vocab_size - original_vocab_size}")
    
    # 填充后的分片计算
    print("")
    print("📊 填充后分片计算:")
    padded_per_partition = padded_vocab_size // tensor_parallel_size
    padded_remainder = padded_vocab_size % tensor_parallel_size
    
    print(f"填充后基础分片大小: {padded_per_partition}")
    print(f"填充后余数: {padded_remainder}")
    
    print("填充后分片分配:")
    for i in range(tensor_parallel_size):
        start_idx = i * padded_per_partition
        if i < padded_remainder:
            end_idx = start_idx + padded_per_partition + 1
        else:
            end_idx = start_idx + padded_per_partition
        shard_size = end_idx - start_idx
        print(f"  分片 {i}: [{start_idx}:{end_idx}] = {shard_size} 个词汇")
    
    print("")
    
    # 问题根源分析
    print("🎯 问题根源分析:")
    print("1. Checkpoint转换时的vocab_size配置:")
    print(f"   - 可能使用了原始大小: {original_vocab_size}")
    print(f"   - 分片大小: {original_vocab_size // tensor_parallel_size} = {vocab_size_per_partition}")
    
    print("")
    print("2. 训练脚本的vocab_size配置:")
    print(f"   - 使用了填充后大小: {padded_vocab_size}")
    print(f"   - 分片大小: {padded_vocab_size // tensor_parallel_size} = {padded_per_partition}")
    
    print("")
    print("3. 实际错误:")
    checkpoint_shard_size = 14962
    expected_shard_size = 14976
    print(f"   - Checkpoint中分片大小: {checkpoint_shard_size}")
    print(f"   - 训练时期望分片大小: {expected_shard_size}")
    print(f"   - 差异: {expected_shard_size - checkpoint_shard_size}")
    
    print("")
    print("🔧 解决方案:")
    print("方案1: 修改训练脚本的vocab_size参数")
    print(f"   - 将 --vocab-size {padded_vocab_size} 改为 --vocab-size {original_vocab_size}")
    print("")
    print("方案2: 重新转换checkpoint使用正确的vocab_size")
    print(f"   - 转换时使用 --vocab-size {padded_vocab_size}")
    print("")
    print("方案3: 调整 make-vocab-size-divisible-by 参数")
    print("   - 使转换和训练使用相同的填充策略")
    
    return {
        'original_vocab_size': original_vocab_size,
        'padded_vocab_size': padded_vocab_size,
        'checkpoint_shard_size': checkpoint_shard_size,
        'expected_shard_size': expected_shard_size,
        'tensor_parallel_size': tensor_parallel_size
    }

if __name__ == "__main__":
    result = analyze_vocab_mismatch()
    
    print("")
    print("=" * 60)
    print("🎯 推荐立即执行:")
    print("1. 检查checkpoint转换时的实际vocab_size配置")
    print("2. 修改训练脚本使用一致的vocab_size")
    print("3. 验证make-vocab-size-divisible-by参数一致性")
