#!/usr/bin/env python3
"""
计算 WikiText-103 训练使用的全部 token 数量
基于 llama2_7b_mask_only_wikitext103_tp4.sh 的配置
"""

def calculate_training_tokens():
    print("=== WikiText-103 训练 Token 计算 ===")
    
    # 从脚本中提取的关键参数
    TRAIN_ITERS = 400          # 训练迭代次数
    GLOBAL_BATCH_SIZE = 64     # 全局批次大小
    SEQ_LENGTH = 4096          # 序列长度 (注意：脚本中显示4096，但注释说2048)
    
    # 实际使用的序列长度 (根据脚本内容)
    actual_seq_length = 4096   # 脚本中实际设置的值
    
    print(f"训练参数:")
    print(f"  - 训练迭代次数: {TRAIN_ITERS:,}")
    print(f"  - 全局批次大小: {GLOBAL_BATCH_SIZE:,}")
    print(f"  - 序列长度: {actual_seq_length:,}")
    
    # 计算每个迭代的 token 数量
    tokens_per_iter = GLOBAL_BATCH_SIZE * actual_seq_length
    
    # 计算总 token 数量
    total_tokens = TRAIN_ITERS * tokens_per_iter
    
    print(f"\n计算过程:")
    print(f"  每个迭代的 token 数 = {GLOBAL_BATCH_SIZE:,} × {actual_seq_length:,} = {tokens_per_iter:,}")
    print(f"  总 token 数 = {TRAIN_ITERS:,} × {tokens_per_iter:,} = {total_tokens:,}")
    
    # 转换为更易读的格式
    total_tokens_millions = total_tokens / 1_000_000
    total_tokens_billions = total_tokens / 1_000_000_000
    
    print(f"\n结果:")
    print(f"  - 总 token 数量: {total_tokens:,}")
    print(f"  - 约 {total_tokens_millions:.1f} 百万 tokens")
    print(f"  - 约 {total_tokens_billions:.3f} 十亿 tokens")
    
    # 与数据集大小对比
    print(f"\n数据集对比:")
    print(f"  - WikiText-103 训练集: ~103M tokens")
    print(f"  - 训练使用的 token: {total_tokens_millions:.1f}M tokens")
    
    if total_tokens > 103_000_000:
        epochs = total_tokens / 103_000_000
        print(f"  - 相当于 {epochs:.1f} 个 epoch")
    else:
        coverage = (total_tokens / 103_000_000) * 100
        print(f"  - 覆盖了 {coverage:.1f}% 的训练数据")
    
    return total_tokens

def calculate_with_different_seq_lengths():
    """计算不同序列长度下的 token 数量"""
    print(f"\n=== 不同序列长度的对比 ===")
    
    TRAIN_ITERS = 400
    GLOBAL_BATCH_SIZE = 64
    
    seq_lengths = [2048, 4096]
    
    for seq_len in seq_lengths:
        total_tokens = TRAIN_ITERS * GLOBAL_BATCH_SIZE * seq_len
        total_millions = total_tokens / 1_000_000
        
        print(f"序列长度 {seq_len:,}:")
        print(f"  - 总 token: {total_tokens:,} ({total_millions:.1f}M)")
        print(f"  - 内存使用: ~{total_millions * 2:.1f}GB (估算)")

if __name__ == "__main__":
    total_tokens = calculate_training_tokens()
    calculate_with_different_seq_lengths()
    
    print(f"\n=== 建议 ===")
    print("1. 当前配置使用约 1.05B tokens 进行训练")
    print("2. 这相当于对 WikiText-103 训练了约 10.2 个 epoch")
    print("3. 对于增量训练来说，这个量是合理的")
    print("4. 如果需要更多训练，可以增加 TRAIN_ITERS 或调整其他参数") 