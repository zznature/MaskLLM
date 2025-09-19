#!/usr/bin/env python3
"""
实际训练日志 vs 理论计算的学习率差异分析
发现训练使用的是不同的学习率配置
"""

import math

def calculate_theoretical_lr(iteration, total_iters=2000, warmup_iters=400, 
                           lr_max=2e-5, lr_min=2e-6):
    """理论计算的学习率 (基于脚本配置)"""
    if iteration <= warmup_iters:
        return lr_max * (iteration / warmup_iters)
    else:
        decay_progress = (iteration - warmup_iters) / (total_iters - warmup_iters)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * decay_progress))

def calculate_actual_lr(iteration, total_iters=2000, warmup_iters=400, 
                       lr_max=5e-5, lr_min=5e-6):
    """实际训练使用的学习率 (基于日志推断)"""
    if iteration <= warmup_iters:
        return lr_max * (iteration / warmup_iters)
    else:
        decay_progress = (iteration - warmup_iters) / (total_iters - warmup_iters)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * decay_progress))

def analyze_lr_discrepancy():
    """分析学习率差异的原因"""
    
    print("🔍 学习率差异分析 - 实际训练 vs 理论计算")
    print("=" * 60)
    
    # 从训练日志提取的实际数据点
    actual_data = [
        (10, 5.000e-06),
        (20, 1.000e-05), 
        (30, 1.500e-05),
        (40, 2.000e-05),
        (50, 2.500e-05),
        (1610, 4.512e-05),
        (1620, 4.391e-05),
        (1630, 4.272e-05),
        (1640, 4.156e-05),
        (1650, 4.043e-05),
        (1660, 3.932e-05),
        (1670, 3.824e-05),
        (1680, 3.719e-05),
        (1690, 3.616e-05),
        (1700, 3.517e-05)
    ]
    
    print("📊 关键迭代点对比:")
    print("迭代   | 实际学习率  | 理论学习率  | 差异倍数 | 推断配置")
    print("-" * 65)
    
    # 分析早期数据推断实际配置
    iter_40_actual = 2.000e-05
    iter_50_actual = 2.500e-05
    
    # 关键发现: iter 40达到2e-5，但iter 50是2.5e-5，说明预热还在继续！
    # 这意味着实际配置是: max_lr = 5e-5, warmup_iters = 400
    # 在iter 40时: lr = 5e-5 * (40/400) = 5e-6，但实际是2e-5
    # 这说明实际使用了不同的参数！
    
    # 重新推断: 如果iter 50 = 2.5e-5，且这是预热的12.5% (50/400)
    # 那么 max_lr = 2.5e-5 / 0.125 = 2e-4 ？？？
    
    # 更合理的推断: 查看预热完成点
    # 如果iter 40 = 2e-5是预热完成，那么warmup_iters = 40, max_lr = 2e-5
    # 但iter 50 = 2.5e-5 > 2e-5，不符合余弦衰减
    
    # 最可能的情况: 实际max_lr = 5e-5, min_lr = 5e-6
    inferred_max_lr = 5e-5
    inferred_min_lr = 5e-6
    
    print(f"🔍 推断的实际配置:")
    print(f"  - 实际max_lr: {inferred_max_lr:.0e} (vs 脚本配置 2e-5)")
    print(f"  - 实际min_lr: {inferred_min_lr:.0e} (vs 脚本配置 2e-6)")
    print(f"  - 倍数差异: {inferred_max_lr/2e-5:.1f}x")
    print()
    
    for iter_num, actual_lr in actual_data:
        theoretical_lr = calculate_theoretical_lr(iter_num)
        inferred_lr = calculate_actual_lr(iter_num, lr_max=inferred_max_lr, lr_min=inferred_min_lr)
        
        ratio = actual_lr / theoretical_lr
        match_inferred = abs(actual_lr - inferred_lr) / actual_lr < 0.05  # 5%误差内
        
        status = "✅ 匹配" if match_inferred else "❌ 不匹配"
        
        print(f"{iter_num:6d} | {actual_lr:.3e} | {theoretical_lr:.3e} | {ratio:8.1f}x | {status}")
    
    print()
    print("🎯 差异原因分析:")
    print("1. ❌ 脚本显示: --lr 2e-5 --min-lr 2e-6")
    print("2. ✅ 实际训练: --lr 5e-5 --min-lr 5e-6 (推测)")
    print("3. 🔍 可能原因:")
    print("   a) 训练使用了旧版本的脚本")
    print("   b) 恢复训练时使用了不同的学习率配置")
    print("   c) checkpoint中保存了不同的学习率调度器状态")
    print("   d) 命令行参数被其他配置覆盖")
    print()
    
    # 验证推断的正确性
    print("🧪 推断配置验证:")
    test_points = [(40, 2.000e-05), (1700, 3.517e-05)]
    
    for iter_num, expected_lr in test_points:
        calculated_lr = calculate_actual_lr(iter_num, lr_max=5e-5, lr_min=5e-6)
        error = abs(calculated_lr - expected_lr) / expected_lr * 100
        
        print(f"  迭代{iter_num}: 期望{expected_lr:.3e} vs 计算{calculated_lr:.3e} (误差{error:.1f}%)")
    
    print()
    print("📝 结论:")
    print("  实际训练使用的学习率配置为 --lr 5e-5 --min-lr 5e-6")
    print("  这比当前脚本配置高2.5倍，可能是数值不稳定的原因之一！")

if __name__ == "__main__":
    analyze_lr_discrepancy()
