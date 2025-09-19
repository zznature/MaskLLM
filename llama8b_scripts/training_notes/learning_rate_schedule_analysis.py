#!/usr/bin/env python3
"""
Llama8b学习率调度分析脚本
分析2000迭代训练过程中学习率的详细演进
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_learning_rate(iteration, total_iters=2000, warmup_iters=400, 
                          lr_max=2e-5, lr_min=2e-6):
    """
    计算指定迭代的学习率
    
    Args:
        iteration: 当前迭代数
        total_iters: 总迭代数 (2000)
        warmup_iters: 预热迭代数 (400) 
        lr_max: 最大学习率 (2e-5)
        lr_min: 最小学习率 (2e-6)
    
    Returns:
        当前迭代的学习率
    """
    if iteration <= warmup_iters:
        # 线性预热阶段
        return lr_max * (iteration / warmup_iters)
    else:
        # 余弦衰减阶段
        decay_progress = (iteration - warmup_iters) / (total_iters - warmup_iters)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * decay_progress))

def analyze_learning_rate_schedule():
    """分析学习率调度的详细信息"""
    
    print("📈 Llama8b学习率调度详细分析")
    print("=" * 50)
    
    # 参数配置
    total_iters = 2000
    warmup_iters = 400
    lr_max = 2e-5
    lr_min = 2e-6
    
    print(f"🔧 配置参数:")
    print(f"  - 总迭代数: {total_iters}")
    print(f"  - 预热迭代数: {warmup_iters}")
    print(f"  - 最大学习率: {lr_max:.0e}")
    print(f"  - 最小学习率: {lr_min:.0e}")
    print(f"  - 衰减方式: 余弦衰减")
    print()
    
    # 关键节点分析
    key_iterations = [0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    
    print("🎯 关键迭代点学习率:")
    print("迭代次数  |  学习率    |  阶段描述")
    print("-" * 40)
    
    for iter_num in key_iterations:
        lr = calculate_learning_rate(iter_num, total_iters, warmup_iters, lr_max, lr_min)
        
        if iter_num <= warmup_iters:
            phase = "预热阶段"
            progress = f"{iter_num/warmup_iters*100:.0f}%预热"
        else:
            phase = "衰减阶段" 
            decay_progress = (iter_num - warmup_iters) / (total_iters - warmup_iters)
            progress = f"{decay_progress*100:.0f}%衰减"
            
        print(f"{iter_num:8d}  |  {lr:.2e}  |  {phase} ({progress})")
    
    print()
    
    # 阶段统计
    print("📊 阶段统计:")
    print(f"🌅 预热阶段 (0-{warmup_iters}): {warmup_iters/total_iters*100:.0f}% 训练时间")
    print(f"🌇 衰减阶段 ({warmup_iters}-{total_iters}): {(total_iters-warmup_iters)/total_iters*100:.0f}% 训练时间")
    print()
    
    # 学习率变化统计
    warmup_end_lr = calculate_learning_rate(warmup_iters, total_iters, warmup_iters, lr_max, lr_min)
    final_lr = calculate_learning_rate(total_iters, total_iters, warmup_iters, lr_max, lr_min)
    
    print("📈 学习率变化统计:")
    print(f"  - 预热增长: 0 → {warmup_end_lr:.2e} (增长{warmup_end_lr/lr_max*100:.0f}%)")
    print(f"  - 衰减幅度: {warmup_end_lr:.2e} → {final_lr:.2e} (衰减{(1-final_lr/warmup_end_lr)*100:.0f}%)")
    print(f"  - 总体衰减: {lr_max:.2e} → {final_lr:.2e} (最终为初始的{final_lr/lr_max*100:.0f}%)")
    print()
    
    # 生成完整曲线数据
    iterations = list(range(0, total_iters + 1, 10))  # 每10步采样
    learning_rates = [calculate_learning_rate(i, total_iters, warmup_iters, lr_max, lr_min) 
                     for i in iterations]
    
    # 保存数据到文件
    with open('llama8b_scripts/learning_rate_curve_data.csv', 'w') as f:
        f.write("iteration,learning_rate,phase\n")
        for i, lr in zip(iterations, learning_rates):
            phase = "warmup" if i <= warmup_iters else "decay"
            f.write(f"{i},{lr:.8e},{phase}\n")
    
    print("💾 学习率曲线数据已保存到: llama8b_scripts/learning_rate_curve_data.csv")
    
    # 计算梯度(学习率变化率)
    print("\n🔍 学习率变化速度分析:")
    sample_points = [50, 200, 350, 450, 800, 1200, 1600, 1950]
    
    for point in sample_points:
        if point < total_iters:
            lr_before = calculate_learning_rate(max(0, point-10), total_iters, warmup_iters, lr_max, lr_min)
            lr_after = calculate_learning_rate(min(total_iters, point+10), total_iters, warmup_iters, lr_max, lr_min)
            gradient = (lr_after - lr_before) / 20  # 20步的变化率
            
            phase = "预热" if point <= warmup_iters else "衰减"
            print(f"  迭代{point:4d} ({phase}): 变化率 = {gradient:.2e}/step")

if __name__ == "__main__":
    analyze_learning_rate_schedule()
