#!/usr/bin/env python3
"""
Analyze MaskLLM checkpoint file to determine data types and sizes
"""

import torch
import os
from collections import defaultdict

def analyze_checkpoint_file(checkpoint_path):
    """分析检查点文件的详细信息"""
    
    print(f"=== 分析检查点文件: {checkpoint_path} ===")
    
    # 检查文件大小
    file_size = os.path.getsize(checkpoint_path)
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    
    try:
        # 加载检查点
        print("正在加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"检查点顶层键: {list(checkpoint.keys())}")
        
        # 分析模型权重
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"\n=== 模型权重分析 ===")
            print(f"模型权重数量: {len(model_state)}")
            
            # 统计参数类型和数量
            dtype_stats = defaultdict(lambda: {'count': 0, 'params': 0, 'size_gb': 0})
            total_params = 0
            
            print("\n权重详细信息:")
            for name, param in model_state.items():
                if hasattr(param, 'dtype') and hasattr(param, 'numel'):
                    dtype = str(param.dtype)
                    num_params = param.numel()
                    param_size_bytes = param.element_size() * num_params
                    
                    dtype_stats[dtype]['count'] += 1
                    dtype_stats[dtype]['params'] += num_params
                    dtype_stats[dtype]['size_gb'] += param_size_bytes / (1024**3)
                    total_params += num_params
                    
                    print(f"  {name}: {param.shape}, dtype={dtype}, params={num_params:,}")
            
            print(f"\n=== 数据类型统计 ===")
            print(f"总参数数量: {total_params:,} ({total_params / 1e9:.2f}B)")
            
            for dtype, stats in dtype_stats.items():
                print(f"{dtype}:")
                print(f"  - 权重数量: {stats['count']}")
                print(f"  - 参数数量: {stats['params']:,} ({stats['params'] / 1e9:.2f}B)")
                print(f"  - 占用空间: {stats['size_gb']:.2f} GB")
        
        # 分析优化器状态
        if 'optimizer' in checkpoint:
            print(f"\n=== 优化器状态分析 ===")
            optimizer_state = checkpoint['optimizer']
            if 'state' in optimizer_state:
                opt_states = optimizer_state['state']
                print(f"优化器状态数量: {len(opt_states)}")
                
                # 计算优化器状态大小
                opt_size_gb = 0
                for param_id, state in opt_states.items():
                    if isinstance(state, dict):
                        for key, value in state.items():
                            if hasattr(value, 'element_size') and hasattr(value, 'numel'):
                                opt_size_gb += value.element_size() * value.numel() / (1024**3)
                
                print(f"优化器状态大小: {opt_size_gb:.2f} GB")
        
        # 分析其他组件
        other_components = [k for k in checkpoint.keys() if k not in ['model', 'optimizer']]
        if other_components:
            print(f"\n=== 其他组件 ===")
            for comp in other_components:
                print(f"- {comp}: {type(checkpoint[comp])}")
                if hasattr(checkpoint[comp], '__len__'):
                    try:
                        print(f"  长度: {len(checkpoint[comp])}")
                    except:
                        pass
        
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        print("尝试使用torch.jit.load...")
        try:
            # 尝试其他加载方式
            model = torch.jit.load(checkpoint_path, map_location='cpu')
            print("成功使用torch.jit.load加载")
        except Exception as e2:
            print(f"torch.jit.load也失败: {e2}")

def main():
    checkpoint_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
    
    print("=== MaskLLM Checkpoint Analysis ===\n")
    
    if os.path.exists(checkpoint_path):
        analyze_checkpoint_file(checkpoint_path)
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
    
    # 理论计算说明
    print(f"\n=== 理论大小计算 ===")
    print("对于8B参数的模型:")
    print("- FP32 (float32): 8B × 4 bytes = 32GB")
    print("- FP16 (float16): 8B × 2 bytes = 16GB") 
    print("- BF16 (bfloat16): 8B × 2 bytes = 16GB")
    print("")
    print("注意: 检查点文件通常包含:")
    print("- 模型权重")
    print("- 优化器状态 (momentum, variance等，通常2-3倍模型大小)")
    print("- 训练状态信息")
    print("因此总大小会比纯模型权重大很多")

if __name__ == "__main__":
    main()
