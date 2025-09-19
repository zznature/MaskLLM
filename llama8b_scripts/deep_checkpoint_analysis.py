#!/usr/bin/env python3
"""
深入分析MaskLLM检查点文件结构
"""

import torch
import os
from collections import defaultdict
import json

def explore_nested_dict(obj, prefix="", max_depth=5, current_depth=0):
    """递归探索嵌套字典结构"""
    if current_depth > max_depth:
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}Dict with {len(obj)} keys:")
        for key in obj.keys():
            print(f"{prefix}  - {key}: {type(obj[key])}")
            if hasattr(obj[key], 'shape'):
                print(f"{prefix}    Shape: {obj[key].shape}, dtype: {obj[key].dtype}")
            elif hasattr(obj[key], '__len__') and not isinstance(obj[key], (str, bytes)):
                try:
                    print(f"{prefix}    Length: {len(obj[key])}")
                except:
                    pass
            
            # 递归探索嵌套结构
            if isinstance(obj[key], dict) and current_depth < 3:
                explore_nested_dict(obj[key], prefix + "  ", max_depth, current_depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}List/Tuple with {len(obj)} items")
        if len(obj) > 0:
            print(f"{prefix}  First item type: {type(obj[0])}")
            if hasattr(obj[0], 'shape'):
                print(f"{prefix}  First item shape: {obj[0].shape}")

def analyze_model_weights(model_state, prefix=""):
    """详细分析模型权重"""
    print(f"{prefix}=== 模型权重详细分析 ===")
    
    if isinstance(model_state, dict):
        print(f"{prefix}模型状态是字典，包含 {len(model_state)} 个键")
        
        total_params = 0
        dtype_stats = defaultdict(lambda: {'count': 0, 'params': 0, 'size_gb': 0})
        
        for key, value in model_state.items():
            print(f"{prefix}{key}: {type(value)}")
            
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                # 这是一个张量
                num_params = value.numel()
                param_size_bytes = value.element_size() * num_params
                dtype = str(value.dtype)
                
                dtype_stats[dtype]['count'] += 1
                dtype_stats[dtype]['params'] += num_params
                dtype_stats[dtype]['size_gb'] += param_size_bytes / (1024**3)
                total_params += num_params
                
                print(f"{prefix}  Shape: {value.shape}")
                print(f"{prefix}  Dtype: {dtype}")
                print(f"{prefix}  Parameters: {num_params:,}")
                print(f"{prefix}  Size: {param_size_bytes / (1024**2):.2f} MB")
                
            elif isinstance(value, dict):
                print(f"{prefix}  嵌套字典，包含键: {list(value.keys())}")
                # 递归分析嵌套的模型权重
                sub_total = analyze_model_weights(value, prefix + "  ")
                total_params += sub_total
            
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}  列表/元组，长度: {len(value)}")
                if len(value) > 0 and hasattr(value[0], 'shape'):
                    print(f"{prefix}  第一个元素形状: {value[0].shape}")
        
        if dtype_stats:
            print(f"{prefix}=== 数据类型统计 ===")
            for dtype, stats in dtype_stats.items():
                print(f"{prefix}{dtype}:")
                print(f"{prefix}  权重数量: {stats['count']}")
                print(f"{prefix}  参数数量: {stats['params']:,} ({stats['params'] / 1e9:.2f}B)")
                print(f"{prefix}  占用空间: {stats['size_gb']:.2f} GB")
        
        return total_params
    
    elif hasattr(model_state, 'state_dict'):
        print(f"{prefix}模型状态有state_dict方法")
        try:
            state_dict = model_state.state_dict()
            return analyze_model_weights(state_dict, prefix)
        except Exception as e:
            print(f"{prefix}调用state_dict()失败: {e}")
            return 0
    
    else:
        print(f"{prefix}模型状态类型: {type(model_state)}")
        if hasattr(model_state, '__dict__'):
            print(f"{prefix}对象属性: {list(model_state.__dict__.keys())}")
        return 0

def main():
    checkpoint_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
    
    print("=== 深入MaskLLM检查点分析 ===\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    
    try:
        print("正在加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"检查点顶层键: {list(checkpoint.keys())}")
        
        # 详细分析每个顶层键
        for key, value in checkpoint.items():
            print(f"\n=== 分析键: {key} ===")
            print(f"类型: {type(value)}")
            
            if key == 'model':
                total_params = analyze_model_weights(value)
                print(f"模型总参数数: {total_params:,} ({total_params / 1e9:.2f}B)")
                
            elif key == 'args':
                if hasattr(value, '__dict__'):
                    args_dict = vars(value)
                    print("训练参数:")
                    for arg_key, arg_value in args_dict.items():
                        if 'tensor_parallel' in arg_key or 'pipeline_parallel' in arg_key:
                            print(f"  {arg_key}: {arg_value}")
                        elif 'dtype' in arg_key or 'precision' in arg_key:
                            print(f"  {arg_key}: {arg_value}")
            
            elif isinstance(value, (dict, list, tuple)):
                explore_nested_dict(value, "  ")
            
            else:
                print(f"值: {value}")
        
        # 尝试手动计算文件中所有张量的总大小
        print(f"\n=== 手动计算所有张量大小 ===")
        total_tensor_size = 0
        
        def calc_tensor_size(obj, path=""):
            nonlocal total_tensor_size
            if hasattr(obj, 'element_size') and hasattr(obj, 'numel'):
                size = obj.element_size() * obj.numel()
                total_tensor_size += size
                print(f"张量 {path}: {size / (1024**2):.2f} MB, dtype: {obj.dtype}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    calc_tensor_size(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    calc_tensor_size(v, f"{path}[{i}]" if path else f"[{i}]")
        
        calc_tensor_size(checkpoint)
        print(f"所有张量总大小: {total_tensor_size / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
