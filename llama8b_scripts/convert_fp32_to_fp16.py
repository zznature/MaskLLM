#!/usr/bin/env python3
"""
将Llama8b-MaskLLM检查点从FP32转换为FP16格式
节省50%存储空间，提升推理效率
"""

import torch
import os
import shutil
import argparse
from pathlib import Path
import sys
import time

def convert_tensors_to_fp16(obj):
    """
    递归转换PyTorch对象中的所有FP32张量为FP16
    """
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float32:
            return obj.half()
        else:
            return obj
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_fp16(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_tensors_to_fp16(x) for x in obj)
    else:
        return obj

def count_tensors_by_dtype(obj, counts=None):
    """
    统计对象中不同数据类型张量的数量
    """
    if counts is None:
        counts = {}
    
    if isinstance(obj, torch.Tensor):
        dtype = str(obj.dtype)
        counts[dtype] = counts.get(dtype, 0) + 1
    elif isinstance(obj, dict):
        for v in obj.values():
            count_tensors_by_dtype(v, counts)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            count_tensors_by_dtype(x, counts)
    
    return counts

def safe_fp32_to_fp16_conversion(
    input_path: str, 
    output_path: str, 
    backup: bool = True,
    verify: bool = True,
    force: bool = False
):
    """
    安全地将FP32检查点转换为FP16
    
    Args:
        input_path: 原始FP32检查点路径
        output_path: 输出FP16检查点路径
        backup: 是否创建备份
        verify: 是否验证转换结果
        force: 是否强制覆盖输出文件
    
    Returns:
        转换后的文件路径
    """
    
    print("=" * 60)
    print("Llama8b-MaskLLM FP32→FP16 转换工具")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 检查输出文件
    if os.path.exists(output_path) and not force:
        response = input(f"输出文件 {output_path} 已存在，是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("转换取消")
            return None
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print()
    
    # 创建备份
    if backup:
        backup_path = input_path + '.fp32_backup'
        if not os.path.exists(backup_path):
            print(f"创建备份: {backup_path}")
            print("备份中... (这可能需要几分钟)")
            shutil.copy2(input_path, backup_path)
            print("✓ 备份完成")
        else:
            print(f"✓ 备份已存在: {backup_path}")
        print()
    
    # 获取原始文件信息
    original_size = os.path.getsize(input_path) / (1024**3)
    print(f"原始文件大小: {original_size:.2f} GB")
    
    # 加载检查点
    print("加载原始检查点... (这可能需要几分钟)")
    start_time = time.time()
    
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
        load_time = time.time() - start_time
        print(f"✓ 加载完成 ({load_time:.1f}s)")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    # 分析原始数据类型分布
    original_counts = count_tensors_by_dtype(checkpoint)
    print("\n原始数据类型分布:")
    for dtype, count in sorted(original_counts.items()):
        print(f"  {dtype}: {count} 个张量")
    
    # 转换权重
    print(f"\n转换权重到FP16... (预计节省 {original_size/2:.2f} GB)")
    start_time = time.time()
    
    try:
        converted_checkpoint = convert_tensors_to_fp16(checkpoint)
        convert_time = time.time() - start_time
        print(f"✓ 转换完成 ({convert_time:.1f}s)")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return None
    
    # 保存FP16检查点
    print(f"\n保存FP16检查点: {output_path}")
    start_time = time.time()
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(converted_checkpoint, output_path)
        save_time = time.time() - start_time
        print(f"✓ 保存完成 ({save_time:.1f}s)")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None
    
    # 验证结果
    if verify:
        print(f"\n验证转换结果...")
        
        try:
            # 检查文件大小
            new_size = os.path.getsize(output_path) / (1024**3)
            compression_ratio = (1 - new_size/original_size) * 100
            
            print(f"转换后文件大小: {new_size:.2f} GB")
            print(f"压缩率: {compression_ratio:.1f}%")
            
            # 验证能否正确加载
            verify_checkpoint = torch.load(output_path, map_location='cpu')
            print("✓ FP16检查点可以正常加载")
            
            # 检查转换后数据类型分布
            converted_counts = count_tensors_by_dtype(verify_checkpoint)
            print("\n转换后数据类型分布:")
            for dtype, count in sorted(converted_counts.items()):
                print(f"  {dtype}: {count} 个张量")
            
            # 检查权重样本
            if 'model' in verify_checkpoint:
                try:
                    sample_key = 'model.language_model.encoder.layers.0.input_norm.weight'
                    sample_path = sample_key.split('.')
                    
                    sample_tensor = verify_checkpoint
                    for key in sample_path:
                        if isinstance(sample_tensor, dict) and key in sample_tensor:
                            sample_tensor = sample_tensor[key]
                        else:
                            sample_tensor = None
                            break
                    
                    if sample_tensor is not None and isinstance(sample_tensor, torch.Tensor):
                        print(f"\n权重样本检查 ({sample_key}):")
                        print(f"  数据类型: {sample_tensor.dtype}")
                        print(f"  形状: {sample_tensor.shape}")
                        print(f"  数值范围: [{sample_tensor.min():.6f}, {sample_tensor.max():.6f}]")
                
                except Exception as e:
                    print(f"权重样本检查失败: {e}")
            
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return None
    
    print(f"\n{'='*60}")
    print("✅ 转换成功!")
    print(f"原始文件: {original_size:.2f} GB")
    print(f"转换后: {new_size:.2f} GB") 
    print(f"节省空间: {original_size - new_size:.2f} GB ({compression_ratio:.1f}%)")
    print(f"{'='*60}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="将Llama8b-MaskLLM检查点从FP32转换为FP16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换默认检查点文件
  python3 convert_fp32_to_fp16.py
  
  # 指定输入和输出文件
  python3 convert_fp32_to_fp16.py -i model.pt -o model_fp16.pt
  
  # 跳过备份和验证以加快速度
  python3 convert_fp32_to_fp16.py --no-backup --no-verify
  
  # 强制覆盖输出文件
  python3 convert_fp32_to_fp16.py --force
        """
    )
    
    # 默认路径
    default_input = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
    default_output = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"
    
    parser.add_argument('-i', '--input', 
                       default=default_input,
                       help='输入FP32检查点文件路径')
    
    parser.add_argument('-o', '--output',
                       default=default_output, 
                       help='输出FP16检查点文件路径')
    
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    
    parser.add_argument('--no-verify', action='store_true',
                       help='跳过验证步骤')
    
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖输出文件')
    
    args = parser.parse_args()
    
    try:
        result = safe_fp32_to_fp16_conversion(
            input_path=args.input,
            output_path=args.output, 
            backup=not args.no_backup,
            verify=not args.no_verify,
            force=args.force
        )
        
        if result:
            print(f"\n输出文件: {result}")
            print("转换完成！可以使用FP16检查点进行推理。")
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
