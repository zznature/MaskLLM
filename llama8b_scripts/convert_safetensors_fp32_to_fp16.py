#!/usr/bin/env python3
"""
将 Llama8b-MaskLLM SafeTensors 格式模型从 FP32 转换为 FP16
专为 HuggingFace SafeTensors 格式设计
"""
import os
import json
import shutil
import torch
from safetensors.torch import save_file, load_file
from collections import defaultdict
import argparse
from tqdm import tqdm

def convert_safetensors_to_fp16(input_dir, output_dir=None, backup=True):
    """
    将SafeTensors格式的模型从FP32转换为FP16
    
    Args:
        input_dir (str): 输入模型目录
        output_dir (str): 输出目录，如果为None则原地转换
        backup (bool): 是否备份原文件
    """
    
    if output_dir is None:
        output_dir = input_dir
        
    print(f"🔄 开始转换SafeTensors模型: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    index_file = os.path.join(input_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"索引文件不存在: {index_file}")
    
    # 创建输出目录
    if output_dir != input_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取索引文件
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    original_size = index['metadata']['total_size']
    print(f"📊 原始模型大小: {original_size / (1024**3):.2f} GB")
    
    # 获取所有safetensors文件
    safetensor_files = list(set(index["weight_map"].values()))
    safetensor_files.sort()
    
    print(f"📦 发现 {len(safetensor_files)} 个SafeTensors文件")
    
    # 转换统计
    conversion_stats = {
        "files_converted": 0,
        "tensors_converted": 0,
        "original_size": 0,
        "new_size": 0,
        "dtype_changes": defaultdict(int)
    }
    
    # 转换每个safetensors文件
    for i, safetensor_file in enumerate(tqdm(safetensor_files, desc="转换文件")):
        input_path = os.path.join(input_dir, safetensor_file)
        output_path = os.path.join(output_dir, safetensor_file)
        
        if not os.path.exists(input_path):
            print(f"⚠️  跳过不存在的文件: {safetensor_file}")
            continue
        
        # 备份原文件
        if backup and input_path != output_path:
            backup_path = input_path + ".fp32_backup"
            if not os.path.exists(backup_path):
                shutil.copy2(input_path, backup_path)
                print(f"💾 备份: {backup_path}")
        
        # 加载tensor
        tensors = load_file(input_path, device="cpu")
        converted_tensors = {}
        
        original_file_size = os.path.getsize(input_path)
        conversion_stats["original_size"] += original_file_size
        
        print(f"\n🔧 处理文件 {i+1}/{len(safetensor_files)}: {safetensor_file}")
        print(f"   原始大小: {original_file_size / (1024**3):.2f} GB")
        
        # 转换每个tensor
        for tensor_name, tensor in tensors.items():
            original_dtype = tensor.dtype
            
            # 转换逻辑
            if original_dtype == torch.float32:
                # FP32 -> FP16
                converted_tensor = tensor.half()
                conversion_stats["dtype_changes"]["fp32_to_fp16"] += 1
                conversion_stats["tensors_converted"] += 1
                
            elif original_dtype == torch.float64:
                # FP64 -> FP16 (如果有的话)
                converted_tensor = tensor.half()
                conversion_stats["dtype_changes"]["fp64_to_fp16"] += 1
                conversion_stats["tensors_converted"] += 1
                
            else:
                # 保持其他类型不变 (int32, int64, bool等)
                converted_tensor = tensor
                conversion_stats["dtype_changes"]["unchanged"] += 1
            
            converted_tensors[tensor_name] = converted_tensor
        
        # 保存转换后的tensor
        save_file(converted_tensors, output_path)
        
        new_file_size = os.path.getsize(output_path)
        conversion_stats["new_size"] += new_file_size
        conversion_stats["files_converted"] += 1
        
        print(f"   转换后大小: {new_file_size / (1024**3):.2f} GB")
        print(f"   压缩率: {(1 - new_file_size/original_file_size)*100:.1f}%")
    
    # 更新配置文件
    config_files = ["config.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"]
    for config_file in config_files:
        input_config = os.path.join(input_dir, config_file)
        output_config = os.path.join(output_dir, config_file)
        
        if os.path.exists(input_config):
            if input_dir != output_dir:
                shutil.copy2(input_config, output_config)
            
            # 特别更新config.json的torch_dtype
            if config_file == "config.json":
                with open(output_config, 'r') as f:
                    config = json.load(f)
                config["torch_dtype"] = "float16"
                
                with open(output_config, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ 更新配置文件: {config_file}")
    
    # 复制其他文件
    other_files = ["vocab.txt", "tokenizer.json"]
    for other_file in other_files:
        input_other = os.path.join(input_dir, other_file)
        output_other = os.path.join(output_dir, other_file)
        
        if os.path.exists(input_other) and input_dir != output_dir:
            shutil.copy2(input_other, output_other)
            print(f"📋 复制文件: {other_file}")
    
    # 更新索引文件
    new_index = index.copy()
    new_index['metadata']['total_size'] = conversion_stats["new_size"]
    
    output_index_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(output_index_file, 'w') as f:
        json.dump(new_index, f, indent=2)
    
    # 打印转换结果
    print("\n" + "=" * 60)
    print("🎉 转换完成!")
    print("-" * 40)
    print(f"文件转换: {conversion_stats['files_converted']}/{len(safetensor_files)}")
    print(f"张量转换: {conversion_stats['tensors_converted']}")
    print(f"原始大小: {conversion_stats['original_size'] / (1024**3):.2f} GB")
    print(f"转换后大小: {conversion_stats['new_size'] / (1024**3):.2f} GB")
    print(f"节省空间: {(conversion_stats['original_size'] - conversion_stats['new_size']) / (1024**3):.2f} GB")
    print(f"压缩比例: {(1 - conversion_stats['new_size']/conversion_stats['original_size'])*100:.1f}%")
    
    print("\n数据类型转换:")
    for change_type, count in conversion_stats["dtype_changes"].items():
        print(f"  {change_type}: {count} 个张量")
    
    # 验证转换
    print("\n🔍 验证转换结果...")
    verify_conversion(output_dir)
    
    return conversion_stats

def verify_conversion(model_dir):
    """验证转换结果"""
    try:
        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # 检查一个文件的数据类型
        first_file = list(set(index["weight_map"].values()))[0]
        first_path = os.path.join(model_dir, first_file)
        
        tensors = load_file(first_path, device="cpu")
        sample_tensor = next(iter(tensors.values()))
        
        print(f"✅ 验证成功!")
        print(f"   示例张量数据类型: {sample_tensor.dtype}")
        print(f"   模型总大小: {index['metadata']['total_size'] / (1024**3):.2f} GB")
        
        # 检查config.json
        config_file = os.path.join(model_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"   配置文件torch_dtype: {config.get('torch_dtype', 'Not specified')}")
        
    except Exception as e:
        print(f"⚠️  验证时出现警告: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensors model from FP32 to FP16")
    parser.add_argument("-i", "--input_dir", 
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4",
                       help="Input model directory")
    parser.add_argument("-o", "--output_dir", 
                       help="Output directory (default: same as input)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup of original files")
    
    args = parser.parse_args()
    
    try:
        conversion_stats = convert_safetensors_to_fp16(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            backup=not args.no_backup
        )
        
        # 保存转换报告
        report_file = "/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/conversion_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "conversion_stats": conversion_stats,
                "input_dir": args.input_dir,
                "output_dir": args.output_dir or args.input_dir,
                "backup_enabled": not args.no_backup
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 转换报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
