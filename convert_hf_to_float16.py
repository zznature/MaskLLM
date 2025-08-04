#!/usr/bin/env python3

"""
将HuggingFace模型转换为float16格式
用于减少模型存储空间和加快推理速度
"""

import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

def convert_to_float16(input_model_path, output_model_path):
    """
    将HuggingFace模型转换为float16格式
    
    Args:
        input_model_path: 输入模型路径
        output_model_path: 输出模型路径
    """
    print(f"正在加载模型: {input_model_path}")
    
    # 加载模型和tokenizer
    model = LlamaForCausalLM.from_pretrained(
        input_model_path,
        torch_dtype=torch.float32,  # 先以float32加载
        device_map="cpu"
    )
    
    print("正在转换为float16格式...")
    # 转换为float16
    model = model.half()  # 等价于 model.to(torch.float16)
    
    print(f"正在保存模型到: {output_model_path}")
    # 创建输出目录
    os.makedirs(output_model_path, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(
        output_model_path,
        torch_dtype=torch.float16,
        safe_serialization=True  # 使用safetensors格式
    )
    
    # 复制tokenizer文件
    try:
        tokenizer = LlamaTokenizer.from_pretrained(input_model_path)
        tokenizer.save_pretrained(output_model_path)
        print("tokenizer文件已复制")
    except Exception as e:
        print(f"复制tokenizer时出现警告: {e}")
        # 手动复制tokenizer文件
        import shutil
        for file in os.listdir(input_model_path):
            if file.startswith('tokenizer') or file.endswith('.json'):
                src = os.path.join(input_model_path, file)
                dst = os.path.join(output_model_path, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        print("tokenizer文件已手动复制")
    
    print("✅ float16转换完成!")
    
    # 显示文件大小对比
    try:
        import glob
        input_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(input_model_path, "*.bin")) + 
                        glob.glob(os.path.join(input_model_path, "*.safetensors")))
        output_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(output_model_path, "*.bin")) + 
                         glob.glob(os.path.join(output_model_path, "*.safetensors")))
        
        if input_size > 0 and output_size > 0:
            compression_ratio = (input_size - output_size) / input_size * 100
            print(f"文件大小对比:")
            print(f"  输入模型: {input_size / 1024**3:.2f} GB")
            print(f"  输出模型: {output_size / 1024**3:.2f} GB")
            print(f"  压缩比例: {compression_ratio:.1f}%")
    except Exception as e:
        print(f"无法计算文件大小: {e}")

def main():
    parser = argparse.ArgumentParser(description='将HuggingFace模型转换为float16格式')
    parser.add_argument('--input', type=str, required=True,
                       help='输入模型路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出模型路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入模型路径不存在: {args.input}")
        return 1
    
    convert_to_float16(args.input, args.output)
    return 0

if __name__ == '__main__':
    exit(main()) 