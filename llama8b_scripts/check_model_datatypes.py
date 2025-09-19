#!/usr/bin/env python3
"""
检查 Llama8b-MaskLLM SafeTensors 模型文件的数据类型
"""
import os
import json
from safetensors import safe_open
from collections import defaultdict

def check_safetensors_dtypes(model_dir):
    """检查SafeTensors文件中的数据类型分布"""
    
    # 读取索引文件
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    print(f"模型目录: {model_dir}")
    print(f"总文件大小: {index['metadata']['total_size'] / (1024**3):.2f} GB")
    print("=" * 60)
    
    # 统计数据类型
    dtype_stats = defaultdict(lambda: {"count": 0, "size": 0})
    file_stats = defaultdict(lambda: {"tensors": 0, "size": 0})
    
    # 获取所有safetensors文件
    safetensor_files = set(index["weight_map"].values())
    
    print("正在检查各个SafeTensors文件...")
    
    for safetensor_file in sorted(safetensor_files):
        filepath = os.path.join(model_dir, safetensor_file)
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {safetensor_file}")
            continue
            
        file_size = os.path.getsize(filepath)
        file_stats[safetensor_file]["size"] = file_size
        
        print(f"\n📁 {safetensor_file} ({file_size / (1024**3):.2f} GB)")
        
        # 检查文件中的tensor
        with safe_open(filepath, framework="pt", device="cpu") as f:
            tensor_names = f.keys()
            
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                dtype = str(tensor.dtype)
                tensor_size = tensor.numel() * tensor.element_size()
                
                dtype_stats[dtype]["count"] += 1
                dtype_stats[dtype]["size"] += tensor_size
                file_stats[safetensor_file]["tensors"] += 1
                
                # 显示前几个tensor的详细信息
                if file_stats[safetensor_file]["tensors"] <= 3:
                    print(f"  └─ {tensor_name}: {tensor.shape} | {dtype} | {tensor_size / (1024**2):.1f} MB")
    
    print("\n" + "=" * 60)
    print("📊 数据类型统计:")
    print("-" * 40)
    
    total_params = 0
    total_size = 0
    
    for dtype, stats in dtype_stats.items():
        count = stats["count"]
        size = stats["size"]
        params = size // (4 if 'float32' in dtype else 2 if 'float16' in dtype or 'bfloat16' in dtype else 1)
        
        total_params += params
        total_size += size
        
        print(f"{dtype:>12}: {count:>3}个张量 | {params/1e9:>6.2f}B参数 | {size/(1024**3):>6.2f} GB")
    
    print("-" * 40)
    print(f"{'总计':>12}: {sum(s['count'] for s in dtype_stats.values()):>3}个张量 | {total_params/1e9:>6.2f}B参数 | {total_size/(1024**3):>6.2f} GB")
    
    # 分析是否需要转换
    print("\n" + "=" * 60)
    print("🔍 分析结果:")
    
    has_fp32 = any('float32' in dtype for dtype in dtype_stats.keys())
    has_fp16 = any('float16' in dtype or 'bfloat16' in dtype for dtype in dtype_stats.keys())
    
    if has_fp32:
        fp32_size = sum(stats["size"] for dtype, stats in dtype_stats.items() if 'float32' in dtype)
        potential_savings = fp32_size / 2  # FP32 -> FP16 节省50%
        print(f"⚠️  检测到FP32权重: {fp32_size/(1024**3):.2f} GB")
        print(f"💾 转换为FP16可节省: {potential_savings/(1024**3):.2f} GB")
        print("✅ 建议进行FP32→FP16转换")
    elif has_fp16:
        print("✅ 模型已经是FP16格式，无需转换")
    else:
        print("❓ 未检测到常见的浮点数格式")
    
    return dtype_stats

def main():
    # 默认模型目录
    model_dir = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4"
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    try:
        dtype_stats = check_safetensors_dtypes(model_dir)
        
        # 保存检查结果
        output_file = "/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/model_dtype_analysis.json"
        analysis_result = {
            "model_dir": model_dir,
            "dtype_statistics": {dtype: {"count": stats["count"], "size_gb": stats["size"]/(1024**3)} 
                               for dtype, stats in dtype_stats.items()},
            "recommendation": "convert_to_fp16" if any('float32' in dtype for dtype in dtype_stats.keys()) else "already_optimized"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 检查失败: {str(e)}")

if __name__ == "__main__":
    main()
