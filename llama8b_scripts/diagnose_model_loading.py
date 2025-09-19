#!/usr/bin/env python3
"""
Llama8b 模型加载诊断脚本
用于分析模型加载错误的根本原因
"""

import os
import sys
import torch
import json
from pathlib import Path

def diagnose_model_loading():
    """诊断模型加载问题"""
    
    model_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4"
    
    print("🔍 Llama8b 模型加载诊断")
    print("=" * 50)
    
    # 1. 检查环境
    print("\n📋 环境信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  PyTorch版本: {torch.__version__}")
    
    try:
        import transformers
        print(f"  Transformers版本: {transformers.__version__}")
    except ImportError:
        print("  ❌ Transformers未安装")
        return
    
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA设备数: {torch.cuda.device_count()}")
        print(f"  CUDA版本: {torch.version.cuda}")
    
    # 2. 检查模型文件
    print(f"\n📁 模型文件检查:")
    print(f"  模型路径: {model_path}")
    print(f"  路径存在: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("  ❌ 模型路径不存在")
        return
    
    # 检查关键文件
    key_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    for file in key_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        print(f"  {file}: {'✅' if exists else '❌'}")
        
        if exists and file.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"    └─ JSON格式: ✅")
            except json.JSONDecodeError as e:
                print(f"    └─ JSON格式: ❌ ({str(e)})")
    
    # 3. 检查SafeTensors文件
    print(f"\n🔧 SafeTensors文件检查:")
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    print(f"  发现{len(safetensors_files)}个SafeTensors文件:")
    
    total_size = 0
    for file in sorted(safetensors_files):
        file_path = os.path.join(model_path, file)
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        total_size += size
        print(f"    {file}: {size_mb:.1f} MB")
    
    print(f"  总大小: {total_size / (1024**3):.2f} GB")
    
    # 4. 尝试加载配置
    print(f"\n⚙️ 配置加载测试:")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        print(f"  ✅ 配置加载成功")
        print(f"    模型类型: {config.model_type}")
        print(f"    架构: {config.architectures}")
        print(f"    隐藏层数: {config.num_hidden_layers}")
        print(f"    隐藏维度: {config.hidden_size}")
        print(f"    词汇表大小: {config.vocab_size}")
        print(f"    数据类型: {getattr(config, 'torch_dtype', 'Not specified')}")
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {str(e)}")
    
    # 5. 测试不同加载方法
    print(f"\n🧪 模型加载方法测试:")
    
    test_methods = [
        ("AutoModelForCausalLM + device_map='auto'", test_load_auto_device),
        ("AutoModelForCausalLM + device_map=None", test_load_no_device_map),
        ("LlamaForCausalLM直接加载", test_load_llama_direct),
        ("仅CPU加载", test_load_cpu_only),
    ]
    
    for method_name, method_func in test_methods:
        print(f"\n  🔬 测试: {method_name}")
        try:
            result = method_func(model_path)
            if result:
                print(f"    ✅ 成功")
                # 清理内存
                del result
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                break
            else:
                print(f"    ❌ 失败 (返回None)")
        except Exception as e:
            print(f"    ❌ 失败: {str(e)}")
    
    # 6. 内存检查
    print(f"\n💾 内存状态:")
    if torch.cuda.is_available():
        print(f"  GPU内存:")
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    设备{i}: {allocated:.1f}GB已用 / {total:.1f}GB总量 (缓存: {cached:.1f}GB)")

def test_load_auto_device(model_path):
    """测试自动设备映射加载"""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return model

def test_load_no_device_map(model_path):
    """测试无设备映射加载"""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )
    return model

def test_load_llama_direct(model_path):
    """测试直接Llama加载"""
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=False,
    )
    return model

def test_load_cpu_only(model_path):
    """测试仅CPU加载"""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=False,
    )
    return model

if __name__ == "__main__":
    diagnose_model_loading()
