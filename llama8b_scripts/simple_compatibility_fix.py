#!/usr/bin/env python3
"""
简单兼容性修复 - 降级transformers版本或使用兼容参数
"""

import os
import sys

def simple_compatibility_test():
    """测试简单的兼容性修复方法"""
    
    print("🔧 简单兼容性修复测试")
    print("=" * 40)
    
    model_path = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4"
    
    # 方法1: 使用兼容性参数
    print("\n🧪 方法1: 使用兼容性参数")
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        print("尝试使用兼容性参数加载...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",  # 自动检测
            device_map=None,     # 不使用自动设备映射  
            trust_remote_code=False,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            local_files_only=True,  # 只使用本地文件
        )
        
        print("✅ 方法1成功：兼容性参数有效！")
        print(f"模型数据类型: {next(model.parameters()).dtype}")
        return True, "compatibility_params"
        
    except Exception as e:
        print(f"❌ 方法1失败: {str(e)}")
    
    # 方法2: 环境变量绕过
    print("\n🧪 方法2: 环境变量绕过")
    try:
        # 设置兼容性环境变量
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=False,  # 尝试不使用safetensors
        )
        
        print("✅ 方法2成功：环境变量绕过有效！")
        return True, "env_bypass"
        
    except Exception as e:
        print(f"❌ 方法2失败: {str(e)}")
    
    # 方法3: 使用较早的transformers方法
    print("\n🧪 方法3: 旧版本加载方法")
    try:
        import torch
        from transformers import LlamaForCausalLM, LlamaConfig
        
        # 手动读配置，但用最简单的方式
        config = AutoConfig.from_pretrained(model_path)
        
        # 使用最基本的加载方式
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,  # 明确指定FP16
            device_map={"": "cpu"},     # 明确指定CPU
        )
        
        print("✅ 方法3成功：旧版本方法有效！")
        return True, "legacy_method"
        
    except Exception as e:
        print(f"❌ 方法3失败: {str(e)}")
    
    return False, None

def recommend_solution():
    """推荐解决方案"""
    
    success, method = simple_compatibility_test()
    
    if success:
        print(f"\n🎉 找到可行方案: {method}")
        
        if method == "compatibility_params":
            print("\n💡 推荐使用兼容性参数:")
            print("""
# 修改推理脚本使用这些参数:
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=None, 
    trust_remote_code=False,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
""")
        
        elif method == "env_bypass":
            print("\n💡 推荐设置环境变量:")
            print("""
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
""")
        
        elif method == "legacy_method":
            print("\n💡 推荐使用传统加载方法")
        
    else:
        print("\n❌ 所有简单方法都失败")
        print("💡 建议:")
        print("1. 降级transformers到4.31.0:")
        print("   pip install transformers==4.31.0")
        print("2. 或使用已有的pytorch_native_inference.py")

if __name__ == "__main__":
    recommend_solution()
