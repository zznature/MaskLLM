#!/usr/bin/env python3
"""
修复导出的HF模型的tokenizer配置问题
将Llama8bTokenizer替换为LlamaTokenizer以确保兼容性
"""

import json
import os
import sys


def fix_tokenizer_config(model_dir: str):
    """修复tokenizer配置文件"""
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    
    if not os.path.exists(tokenizer_config_path):
        print(f"❌ tokenizer_config.json 未找到: {tokenizer_config_path}")
        return False
    
    try:
        # 读取配置
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)
        
        print(f"原始tokenizer_class: {config.get('tokenizer_class', 'None')}")
        
        # 修复tokenizer_class
        if config.get("tokenizer_class") == "Llama8bTokenizer":
            config["tokenizer_class"] = "LlamaTokenizer"
            
            # 写入修复后的配置
            with open(tokenizer_config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ tokenizer_class已修复为: LlamaTokenizer")
            return True
        else:
            print(f"ℹ️ tokenizer_class无需修复: {config.get('tokenizer_class')}")
            return True
            
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("用法: python fix_tokenizer_config.py <model_directory>")
        print("示例: python fix_tokenizer_config.py output/checkpoints/llama8b_hf_maskllm_c4")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    print(f"🔧 修复tokenizer配置: {model_dir}")
    
    success = fix_tokenizer_config(model_dir)
    
    if success:
        print("✅ tokenizer配置修复完成!")
        sys.exit(0)
    else:
        print("❌ tokenizer配置修复失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
