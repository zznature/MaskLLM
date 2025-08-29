#!/usr/bin/env python3
"""
诊断Llama8b tokenizer路径配置问题
基于Llama2Tokenizer成功方案进行优化
"""

import os
import sys

def diagnose_paths():
    """诊断所有相关路径"""
    
    print("🔍 诊断Llama8b tokenizer和数据路径...")
    
    # 1. 检查C4数据路径
    c4_path = "./assets/data/c4/en/c4-train.00000-of-01024.json"
    print(f"\n1️⃣ C4数据文件检查:")
    print(f"   路径: {c4_path}")
    
    if os.path.exists(c4_path):
        if os.path.isfile(c4_path):
            print(f"   ✅ C4文件存在且为文件")
            print(f"   📊 文件大小: {os.path.getsize(c4_path)} bytes")
        else:
            print(f"   ❌ C4路径存在但不是文件 (可能是目录)")
    else:
        print(f"   ❌ C4文件不存在")
        
        # 检查上级目录
        c4_dir = "./assets/data/c4/en/"
        if os.path.exists(c4_dir):
            print(f"   📁 C4目录存在，列出内容:")
            try:
                files = os.listdir(c4_dir)[:10]  # 显示前10个文件
                for f in files:
                    print(f"     - {f}")
            except Exception as e:
                print(f"     ❌ 无法列出目录内容: {e}")
        else:
            print(f"   ❌ C4目录也不存在: {c4_dir}")
    
    # 2. 检查Llama8b tokenizer路径
    tokenizer_dir = "./assets/checkpoints/Llama8b"
    print(f"\n2️⃣ Llama8b tokenizer检查:")
    print(f"   路径: {tokenizer_dir}")
    
    if os.path.exists(tokenizer_dir):
        if os.path.isdir(tokenizer_dir):
            print(f"   ✅ Tokenizer目录存在")
            
            # 列出tokenizer文件
            try:
                files = os.listdir(tokenizer_dir)
                print(f"   📁 目录内容:")
                for f in files:
                    file_path = os.path.join(tokenizer_dir, f)
                    if os.path.isfile(file_path):
                        print(f"     📄 {f} ({os.path.getsize(file_path)} bytes)")
                    else:
                        print(f"     📁 {f}/")
                        
                # 检查关键文件
                key_files = ['tokenizer.py', 'vocab.txt', 'tokenizer.model', 'tokenizer.json']
                print(f"   🔑 关键文件检查:")
                for key_file in key_files:
                    key_path = os.path.join(tokenizer_dir, key_file)
                    if os.path.exists(key_path):
                        print(f"     ✅ {key_file}")
                    else:
                        print(f"     ❌ {key_file} (缺失)")
                        
            except Exception as e:
                print(f"   ❌ 无法访问tokenizer目录: {e}")
        else:
            print(f"   ❌ Tokenizer路径存在但不是目录")
    else:
        print(f"   ❌ Tokenizer目录不存在")
    
    # 3. 生成修复建议
    print(f"\n3️⃣ 修复建议:")
    
    # 基于Llama2参考生成正确配置
    print(f"   📋 基于Llama2Tokenizer成功方案的建议配置:")
    
    if os.path.exists("./assets/checkpoints/Llama8b/tokenizer.model"):
        print(f"   ✅ 推荐使用: --tokenizer-model ./assets/checkpoints/Llama8b/tokenizer.model")
    elif os.path.exists("./assets/checkpoints/Llama8b"):
        print(f"   ✅ 推荐使用: --tokenizer-model ./assets/checkpoints/Llama8b")
    else:
        print(f"   ❌ 需要确认tokenizer文件位置")
    
    # 4. 生成测试命令
    print(f"\n4️⃣ 建议测试命令:")
    print(f"""
# 基于诊断结果的测试命令:
source container_environment_setup.sh && python3 tools/preprocess_data.py \\
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \\
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \\
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \\
    --tokenizer-type Llama2Tokenizer \\
    --tokenizer-model ./assets/checkpoints/Llama8b \\
    --append-eod \\
    --workers 1
""")

if __name__ == "__main__":
    diagnose_paths()
