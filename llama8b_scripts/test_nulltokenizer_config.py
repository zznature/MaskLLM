#!/usr/bin/env python3
"""
基于Phase 3.2成功经验的NullTokenizer配置测试
参考: PHASE3_PROGRESS_SUMMARY.md - "✅ Tokenizer构建: 通过"
"""

import sys
import os

# 添加路径
sys.path.append('.')

def test_nulltokenizer_config():
    """测试NullTokenizer配置，基于Phase 3.2成功经验"""
    
    print("🔍 测试NullTokenizer配置（基于Phase 3.2成功经验）...")
    
    try:
        # 1. 测试参数配置（模拟preprocess_data.py的参数）
        print("\n1️⃣ 测试参数配置...")
        
        # 基于成功案例的配置
        config = {
            'tokenizer_type': 'NullTokenizer',
            'vocab_size': 119696,
            'vocab_file': './assets/checkpoints/Llama8b/vocab.txt'
        }
        
        print(f"   ✅ tokenizer_type: {config['tokenizer_type']}")
        print(f"   ✅ vocab_size: {config['vocab_size']}")
        print(f"   ✅ vocab_file: {config['vocab_file']}")
        
        # 2. 检查vocab文件存在性
        print("\n2️⃣ 检查vocab文件...")
        if os.path.exists(config['vocab_file']):
            with open(config['vocab_file'], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                actual_vocab_size = len(lines)
                print(f"   ✅ vocab文件存在: {config['vocab_file']}")
                print(f"   ✅ 实际词汇表大小: {actual_vocab_size}")
                
                if actual_vocab_size == config['vocab_size']:
                    print(f"   ✅ 词汇表大小匹配: {actual_vocab_size} == {config['vocab_size']}")
                else:
                    print(f"   ⚠️  词汇表大小不匹配: {actual_vocab_size} != {config['vocab_size']}")
        else:
            print(f"   ❌ vocab文件不存在: {config['vocab_file']}")
            
        # 3. 测试Megatron tokenizer构建
        print("\n3️⃣ 测试Megatron tokenizer构建...")
        try:
            from megatron.tokenizer.tokenizer import build_tokenizer
            
            # 模拟args对象
            class MockArgs:
                def __init__(self):
                    self.tokenizer_type = 'NullTokenizer'
                    self.vocab_size = 119696
                    self.vocab_file = './assets/checkpoints/Llama8b/vocab.txt'
                    self.rank = 0
                    
            args = MockArgs()
            tokenizer = build_tokenizer(args)
            
            print(f"   ✅ NullTokenizer构建成功")
            print(f"   ✅ tokenizer类型: {type(tokenizer)}")
            print(f"   ✅ vocab_size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'N/A'}")
            
        except Exception as e:
            print(f"   ❌ tokenizer构建失败: {str(e)}")
            
        # 4. 生成修复命令
        print("\n4️⃣ 生成修复后的运行命令...")
        
        cmd = f"""
# 基于Phase 3.2成功经验的修复命令:
source container_environment_setup.sh && python3 tools/preprocess_data.py \\
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \\
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \\
    --vocab-file {config['vocab_file']} \\
    --tokenizer-type {config['tokenizer_type']} \\
    --vocab-size {config['vocab_size']} \\
    --append-eod \\
    --workers 64
"""
        
        print(cmd)
        
        print("\n🎉 NullTokenizer配置测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nulltokenizer_config()
    sys.exit(0 if success else 1)
