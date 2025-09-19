#!/usr/bin/env python3
"""
仅测试Llama8bTokenizer加载功能
用于快速验证tokenizer修复
"""

import os
import sys
import importlib


def test_llama8b_tokenizer():
    """测试自定义Llama8bTokenizer加载"""
    print("🧪 测试Llama8bTokenizer加载...")
    
    # 检查原始目录
    original_llama8b_path = "assets/checkpoints/Llama8b"
    if not os.path.exists(original_llama8b_path):
        print(f"❌ 原始Llama8b目录未找到: {original_llama8b_path}")
        return False
    
    vocab_file = os.path.join(original_llama8b_path, "vocab.txt")
    if not os.path.exists(vocab_file):
        print(f"❌ vocab.txt未找到: {vocab_file}")
        return False
    
    print(f"✅ 找到vocab.txt: {vocab_file}")
    
    # 添加到Python路径
    if original_llama8b_path not in sys.path:
        sys.path.insert(0, original_llama8b_path)
        print(f"✅ 添加路径到sys.path: {original_llama8b_path}")
    
    try:
        # 导入自定义tokenizer
        from llama8b.tokenizer import Llama8bTokenizer
        print("✅ 成功导入Llama8bTokenizer类")
        
        # 创建tokenizer实例
        tokenizer = Llama8bTokenizer(vocab_file=vocab_file)
        print(f"✅ 成功创建tokenizer实例")
        print(f"   - vocab_size: {tokenizer.vocab_size}")
        print(f"   - bos_id: {tokenizer.bos_id}")
        print(f"   - eos_id: {tokenizer.eos_id}")
        print(f"   - unk_id: {tokenizer.unk_id}")
        
        # 测试基本功能
        test_text = "Hello, world!"
        print(f"\n🧪 测试编码/解码: '{test_text}'")
        
        # 测试tokenize
        tokens = tokenizer._tokenize(test_text)
        print(f"   tokens: {tokens[:10]}...")  # 显示前10个
        
        # 测试convert
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
        print(f"   token_ids: {token_ids[:10]}...")  # 显示前10个
        
        # 测试解码
        if hasattr(tokenizer, 'raw_decode'):
            decoded = tokenizer.raw_decode(token_ids)
            print(f"   decoded (raw_decode): '{decoded[:50]}...'")
        
        print("✅ 基本功能测试通过")
        
        # 测试HF接口兼容性
        print(f"\n🧪 测试HF接口兼容性...")
        try:
            result = tokenizer(test_text, return_tensors="pt")
            print(f"   ✅ __call__ 方法可用: {result['input_ids'].shape}")
        except Exception as e:
            print(f"   ⚠️ __call__ 方法不可用: {e}")
            print("   这是预期的，我们会使用自定义接口")
        
        return True
        
    except Exception as e:
        print(f"❌ tokenizer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🔧 Llama8bTokenizer 独立测试")
    print("=" * 50)
    
    success = test_llama8b_tokenizer()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Llama8bTokenizer测试成功！")
        print("可以进行完整的推理测试了")
        return 0
    else:
        print("❌ Llama8bTokenizer测试失败")
        print("需要先解决tokenizer问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())
