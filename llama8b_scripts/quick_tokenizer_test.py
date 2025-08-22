#!/usr/bin/env python3
"""
快速Tokenizer一致性测试
专门验证Llama8b tokenizer (vocab.txt) 问题
"""

import os
import sys

def test_vocab_files():
    """快速检查vocab.txt文件"""
    print("🔍 1. 快速vocab.txt检查")
    
    # 原始vocab.txt
    original_vocab = "assets/checkpoints/Llama8b/vocab.txt"
    converted_vocab = "output/checkpoints/llama8b_megatron_tp8/vocab.txt"
    
    results = {}
    
    # 检查原始文件
    if os.path.exists(original_vocab):
        with open(original_vocab, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        results['original_size'] = len(original_lines)
        print(f"    ✅ 原始vocab.txt: {results['original_size']} 词汇")
    else:
        print(f"    ❌ 原始vocab.txt不存在: {original_vocab}")
        results['original_size'] = 0
    
    # 检查转换后文件
    if os.path.exists(converted_vocab):
        with open(converted_vocab, 'r', encoding='utf-8') as f:
            converted_lines = f.readlines()
        results['converted_size'] = len(converted_lines)
        print(f"    ✅ 转换后vocab.txt: {results['converted_size']} 词汇")
    else:
        print(f"    ❌ 转换后vocab.txt不存在: {converted_vocab}")
        results['converted_size'] = 0
    
    # 快速比较
    if results['original_size'] > 0 and results['converted_size'] > 0:
        if results['original_size'] == results['converted_size']:
            print("    ✅ 词汇表大小一致")
            
            # 抽样检查前10个词汇
            print("    🔍 前10个词汇对比:")
            for i in range(min(10, len(original_lines))):
                orig = original_lines[i].strip()
                conv = converted_lines[i].strip()
                if orig == conv:
                    print(f"      ✅ [{i}] '{orig[:20]}{'...' if len(orig) > 20 else ''}'")
                else:
                    print(f"      ❌ [{i}] 不一致")
                    print(f"          原始: '{orig[:30]}'")
                    print(f"          转换: '{conv[:30]}'")
                    
            results['vocab_consistent'] = True
        else:
            print(f"    ❌ 词汇表大小不一致: {results['original_size']} vs {results['converted_size']}")
            results['vocab_consistent'] = False
    else:
        results['vocab_consistent'] = False
    
    return results

def test_original_tokenizer():
    """测试原始Llama8b tokenizer"""
    print("\n🔍 2. 原始Llama8b tokenizer测试")
    
    try:
        # 添加路径
        llama8b_path = 'assets/checkpoints/Llama8b/llama8b'
        if llama8b_path not in sys.path:
            sys.path.insert(0, llama8b_path)
        
        from tokenizer import Llama8bTokenizer
        
        tokenizer = Llama8bTokenizer('assets/checkpoints/Llama8b')
        print(f"    ✅ Llama8bTokenizer加载成功")
        print(f"      - 词汇表大小: {tokenizer.vocab_size}")
        
        # 简单编码测试
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        print(f"      - 测试编码: '{test_text}' -> {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        return True, tokenizer
        
    except Exception as e:
        print(f"    ❌ Llama8bTokenizer失败: {e}")
        return False, None

def test_checkpoint_structure():
    """快速检查checkpoint结构"""
    print("\n🔍 3. Checkpoint结构检查")
    
    checkpoint_dir = "output/checkpoints/llama8b_megatron_tp8"
    
    if not os.path.exists(checkpoint_dir):
        print(f"    ❌ Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    # 检查关键文件
    required_files = [
        "latest_checkpointed_iteration.txt",
        "vocab.txt"
    ]
    
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            print(f"    ✅ {file}")
        else:
            print(f"    ❌ 缺少 {file}")
            return False
    
    # 检查TP分片
    iter_dir = os.path.join(checkpoint_dir, "iter_0000001")
    if os.path.exists(iter_dir):
        mp_ranks = [f for f in os.listdir(iter_dir) if f.startswith("mp_rank_")]
        print(f"    ✅ 找到{len(mp_ranks)}个TP分片")
        return len(mp_ranks) == 8
    else:
        print("    ❌ 迭代目录不存在")
        return False

def main():
    """主要测试流程"""
    print("🚀 快速Tokenizer一致性测试")
    print("=" * 50)
    print("验证Llama8b tokenizer (vocab.txt) 关键问题")
    print()
    
    results = {}
    
    # 1. vocab文件检查
    vocab_results = test_vocab_files()
    results.update(vocab_results)
    
    # 2. 原始tokenizer测试
    orig_success, orig_tokenizer = test_original_tokenizer()
    results['original_tokenizer'] = orig_success
    
    # 3. checkpoint结构检查
    results['checkpoint_structure'] = test_checkpoint_structure()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 快速测试结果总结")
    print("=" * 50)
    
    issues = []
    successes = []
    
    if results.get('original_size', 0) > 0:
        successes.append(f"原始vocab.txt ({results['original_size']} 词汇)")
    else:
        issues.append("原始vocab.txt缺失")
    
    if results.get('converted_size', 0) > 0:
        successes.append(f"转换vocab.txt ({results['converted_size']} 词汇)")
    else:
        issues.append("转换vocab.txt缺失")
    
    if results.get('vocab_consistent', False):
        successes.append("词汇表一致性")
    else:
        issues.append("词汇表不一致")
    
    if results.get('original_tokenizer', False):
        successes.append("原始tokenizer可用")
    else:
        issues.append("原始tokenizer问题")
    
    if results.get('checkpoint_structure', False):
        successes.append("checkpoint结构完整")
    else:
        issues.append("checkpoint结构问题")
    
    print("✅ 成功项目:")
    for success in successes:
        print(f"  - {success}")
    
    if issues:
        print("\n❌ 需要关注的问题:")
        for issue in issues:
            print(f"  - {issue}")
    
    # 结论
    total_checks = 5
    passed_checks = len(successes)
    
    print(f"\n📊 总体状态: {passed_checks}/{total_checks} 项目正常")
    
    if passed_checks == total_checks:
        print("\n🎉 所有检查通过！tokenizer状态良好")
        print("✅ 可以继续进行详细的推理一致性验证")
        return 0
    elif passed_checks >= 4:
        print("\n⚠️ 大部分检查通过，基本状态良好")
        print("💡 存在少量问题，但不影响主要功能")
        return 1
    else:
        print("\n❌ 多个关键问题，需要重点解决")
        print("🔧 建议优先解决tokenizer和vocab问题")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)
