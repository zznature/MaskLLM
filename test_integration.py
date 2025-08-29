#!/usr/bin/env python3

# 简单的集成测试脚本
print("=== 环境集成测试 ===")

try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
    print(f"   路径: {transformers.__file__}")
    if '.ext_pkgs' in transformers.__file__:
        print("   📦 来源: .ext_pkgs (外部包)")
    else:
        print("   🐳 来源: 容器内")
except ImportError as e:
    print(f"❌ transformers导入失败: {e}")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
    print(f"   路径: {torch.__file__}")
    if 'usr/local' in torch.__file__:
        print("   🐳 来源: 容器内 (正确)")
    else:
        print("   ⚠️ 来源: 外部 (可能有问题)")
    
    # 测试张量运算
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print("   ✅ 张量运算正常")
    
except Exception as e:
    if 'ucc' in str(e).lower():
        print(f"❌ UCC冲突: {e}")
    else:
        print(f"❌ torch导入失败: {e}")

try:
    from megatron.tokenizer import build_tokenizer
    print("✅ megatron.tokenizer: 正常")
except Exception as e:
    print(f"❌ megatron.tokenizer: {e}")

print("=== 测试完成 ===")
