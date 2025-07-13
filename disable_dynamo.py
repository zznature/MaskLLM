#!/usr/bin/env python3
"""
禁用 PyTorch Dynamo 编译器的脚本
在运行 MaskLLM 之前导入此脚本可以避免 Dynamo 编译器的问题
"""

import os
import torch

def disable_dynamo():
    """禁用 PyTorch Dynamo 编译器"""
    print("=== 禁用 PyTorch Dynamo 编译器 ===")
    
    # 方法1: 设置环境变量
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    print("✓ 设置 TORCHDYNAMO_DISABLE=1")
    
    # 方法2: 在代码中禁用
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.suppress_errors = True
        print("✓ 设置 torch._dynamo.config.suppress_errors = True")
        
        # 禁用所有后端
        torch._dynamo.config.disable = True
        print("✓ 设置 torch._dynamo.config.disable = True")
    
    # 方法3: 设置编译模式为 eager
    torch._dynamo.config.backend = "eager"
    print("✓ 设置 torch._dynamo.config.backend = 'eager'")
    
    print("Dynamo 编译器已禁用！")

def test_dynamo_disabled():
    """测试 Dynamo 是否已禁用"""
    print("\n=== 测试 Dynamo 状态 ===")
    
    # 检查环境变量
    if os.environ.get('TORCHDYNAMO_DISABLE') == '1':
        print("✓ TORCHDYNAMO_DISABLE=1 已设置")
    else:
        print("✗ TORCHDYNAMO_DISABLE 未设置")
    
    # 检查 PyTorch 配置
    if hasattr(torch, '_dynamo'):
        print(f"✓ torch._dynamo 可用")
        print(f"  - suppress_errors: {torch._dynamo.config.suppress_errors}")
        print(f"  - disable: {torch._dynamo.config.disable}")
        print(f"  - backend: {torch._dynamo.config.backend}")
    else:
        print("✗ torch._dynamo 不可用")
    
    # 测试简单的 CUDA 操作
    try:
        if torch.cuda.is_available():
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.mm(x, y)
            print("✓ CUDA 操作正常")
        else:
            print("✗ CUDA 不可用")
    except Exception as e:
        print(f"✗ CUDA 操作失败: {e}")

if __name__ == "__main__":
    disable_dynamo()
    test_dynamo_disabled()
else:
    # 当作为模块导入时，自动禁用 Dynamo
    disable_dynamo() 