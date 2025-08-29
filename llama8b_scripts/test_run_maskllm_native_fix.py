#!/usr/bin/env python3.10

"""
测试修复后的run_maskllm_native.sh环境设置
验证PyTorch和相关组件是否正常工作
"""

print("🔧 测试修复后的run_maskllm_native.sh环境")
print("=" * 60)

# 测试1: PyTorch基本导入
print("\n📋 测试1: PyTorch基本导入")
try:
    import torch
    print("✅ PyTorch导入成功")
    print(f"✅ 版本: {torch.__version__}")
    print(f"✅ 路径: {torch.__file__}")
    
    # 测试CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA可用: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA设备数: {device_count}")
        if device_count > 0:
            print(f"✅ 当前设备: {torch.cuda.current_device()}")
            print(f"✅ 设备名称: {torch.cuda.get_device_name()}")
except Exception as e:
    print(f"❌ PyTorch导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 基本张量运算
print("\n📋 测试2: 基本张量运算")
try:
    x = torch.randn(3, 3)
    y = torch.mm(x, x)
    print("✅ CPU张量运算正常")
    print(f"✅ 输入张量形状: {x.shape}")
    print(f"✅ 输出张量形状: {y.shape}")
except Exception as e:
    print(f"❌ 张量运算失败: {e}")

# 测试3: 分布式模块
print("\n📋 测试3: 分布式模块")
try:
    import torch.distributed as dist
    print("✅ torch.distributed导入成功")
    print(f"✅ 分布式模块路径: {dist.__file__}")
except Exception as e:
    print(f"❌ 分布式模块导入失败: {e}")

# 测试4: transformer_engine
print("\n📋 测试4: transformer_engine")
try:
    import transformer_engine as te
    print("✅ transformer_engine导入成功")
    print(f"✅ 版本: {getattr(te, '__version__', 'unknown')}")
    print(f"✅ 路径: {te.__file__}")
    
    # 测试pytorch子模块
    if hasattr(te, 'pytorch'):
        print("✅ transformer_engine.pytorch可用")
        if hasattr(te.pytorch, 'Linear'):
            print("✅ transformer_engine.pytorch.Linear可用")
    else:
        print("⚠️ transformer_engine.pytorch不可用")
        
except Exception as e:
    print(f"❌ transformer_engine导入失败: {e}")

# 测试5: megatron.tokenizer
print("\n📋 测试5: megatron.tokenizer")
try:
    from megatron.tokenizer import build_tokenizer
    print("✅ megatron.tokenizer.build_tokenizer导入成功")
    print("✅ megatron包路径检查:")
    
    import megatron
    print(f"  - megatron: {megatron.__file__}")
    
    import megatron.tokenizer
    print(f"  - megatron.tokenizer: {megatron.tokenizer.__file__}")
    
except Exception as e:
    print(f"❌ megatron.tokenizer导入失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 环境变量检查
print("\n📋 测试6: 环境变量检查")
import os

key_env_vars = [
    'LD_LIBRARY_PATH',
    'PYTHONPATH', 
    'CUDA_HOME',
    'CUDA_ROOT',
    'TORCH_DISTRIBUTED_BACKEND',
    'NCCL_IB_DISABLE',
    'OMPI_MCA_pml'
]

for var in key_env_vars:
    value = os.environ.get(var)
    if value:
        print(f"✅ {var}: {value[:100]}{'...' if len(value) > 100 else ''}")
    else:
        print(f"⚠️ {var}: 未设置")

# 测试7: 关键库文件检查
print("\n📋 测试7: 关键库文件检查")
import ctypes.util

key_libs = ['cudnn', 'cupti', 'nccl', 'cuda']
for lib in key_libs:
    lib_path = ctypes.util.find_library(lib)
    if lib_path:
        print(f"✅ {lib}: {lib_path}")
    else:
        print(f"❌ {lib}: 未找到")

print("\n🎉 环境测试完成!")
print("=" * 60)
