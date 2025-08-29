#!/usr/bin/env python3.10

"""
PyTorch导入测试脚本
用于验证HPC-X库冲突修复效果
"""

import sys
import os

def test_pytorch_import():
    """测试PyTorch导入和基本功能"""
    print("🔍 PyTorch导入和功能测试")
    print("=" * 50)
    
    # 1. 基本导入测试
    print("1. 测试PyTorch基本导入...")
    try:
        import torch
        print(f"✅ PyTorch导入成功，版本: {torch.__version__}")
        print(f"   路径: {torch.__file__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    # 2. CUDA可用性测试
    print("\n2. 测试CUDA可用性...")
    try:
        cuda_available = torch.cuda.is_available()
        print(f"✅ CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"   CUDA设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.current_device()}")
            print(f"   设备名称: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        return False
    
    # 3. 分布式功能测试
    print("\n3. 测试分布式模块导入...")
    try:
        import torch.distributed as dist
        print(f"✅ torch.distributed导入成功")
        print(f"   路径: {dist.__file__}")
        
        # 测试后端可用性
        available_backends = []
        for backend in ['nccl', 'mpi', 'gloo']:
            if dist.is_backend_available(backend):
                available_backends.append(backend)
        print(f"   可用后端: {available_backends}")
        
    except ImportError as e:
        print(f"❌ torch.distributed导入失败: {e}")
        return False
    
    # 4. 基本张量操作测试
    print("\n4. 测试基本张量操作...")
    try:
        # CPU张量
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✅ CPU张量运算成功，结果形状: {z.shape}")
        
        # GPU张量（如果CUDA可用）
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print(f"✅ GPU张量运算成功，结果形状: {z_gpu.shape}")
        
    except Exception as e:
        print(f"❌ 张量操作失败: {e}")
        return False
    
    # 5. 编译器功能测试
    print("\n5. 测试编译相关功能...")
    try:
        print(f"   JIT可用: {torch.jit.is_scripting()}")
        print(f"   编译模式: {torch.is_grad_enabled()}")
        
        # 检查dynamo设置
        dynamo_disable = os.environ.get('TORCHDYNAMO_DISABLE', 'Not Set')
        print(f"   TORCHDYNAMO_DISABLE: {dynamo_disable}")
        
    except Exception as e:
        print(f"⚠️  编译功能检查异常: {e}")
    
    print("\n✅ PyTorch功能测试完成！")
    return True

def test_environment_info():
    """显示环境信息"""
    print("\n🌍 环境信息")
    print("=" * 50)
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    print(f"\nPYTHONPATH前5个路径:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i+1}. {path}")
    
    print(f"\n关键环境变量:")
    env_vars = [
        'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_ROOT',
        'OMPI_MCA_pml', 'OMPI_MCA_btl', 'UCX_TLS', 'UCC_TLS',
        'TORCHDYNAMO_DISABLE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        # 截断过长的路径
        if len(value) > 100:
            value = value[:100] + "..."
        print(f"  {var}: {value}")

def main():
    """主函数"""
    print("🚀 开始PyTorch环境验证测试")
    print("时间:", os.popen('date').read().strip())
    print()
    
    # 环境信息
    test_environment_info()
    
    # PyTorch功能测试
    success = test_pytorch_import()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！PyTorch环境正常")
        print("💡 可以安全地运行MaskLLM相关任务")
        exit(0)
    else:
        print("❌ 测试失败！需要进一步修复环境")
        print("💡 建议检查HPC-X库冲突修复是否生效")
        exit(1)

if __name__ == "__main__":
    main()
