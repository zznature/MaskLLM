#!/usr/bin/env python3.10

"""
PyTorch隔离包装器
通过控制动态链接来避免HPC-X库冲突
"""

import os
import sys
import ctypes
from ctypes.util import find_library

def setup_library_isolation():
    """设置库隔离环境"""
    print("🔧 设置PyTorch库隔离环境...")
    
    # 1. 清理可能的HPC-X环境污染
    hpcx_env_vars = [
        'HPCX_DIR', 'HPCX_HOME', 'HPCX_ROOT',
        'HPCX_MPI_DIR', 'HPCX_UCX_DIR', 'HPCX_UCC_DIR',
        'OMPI_HOME', 'OMPI_ROOT', 'MPI_HOME', 'MPI_ROOT'
    ]
    
    for var in hpcx_env_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"  清除环境变量: {var}")
    
    # 2. 设置强制禁用HPC-X的环境变量
    isolation_vars = {
        'OMPI_MCA_pml': '^ucx,^hcoll',
        'OMPI_MCA_btl': '^openib,^uct',
        'OMPI_MCA_coll': '^hcoll,^ucx',
        'UCX_TLS': '^ud,^dc,^rc',
        'UCC_TLS': '^ucp,^sharp',
        'UCX_LOG_LEVEL': 'error',
        'UCC_LOG_LEVEL': 'error',
        'TORCH_DISTRIBUTED_BACKEND': 'gloo',
        'NCCL_SOCKET_IFNAME': 'lo',
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1'
    }
    
    for var, value in isolation_vars.items():
        os.environ[var] = value
        print(f"  设置隔离变量: {var}={value}")
    
    # 3. 预加载CUDA库以避免冲突
    cuda_libs = [
        'libcuda.so.1',
        'libcudart.so.12',
        'libnccl.so.2'
    ]
    
    preload_libs = []
    for lib in cuda_libs:
        lib_path = find_library(lib.split('.')[0])
        if lib_path:
            preload_libs.append(lib_path)
            print(f"  发现CUDA库: {lib} -> {lib_path}")
    
    if preload_libs:
        current_preload = os.environ.get('LD_PRELOAD', '')
        new_preload = ':'.join(preload_libs)
        if current_preload:
            new_preload = f"{new_preload}:{current_preload}"
        os.environ['LD_PRELOAD'] = new_preload
        print(f"  设置LD_PRELOAD: {new_preload}")
    
    print("✅ 库隔离环境设置完成")

def test_torch_import():
    """测试PyTorch导入"""
    print("🔍 测试PyTorch导入...")
    
    try:
        # 使用隔离环境导入PyTorch
        import torch
        print(f"✅ PyTorch导入成功: {torch.__version__}")
        
        # 测试CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.device_count()} 设备")
        else:
            print("⚠️  CUDA不可用，但PyTorch导入成功")
        
        # 测试分布式
        import torch.distributed as dist
        print(f"✅ torch.distributed导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def run_target_script():
    """运行目标脚本"""
    if len(sys.argv) < 2:
        print("用法: python3.10 torch_isolation_wrapper.py <目标脚本> [参数...]")
        sys.exit(1)
    
    target_script = sys.argv[1]
    target_args = sys.argv[2:]
    
    print(f"🚀 运行目标脚本: {target_script}")
    print(f"📋 参数: {target_args}")
    
    # 设置隔离环境
    setup_library_isolation()
    
    # 测试PyTorch
    if not test_torch_import():
        print("❌ PyTorch环境验证失败，停止执行")
        sys.exit(1)
    
    # 执行目标脚本
    import subprocess
    cmd = ['python3.10', target_script] + target_args
    
    print(f"🔄 执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)

if __name__ == "__main__":
    # 如果没有提供参数，只做测试
    if len(sys.argv) == 1:
        setup_library_isolation()
        success = test_torch_import()
        sys.exit(0 if success else 1)
    else:
        run_target_script()
