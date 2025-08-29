#!/usr/bin/env python3.10

"""
深度库依赖分析脚本
分析PyTorch导入失败的根本原因
"""

import os
import sys
import subprocess
import ctypes
from ctypes.util import find_library

def run_command(cmd, shell=True):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "命令超时"
    except Exception as e:
        return -1, "", str(e)

def check_library_dependencies():
    """检查关键库的依赖关系"""
    print("🔍 检查关键库的依赖关系...")
    
    # 查找PyTorch核心库
    torch_paths = [
        "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so",
        "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so",
        "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so"
    ]
    
    for torch_lib in torch_paths:
        if os.path.exists(torch_lib):
            print(f"\n📋 分析 {torch_lib}:")
            
            # 使用ldd查看依赖
            ret, stdout, stderr = run_command(f"ldd {torch_lib}")
            if ret == 0:
                lines = stdout.split('\n')
                hpcx_deps = [line for line in lines if 'hpcx' in line or 'ucc' in line or 'ucx' in line]
                if hpcx_deps:
                    print("  ❌ 发现HPC-X依赖:")
                    for dep in hpcx_deps:
                        print(f"    {dep.strip()}")
                else:
                    print("  ✅ 未发现直接HPC-X依赖")
                
                # 查看是否有间接的通信库依赖
                comm_deps = [line for line in lines if any(lib in line for lib in ['nccl', 'mpi', 'ibverbs'])]
                if comm_deps:
                    print("  📋 通信库依赖:")
                    for dep in comm_deps:
                        print(f"    {dep.strip()}")
            else:
                print(f"  ❌ ldd失败: {stderr}")

def check_dynamic_linker_cache():
    """检查动态链接器缓存"""
    print("\n🔍 检查动态链接器缓存...")
    
    ret, stdout, stderr = run_command("ldconfig -p | grep -E '(ucc|ucx|hpcx)'")
    if ret == 0 and stdout.strip():
        print("❌ 在ldconfig缓存中发现HPC-X库:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("✅ ldconfig缓存中未发现HPC-X库")

def check_environment_variables():
    """检查所有相关环境变量"""
    print("\n🔍 检查环境变量...")
    
    # 关键环境变量
    key_vars = [
        'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LIBRARY_PATH',
        'PYTHONPATH', 'PATH',
        'OMPI_MCA_pml', 'OMPI_MCA_btl', 'UCX_TLS', 'UCC_TLS',
        'TORCH_DISTRIBUTED_BACKEND', 'NCCL_SOCKET_IFNAME'
    ]
    
    for var in key_vars:
        value = os.environ.get(var, 'Not Set')
        if len(value) > 150:
            value = value[:150] + "..."
        print(f"  {var}: {value}")

def test_minimal_torch_import():
    """测试最小化的torch导入"""
    print("\n🔍 测试最小化torch导入...")
    
    # 尝试不同的导入策略
    import_tests = [
        ("基础torch导入", "import torch"),
        ("torch._C导入", "import torch._C"),
        ("torch.version导入", "import torch; print(torch.__version__)"),
    ]
    
    for test_name, import_code in import_tests:
        print(f"\n📋 {test_name}:")
        ret, stdout, stderr = run_command(f'python3.10 -c "{import_code}"')
        if ret == 0:
            print(f"  ✅ 成功: {stdout.strip()}")
        else:
            print(f"  ❌ 失败: {stderr.strip()}")
            # 分析错误信息
            if "libucc.so.1" in stderr:
                print("    🔍 确认是UCC库冲突")
            if "undefined symbol" in stderr:
                print("    🔍 符号未定义错误")

def analyze_library_search_paths():
    """分析库搜索路径"""
    print("\n🔍 分析库搜索路径...")
    
    # 检查/etc/ld.so.conf.d/中的配置
    ret, stdout, stderr = run_command("find /etc/ld.so.conf.d/ -name '*.conf' -exec grep -l 'hpcx\\|ucx\\|ucc' {} \\; 2>/dev/null")
    if ret == 0 and stdout.strip():
        print("❌ 在系统配置中发现HPC-X路径:")
        for conf_file in stdout.split('\n'):
            if conf_file.strip():
                print(f"  配置文件: {conf_file.strip()}")
                ret2, content, _ = run_command(f"cat {conf_file.strip()}")
                if ret2 == 0:
                    for line in content.split('\n'):
                        if any(term in line for term in ['hpcx', 'ucx', 'ucc']):
                            print(f"    {line.strip()}")
    else:
        print("✅ 系统配置中未发现HPC-X路径")

def check_alternative_torch_locations():
    """检查其他位置的PyTorch"""
    print("\n🔍 检查其他位置的PyTorch...")
    
    # 可能的PyTorch位置
    torch_locations = [
        "/data/home/zdhs0054/zzhou/MaskLLM/.ext_pkgs/torch",
        "/usr/local/lib/python3.10/dist-packages/torch",
        "/usr/local/lib/python3.10/site-packages/torch",
    ]
    
    for location in torch_locations:
        if os.path.exists(location):
            print(f"📋 发现PyTorch: {location}")
            # 检查版本
            init_file = os.path.join(location, "__init__.py")
            if os.path.exists(init_file):
                ret, stdout, stderr = run_command(f'cd {os.path.dirname(location)} && python3.10 -c "import torch; print(torch.__version__, torch.__file__)"')
                if ret == 0:
                    print(f"  ✅ 版本: {stdout.strip()}")
                else:
                    print(f"  ❌ 导入失败: {stderr.strip()}")

def suggest_fixes():
    """建议修复方案"""
    print("\n💡 建议的修复方案:")
    print("1. 强制使用.ext_pkgs中的PyTorch:")
    print("   export PYTHONPATH='/data/home/zdhs0054/zzhou/MaskLLM/.ext_pkgs:$PYTHONPATH'")
    print("")
    print("2. 使用LD_PRELOAD强制加载正确的库:")
    print("   export LD_PRELOAD='/usr/local/cuda/lib64/libcudart.so'")
    print("")
    print("3. 完全禁用HPC-X:")
    print("   unset $(env | grep -E '^(HPCX|OMPI|UCX|UCC)' | cut -d= -f1)")
    print("")
    print("4. 使用Python虚拟环境隔离:")
    print("   创建专门的虚拟环境避免系统库冲突")

def main():
    """主函数"""
    print("🚀 开始深度库依赖分析")
    print("=" * 60)
    
    # 环境信息
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    print("")
    
    # 各项检查
    check_environment_variables()
    check_library_dependencies()
    check_dynamic_linker_cache()
    analyze_library_search_paths()
    check_alternative_torch_locations()
    test_minimal_torch_import()
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("🎯 分析完成")

if __name__ == "__main__":
    main()
