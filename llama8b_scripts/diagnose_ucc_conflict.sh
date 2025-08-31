#!/bin/bash
# HPC-X冲突诊断脚本
# 用于分析UCC库冲突的根本原因
# 运行方式: bash run_maskllm_native.sh diagnose_ucc_conflict.sh

echo "=================================="
echo "HPC-X冲突诊断开始"
echo "=================================="

# 1. 检查Python和PyTorch环境
echo "1. Python环境信息："
python3 --version
echo "PyTorch路径："
python3 -c "import torch; print(torch.__file__)" 2>/dev/null || echo "PyTorch导入失败"

# 2. 检查HPC-X相关库的位置和版本
echo -e "\n2. HPC-X库位置检查："
echo "主机HPC-X路径："
ls -la /opt/hpcx/ 2>/dev/null || echo "/opt/hpcx/ 不存在"

echo "容器内UCX/UCC库："
find /usr -name "*ucx*" 2>/dev/null | head -10
find /usr -name "*ucc*" 2>/dev/null | head -10

# 3. 检查动态链接库路径
echo -e "\n3. 动态链接库路径："
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "当前ldconfig缓存中的UCX/UCC："
ldconfig -p | grep -E "(ucx|ucc)" || echo "未找到UCX/UCC在系统库缓存中"

# 4. 检查符号定义
echo -e "\n4. 符号检查："
echo "检查ucs_mpool_params_reset符号："
# 在所有相关库中搜索这个符号
for lib in $(find /opt/hpcx/ /usr -name "*.so*" 2>/dev/null | grep -E "(ucx|ucc)" | head -20); do
    if [ -f "$lib" ]; then
        echo "检查 $lib："
        nm -D "$lib" 2>/dev/null | grep "ucs_mpool_params_reset" && echo "  找到符号！" || echo "  未找到符号"
    fi
done

# 5. 检查版本兼容性
echo -e "\n5. 版本信息："
echo "HPC-X版本（如果可用）："
if [ -f /opt/hpcx/VERSION ]; then
    cat /opt/hpcx/VERSION
elif [ -d /opt/hpcx ]; then
    ls /opt/hpcx/
fi

# 6. 测试最小PyTorch导入
echo -e "\n6. PyTorch导入测试："
python3 -c "
import sys
print('Python路径:', sys.path)
print('尝试导入torch...')
try:
    import torch
    print('PyTorch导入成功，版本:', torch.__version__)
except Exception as e:
    print('PyTorch导入失败:', str(e))
    print('错误类型:', type(e).__name__)
"

# 7. 环境变量检查
echo -e "\n7. 相关环境变量："
env | grep -E "(HPC|UCX|UCC|MPI)" | sort

echo -e "\n=================================="
echo "诊断完成，请将结果发送给助手分析"
echo "=================================="
