#!/bin/bash

# 快速Apptainer环境分析脚本
# 避免长时间等待，批量收集信息

echo "🔍 快速Apptainer环境分析"
echo "时间: $(date)"
echo "=" * 50

# 基础环境信息
echo "📋 第1部分: 基础环境信息"
echo "容器名称: $APPTAINER_NAME"
echo "当前目录: $(pwd)"
echo "用户: $(whoami)"
echo ""

# HPC-X库位置和时间戳分析
echo "📋 第2部分: HPC-X库分析"
echo "🔍 HPC-X目录结构:"
ls -la /opt/hpcx/ 2>/dev/null | head -5
echo ""

echo "🔍 UCC库详细信息:"
if [ -d "/opt/hpcx/ucc" ]; then
    echo "  ✅ UCC目录存在: /opt/hpcx/ucc"
    ls -la /opt/hpcx/ucc/lib/libucc* 2>/dev/null || echo "  ❌ UCC库文件不存在"
    
    # 检查文件时间戳
    echo "  📅 UCC目录时间戳:"
    stat /opt/hpcx/ucc/ 2>/dev/null | grep -E "(Modify|Change|Birth)" || echo "    ❌ 无法获取时间戳"
else
    echo "  ❌ UCC目录不存在"
fi
echo ""

echo "🔍 HPC-X版本信息:"
cat /opt/hpcx/VERSION 2>/dev/null || echo "  ❌ 无法读取版本信息"
echo ""

# PyTorch和库依赖分析
echo "📋 第3部分: PyTorch库依赖分析"
echo "🔍 PyTorch位置:"
find /usr/local/lib/python3.10/dist-packages -name "torch" -type d 2>/dev/null | head -1

echo "🔍 PyTorch核心库依赖:"
if [ -f "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so" ]; then
    echo "  ✅ PyTorch CUDA库存在"
    # 检查库的符号依赖
    ldd /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so 2>/dev/null | grep -E "(ucc|hpcx)" | head -3 || echo "  💡 无UCC/HPC-X直接依赖"
else
    echo "  ❌ PyTorch CUDA库不存在"
fi
echo ""

# 系统UCC库检查
echo "📋 第4部分: 系统UCC库检查"
echo "🔍 系统中的UCC库:"
find /usr -name "libucc*" -type f 2>/dev/null | head -3 || echo "  ❌ 系统无UCC库"
find /lib -name "libucc*" -type f 2>/dev/null | head -3 || echo "  ❌ /lib无UCC库"
echo ""

# 最近文件变更检查
echo "📋 第5部分: 最近10天文件变更检查"
echo "🔍 HPC-X相关文件最近变更:"
find /opt/hpcx -type f -mtime -10 2>/dev/null | head -5 || echo "  💡 无最近变更文件"

echo "🔍 PyTorch相关文件最近变更:"
find /usr/local/lib/python3.10/dist-packages/torch -type f -mtime -10 2>/dev/null | head -3 || echo "  💡 无最近变更文件"
echo ""

# 当前错误复现
echo "📋 第6部分: 当前错误状态"
echo "🧪 尝试导入PyTorch:"
python3.10 -c "
try:
    import torch
    print('✅ PyTorch导入成功')
    print(f'版本: {torch.__version__}')
except Exception as e:
    print('❌ PyTorch导入失败')
    error_msg = str(e)
    if 'ucc' in error_msg.lower():
        print('🎯 确认UCC符号问题')
        # 提取具体的符号错误
        import re
        symbol_match = re.search(r'undefined symbol: ([a-zA-Z_][a-zA-Z0-9_]*)', error_msg)
        if symbol_match:
            print(f'缺失符号: {symbol_match.group(1)}')
        else:
            print(f'错误详情: {error_msg[:100]}...')
    else:
        print(f'其他错误: {error_msg[:100]}...')
"

echo ""
echo "🎯 快速分析完成"
echo "💡 基于以上信息确定最佳解决策略"
echo "=" * 50
