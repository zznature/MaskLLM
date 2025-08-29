#!/bin/bash

# 环境问题深度诊断脚本
# 诊断UCC冲突、PYTHONPATH和Python执行器问题

echo "🔍 环境问题深度诊断"
echo "时间: $(date)"
echo "目标: 诊断UCC冲突、PYTHONPATH和Python执行器问题"
echo "=================================="

# 1. 诊断Python执行器问题
echo "1️⃣ 诊断Python执行器问题"
echo "当前Python路径检查:"
echo "  which python: $(which python 2>/dev/null || echo 'not found')"
echo "  which python3: $(which python3 2>/dev/null || echo 'not found')"
echo "  which python3.10: $(which python3.10 2>/dev/null || echo 'not found')"
echo ""

echo "Python版本检查:"
python --version 2>/dev/null || echo "python: command not found"
python3 --version 2>/dev/null || echo "python3: command not found"
python3.10 --version 2>/dev/null || echo "python3.10: command not found"
echo ""

echo "文件属性检查:"
ls -la /usr/bin/python* | head -5
echo ""

# 2. 诊断PYTHONPATH问题
echo "2️⃣ 诊断PYTHONPATH问题"
echo "当前PYTHONPATH:"
echo "$PYTHONPATH" | tr ':' '\n' | nl
echo ""

echo "sys.path检查:"
python3 -c "
import sys
print('Python sys.path:')
for i, path in enumerate(sys.path):
    print(f'  {i+1}: {path}')
" 2>/dev/null || echo "无法执行Python检查"
echo ""

# 3. 诊断.ext_pkgs路径问题
echo "3️⃣ 诊断.ext_pkgs路径问题"
EXT_PKGS_DIR="$(pwd)/.ext_pkgs"
echo "EXT_PKGS_DIR: $EXT_PKGS_DIR"
echo "目录是否存在: $([ -d "$EXT_PKGS_DIR" ] && echo '✅ 存在' || echo '❌ 不存在')"

if [ -d "$EXT_PKGS_DIR" ]; then
    echo "transformers路径检查:"
    echo "  $EXT_PKGS_DIR/transformers: $([ -d "$EXT_PKGS_DIR/transformers" ] && echo '✅ 存在' || echo '❌ 不存在')"
    
    if [ -d "$EXT_PKGS_DIR/transformers" ]; then
        echo "  transformers/__init__.py: $([ -f "$EXT_PKGS_DIR/transformers/__init__.py" ] && echo '✅ 存在' || echo '❌ 不存在')"
        echo "  transformers目录内容:"
        ls -la "$EXT_PKGS_DIR/transformers" | head -5
    fi
fi
echo ""

# 4. 诊断UCC冲突问题
echo "4️⃣ 诊断UCC冲突问题"
echo "当前LD_LIBRARY_PATH中的HPC-X相关路径:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i -E "(hpcx|ucc|ucx)" | nl || echo "无HPC-X路径"
echo ""

echo "UCC库文件检查:"
find /opt -name "*ucc*" 2>/dev/null | head -5 || echo "无法访问/opt目录"
echo ""

echo "环境变量中的UCC相关设置:"
env | grep -i ucc || echo "无UCC环境变量"
echo ""

# 5. 生成修复建议
echo "5️⃣ 修复建议生成"
echo "=================================="

echo "问题A: Python执行器问题"
if ! python3 --version >/dev/null 2>&1; then
    echo "  🔴 Python3不可用"
    echo "  💡 建议: 使用 /usr/bin/python3 或检查PATH设置"
else
    echo "  ✅ Python3可用"
fi

echo ""

echo "问题B: PYTHONPATH配置问题"
if echo "$PYTHONPATH" | grep -q ".ext_pkgs"; then
    echo "  ✅ PYTHONPATH包含.ext_pkgs"
else
    echo "  🔴 PYTHONPATH缺少.ext_pkgs路径"
    echo "  💡 建议: 检查PYTHONPATH构建逻辑"
fi

echo ""

echo "问题C: UCC冲突问题"
if echo "$LD_LIBRARY_PATH" | grep -q -i "hpcx"; then
    echo "  🔴 LD_LIBRARY_PATH仍包含HPC-X路径"
    echo "  💡 建议: 更彻底的HPC-X清理"
else
    echo "  ✅ LD_LIBRARY_PATH无明显HPC-X路径"
    echo "  💡 但UCC符号冲突可能来自其他源"
fi

echo ""
echo "🎯 诊断完成"
