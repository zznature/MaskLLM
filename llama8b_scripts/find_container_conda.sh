#!/bin/bash

# 容器内conda环境检测脚本
# 在NGC PyTorch容器中查找conda环境

echo "🔍 开始检测容器内conda环境..."
echo "容器: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# 1. 检查当前环境变量
echo "📋 步骤1: 检查现有环境变量"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-未设置}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-未设置}"
echo "PATH: $PATH"
echo ""

# 2. 搜索可能的conda位置
echo "📋 步骤2: 搜索conda安装位置"
CONDA_SEARCH_PATHS=(
    "/opt/conda"
    "/usr/local/conda" 
    "/opt/miniconda3"
    "/usr/local/miniconda3"
    "/conda"
    "/opt/miniconda"
    "/usr/conda"
)

echo "搜索常见conda路径:"
for path in "${CONDA_SEARCH_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ✅ 发现: $path"
        if [ -f "$path/bin/conda" ]; then
            echo "    - conda可执行文件存在"
        fi
        if [ -d "$path/lib" ]; then
            echo "    - lib目录存在"
        fi
        if [ -d "$path/lib/python3.10" ]; then
            echo "    - Python 3.10目录存在"
        fi
    else
        echo "  ❌ 不存在: $path"
    fi
done
echo ""

# 3. 搜索conda可执行文件
echo "📋 步骤3: 搜索conda可执行文件"
echo "使用which命令:"
which conda 2>/dev/null && echo "  ✅ conda命令可用: $(which conda)" || echo "  ❌ conda命令不可用"

echo "使用find命令搜索conda:"
find /opt /usr/local /usr /conda -name "conda" -type f -executable 2>/dev/null | head -5

echo ""

# 4. 搜索关键库文件位置
echo "📋 步骤4: 搜索关键库文件"

echo "🔍 搜索libcudnn.so.8.9.7:"
find /opt /usr/local /usr /conda -name "libcudnn.so.8.9.7" 2>/dev/null | head -3

echo "🔍 搜索libcudnn相关文件:"
find /opt /usr/local /usr /conda -name "*libcudnn*8.9*" 2>/dev/null | head -5

echo "🔍 搜索libnccl相关文件:"
find /opt /usr/local /usr /conda -name "*libnccl*" 2>/dev/null | head -3

echo "🔍 搜索libcupti相关文件:"
find /opt /usr/local /usr /conda -name "*libcupti*" 2>/dev/null | head -3

echo ""

# 5. 检查Python包位置
echo "📋 步骤5: 检查Python包位置"
echo "Python可执行文件: $(which python3.10)"
echo "Python库路径:"
python3.10 -c "import sys; [print(f'  {i+1}. {path}') for i, path in enumerate(sys.path[:8])]" 2>/dev/null

echo ""
echo "搜索nvidia相关Python包:"
find /opt /usr/local /usr /conda -path "*/site-packages/nvidia*" -type d 2>/dev/null | head -5

echo ""

# 6. 检查environment.yml或conda-meta
echo "📋 步骤6: 检查conda环境信息"
find /opt /usr/local /usr /conda -name "conda-meta" -type d 2>/dev/null | head -3
find /opt /usr/local /usr /conda -name "environment.yml" 2>/dev/null | head -3

echo ""

# 7. 建议可能的CONDA_PREFIX
echo "📋 步骤7: 环境设置建议"

# 尝试自动检测最可能的conda路径
BEST_CONDA_PATH=""
for path in "${CONDA_SEARCH_PATHS[@]}"; do
    if [ -d "$path/lib" ] && [ -d "$path/bin" ]; then
        # 检查是否有关键库文件
        if find "$path" -name "*libcudnn*" 2>/dev/null | grep -q .; then
            BEST_CONDA_PATH="$path"
            echo "🎯 推荐设置: export CONDA_PREFIX=\"$path\""
            echo "   原因: 目录存在且包含cuDNN库文件"
            break
        elif [ -d "$path/lib/python3.10" ]; then
            BEST_CONDA_PATH="$path" 
            echo "🎯 备选设置: export CONDA_PREFIX=\"$path\""
            echo "   原因: 目录存在且有Python 3.10"
        fi
    fi
done

if [ -z "$BEST_CONDA_PATH" ]; then
    echo "⚠️  无法自动检测conda环境"
    echo "💡 可能的解决方案:"
    echo "   1. 检查容器是否正确启动"
    echo "   2. 使用其他方法定位库文件"
    echo "   3. 直接设置库文件路径而不依赖conda"
fi

echo ""
echo "🎯 推荐下一步:"
if [ -n "$BEST_CONDA_PATH" ]; then
    echo "1. 设置环境: export CONDA_PREFIX=\"$BEST_CONDA_PATH\""
    echo "2. 重新运行: bash llama8b_scripts/fix_container_cuda_libs.sh"
else
    echo "1. 手动查找库文件位置"
    echo "2. 创建直接修复脚本"
fi
