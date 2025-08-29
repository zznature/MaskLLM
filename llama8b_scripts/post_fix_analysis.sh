#!/bin/bash

# 方案B后的问题深入分析
# 分析移除.ext_pkgs后暴露的cuDNN库配置问题

echo "🔍 方案B后的问题深入分析"
echo "时间: $(date)"
echo "问题: 移除.ext_pkgs后出现cuDNN库缺失"
echo "=================================="

# 1. 问题演进分析
echo "1️⃣ 问题演进分析"
echo "📋 修复效果验证:"
echo "   ✅ UCC符号冲突 → 已解决 (PyTorch现在是容器内版本)"
echo "   ❌ cuDNN库缺失 → 新问题暴露"
echo ""

echo "🔍 问题根源推测:"
echo "   .ext_pkgs之前提供了cuDNN库文件"
echo "   移除后暴露了容器内cuDNN配置问题"
echo "   这个问题之前被.ext_pkgs掩盖了"
echo ""

# 2. 检查容器内cuDNN库状态
echo "2️⃣ 检查容器内cuDNN库状态"

echo "🔍 查找cuDNN库文件:"
CUDNN_PATHS=(
    "/usr/local/lib/python3.10/dist-packages/torch/lib"
    "/usr/local/cuda/lib64"
    "/opt/conda_libs"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/lib"
)

FOUND_CUDNN=false
for path in "${CUDNN_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "   📁 检查路径: $path"
        cudnn_files=$(find "$path" -name "*cudnn*" 2>/dev/null | head -5)
        if [ -n "$cudnn_files" ]; then
            echo "$cudnn_files" | while read file; do
                echo "     ✅ $file"
            done
            FOUND_CUDNN=true
        else
            echo "     ❌ 未找到cuDNN文件"
        fi
    else
        echo "   ⚪ 路径不存在: $path"
    fi
done

echo ""

# 3. 检查当前LD_LIBRARY_PATH
echo "3️⃣ 检查当前LD_LIBRARY_PATH"
echo "🔍 当前LD_LIBRARY_PATH设置:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | while read path; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        cudnn_in_path=$(find "$path" -name "*cudnn*" 2>/dev/null | wc -l)
        if [ "$cudnn_in_path" -gt 0 ]; then
            echo "   ✅ $path (包含cuDNN: $cudnn_in_path个文件)"
        else
            echo "   ⚪ $path (无cuDNN文件)"
        fi
    elif [ -n "$path" ]; then
        echo "   ❌ $path (路径不存在)"
    fi
done

echo ""

# 4. 检查.ext_pkgs中的cuDNN
echo "4️⃣ 检查.ext_pkgs中之前提供的cuDNN"
if [ -d ".ext_pkgs" ]; then
    echo "🔍 .ext_pkgs中的cuDNN相关文件:"
    find .ext_pkgs -name "*cudnn*" 2>/dev/null | head -10 | while read file; do
        echo "   📦 $file"
    done
    
    echo ""
    echo "🔍 .ext_pkgs中的NVIDIA库:"
    if [ -d ".ext_pkgs/nvidia" ]; then
        ls -la .ext_pkgs/nvidia/ | head -10
    else
        echo "   ❌ 无.ext_pkgs/nvidia目录"
    fi
else
    echo "❌ .ext_pkgs目录不存在"
fi

echo ""

# 5. 对比Project_Tasks.md中的已知解决方案
echo "5️⃣ 检查Project_Tasks.md中的已知解决方案"
echo "📋 Project_Tasks.md记录的cuDNN解决方案:"

if [ -f "Project_Tasks.md" ]; then
    echo "🔍 搜索cuDNN相关解决方案:"
    grep -n -A 5 -B 5 "libcudnn\|cuDNN\|fix_container_libs" Project_Tasks.md | head -20
else
    echo "❌ Project_Tasks.md文件不存在"
fi

echo ""

# 6. 生成修复策略
echo "6️⃣ 生成修复策略"

echo "📋 修复策略分析:"
echo ""

echo "策略A: 重新应用Project_Tasks.md中的cuDNN修复方案"
echo "   描述: 使用已知的无符号链接LD_LIBRARY_PATH方案"
echo "   文件: llama8b_scripts/fix_container_libs_no_symlink.sh"
echo "   状态: $([ -f "llama8b_scripts/fix_container_libs_no_symlink.sh" ] && echo "存在" || echo "不存在")"
echo ""

echo "策略B: 有选择性地恢复.ext_pkgs中的cuDNN"
echo "   描述: 只恢复cuDNN相关库，避免重新引入UCC冲突"
echo "   方法: 创建专门的cuDNN库路径"
echo ""

echo "策略C: 混合方案 - 容器内PyTorch + 外部cuDNN"
echo "   描述: 保持容器内PyTorch，但使用.ext_pkgs提供cuDNN"
echo "   风险: 可能重新引入部分冲突"
echo ""

# 7. 快速诊断命令
echo "7️⃣ 快速诊断和修复"

echo "💡 推荐执行顺序:"
echo "1. 首先尝试策略A (应用已知的容器内cuDNN修复):"
echo "   bash llama8b_scripts/fix_container_libs_no_symlink.sh"
echo ""

echo "2. 如果策略A失败，尝试策略B (有选择性恢复cuDNN):"
echo "   创建专门的cuDNN环境变量"
echo ""

echo "3. 验证修复:"
echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""

# 8. 检查是否存在已知修复脚本
echo "8️⃣ 检查现有修复脚本"

KNOWN_FIX_SCRIPTS=(
    "llama8b_scripts/fix_container_libs_no_symlink.sh"
    "llama8b_scripts/fix_container_cuda_libs.sh"
    "llama8b_scripts/fix_container_cuda_libs_direct.sh"
)

echo "🔍 检查已知的cuDNN修复脚本:"
for script in "${KNOWN_FIX_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "   ✅ $script (存在)"
        # 检查脚本的修改时间
        mod_time=$(stat -c "%Y" "$script")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        echo "      📅 修改于 $days_ago 天前"
    else
        echo "   ❌ $script (不存在)"
    fi
done

echo ""
echo "🎯 分析完成"
echo "💡 结论: 问题从UCC冲突转变为cuDNN库配置问题"
echo "💡 下一步: 应用Project_Tasks.md中已验证的cuDNN解决方案"
