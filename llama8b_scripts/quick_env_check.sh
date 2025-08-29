#!/bin/bash

# 快速环境检查脚本 - 专注于UCC冲突相关的环境变化
# 用途: 快速诊断导致UCC符号冲突的关键文件变更

echo "⚡ 快速环境检查"
echo "时间: $(date)"
echo "目标: 快速定位UCC冲突相关的环境变化"
echo "=================================="

cd /data/home/zdhs0054/zzhou/MaskLLM || exit 1

# 1. 检查关键脚本文件的最近修改
echo "1️⃣ 检查关键环境脚本..."

KEY_SCRIPTS=(
    "run_maskllm_native.sh"
    "run_apptainer_extended_libs.sh" 
    "fix_transformers.sh"
    "fix_transformers_safe.sh"
    "llama8b_scripts/radical_efficient_solution.sh"
    "llama8b_scripts/persistent_environment_fix.sh"
    "llama8b_scripts/container_environment_diagnosis.sh"
)

RECENT_CHANGES=()

for script in "${KEY_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mod_time=$(stat -c "%Y" "$script")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        
        if [ $days_ago -le 7 ]; then
            RECENT_CHANGES+=("$script:$days_ago")
            echo "🔴 $script - 修改于 $days_ago 天前"
            echo "   详情: $(stat -c "%y" "$script")"
        else
            echo "🟢 $script - 修改于 $days_ago 天前 (稳定)"
        fi
    else
        echo "⚪ $script - 文件不存在"
    fi
done

echo ""

# 2. 检查最近的git提交，特别关注环境相关的变更
echo "2️⃣ 检查最近的环境相关提交..."

if [ -d ".git" ]; then
    echo "🔍 最近7天的提交记录:"
    git log --since="7 days ago" --oneline --grep="环境\|env\|container\|ucc\|hpc\|fix" | head -5
    
    echo ""
    echo "🔍 最近修改的环境脚本文件:"
    git log --since="7 days ago" --name-only --pretty=format: | grep -E "\.(sh|py)$" | grep -E "(run_|fix_|env|container)" | sort | uniq
else
    echo "⚠️ 不是Git仓库"
fi

echo ""

# 3. 快速检查容器sif文件
echo "3️⃣ 检查容器文件状态..."

if [ -f "pytorch_24.01-py3.sif" ]; then
    sif_mod=$(stat -c "%Y" "pytorch_24.01-py3.sif")
    current_time=$(date +%s)
    sif_days_ago=$(( (current_time - sif_mod) / 86400 ))
    
    echo "📦 pytorch_24.01-py3.sif - 修改于 $sif_days_ago 天前"
    if [ $sif_days_ago -le 7 ]; then
        echo "🔴 容器文件最近被修改！这可能是问题根源"
    else
        echo "🟢 容器文件稳定"
    fi
else
    echo "❌ 容器文件不存在！"
fi

echo ""

# 4. 检查HPC-X和UCC相关环境变量设置
echo "4️⃣ 检查UCC相关环境配置..."

if [ -f "$HOME/.bashrc" ]; then
    bashrc_mod=$(stat -c "%Y" "$HOME/.bashrc")
    current_time=$(date +%s)
    bashrc_days_ago=$(( (current_time - bashrc_mod) / 86400 ))
    
    echo "📝 ~/.bashrc - 修改于 $bashrc_days_ago 天前"
    
    if [ $bashrc_days_ago -le 7 ]; then
        echo "🔴 .bashrc最近被修改，检查UCC相关设置:"
        grep -n -i "ucc\|hpcx\|LD_LIBRARY_PATH" "$HOME/.bashrc" | head -5 || echo "   无UCC相关设置"
    fi
fi

echo ""

# 5. 生成快速诊断结果
echo "5️⃣ 快速诊断结果..."

if [ ${#RECENT_CHANGES[@]} -gt 0 ]; then
    echo "⚠️ 发现最近修改的关键文件:"
    for change in "${RECENT_CHANGES[@]}"; do
        file=$(echo "$change" | cut -d: -f1)
        days=$(echo "$change" | cut -d: -f2)
        echo "   🔸 $file ($days 天前)"
    done
    
    echo ""
    echo "💡 建议操作:"
    echo "1. 运行完整诊断: bash llama8b_scripts/environment_change_diagnosis.sh"
    echo "2. 检查最近修改的文件内容变化"
    echo "3. 考虑回退到7天前的稳定版本"
    
else
    echo "✅ 没有发现最近修改的关键环境文件"
    echo "💡 问题可能来自:"
    echo "   - 系统级更新"
    echo "   - 容器镜像变化" 
    echo "   - 外部依赖更新"
    echo "   - HPC集群环境变化"
fi

echo ""

# 6. 提供immediate action建议
echo "6️⃣ 立即可尝试的解决方案..."

echo "方案A: 快速回退最近修改的脚本"
echo "   git checkout HEAD~1 -- run_maskllm_native.sh"
echo ""

echo "方案B: 使用7天前的工作版本"
echo "   git log --since=\"7 days ago\" --oneline"
echo "   git checkout <稳定版本hash> -- <有问题的文件>"
echo ""

echo "方案C: 重新下载原始容器"
echo "   # 如果容器文件被修改"
echo ""

echo "⚡ 快速检查完成"
echo "如需详细分析，请运行: bash llama8b_scripts/environment_change_diagnosis.sh"
