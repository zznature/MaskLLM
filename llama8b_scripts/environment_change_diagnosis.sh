#!/bin/bash

# 环境变化诊断脚本 - 检查7天内环境文件变更并提供复原方案
# 用途: 诊断为什么上周正常的容器环境现在出现问题

echo "🔍 环境变化诊断脚本"
echo "时间: $(date)"
echo "目标: 检查7天内环境文件变更，定位问题根源"
echo "=================================="

# 设置工作目录
cd /data/home/zdhs0054/zzhou/MaskLLM || exit 1

# 创建诊断结果目录
DIAGNOSIS_DIR="environment_diagnosis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAGNOSIS_DIR"

echo "📋 诊断结果将保存到: $DIAGNOSIS_DIR"
echo ""

# 1. 检查项目根目录7天内修改的文件
echo "1️⃣ 检查项目根目录7天内修改的文件..."
echo "📁 检查范围: 项目根目录"
echo "⏰ 时间范围: 最近7天"

find . -maxdepth 2 -type f -mtime -7 -not -path "./.git/*" | sort > "$DIAGNOSIS_DIR/recently_modified_files.txt"

echo "🔍 发现修改的文件数量: $(wc -l < "$DIAGNOSIS_DIR/recently_modified_files.txt")"
echo "详细列表已保存到: $DIAGNOSIS_DIR/recently_modified_files.txt"

# 显示关键文件的修改情况
echo ""
echo "🎯 关键环境文件修改情况:"
echo "-----------------------------------"

# 检查关键脚本文件
KEY_FILES=(
    "run_maskllm_native.sh"
    "run_apptainer_extended_libs.sh" 
    "check_and_enter_apptainer.sh"
    "fix_transformers.sh"
    "fix_transformers_safe.sh"
)

for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        mod_time=$(stat -c "%Y" "$file")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        
        if [ $days_ago -le 7 ]; then
            echo "🔴 $file - 修改于 $days_ago 天前 $(stat -c "%y" "$file")"
            ls -la "$file" >> "$DIAGNOSIS_DIR/key_files_details.txt"
        else
            echo "🟢 $file - 修改于 $days_ago 天前 (超过7天)"
        fi
    else
        echo "⚪ $file - 文件不存在"
    fi
done

echo ""

# 2. 检查git仓库变更
echo "2️⃣ 检查Git仓库变更..."
echo "📋 检查最近7天的提交记录"

if [ -d ".git" ]; then
    echo "Git日志 (最近7天):" > "$DIAGNOSIS_DIR/git_changes.txt"
    git log --since="7 days ago" --oneline --decorate >> "$DIAGNOSIS_DIR/git_changes.txt" 2>/dev/null || echo "无法获取git日志" >> "$DIAGNOSIS_DIR/git_changes.txt"
    
    echo "修改的文件 (最近7天):" >> "$DIAGNOSIS_DIR/git_changes.txt"
    git log --since="7 days ago" --name-only --pretty=format: | sort | uniq >> "$DIAGNOSIS_DIR/git_changes.txt" 2>/dev/null || echo "无法获取修改文件列表" >> "$DIAGNOSIS_DIR/git_changes.txt"
    
    # 显示最近的提交
    echo "🔍 最近的Git提交:"
    git log --since="7 days ago" --oneline --decorate | head -10 || echo "无法获取git日志"
else
    echo "⚠️ 不是Git仓库，跳过Git检查"
fi

echo ""

# 3. 检查容器相关文件
echo "3️⃣ 检查容器相关文件状态..."

CONTAINER_FILES=(
    "pytorch_24.01-py3.sif"
    "container_info"
    "verify_container_env.sh"
    "check_container_info.sh"
)

echo "容器文件状态:" > "$DIAGNOSIS_DIR/container_files_status.txt"

for file in "${CONTAINER_FILES[@]}"; do
    if [ -e "$file" ]; then
        ls -la "$file" >> "$DIAGNOSIS_DIR/container_files_status.txt"
        
        # 检查是否最近修改
        mod_time=$(stat -c "%Y" "$file" 2>/dev/null || echo "0")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        
        if [ $days_ago -le 7 ]; then
            echo "🔴 $file - 最近 $days_ago 天内被修改"
        else
            echo "🟢 $file - 修改于 $days_ago 天前"
        fi
    else
        echo "❌ $file - 文件不存在" >> "$DIAGNOSIS_DIR/container_files_status.txt"
        echo "❌ $file - 文件不存在"
    fi
done

echo ""

# 4. 检查llama8b_scripts目录变更
echo "4️⃣ 检查llama8b_scripts目录变更..."

if [ -d "llama8b_scripts" ]; then
    find llama8b_scripts -type f -mtime -7 | sort > "$DIAGNOSIS_DIR/llama8b_scripts_changes.txt"
    
    echo "🔍 llama8b_scripts目录中7天内修改的文件:"
    if [ -s "$DIAGNOSIS_DIR/llama8b_scripts_changes.txt" ]; then
        while read -r file; do
            days_ago=$(( ($(date +%s) - $(stat -c "%Y" "$file")) / 86400 ))
            echo "📝 $file - $days_ago 天前"
        done < "$DIAGNOSIS_DIR/llama8b_scripts_changes.txt"
    else
        echo "✅ 没有发现最近修改的文件"
    fi
else
    echo "⚠️ llama8b_scripts目录不存在"
fi

echo ""

# 5. 检查环境变量和配置文件
echo "5️⃣ 检查环境配置文件..."

ENV_FILES=(
    "$HOME/.bashrc"
    "$HOME/.bash_profile" 
    "$HOME/.profile"
    "/etc/environment"
    "$HOME/.modulerc"
)

echo "环境配置文件状态:" > "$DIAGNOSIS_DIR/env_files_status.txt"

for file in "${ENV_FILES[@]}"; do
    if [ -f "$file" ]; then
        mod_time=$(stat -c "%Y" "$file" 2>/dev/null || echo "0")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        
        echo "$file - 修改于 $days_ago 天前" >> "$DIAGNOSIS_DIR/env_files_status.txt"
        
        if [ $days_ago -le 7 ]; then
            echo "🔴 $file - 最近 $days_ago 天内被修改"
            # 备份最近修改的环境文件
            cp "$file" "$DIAGNOSIS_DIR/$(basename $file).backup" 2>/dev/null
        else
            echo "🟢 $file - 修改于 $days_ago 天前"
        fi
    else
        echo "⚪ $file - 文件不存在" >> "$DIAGNOSIS_DIR/env_files_status.txt"
    fi
done

echo ""

# 6. 检查系统库文件变更（如果有权限）
echo "6️⃣ 检查系统相关文件..."

SYSTEM_PATHS=(
    "/usr/local/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/opt"
)

echo "系统路径检查:" > "$DIAGNOSIS_DIR/system_paths_check.txt"

for path in "${SYSTEM_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "检查路径: $path" >> "$DIAGNOSIS_DIR/system_paths_check.txt"
        find "$path" -name "*ucc*" -o -name "*hpcx*" -o -name "*mpi*" 2>/dev/null | head -20 >> "$DIAGNOSIS_DIR/system_paths_check.txt" || echo "无权限或无相关文件" >> "$DIAGNOSIS_DIR/system_paths_check.txt"
    else
        echo "$path - 路径不存在" >> "$DIAGNOSIS_DIR/system_paths_check.txt"
    fi
done

echo ""

# 7. 生成诊断报告
echo "7️⃣ 生成诊断报告..."

REPORT_FILE="$DIAGNOSIS_DIR/diagnosis_report.md"

cat > "$REPORT_FILE" << 'EOF'
# 环境变化诊断报告

## 诊断概要
- 诊断时间: 
- 诊断范围: 7天内文件变更
- 问题描述: 上周正常的容器环境现在出现UCC符号冲突问题

## 关键发现

### 1. 最近修改的关键文件
EOF

echo "- 诊断时间: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 添加最近修改文件的详细信息
if [ -s "$DIAGNOSIS_DIR/recently_modified_files.txt" ]; then
    echo "### 最近7天修改的文件:" >> "$REPORT_FILE"
    while read -r file; do
        days_ago=$(( ($(date +%s) - $(stat -c "%Y" "$file")) / 86400 ))
        echo "- $file (修改于 $days_ago 天前)" >> "$REPORT_FILE"
    done < "$DIAGNOSIS_DIR/recently_modified_files.txt"
else
    echo "- 没有发现最近修改的文件" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"

# 8. 生成复原建议
echo "## 复原建议" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 生成复原脚本
RESTORE_SCRIPT="$DIAGNOSIS_DIR/restore_environment.sh"

cat > "$RESTORE_SCRIPT" << 'EOF'
#!/bin/bash

echo "🔄 环境复原脚本"
echo "时间: $(date)"
echo "目标: 复原7天前的环境状态"
echo "======================="

# 1. 备份当前状态
echo "1️⃣ 备份当前环境状态..."
BACKUP_DIR="current_env_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份关键文件
cp run_maskllm_native.sh "$BACKUP_DIR/" 2>/dev/null
cp run_apptainer_extended_libs.sh "$BACKUP_DIR/" 2>/dev/null
cp -r llama8b_scripts "$BACKUP_DIR/" 2>/dev/null

echo "✅ 当前环境已备份到: $BACKUP_DIR"

# 2. Git reset选项
echo ""
echo "2️⃣ Git复原选项:"
echo "选择以下复原方式之一:"
echo ""
echo "方式A: 软复原 (保留工作目录更改)"
echo "git reset --soft HEAD~N  # N为要回退的提交数"
echo ""
echo "方式B: 混合复原 (保留工作目录，重置暂存区)"  
echo "git reset --mixed HEAD~N"
echo ""
echo "方式C: 硬复原 (完全回退，丢失所有更改)"
echo "git reset --hard HEAD~N"
echo ""
echo "方式D: 回退到特定日期"
echo 'git log --since="7 days ago" --oneline  # 查看最近提交'
echo 'git reset --hard <commit_hash>  # 回退到指定提交'

# 3. 文件级复原
echo ""
echo "3️⃣ 文件级复原 (如果不想使用git):"

EOF

# 检查哪些关键文件最近被修改，生成具体的复原命令
for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        mod_time=$(stat -c "%Y" "$file")
        current_time=$(date +%s)
        days_ago=$(( (current_time - mod_time) / 86400 ))
        
        if [ $days_ago -le 7 ]; then
            echo "echo '🔄 复原 $file...'" >> "$RESTORE_SCRIPT"
            echo "git checkout HEAD~1 -- $file  # 或选择具体的提交hash" >> "$RESTORE_SCRIPT"
        fi
    fi
done

cat >> "$RESTORE_SCRIPT" << 'EOF'

# 4. 清理临时文件
echo ""
echo "4️⃣ 清理可能的临时文件..."
find . -name "*.tmp" -mtime -7 -delete 2>/dev/null
find . -name "*.cache" -mtime -7 -delete 2>/dev/null

# 5. 重新设置权限
echo ""
echo "5️⃣ 重新设置脚本权限..."
chmod +x run_maskllm_native.sh
chmod +x run_apptainer_extended_libs.sh
chmod +x llama8b_scripts/*.sh

echo ""
echo "✅ 环境复原完成"
echo "💡 建议测试容器环境: bash run_apptainer_extended_libs.sh"

EOF

chmod +x "$RESTORE_SCRIPT"

echo "### 推荐复原步骤:" >> "$REPORT_FILE"
echo "1. 备份当前环境状态" >> "$REPORT_FILE"
echo "2. 使用git回退到7天前的稳定版本" >> "$REPORT_FILE"
echo "3. 重新设置脚本权限" >> "$REPORT_FILE"
echo "4. 测试容器环境" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "详细复原脚本: $RESTORE_SCRIPT" >> "$REPORT_FILE"

echo "📋 诊断完成！"
echo "=================================="
echo "📁 诊断结果目录: $DIAGNOSIS_DIR"
echo "📄 详细报告: $REPORT_FILE"
echo "🔄 复原脚本: $RESTORE_SCRIPT"
echo ""
echo "💡 下一步建议:"
echo "1. 查看诊断报告: cat $REPORT_FILE"
echo "2. 检查最近修改的文件列表"
echo "3. 如确认需要复原，运行: bash $RESTORE_SCRIPT"
echo ""
echo "⚠️ 复原前请确保重要工作已备份！"
