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

echo '🔄 复原 run_maskllm_native.sh...'
git checkout HEAD~1 -- run_maskllm_native.sh  # 或选择具体的提交hash

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

