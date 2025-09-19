#!/bin/bash

# Llama8b学习率checkpoint冲突修复脚本
# 问题: checkpoint中保存了旧的学习率配置(5e-5)，与新脚本配置(2e-5)冲突

echo "🔧 Llama8b学习率checkpoint冲突修复"
echo "=================================="

PROJECT_DIR=$(pwd)
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
BACKUP_DIR="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_backup_$(date +%Y%m%d_%H%M%S)"

echo "📊 当前状态分析:"
echo "  - 当前checkpoint: $TRAINING_CHECKPOINT"
echo "  - 最新迭代: $(cat $TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt 2>/dev/null || echo 'N/A')"
echo "  - 脚本配置: --lr 2e-5 --min-lr 2e-6"
echo "  - 实际训练: --lr 5e-5 --min-lr 5e-6 (从日志推断)"
echo ""

echo "🎯 修复方案选择:"
echo "方案A: 强制使用新学习率 (推荐)"
echo "  - 优点: 使用更保守的学习率(2e-5)，降低数值不稳定风险"
echo "  - 缺点: 需要重新初始化optimizer，可能影响训练连续性"
echo "  - 实现: 添加 --no-load-optim --finetune 参数"
echo ""

echo "方案B: 保持checkpoint学习率"
echo "  - 优点: 保持训练连续性"
echo "  - 缺点: 继续使用高学习率(5e-5)，可能再次遇到数值不稳定"
echo "  - 实现: 修改脚本配置匹配checkpoint"
echo ""

echo "方案C: 清理checkpoint重新开始"
echo "  - 优点: 完全使用新配置，最彻底的解决方案"
echo "  - 缺点: 丢失已有训练进度(1600步)"
echo "  - 实现: 备份并删除training checkpoint"
echo ""

read -p "请选择修复方案 (A/B/C): " choice

case $choice in
    [Aa])
        echo "✅ 选择方案A: 强制使用新学习率"
        echo ""
        echo "🔧 实施步骤:"
        echo "1. 备份现有checkpoint"
        echo "2. 修改训练脚本添加强制参数"
        echo "3. 训练将从iter 1600恢复，但使用新的学习率配置"
        
        # 备份checkpoint
        echo "📁 备份checkpoint到: $BACKUP_DIR"
        cp -r "$TRAINING_CHECKPOINT" "$BACKUP_DIR"
        
        # 修改训练脚本
        echo "⚙️  修改训练脚本..."
        sed -i 's/EXTRA_CMD="$EXTRA_CMD --no-load-optim --no-load-rng --finetune --enable-partial-load"/EXTRA_CMD="$EXTRA_CMD --no-load-optim --no-load-rng --finetune --enable-partial-load --override-opt_param-scheduler"/' llama8b_scripts/llama8b_presparse_training_tp8.sh
        
        echo "✅ 修复完成！"
        echo "📝 下次训练将:"
        echo "  - 从iter 1600恢复模型权重"
        echo "  - 重新初始化optimizer和学习率调度器"
        echo "  - 使用新的学习率配置: 2e-5 → 2e-6"
        ;;
        
    [Bb])
        echo "✅ 选择方案B: 保持checkpoint学习率"
        echo ""
        echo "🔧 实施步骤:"
        echo "1. 修改脚本配置匹配checkpoint"
        echo "2. 保持训练连续性"
        
        # 修改脚本配置
        echo "⚙️  修改脚本学习率配置..."
        sed -i 's/--lr 2e-5/--lr 5e-5/' llama8b_scripts/llama8b_presparse_training_tp8.sh
        sed -i 's/--min-lr 2e-6/--min-lr 5e-6/' llama8b_scripts/llama8b_presparse_training_tp8.sh
        sed -i 's/echo "  - 学习率: 2e-5 → 2e-6"/echo "  - 学习率: 5e-5 → 5e-6"/' llama8b_scripts/llama8b_presparse_training_tp8.sh
        
        echo "✅ 修复完成！"
        echo "📝 脚本现在匹配checkpoint配置"
        echo "⚠️  注意: 仍使用较高学习率，需要监控数值稳定性"
        ;;
        
    [Cc])
        echo "✅ 选择方案C: 清理checkpoint重新开始"
        echo ""
        echo "🔧 实施步骤:"
        echo "1. 备份现有checkpoint"
        echo "2. 清理training checkpoint"
        echo "3. 训练将从预稀疏checkpoint重新开始"
        
        # 备份checkpoint
        echo "📁 备份checkpoint到: $BACKUP_DIR"
        cp -r "$TRAINING_CHECKPOINT" "$BACKUP_DIR"
        
        # 清理training checkpoint (保留wandb_id)
        echo "🧹 清理training checkpoint..."
        find "$TRAINING_CHECKPOINT" -name "iter_*" -type d -exec rm -rf {} \; 2>/dev/null || true
        rm -f "$TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt"
        
        echo "✅ 修复完成！"
        echo "📝 下次训练将:"
        echo "  - 从预稀疏checkpoint重新开始"
        echo "  - 使用新的学习率配置: 2e-5 → 2e-6"
        echo "  - 重新训练1600步的进度"
        ;;
        
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "🎉 修复完成！现在可以运行训练:"
echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh"
