#!/bin/bash
set -e

echo "=== 同步代码到 zznature GitHub 账号 ==="

# 配置 Git 用户信息为 zznature
echo "配置 Git 用户信息..."
git config user.name "Zhou Zhang"
git config user.email "zhouzhang92@gmail.com"  # 使用 GitHub 提供的 noreply 邮箱

# 查看当前配置
echo "当前 Git 配置:"
git config user.name
git config user.email

# 查看当前状态
echo ""
echo "当前 Git 状态:"
git status --short

# 添加所有有用的更改
echo ""
echo "添加更改到暂存区..."

# 添加新的脚本和配置文件
git add WIKITEXT103_TRAINING_OPTIMIZATIONS.md
git add scripts/data/prepare_wikitext103_megatron_llama2.sh
git add scripts/incremental/llama2_7b_mask_only_wikitext103_tp4.sh

# 添加其他有用的脚本
git add prepare_wikitext_c4_format_final.sh
git add run_incremental_training_wikitext103.sh

# 添加修改过的文件
git add .gitignore
git add Project_Tasks.md

# 提交更改
echo ""
echo "提交更改..."
git commit -m "feat: Add WikiText-103 C4-format preprocessing and optimized training

- Add prepare_wikitext103_megatron_llama2.sh: C4-format preprocessing for WikiText-103
- Update llama2_7b_mask_only_wikitext103_tp4.sh: Optimized for WikiText-103 training
  * Updated dataset path to C4-format processed data
  * Optimized SEQ_LENGTH: 4096 → 2048 (better for WikiText-103)
  * Adjusted GLOBAL_BATCH_SIZE: 128 → 64 (prevent overfitting)
  * Increased TRAIN_ITERS: 400 → 800 (more epochs for smaller dataset)
  * Updated intervals for better monitoring
- Add WIKITEXT103_TRAINING_OPTIMIZATIONS.md: Detailed optimization guide
- Add helper scripts for dataset preparation and training execution
- Fix dataset indexing issues with proper C4-format preprocessing

This resolves the ValueError: offset must be non-negative error and enables
proper WikiText-103 incremental training for MaskLLM sparsity learning."

# 推送到 personal remote (zznature)
echo ""
echo "推送到 GitHub (zznature)..."
git push personal H100-S1

echo ""
echo "=== 同步完成 ==="
echo "✅ 代码已成功推送到 https://github.com/zznature/MaskLLM.git"
echo "✅ 分支: H100-S1"
echo ""
echo "主要更新内容:"
echo "- WikiText-103 C4格式预处理脚本"
echo "- 优化的 WikiText-103 训练配置"
echo "- 修复数据集索引错误"
echo "- 详细的优化说明文档" 