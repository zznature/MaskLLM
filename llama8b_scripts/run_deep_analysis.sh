#!/bin/bash
#
# 深入分析MaskLLM检查点文件结构
#

echo "=== 深入检查点分析脚本 ==="
echo "详细探查模型权重的嵌套结构"
echo

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 设置Python路径
export PYTHONPATH=/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH

# 运行深入分析脚本
echo "运行深入分析..."
python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/deep_checkpoint_analysis.py

echo
echo "深入分析完成！"
