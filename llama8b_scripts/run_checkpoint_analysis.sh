#!/bin/bash
#
# 分析MaskLLM检查点文件的数据类型和大小
#

echo "=== MaskLLM Checkpoint Analysis Script ==="
echo "分析检查点文件的数据类型和大小信息"
echo

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 设置Python路径
export PYTHONPATH=/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH

# 运行分析脚本
echo "运行检查点分析..."
python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/analyze_checkpoint_info.py

echo
echo "分析完成！"
