#!/bin/bash

echo "=== 进入正在运行的容器 ==="

# 检查运行的apptainer进程
echo "1. 发现正在运行的容器进程:"
ps aux | grep "run_apptainer_extended_libs.sh" | grep -v grep

echo ""
echo "2. 尝试通过多种方式进入容器:"

# 方法1: 直接运行run_maskllm_native.sh进入容器
echo "方法1: 使用run_maskllm_native.sh进入容器"
if [ -f "run_maskllm_native.sh" ]; then
    echo "找到run_maskllm_native.sh，执行: bash run_maskllm_native.sh"
    bash run_maskllm_native.sh
else
    echo "未找到run_maskllm_native.sh文件"
fi

echo ""
echo "如果上述方法失败，请手动尝试以下命令:"
echo "1. bash run_maskllm_native.sh"
echo "2. 或者查找容器启动脚本: ls -la *apptainer* *container*"
echo "3. 或者终止当前容器进程并重新启动"
