#!/bin/bash

echo "=== 检查和进入 Apptainer 容器 ==="

echo "1. 检查当前节点上的 Apptainer 进程..."
echo "执行命令: ps aux | grep apptainer"
ps aux | grep apptainer | grep -v grep

echo ""
echo "2. 检查容器相关进程..."
echo "执行命令: ps aux | grep pytorch"
ps aux | grep pytorch | grep -v grep

echo ""
echo "3. 检查 SLURM 作业进程..."
echo "执行命令: ps aux | grep slurmstepd"
ps aux | grep slurmstepd | grep -v grep

echo ""
echo "4. 检查当前用户的所有进程..."
echo "执行命令: ps -u $(whoami) -o pid,ppid,cmd"
ps -u $(whoami) -o pid,ppid,cmd

echo ""
echo "=== 进入容器的方法 ==="

echo "方法 1: 如果容器还在运行，查找容器 PID 并进入"
CONTAINER_PID=$(ps aux | grep "apptainer run" | grep pytorch | grep -v grep | awk '{print $2}' | head -1)

if [ ! -z "$CONTAINER_PID" ]; then
    echo "找到容器进程 PID: $CONTAINER_PID"
    echo "尝试进入容器命名空间:"
    echo "sudo nsenter -t $CONTAINER_PID -p -m -u -i -n /bin/bash"
    echo ""
    echo "或者使用 apptainer shell:"
    echo "apptainer shell instance://pytorch_instance"
else
    echo "未找到运行中的容器进程"
fi

echo ""
echo "方法 2: 重新启动容器 (推荐)"
echo "cd /data/home/zdhs0054/zzhou/MaskLLM"
echo "./run_apptainer_extended_libs.sh"

echo ""
echo "方法 3: 检查是否有命名实例在运行"
echo "apptainer instance list"

echo ""
echo "方法 4: 如果有命名实例，直接进入"
echo "apptainer shell instance://实例名"

echo ""
echo "=== 自动检测和操作 ==="

# 检查是否在 MaskLLM 目录
if [ -f "run_apptainer_extended_libs.sh" ]; then
    echo "✅ 检测到在 MaskLLM 项目目录中"
    
    # 检查是否有运行中的容器
    if [ ! -z "$CONTAINER_PID" ]; then
        echo "✅ 检测到运行中的容器 (PID: $CONTAINER_PID)"
        echo ""
        echo "推荐操作:"
        echo "1. 尝试进入现有容器: nsenter -t $CONTAINER_PID -p -m -u -i -n /bin/bash"
        echo "2. 或重新启动容器: ./run_apptainer_extended_libs.sh"
    else
        echo "❌ 未检测到运行中的容器"
        echo ""
        echo "推荐操作:"
        echo "重新启动容器: ./run_apptainer_extended_libs.sh"
    fi
else
    echo "❌ 请先导航到 MaskLLM 目录:"
    echo "cd /data/home/zdhs0054/zzhou/MaskLLM"
fi 