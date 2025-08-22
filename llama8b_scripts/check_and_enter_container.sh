#!/bin/bash

echo "=== 检查Apptainer容器状态 ==="

# 检查是否有正在运行的apptainer进程
echo "1. 检查正在运行的apptainer进程:"
ps aux | grep apptainer | grep -v grep

echo ""
echo "2. 检查系统中所有apptainer相关进程:"
pgrep -af apptainer

echo ""
echo "3. 检查是否有活动的容器实例:"
apptainer instance list

echo ""
echo "4. 检查当前工作目录:"
pwd

echo ""
echo "5. 检查MaskLLM项目目录是否存在:"
ls -la /data/home/zdhs0054/zzhou/MaskLLM/

echo ""
echo "6. 如果发现运行中的容器实例，尝试进入容器:"
echo "   (如果看到实例名称，请手动运行: apptainer shell instance://实例名称)"

echo ""
echo "7. 如果没有运行的容器，启动新的容器:"
echo "   请运行以下命令启动容器:"
echo "   cd /data/home/zdhs0054/zzhou/MaskLLM"
echo "   apptainer shell --nv /path/to/your/container.sif"
echo "   或者使用之前的启动脚本: ./run_maskllm_native.sh"

echo ""
echo "=== 完成检查 ==="
