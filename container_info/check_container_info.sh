#!/bin/bash

# 检查 pytorch_24.01-py3.sif 容器详细信息

echo "=== 检查 pytorch_24.01-py3.sif 容器信息 ==="

# 检查文件是否存在
if [ ! -f "pytorch_24.01-py3.sif" ]; then
    echo "错误: pytorch_24.01-py3.sif 文件不存在"
    exit 1
fi

echo ""
echo "1. 文件基本信息:"
ls -lh pytorch_24.01-py3.sif

echo ""
echo "2. 文件修改时间:"
stat pytorch_24.01-py3.sif

echo ""
echo "3. 容器标签信息:"
apptainer inspect --labels pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取标签信息"

echo ""
echo "4. 容器环境变量:"
apptainer inspect --environment pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取环境变量"

echo ""
echo "5. 容器运行脚本:"
apptainer inspect --runscript pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取运行脚本"

echo ""
echo "6. 容器启动脚本:"
apptainer inspect --startscript pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取启动脚本"

echo ""
echo "7. 容器测试脚本:"
apptainer inspect --test pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取测试脚本"

echo ""
echo "8. 容器定义文件:"
apptainer inspect --deffile pytorch_24.01-py3.sif 2>/dev/null || echo "无法获取定义文件"

echo ""
echo "9. 检查容器内的 Python 环境:"
apptainer exec pytorch_24.01-py3.sif python --version 2>/dev/null || echo "无法执行 Python"

echo ""
echo "10. 检查容器内的 PyTorch:"
apptainer exec pytorch_24.01-py3.sif python -c "import torch; print('PyTorch 版本:', torch.__version__)" 2>/dev/null || echo "PyTorch 不可用"

echo ""
echo "11. 检查容器内的 transformer_engine:"
apptainer exec pytorch_24.01-py3.sif python -c "import transformer_engine; print('transformer_engine 可用')" 2>/dev/null || echo "transformer_engine 不可用"

echo ""
echo "=== 容器信息检查完成 ===" 