#!/bin/bash
# CUDA配置诊断脚本
# 系统性检查容器内CUDA环境和GPU可见性问题

set -euo pipefail

echo "🔍 CUDA配置诊断 - 容器环境GPU可见性检查"
echo "============================================================"
echo ""

echo "📋 1. 宿主机GPU检查 (nvidia-smi):"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    echo ""
    echo "GPU总数: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
else
    echo "❌ nvidia-smi 不可用"
fi
echo ""

echo "📋 2. 环境变量检查:"
echo "----------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo "CUDA_HOME: ${CUDA_HOME:-未设置}"
echo "CUDA_ROOT: ${CUDA_ROOT:-未设置}"
echo "CUDA_DEVICE_MAX_CONNECTIONS: ${CUDA_DEVICE_MAX_CONNECTIONS:-未设置}"
echo ""

echo "📋 3. CUDA工具链检查:"
echo "----------------------------------------"
if command -v nvcc &> /dev/null; then
    nvcc --version | head -4
else
    echo "❌ nvcc 不可用"
fi
echo ""

echo "📋 4. Python PyTorch CUDA检查:"
echo "----------------------------------------"
python3.10 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU设备数: {torch.cuda.device_count()}')
    print('GPU设备详情:')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}, 内存: {props.total_memory//1024//1024//1024}GB')
        print(f'    计算能力: {props.major}.{props.minor}')
else:
    print('❌ PyTorch无法访问CUDA')
"
echo ""

echo "📋 5. 容器CUDA库检查:"
echo "----------------------------------------"
echo "CUDA库路径:"
find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null | head -5 || echo "未找到CUDA库"
echo ""

echo "LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-未设置}" | tr ':' '\n' | grep -i cuda || echo "LD_LIBRARY_PATH中无CUDA路径"
echo ""

echo "📋 6. 容器GPU设备文件检查:"
echo "----------------------------------------"
echo "GPU设备文件:"
ls -la /dev/nvidia* 2>/dev/null || echo "❌ 未找到GPU设备文件"
echo ""

echo "nvidia-ml库:"
ls -la /usr/lib/x86_64-linux-gnu/libnvidia-ml* 2>/dev/null || echo "未找到nvidia-ml库"
echo ""

echo "📋 7. 分布式通信检查:"
echo "----------------------------------------"
python3.10 -c "
try:
    import torch.distributed as dist
    print('torch.distributed可用')
    
    # 检查NCCL
    if torch.cuda.is_available():
        print(f'NCCL版本: {torch.cuda.nccl.version()}')
        
        # 检查每个GPU的可用性
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                tensor = torch.cuda.FloatTensor([1.0])
                print(f'GPU {i}: 可用 ✓')
            except Exception as e:
                print(f'GPU {i}: 错误 - {e}')
    else:
        print('❌ CUDA不可用，无法检查NCCL')
        
except Exception as e:
    print(f'分布式检查失败: {e}')
"
echo ""

echo "📋 8. 容器运行时检查:"
echo "----------------------------------------"
echo "容器运行时:"
if [ -f /.dockerenv ]; then
    echo "Docker容器环境"
elif command -v apptainer &> /dev/null; then
    echo "Apptainer容器环境"
    echo "APPTAINER版本: $(apptainer --version)"
elif command -v singularity &> /dev/null; then
    echo "Singularity容器环境"
    echo "SINGULARITY版本: $(singularity --version)"
else
    echo "原生环境或未知容器"
fi
echo ""

echo "📋 9. GPU拓扑检查:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU拓扑:"
    nvidia-smi topo -m || echo "无法获取GPU拓扑"
else
    echo "❌ 无法检查GPU拓扑"
fi
echo ""

echo "📋 10. 进程GPU占用检查:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "当前GPU进程:"
    nvidia-smi pmon -c 1 || nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv || echo "无活跃GPU进程"
else
    echo "❌ 无法检查GPU进程"
fi
echo ""

echo "============================================================"
echo "🎯 诊断总结建议"
echo "============================================================"

# 检查关键问题
GPU_COUNT=$(python3.10 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
CUDA_AVAILABLE=$(python3.10 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

echo "当前状态:"
echo "  CUDA可用: $CUDA_AVAILABLE"
echo "  GPU数量: $GPU_COUNT"
echo ""

if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo "❌ 主要问题: CUDA不可用"
    echo "🔧 建议:"
    echo "  1. 检查容器是否正确启动GPU支持"
    echo "  2. 验证NVIDIA驱动安装"
    echo "  3. 检查容器运行时配置"
elif [ "$GPU_COUNT" -lt 8 ]; then
    echo "⚠️ 主要问题: GPU数量不足($GPU_COUNT/8)"
    echo "🔧 建议:"
    echo "  1. 检查CUDA_VISIBLE_DEVICES设置"
    echo "  2. 验证容器GPU绑定配置"
    echo "  3. 检查物理GPU可用性"
else
    echo "✅ CUDA配置正常"
    echo "💡 可能的分布式问题:"
    echo "  1. NCCL配置问题"
    echo "  2. GPU设备映射冲突"
    echo "  3. 网络配置问题"
fi
echo ""

echo "🚀 下一步操作建议已生成，请查看诊断结果"
