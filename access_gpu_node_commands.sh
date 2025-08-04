#!/bin/bash

echo "=== 进入 hd02-gpu1-0017 节点的方法 ==="

echo "方法 1: 通过现有作业进入节点"
echo "您的作业 JOBID=4880 正在 hd02-gpu1-0017 上运行"
echo ""

echo "尝试进入作业环境:"
echo "srun --jobid=4880 --pty bash -i"
echo ""

echo "如果上面失败，尝试:"
echo "srun --jobid=4880 --overlap --pty bash -i"
echo ""

echo "方法 2: 检查作业状态和节点信息"
echo "squeue -j 4880 -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
echo ""

echo "方法 3: 直接分配到该节点 (如果节点空闲)"
echo "srun --nodelist=hd02-gpu1-0017 --pty bash -i"
echo ""

echo "方法 4: 查看节点详细信息"
echo "scontrol show node hd02-gpu1-0017"
echo ""

echo "方法 5: 如果需要新的作业会话"
echo "sbatch --nodelist=hd02-gpu1-0017 --wrap='sleep 3600'"
echo "# 然后使用新的 JOBID 进入"
echo ""

echo "=== 当前推荐操作 ==="
echo "1. 先尝试方法1中的 srun 命令"
echo "2. 如果失败，检查作业是否还在运行"
echo "3. 考虑在原有容器环境中直接操作" 