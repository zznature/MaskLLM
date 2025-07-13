#!/bin/bash
#SBATCH --job-name=pytorch_gpu
#SBATCH --output=pytorch_gpu_%j.out
#SBATCH --error=pytorch_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=project1

# 加载必要的模块
module load apptainer
module load cuda/12.4

# 设置工作目录
cd $SLURM_SUBMIT_DIR

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置系统限制
ulimit -l unlimited
ulimit -s 67108864

# 运行 Apptainer 容器
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    ../pytorch_24.01-py3.sif 