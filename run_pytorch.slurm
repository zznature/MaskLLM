#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --partition=project4-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=pytorch_%j.out
#SBATCH --error=pytorch_%j.err

# 加载必要的模块
module load apptainer
module load cuda/12.4

# 设置工作目录
cd $SLURM_SUBMIT_DIR

# 设置 NVIDIA 相关环境变量
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行 apptainer 容器
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    pytorch_24.01-py3.sif

#   4661 project4- run_sbat zdhs0054  R 24-09:13:29      1 hd02-gpu1-0056
#   4660 project4- run_sbat zdhs0054  R 24-09:13:32      1 hd02-gpu1-0024 