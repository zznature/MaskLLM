# 容器外运行指南

## 问题分析
NGC容器的PyTorch与HPC-X深度集成，导致符号依赖问题难以根本解决。

## 解决方案
在容器外的标准Linux环境中运行MaskLLM。

## 环境准备步骤

### 1. 安装标准PyTorch
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安装CUDA工具链
```bash
# 确保CUDA 11.8+可用
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. 安装MaskLLM依赖
```bash
pip install transformers datasets tokenizers
pip install flash-attn --no-build-isolation
```

### 4. 配置环境
```bash
export PYTHONPATH=/path/to/MaskLLM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 优势
- 完全避开HPC-X符号问题
- 环境更纯净，问题更少
- 更容易调试和维护

## 劣势
- 需要重新配置环境
- 可能缺少某些NGC容器的优化
