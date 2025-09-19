# Llama8b WIKITEXT2 Perplexity Evaluation Guide

## 概述

`evaluate_llama8b_wikitext2.sh` 是专为Llama8b模型设计的WIKITEXT2数据集perplexity评测脚本。该脚本基于MaskLLM框架，支持密集和稀疏模型的性能评测。

## 功能特性

- ✅ **Llama8b专用配置** - 针对8B参数模型优化的评测参数
- ✅ **稀疏模型支持** - 支持MaskLLM训练的稀疏模型评测
- ✅ **多GPU并行** - 支持Tensor Parallel (TP=1,2,4,8)
- ✅ **自定义Tokenizer** - 使用Llama8b专用tokenizer
- ✅ **标准数据集** - WIKITEXT2标准perplexity评测

## 脚本使用方法

### 基本语法
```bash
bash evaluate_llama8b_wikitext2.sh <MODEL_PATH> <TP> <MODE>
```

### 参数说明

| 参数 | 说明 | 可选值 | 示例 |
|------|------|--------|------|
| `MODEL_PATH` | 模型checkpoint路径 | Megatron格式路径 | `output/checkpoints/llama8b_maskllm_training_tp8/` |
| `TP` | Tensor Parallel大小 | `1`, `2`, `4`, `8` | `1` |
| `MODE` | 评测模式 | `dense`, `sparse` | `sparse` |

### 模型路径要求

脚本支持以下格式的模型：

1. **Megatron格式** (推荐)
   ```
   output/checkpoints/llama8b_maskllm_training_tp8/
   ├── latest_checkpointed_iteration.txt
   ├── iter_0002000/
   │   ├── mp_rank_00/
   │   │   └── model_optim_rng.pt
   │   └── ...
   ```

2. **单文件TP=1格式**
   ```
   output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/
   ```

## 使用示例

### 示例1：稀疏模型评测 (TP=1)
```bash
# 评测转换后的TP=1稀疏模型
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000 \
    1 \
    sparse
```

### 示例2：原始训练模型评测 (TP=8)
```bash
# 评测原始训练checkpoint (需要8卡)
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp8 \
    8 \
    sparse
```

### 示例3：密集模式对比评测
```bash
# 以密集模式评测同一模型（对比基准）
bash llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000 \
    1 \
    dense
```

### 示例4：使用run_maskllm_native.sh
```bash
# 在MaskLLM环境中运行
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000 \
    1 \
    sparse
```

## 模型配置参数

脚本已针对Llama8b进行预配置：

```bash
HIDDEN_SIZE=4096        # 隐藏层大小
NUM_LAYERS=32          # 层数
NUM_ATTN_HEADS=32      # 注意力头数
FFN_HIDDEN_SIZE=14336  # FFN隐藏层大小
SEQ_LENGTH=4096        # 序列长度
```

## 输出解释

### 成功运行输出示例
```
🚀 Llama8b WIKITEXT2 Evaluation
Model Path: output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000
Tensor Parallel: 1
Mode: sparse

🔸 Running in SPARSE mode
📁 Project Directory: /data/home/zdhs0054/zzhou/MaskLLM
📋 Starting evaluation with the following configuration:
   - Hidden Size: 4096
   - Layers: 32
   - Attention Heads: 32
   - FFN Hidden Size: 14336
   - Sequence Length: 4096
   - Tensor Parallel: 1
   - Tokenizer: Llama8bTokenizer

🔥 Running torchrun with 1 processes...
...
 validation loss at iteration 1 | lm loss value: 2.234 | lm loss PPL: 9.34
...
✅ Llama8b WIKITEXT2 evaluation completed!
```

### 关键指标
- **lm loss value**: 语言模型损失值（越低越好）
- **lm loss PPL**: Perplexity值（越低越好）
- **处理速度**: tokens/sec

## 依赖要求

### 必需文件
1. **Llama8b Tokenizer Python模块**: `assets/checkpoints/Llama8b/llama8b/tokenizer.py`
2. **词汇表文件**: `assets/checkpoints/Llama8b/vocab.txt`  
3. **Megatron Tokenizer集成**: `megatron/tokenizer/llama8b_tokenizer.py` (已集成)
4. **WIKITEXT2数据集**: 自动下载或使用缓存

**注意**: 评测脚本会自动验证tokenizer Python模块和vocab文件是否存在。

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- Transformers库
- MaskLLM环境

## 故障排除

### 常见问题

#### 1. Tokenizer文件未找到
```
❌ Llama8b tokenizer Python module not found: assets/checkpoints/Llama8b/llama8b/tokenizer.py
❌ Llama8b vocab file not found: assets/checkpoints/Llama8b/vocab.txt
```
**解决**: 确保原始Llama8b模型文件在正确位置
- Python模块: `assets/checkpoints/Llama8b/llama8b/tokenizer.py`
- 词汇表: `assets/checkpoints/Llama8b/vocab.txt`

#### 2. 缺少latest_checkpointed_iteration.txt
```
WARNING: could not find the metadata file output/checkpoints/.../latest_checkpointed_iteration.txt
```
**解决**: 运行修复脚本
```bash
bash llama8b_scripts/fix_tp1_checkpoint_metadata.sh
```

#### 3. 模型路径错误
```
ERROR: Could not find model checkpoint
```
**解决**: 检查模型路径，确保包含正确的checkpoint文件

#### 3. GPU内存不足
```
CUDA out of memory
```
**解决**: 
- 减少batch size
- 使用更小的TP值
- 使用更少的GPU卡数

#### 4. TP配置不匹配
```
Tensor parallel size mismatch
```
**解决**: 确保TP参数与模型训练时的TP配置匹配

## 性能基准

### 期望的Perplexity范围
- **密集模型**: PPL < 10.0
- **2:4稀疏模型**: PPL < 12.0 (相比密集模型略有下降)
- **更高稀疏度**: PPL可能进一步增加

### 推荐配置
- **单卡评测**: TP=1，使用转换后的checkpoint
- **多卡评测**: TP=8，使用原始训练checkpoint
- **对比评测**: 同一模型分别用dense和sparse模式评测

## 集成到评测流水线

可以将此脚本集成到自动化评测流水线中：

```bash
#!/bin/bash
# 自动化Llama8b评测流水线

MODELS=(
    "output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000"
    "output/checkpoints/llama8b_baseline/iter_0002000"
)

MODES=("dense" "sparse")

for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "评测: $model ($mode)"
        bash llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
            "$model" 1 "$mode"
        echo "完成: $model ($mode)"
        echo "---"
    done
done
```

## 相关文档

- [Checkpoint转换指南](../training_notes/checkpoint_conversion.md)
- [推理测试指南](llama8b_inference_guide.md)
- [MaskLLM训练文档](../training_notes/training_logs.md)
