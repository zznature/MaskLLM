# Llama8b Model Conversion and HuggingFace Export

This document outlines the complete process for converting trained Llama8b MaskLLM checkpoints to HuggingFace format for deployment.

## Overview

The conversion pipeline follows these stages:
```
Trained Checkpoint (TP=8, with .mask) 
    ↓ [Stage 1: Trim learnable sparsity]
Release Checkpoint (TP=8, winner masks only)
    ↓ [Stage 2: Apply sparsity] 
Sparse Checkpoint (TP=8, 0s for masked weights)
    ↓ [Stage 3: Convert TP=8 to TP=1]
Single Device Checkpoint (TP=1, sparse weights)
    ↓ [Stage 4: Export to HuggingFace]
HuggingFace Format (ready for deployment)
```

## Llama8b Adaptation

```bash
# Stage 1: 提取获胜掩码(跳过，直接从原始训练checkpoint开始)
bash run_maskllm_native.sh python llama8b_scripts/tool_trim_learnable_sparsity_llama8b.py --ckpt_dir output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000

# Stage 2: 应用稀疏性  
bash run_maskllm_native.sh llama8b_scripts/tool_create_sparse_llama8b.py --ckpt_dir output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000

# Stage 3: TP转换
bash run_maskllm_native.sh llama8b_scripts/convert_llama8b_tp8_to_tp1.sh

# Stage 4: HF导出
bash run_maskllm_native.sh llama8b_scripts/export_llama8b_to_hf.sh
```

### Key Differences for Llama8b:
- Checkpoint directory: `output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000`
- All weights converted to `torch_dtype: float16` for memory efficiency
- Custom tokenizer handling (tokenizer.py + vocab.txt)

### Stage 1: Trim Learnable Sparsity (跳过，直接从原始训练checkpoint开始)
Extract winner masks with highest probability for inference:
```bash
# Create release checkpoint
bash run_maskllm_native.sh python llama8b_scripts/tool_trim_learnable_sparsity_llama8b.py --ckpt_dir output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000
```
**Output**: `output/checkpoints/llama8b_maskllm_training_tp8/release/`

### Stage 2: Apply Sparsity to Weights
Apply winner masks to actual weights (creates 0s for masked parameters):
```bash
bash run_maskllm_native.sh llama8b_scripts/tool_create_sparse_llama8b.py --ckpt_dir output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000
```
**Input**: `iter_0002000/` (原始训练checkpoint，包含完整的 `diff_mask.gate` 结构)  
**Output**: `iter_0000001/` (稀疏权重，0值代表被掩码的参数)

### Stage 3: Convert TP=8 to TP=1
Convert from tensor parallel (8 GPUs) to single device format:
```bash
bash run_maskllm_native.sh llama8b_scripts/convert_llama8b_tp8_to_tp1.sh
```
**Input**: `output/checkpoints/llama8b_maskllm_training_tp8/` (包含 `latest_checkpointed_iteration.txt` 的目录)  
**Output**: `output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/` (33GB单文件)

### Stage 4: Export to HuggingFace Format
Convert Megatron checkpoint to HF format with tokenizer files:
```bash
bash run_maskllm_native.sh llama8b_scripts/export_llama8b_to_hf.sh
```
**Output**: `output/checkpoints/llama8b_hf_maskllm_c4/` (33GB, 8个safetensors分片文件)

## Complete Pipeline (Recommended)

Run the entire conversion pipeline with a single command:
```bash
bash run_maskllm_native.sh llama8b_scripts/complete_hf_export_pipeline.sh
```

This script will:
1. ✅ Check and create release checkpoint if needed
2. ✅ Apply sparsity to create sparse weights
3. ✅ Convert from TP=8 to TP=1
4. ✅ Export to HuggingFace format
5. ✅ Evaluate the exported model
6. ✅ Generate precomputed mask files (optional)

## Verification and Testing

### Evaluate HF Model
```bash
bash run_maskllm_native.sh python eval_llama_ppl.py --model output/checkpoints/llama8b_hf_maskllm_c4/
```

### Generate Precomputed Masks (Optional)
For deployment with dynamic mask loading:
```bash
# Compute mask file
bash run_maskllm_native.sh python tool_compute_mask_hf.py \
    --dense assets/checkpoints/Llama8b \
    --sparse output/checkpoints/llama8b_hf_maskllm_c4 \
    --save output/checkpoints/llama8b_hf_maskllm_c4_mask.pt

# Compress mask file
bash run_maskllm_native.sh python tool_compress_mask.py \
    --mask_ckpt output/checkpoints/llama8b_hf_maskllm_c4_mask.pt \
    --output output/precomputed_masks/llama8b_hf_maskllm_c4_mask_compressed.npz
```

## Files Generated

| Stage | Output Location | Description |
|-------|------------------|-------------|
| 1 | `output/checkpoints/llama8b_maskllm_training_tp8/release/` | Winner masks only, TP=8 |
| 2 | `output/checkpoints/llama8b_maskllm_training_tp8/iter_0000001/` | Sparse weights, TP=8 |
| 3 | `output/checkpoints/llama8b_maskllm_training_tp1/` | Sparse weights, TP=1 |
| 4 | `output/checkpoints/llama8b_hf_maskllm_c4/` | HF format model |
| Optional | `output/precomputed_masks/` | Compressed mask files |

## Technical Notes

### Float16 Optimization
All Llama8b weights are converted to float16 to:
- Reduce memory footprint (50% smaller)
- Maintain inference speed
- Ensure compatibility with deployment frameworks

### Sparsity Pattern
The conversion preserves 2:4 structured sparsity patterns:
- Every group of 4 consecutive elements has exactly 2 zeros
- Hardware acceleration compatible (NVIDIA Ampere+)
- Maintains model quality while reducing compute

### Error Handling
Each stage includes validation checks:
- Checkpoint directory existence
- File integrity verification
- Conversion success validation
- Output format compliance

## Inference Testing

After Stage 4 completion, test the exported HuggingFace sparse model:

### Automated Inference Tests
```bash
# Run comprehensive inference tests
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/run_inference_test.sh
```

### Fixed Inference (推荐)
```bash
# 自动修复tokenizer配置并运行完整测试
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/run_llama8b_inference_fixed.sh

# 快速修复和测试
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/quick_tokenizer_test.sh
```

### Manual Steps
如果需要手动修复：
```bash
# Step 1: 修复tokenizer配置
bash run_maskllm_native.sh llama8b_scripts/fix_tokenizer_config.py output/checkpoints/llama8b_hf_maskllm_c4

# Step 2: 运行推理测试
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/test_llama8b_sparse_inference.py \
    --model output/checkpoints/llama8b_hf_maskllm_c4 \
    --prompt "Your prompt here" \
    --max-new-tokens 256 \
    --use-gpu
```

### Interactive Inference
```bash
# Quick single prompt inference (需要先修复tokenizer)
bash llama8b_scripts/llama8b_inference/llama8b_inference_simple.sh "What is MaskLLM?"
```

### Expected Performance
- **模型大小**: 33GB (8个safetensors分片)
- **内存优化**: float16权重 (50%内存节省)
- **计算优化**: 2:4结构化稀疏 (硬件加速兼容)
- **推理速度**: 相比密集模型显著提升

## Model Evaluation

### WIKITEXT2 Perplexity Evaluation

After successful conversion, evaluate model quality using the dedicated evaluation script:

```bash
# 评测TP=1稀疏模型
bash llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000 \
    1 \
    sparse

# 评测原始TP=8模型 (需要8卡)  
bash llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp8 \
    8 \
    sparse

# 密集模式对比评测
bash llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000 \
    1 \
    dense
```

### Expected Evaluation Results
- **稀疏模型PPL**: < 12.0 (WIKITEXT2)
- **密集模型PPL**: < 10.0 (对比基准)
- **性能保持率**: > 85% (相比密集模型)

详细使用说明请参考: [llama8b_eval.md](llama8b_inference/llama8b_eval.md)

## Generated Scripts

The following scripts have been created:

**Conversion Scripts** (`llama8b_scripts/`):
- `tool_trim_learnable_sparsity_llama8b.py` - Stage 1 implementation
- `tool_create_sparse_llama8b.py` - Stage 2 implementation  
- `convert_llama8b_tp8_to_tp1.sh` - Stage 3 wrapper
- `export_llama8b_to_hf.sh` - Stage 4 wrapper
- `complete_hf_export_pipeline.sh` - Full pipeline automation
- `fix_tokenizer_config.py` - 修复HF模型tokenizer配置

**Inference Scripts** (`llama8b_scripts/llama8b_inference/`):
- `test_llama8b_sparse_inference.py` - HF格式推理测试
- `run_inference_test.sh` - 自动化推理测试套件
- `llama8b_inference_simple.sh` - 快速交互式推理
- `run_llama8b_inference_fixed.sh` - 集成修复的推理测试
- `quick_tokenizer_test.sh` - 快速tokenizer修复测试
- `test_tokenizer_only.py` - 独立tokenizer测试
- `evaluate_llama8b_wikitext2.sh` - WIKITEXT2 perplexity评测脚本
- `llama8b_eval.md` - 评测脚本使用说明

## 问题解决记录

### 🔧 Tokenizer兼容性问题
**问题**: HF导出的模型使用`tokenizer_class: "Llama8bTokenizer"`，但transformers无法找到此类。
**错误**: `Tokenizer class Llama8bTokenizer does not exist or is not currently imported.`

**根本原因**: 
- 导出过程将自定义tokenizer类名写入配置
- HuggingFace AutoTokenizer无法自动导入非标准tokenizer类
- 需要将配置修复为标准的LlamaTokenizer

**解决方案**:
1. **自动修复脚本**: `fix_tokenizer_config.py` - 将tokenizer_class修改为LlamaTokenizer
2. **集成测试脚本**: `run_llama8b_inference_fixed.sh` - 自动修复+推理测试
3. **兼容性验证**: 确保词汇表大小(119,696)和特殊token正确保持

**解决方案更新**:
4. **使用自定义Llama8bTokenizer**: 修改推理脚本直接使用`assets/checkpoints/Llama8b/llama8b/tokenizer.py`中的原始tokenizer
5. **多方法兼容**: 实现多种tokenizer加载和使用方法，确保向后兼容
6. **编码解码适配**: 适配自定义tokenizer的encode/decode接口

**验证状态**: ✅ 已完成代码修复，待测试验证

### 🔧 评测阶段错误修复 (Stage 5)
**问题1**: `latest_checkpointed_iteration.txt` 文件缺失
**错误**: `WARNING: could not find the metadata file ...latest_checkpointed_iteration.txt`
**原因**: TP转换后未创建Megatron所需的元数据文件
**解决**: 
- 创建 `fix_tp1_checkpoint_metadata.sh` 修复脚本
- 自动创建包含迭代数(2000)的元数据文件

**问题2**: 重复添加sparse masks
**错误**: `Adding sparse masks for module language_model.encoder.layers.xx`
**原因**: 
- 评测脚本使用 `--enable-sparsity` 参数
- 但Stage 2后权重已应用稀疏性，不需要重新添加掩码
**解决**:
- 修改评测脚本逻辑，对已应用稀疏性的checkpoint使用dense模式加载
- 添加说明注释：稀疏性已"烧录"到权重中

**问题3**: torchrun路径错误  
**错误**: `torchrun: command not found`
**原因**: 容器环境中torchrun位于非标准路径
**解决**: 修正为 `/usr/local/bin/torchrun` 路径，添加fallback机制

**验证状态**: ✅ 所有修复已完成并验证