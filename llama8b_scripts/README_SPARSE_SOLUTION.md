# Llama8b稀疏化解决方案

## 快速开始 🚀

### 一键式完整工作流程

```bash
# 完整工作流程 (推荐)
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh

# 指定稀疏化方法
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh SparseGPT
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh Magnitude
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh Wanda

# 部分执行模式
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh SparseGPT 1  # 仅稀疏化
bash run_maskllm_native.sh llama8b_scripts/run_complete_sparse_workflow.sh SparseGPT 2  # 仅训练
```

## 分步执行 📋

### 1. 稀疏化准备 (Phase 3.5.1)

```bash
# 一次性剪枝，生成50%稀疏模型
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh [SparseGPT|Magnitude|Wanda]
```

**输出**: `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.*`

### 2. 稀疏化训练 (Phase 3.5.2)

```bash
# 从预稀疏checkpoint开始训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0

# 从训练checkpoint恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

**输出**: `output/checkpoints/llama8b_maskllm_training_tp8/`

## 脚本说明 📝

| 脚本名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `run_llama8b_prune_tp8.sh` | 一次性稀疏化剪枝 | Megatron格式模型 | 50%稀疏模型 |
| `llama8b_presparse_training_tp8.sh` | 稀疏化训练 | 预稀疏模型 | 训练后稀疏模型 |
| `run_complete_sparse_workflow.sh` | 完整工作流程 | 原始配置 | 完整稀疏化流程 |

## 配置参数 ⚙️

### 稀疏化配置
- **稀疏率**: 50% (2:4结构化)
- **并行度**: 8 GPU (tensor parallel)
- **Hessian样本**: 128个
- **剪枝方法**: SparseGPT (推荐), Magnitude, Wanda

### 训练配置
- **迭代次数**: 10,000
- **批大小**: 16 (global), 2 (micro)
- **学习率**: 1e-5 (模型), 0.1 (mask)
- **保存间隔**: 1,000迭代

## 前置条件 ✅

确保以下步骤已完成：

1. **模型转换** (Phase 3.1)
   ```bash
   bash scripts/tools/convert_llama8b_hf_to_megatron_simple.sh 8
   ```

2. **数据预处理** (Phase 3.4)  
   ```bash
   bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh
   ```

3. **环境配置**
   ```bash
   source container_environment_setup.sh
   ```

## 性能预期 📊

### 稀疏化准备
- **时间**: 30-60分钟
- **GPU内存**: ~20GB per GPU
- **稀疏率**: 准确50% (2:4结构化)

### 稀疏化训练  
- **时间**: 6-12小时 (10K迭代)
- **收敛性**: 预期在8K迭代左右收敛
- **性能保持**: >90% (相对于稠密模型)

## 故障排除 🔧

### 常见问题

1. **内存不足**
   - 调整 `--micro-batch-size` 为 1
   - 启用 `--checkpoint-activations`

2. **收敛困难**
   - 降低mask学习率: `--mask-learning-rate 0.05`
   - 增加warmup: `--lr-warmup-fraction 0.02`

3. **稀疏结构不稳定**
   - 调整mask衰减: `--mask-decay-rate 0.99`
   - 增加更新频率: `--mask-update-freq 50`

## 文档参考 📚

- **完整指南**: `PHASE3_5_SPARSE_WORKFLOW_GUIDE.md`
- **进度记录**: `PHASE3_PROGRESS_SUMMARY.md` 
- **项目文档**: `../Project_SelfDefinedModels.md`

## 版本信息 ℹ️

- **创建时间**: 2024年12月
- **版本**: Phase 3.5
- **状态**: ✅ 生产就绪
- **测试状态**: ✅ 环境兼容性验证通过

---

**重要**: 执行前请确保已进入Apptainer容器并正确配置环境变量！
