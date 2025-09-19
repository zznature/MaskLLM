# Llama8b稀疏化错误分析总结

## 错误1: GPU重复分配错误 ❌ (已解决)

### 问题描述
```
Duplicate GPU detected : rank 0 and rank 4 both on CUDA device 18000
NCCL error: ncclInvalidUsage
```

### 根本原因
- **环境问题**: 在宿主机而非Apptainer容器内运行
- **GPU不可见**: `nvidia-smi不可用`，`No module named 'torch'`

### 解决方案 ✅
```bash
# 正确流程
bash run_apptainer_extended_libs.sh          # 进入容器
source container_environment_setup.sh        # 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 设置GPU可见性
```

---

## 错误2: 模型配置不匹配错误 ❌ (已修复)

### 问题描述
```
size mismatch for layers.X.mlp.dense_h_to_4h.weight: 
copying a param with shape torch.Size([3584, 4096]) from checkpoint, 
the shape in current model is torch.Size([2752, 4096])
```

### 根本原因
- **FFN配置错误**: 脚本使用`--ffn-hidden-size 11008` (Llama2配置)
- **实际Llama8b**: `intermediate_size = 14336`
- **维度计算错误**: 
  - TP8分割: `14336 / 8 = 1792` (不是1376)
  - TP4分割: `14336 / 4 = 3584` (不是2752)

### 配置对比

| 参数 | Llama2-7B | Llama8b | 脚本原配置 | 修复后 |
|------|-----------|---------|-----------|--------|
| `intermediate_size` | 11008 | **14336** | 11008 ❌ | 14336 ✅ |
| TP8分割 | 1376 | **1792** | 1376 ❌ | 1792 ✅ |
| TP4分割 | 2752 | **3584** | 2752 ❌ | 3584 ✅ |

### 解决方案 ✅

已修复以下文件：
- `llama8b_scripts/run_llama8b_prune_tp8.sh`: `--ffn-hidden-size 14336`
- `llama8b_scripts/llama8b_presparse_training_tp8.sh`: `--ffn-hidden-size 14336`

---

## Llama8b完整配置参数 📋

从`assets/checkpoints/Llama8b/config.json`提取的真实配置：

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336,    # 关键修复点
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "vocab_size": 119696
}
```

### Tensor Parallel分割计算

```bash
# TP8 (8 GPU)
FFN_SIZE_TP8 = 14336 / 8 = 1792

# TP4 (4 GPU) 
FFN_SIZE_TP4 = 14336 / 4 = 3584
```

---

## 修复验证 ✅

### 1. 环境检查
```bash
# 在容器内验证
nvidia-smi                    # 应显示8个GPU
python3 -c "import torch; print(torch.cuda.device_count())"  # 应输出8
```

### 2. 配置验证
```bash
# 检查修复后的脚本
grep "ffn-hidden-size" llama8b_scripts/run_llama8b_prune_tp8.sh
# 应输出: --ffn-hidden-size 14336
```

### 3. 重新运行
```bash
# 在容器内正确执行
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT
```

---

## 经验总结 📚

### 关键教训
1. **环境第一**: 必须在正确的容器环境内运行
2. **配置匹配**: 不能直接套用其他模型的配置参数  
3. **架构理解**: 需要理解模型的真实架构参数
4. **维度计算**: Tensor Parallel分割需要精确计算

### 预防措施
1. **配置检查**: 运行前验证模型配置文件
2. **环境验证**: 确保GPU和PyTorch正确可用
3. **参数匹配**: 根据实际模型调整所有相关参数
4. **测试先行**: 小规模测试验证配置正确性

---

## 状态更新 🎯

- ✅ **错误1 (GPU分配)**: 已解决 - 环境配置问题
- ✅ **错误2 (配置不匹配)**: 已修复 - FFN维度更新为14336
- 🚀 **下一步**: 在容器内重新运行稀疏化脚本

**当前状态**: 准备就绪，可以开始正常的Llama8b稀疏化流程！
