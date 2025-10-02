# Llama8b vs Llama2 稀疏训练参数对比分析

## 1. 关键修复：预稀疏 Checkpoint 路径 ✅

### 问题
稀疏化脚本修复后，使用 `--update-weight` 标志会自动在保存路径后添加 `.update_weight` 后缀。

### 修改
```bash
# 修改前
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${SPARSE_NAME}"

# 修改后 ✅
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${SPARSE_NAME}.update_weight"
```

实际路径：
- ❌ 旧路径（稠密权重）: `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0`
- ✅ 新路径（真正稀疏）: `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight`

---

## 2. 训练参数对比分析

### 2.1 训练间隔参数

| 参数 | Llama2 (7B) | Llama8b (8B) 原始 | Llama8b (8B) 修正 | 分析 |
|------|-------------|-------------------|-------------------|------|
| `SAVE_INTERVAL` | 500 | 100 | **500** ✅ | 原设置100太频繁，增加I/O开销 |
| `EVAL_INTERVAL` | 100 | 50 | **100** ✅ | 原设置50太频繁，减慢训练速度 |
| `LOG_INTERVAL` | 10 | 1 | **10** ✅ | 每次迭代打印日志过于冗余 |
| `WARMUP_ITERS` | 0 | 400 | **0** ✅ | 从预稀疏checkpoint开始不需要warmup |

**理由**：
- 从预稀疏模型开始训练，模型已经是合理初始化，不需要大量warmup
- 与Llama2验证配置保持一致，避免引入不必要的差异
- 减少I/O和日志开销，提高训练效率

### 2.2 学习率参数

| 参数 | Llama2 (7B) | Llama8b (8B) 原始 | Llama8b (8B) 修正 | 分析 |
|------|-------------|-------------------|-------------------|------|
| `--lr` | 5e-5 | 2e-5 | **5e-5** ✅ | 更高学习率加速收敛 |
| `--min-lr` | 5e-6 | 2e-6 | **5e-6** ✅ | 保持10:1的衰减比例 |

**理由**：
1. **稀疏训练需要较高学习率**：
   - Mask优化需要足够的梯度信号
   - SparseGPT产生的初始稀疏模式可能需要调整
   - 较高学习率有助于跳出局部最优

2. **模型规模差异小**：
   - Llama8b (8B) vs Llama2 (7B) 差异仅14%
   - 不需要大幅调整学习率
   - 保持与验证配置一致降低风险

### 2.3 MaskLLM 稀疏训练参数

| 参数 | Llama2 (7B) | Llama8b (8B) 原始 | Llama8b (8B) 修正 | 分析 |
|------|-------------|-------------------|-------------------|------|
| `--gumbel-scale-range` | 1e2 5e2 | 5e1 2.5e2 | **1e2 5e2** ✅ | Gumbel softmax的scale范围 |
| `--weight-reg` | 1e-5 | 2e-6 | **1e-5** ✅ | 权重正则化强度 |

**详细分析**：

#### Gumbel Scale Range
```bash
# Llama2: --gumbel-scale-range 1e2 5e2
# 含义: scale从100开始，线性增长到500
# 作用: 控制Gumbel softmax从软到硬的转换速度

# Llama8b 原始: --gumbel-scale-range 5e1 2.5e2  
# 问题: 起点太低(50)，结束点太低(250)
#      → mask更新太平滑，可能无法收敛到明确的2:4模式

# Llama8b 修正: --gumbel-scale-range 1e2 5e2 ✅
# 理由: 与Llama2保持一致，已经过验证
```

**Gumbel Softmax 原理**：
- 低scale (50-100): mask概率分布平滑，探索阶段
- 高scale (500): mask概率分布尖锐，接近硬mask，exploitation阶段
- 训练过程中逐渐从探索转向利用

#### Weight Regularization
```bash
# Llama2: --weight-reg 1e-5
# 含义: 对权重施加1e-5的L2正则化

# Llama8b 原始: --weight-reg 2e-6
# 问题: 正则化太弱，可能导致过拟合

# Llama8b 修正: --weight-reg 1e-5 ✅
# 理由: 
#   1. Llama8b参数量更大(8B vs 7B) → 更需要正则化
#   2. 与Llama2保持一致，降低调参风险
```

### 2.4 不变参数（保持与Llama2一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--N` | 2 | 2:4稀疏模式中的N |
| `--M` | 4 | 2:4稀疏模式中的M |
| `--mask-only` | - | 只优化mask，权重固定 |
| `--prior-strength` | 3.0 | 先验强度 |
| `--lr-mult` | 10 | mask学习率相对权重学习率的倍数 |
| `--gumbel-temperature-range` | 4 0.05 | Gumbel温度从4降到0.05 |

---

## 3. 批大小和序列长度

### 相同配置
| 参数 | Llama2 | Llama8b | 说明 |
|------|--------|---------|------|
| `--micro-batch-size` | 1 | 1 | 单GPU批大小 |
| `--global-batch-size` | 256 | 256 | 全局批大小 |
| `--seq-length` | 4096 | 4096 | 序列长度 |

**梯度累积步数**：
```
gradient_accumulation_steps = global_batch_size / (micro_batch_size × world_size)
                            = 256 / (1 × 8)
                            = 32
```

每32个micro-batch更新一次参数。

---

## 4. 优化器参数对比

### 相同配置（均采用AdamW）

| 参数 | Llama2 | Llama8b | 说明 |
|------|--------|---------|------|
| `--adam-beta1` | 0.9 | 0.9 | 一阶矩估计衰减率 |
| `--adam-beta2` | 0.95 | 0.95 | 二阶矩估计衰减率 |
| `--adam-eps` | 1e-5 | 1e-5 | 数值稳定性参数 |
| `--weight-decay` | 0.1 | 0.1 | 权重衰减 |
| `--clip-grad` | 1.0 | 1.0 | 梯度裁剪阈值 |

**分析**：这些优化器参数已经是大模型训练的标准配置，Llama8b无需调整。

---

## 5. 特殊参数分析

### 5.1 兼容性参数（首次训练时使用）

```bash
# 从预稀疏checkpoint开始训练时需要
--no-load-optim             # 不加载优化器状态（预稀疏checkpoint中没有）
--no-load-rng               # 不加载随机数状态
--finetune                  # 微调模式
--enable-partial-load       # 允许部分加载（key不完全匹配时）
```

**原因**：
- 预稀疏checkpoint只包含模型权重和mask
- 不包含优化器状态、学习率调度器状态、RNG状态
- 这些标志防止加载失败

### 5.2 恢复训练时（resume=1）

```bash
# 从训练checkpoint恢复时，这些参数需要移除
# 以便加载完整的训练状态
```

---

## 6. 数据配置

### 相同的C4数据blend策略

```bash
# Llama2
C4_HOME=assets/data/c4_llama2_pretokenized
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama2_${i}_text_document"
done

# Llama8b
C4_HOME=assets/data/c4_llama8b_pretokenized
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
```

- 20个文件，每个权重0.05
- 总权重: 20 × 0.05 = 1.0 ✅
- 数据分布均匀

---

## 7. 关键差异总结

| 类别 | 差异点 | 原因 |
|------|--------|------|
| **模型架构** | `--vocab-size 119696` vs 32000 | Llama8b使用更大的词汇表 |
| **模型架构** | `--ffn-hidden-size 14336` vs 11008 | Llama8b的FFN层更宽 |
| **Tokenizer** | `Llama8bTokenizer` vs `Llama2Tokenizer` | 自定义tokenizer |
| **Checkpoint路径** | 添加`.update_weight`后缀 | 修复后的稀疏化脚本行为 |

---

## 8. 修改总结

### ✅ 已应用的修改

1. **Checkpoint路径**: 添加 `.update_weight` 后缀
2. **训练间隔**: 与Llama2保持一致 (500/100/10/0)
3. **学习率**: 提升至 5e-5 → 5e-6
4. **Gumbel scale**: 调整至 1e2 → 5e2
5. **Weight regularization**: 增强至 1e-5

### 📊 预期效果

1. **训练效率**: 减少不必要的保存/评估/日志开销
2. **收敛速度**: 更高学习率加速mask优化
3. **稀疏质量**: 更强的Gumbel scale促进明确的2:4模式
4. **泛化性能**: 适当的正则化防止过拟合

---

## 9. 使用方法

### 首次训练（从预稀疏checkpoint）
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0
```

### 恢复训练（从训练checkpoint）
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

---

## 10. 监控指标

训练过程中重点关注：

1. **训练loss**: 应该稳定下降
2. **验证loss**: 每100步评估一次
3. **Mask变化**: `--log-diff-mask` 记录mask更新幅度
4. **梯度范数**: `--log-num-zeros-in-grad` 监控稀疏梯度
5. **Gumbel scale**: 从100线性增长到500

预期训练2000步后：
- 验证loss降低
- Mask收敛到稳定的2:4模式
- 模型保持50%稀疏率

---

## 参考

- Llama2验证脚本: `scripts/learnable_sparsity/llama2_7b_mask_only_tp8_c4.sh`
- MaskLLM论文: 详细介绍Gumbel softmax和mask优化策略
- SparseGPT: 预稀疏模型生成方法
