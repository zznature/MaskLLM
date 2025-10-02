# Llama8b 稀疏训练快速开始

## 🎯 前置条件检查

### 1. 预稀疏模型是否就绪？

检查路径是否存在：
```bash
ls -lh output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight/
```

**注意**：必须是 `.update_weight` 后缀的目录（修复后生成的真正稀疏模型）

如果不存在，先运行稀疏化：
```bash
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT
```

### 2. 预处理数据是否就绪？

检查数据文件：
```bash
ls assets/data/c4_llama8b_pretokenized/c4_llama8b_00000_text_document.bin
```

如果不存在，运行数据预处理：
```bash
bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh
```

---

## 🚀 启动训练

### 首次训练（从预稀疏checkpoint开始）

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0
```

参数说明：
- `0` = 从预稀疏checkpoint开始
- 自动添加兼容性参数：`--no-load-optim --no-load-rng --finetune --enable-partial-load`

### 恢复训练（从训练checkpoint继续）

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

参数说明：
- `1` = 从训练checkpoint恢复
- 加载完整训练状态（优化器、学习率调度器等）

---

## 📊 关键修改（相比之前版本）

### ✅ 修复1：Checkpoint路径
```bash
# 修改前（错误）
PRESPARSE_CHECKPOINT=".../llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0"

# 修改后（正确）✅
PRESPARSE_CHECKPOINT=".../llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight"
```

### ✅ 修改2：训练间隔参数（对齐Llama2验证配置）
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `SAVE_INTERVAL` | 100 | **500** | 减少I/O开销 |
| `EVAL_INTERVAL` | 50 | **100** | 减慢训练阻塞 |
| `LOG_INTERVAL` | 1 | **10** | 减少日志冗余 |
| `WARMUP_ITERS` | 400 | **0** | 预稀疏模型不需要warmup |

### ✅ 修改3：学习率（对齐Llama2）
```bash
# 旧配置
--lr 2e-5 --min-lr 2e-6

# 新配置 ✅
--lr 5e-5 --min-lr 5e-6
```

### ✅ 修改4：MaskLLM参数（对齐Llama2）
```bash
# 旧配置
--gumbel-scale-range 5e1 2.5e2 --weight-reg 2e-6

# 新配置 ✅
--gumbel-scale-range 1e2 5e2 --weight-reg 1e-5
```

---

## 📈 监控训练进度

### 1. 实时查看日志

```bash
tail -f output/logs/llama8b_presparse_training/training_*.log
```

### 2. 关键指标

**训练loss**：
```
iteration   100/2000 | consumed samples:   25600 | ... | loss: 2.xxxE+00
```
- 应该稳定下降
- 典型范围：2.0 → 1.8（2000步）

**验证loss**（每100步）：
```
validation loss at iteration 100 | validation loss: 2.xxxE+00 | ppl: x.xx
```
- 困惑度(ppl)应该下降
- 稀疏模型ppl通常高于稠密模型10-20%

**Mask变化**：
```
[Mask Diff] iteration 100 | avg_diff: 0.xxx
```
- 初期应该较大（~0.1-0.2）
- 后期应该收敛（<0.01）

### 3. Checkpoint位置

```bash
ls -lh output/checkpoints/llama8b_maskllm_training_tp8/
```

每500步保存一次：
- `iter_0000500/`
- `iter_0001000/`
- `iter_0001500/`
- `iter_0002000/`

---

## ⚠️ 常见问题

### Q1: `FileNotFoundError: .../llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0`

**原因**：缺少 `.update_weight` 后缀

**解决**：
1. 检查是否运行了修复后的稀疏化脚本
2. 确认预稀疏checkpoint路径包含 `.update_weight`

### Q2: `RuntimeError: Error(s) in loading state_dict`

**原因**：首次训练时未使用兼容性参数

**解决**：确认 `RESUME_FLAG=0` 时自动添加了：
```bash
--no-load-optim --no-load-rng --finetune --enable-partial-load
```

### Q3: 训练loss不下降或NaN

**可能原因**：
1. 学习率过高 → 降低至 2e-5
2. 梯度爆炸 → 检查 `--clip-grad 1.0` 是否生效
3. 数据问题 → 检查预处理数据完整性

**调试**：
```bash
# 检查梯度范数
grep "grad norm" output/logs/.../training_*.log

# 检查权重更新
grep "weight norm" output/logs/.../training_*.log
```

### Q4: OOM (Out of Memory)

**解决方案**：
1. 降低 `--micro-batch-size` 从1到1（已经是最小）
2. 减少 `--seq-length` 从4096到2048
3. 使用gradient checkpointing（需要修改代码）

### Q5: 训练太慢

**优化**：
1. 确认使用了 `--use-flash-attn`
2. 检查 `EVAL_INTERVAL=100`（不要设太小）
3. 检查 `LOG_INTERVAL=10`（不要设太小）

---

## 📋 训练完成后

### 1. 评估稀疏模型

```bash
bash run_maskllm_native.sh \
    llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000 \
    8 \
    sparse
```

### 2. 对比稠密基线

| 模型 | PPL (WIKITEXT2) | 稀疏率 |
|------|-----------------|--------|
| 稠密基线 | ~8.0 | 0% |
| 预稀疏模型 (SparseGPT) | ~12-15 | 50% |
| 稀疏训练后 | **目标: 9-11** | 50% |

### 3. 导出为HuggingFace格式（可选）

```bash
# TODO: 添加Megatron→HF转换脚本
```

---

## 🔗 相关文档

- 详细参数分析：`llama8b_scripts/training_params_analysis.md`
- 稀疏化修复：`llama8b_scripts/llama8b_inference/llama8b_sparse_eval.md`
- 推理评测：`llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh`

---

## 📞 问题排查清单

训练前检查：
- [ ] 预稀疏checkpoint存在且包含 `.update_weight` 后缀
- [ ] 预处理数据存在（20个文件）
- [ ] GPU内存充足（建议每卡>40GB）
- [ ] 磁盘空间充足（checkpoint ~15GB/500步）

训练中监控：
- [ ] Loss稳定下降
- [ ] 无NaN/Inf
- [ ] GPU利用率高（>80%）
- [ ] Mask变化逐渐收敛

训练后验证：
- [ ] 验证loss低于预稀疏模型
- [ ] 稀疏度保持在50%
- [ ] Checkpoint完整保存
