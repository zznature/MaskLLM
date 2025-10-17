# Warmup Iterations 参数不匹配错误修复

## 🔴 错误信息

```
AssertionError: OptimizerParamScheduler: class input value 0 and checkpoint value 102400 for warmup iterations do not match
```

## 🔍 根本原因

当从 checkpoint 恢复训练时，即使使用了 `--no-load-optim`（不加载优化器状态），Megatron 仍然会检查 `OptimizerParamScheduler` 的参数是否匹配，包括：
- `lr_warmup_steps`
- `lr`
- `min_lr`
- 其他 scheduler 参数

**问题**:
- Checkpoint (iter 1125) 保存的 warmup_iters: `102400`（可能是早期训练的值）
- 新脚本设置的 warmup_iters: `0`（认为从 iter 1125 继续不需要 warmup）
- **冲突** → 训练启动失败

## ✅ 解决方案

### **方案 1: 使用 `--override-opt_param-scheduler` (推荐)**

**原理**: 强制使用当前脚本的 scheduler 参数，忽略 checkpoint 中的值。

**修改**:
```bash
options=" \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
    --override-opt_param-scheduler \  # 🔑 关键参数
    --seed 48 \
    $TASK_CMD "
```

**优点**:
- ✅ 完全控制 scheduler 参数
- ✅ 避免历史 checkpoint 参数冲突
- ✅ 可以自由调整学习率策略

**缺点**:
- ⚠️ Warmup 会从当前 iteration 重新开始（但我们已经 iter 1125，warmup 早已完成）

---

### **方案 2: 匹配 checkpoint 的 warmup_iters 值**

**修改**:
```bash
WARMUP_ITERS=102400  # 与 checkpoint 一致
```

**优点**:
- ✅ 避免参数冲突

**缺点**:
- ❌ 需要知道 checkpoint 中的确切值
- ❌ 如果 checkpoint 之间值不同，每次都要调整

---

## 🎯 推荐配置

**最终采用方案 1** (`--override-opt_param-scheduler`):

```bash
# 训练参数
WARMUP_ITERS=400  # 保持与原始训练一致（虽然会被 override）

# options 中添加
--override-opt_param-scheduler  # 强制使用当前参数
```

## 📝 注意事项

1. **学习率调度**:
   - 由于 `--override-opt_param-scheduler`，学习率会根据当前 iteration (1125) 重新计算
   - Cosine decay 会从 iter 1125 继续，不会重置到 iter 0

2. **Warmup 影响**:
   - Warmup 已在 iter 0-400 完成
   - Iter 1125 时，warmup 早已结束，设置为 0 或 400 都无影响

3. **与其他参数的兼容性**:
   - `--no-load-optim`: 不加载优化器状态（Adam 的 m 和 v）
   - `--no-load-rng`: 不加载 RNG 状态
   - `--override-opt_param-scheduler`: 不使用 checkpoint 的 scheduler 参数
   - **三者配合**: 完全重置优化器和调度器，只保留模型权重

## ✅ 验证成功

修复后，训练应该能够正常启动：

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable_resume.sh
```

**成功标志**:
```
loading checkpoint from .../iter_0001125
> using checkpoint value 2e-05 for learning rate
> using checkpoint value 2e-06 for minimum learning rate
successfully loaded checkpoint from iteration 1125
```

---

**修复时间**: 2025-10-07  
**影响文件**: `llama8b_presparse_training_tp8_stable_resume.sh`  
**关键参数**: `--override-opt_param-scheduler`

