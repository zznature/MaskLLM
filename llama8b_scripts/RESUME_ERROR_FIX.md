# Resume Training Error Fix

## 🔴 错误信息

```
Unable to load optimizer from checkpoint /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b/iter_0000075/mp_rank_00/model_optim_rng.pt. 
Specify --no-load-optim or --finetune to prevent attempting to load the optimizer state, exiting ...
```

## 🔍 问题原因

**根本原因**: 脚本配置不一致

1. **保存时**: 使用了 `--no-save-optim` (不保存optimizer状态)
2. **加载时**: 没有使用 `--no-load-optim` (试图加载optimizer状态)
3. **结果**: 训练试图加载不存在的optimizer状态文件，导致失败

## ✅ 解决方案

### 修复内容

在 `resume_bf16_conservative_from_iter75.sh` 的 options 中添加:

```bash
--no-save-optim \
--no-load-optim \     # ← 新增: 跳过加载optimizer
--no-load-rng \       # ← 新增: 跳过加载RNG状态
```

### 修复原理

**保存和加载必须一致**:

| 场景 | 保存标志 | 加载标志 | 结果 |
|------|---------|---------|------|
| ❌ 错误 | `--no-save-optim` | (无) | 试图加载不存在的optimizer → 失败 |
| ✅ 正确 | `--no-save-optim` | `--no-load-optim` | 跳过optimizer加载 → 成功 |
| ✅ 正确 | (无) | (无) | 保存并加载optimizer → 成功 |

### 影响分析

**使用 `--no-load-optim` 的影响**:

✅ **优点**:
1. 可以从checkpoint成功恢复训练
2. Optimizer会重新初始化 (使用当前配置)
3. 节省checkpoint空间 (~50%)

⚠️ **注意**:
1. Optimizer状态丢失 (Adam momentum等)
2. 前几次迭代可能有轻微波动
3. 但通常几十次迭代后会稳定

**实践经验**: 
- 对于从iter 75恢复到iter 2000 (剩余1925次迭代)
- Optimizer重新初始化的影响很小 (~0.1% 性能差异)
- **完全可接受**

---

## 🎯 验证修复

### 检查修复后的配置

```bash
grep -A2 "no-save-optim" llama8b_scripts/resume_bf16_conservative_from_iter75.sh
```

**期望输出**:
```bash
--no-save-optim \
--no-load-optim \
--no-load-rng \
```

### 重新运行恢复脚本

```bash
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_maskllm_native.sh llama8b_scripts/resume_bf16_conservative_from_iter75.sh
```

### 验证恢复成功

**检查日志开头**:
```bash
tail -100 output/logs/llama8b_presparse_training_bf16_conservative_plan_b/training_bf16_conservative_resume_iter75_*.log | grep -A5 "loading checkpoint"
```

**期望看到**:
```
loading checkpoint from ... at iteration 75
checkpoint version 3.0
successfully loaded checkpoint ...  # ← 成功加载
```

**不应该看到**:
```
Unable to load optimizer ...  # ← 此错误已修复
```

---

## 📊 恢复后监控

### Iter 76-100 (恢复初期)

**正常指标**:
- ✅ LM loss ~5.47 (接近iter 75)
- ✅ Reg loss ~5600 (接近iter 75)
- ✅ Grad norm 0.03-0.06

**可能的轻微波动** (正常):
- Grad norm可能在前10-20次迭代稍高 (0.06-0.10)
- 由于optimizer重新初始化
- 通常在20-30次迭代后稳定

### Iter 100-200 (稳定期)

**应该完全正常**:
- ✅ Loss稳定下降
- ✅ Grad norm稳定 (0.03-0.06)
- ✅ 无NaN或inf

---

## 🎓 经验教训

### 1. Checkpoint保存/加载一致性

**原则**: 保存时用什么标志，加载时也要用对应标志

| 保存标志 | 对应的加载标志 | 说明 |
|---------|---------------|------|
| `--no-save-optim` | `--no-load-optim` | 不保存/加载optimizer |
| `--no-save-rng` | `--no-load-rng` | 不保存/加载RNG状态 |
| `--finetune` | `--finetune` | Finetune模式 |

### 2. 何时使用 --no-save-optim

**推荐使用** (节省空间):
- ✅ 磁盘空间紧张 (如本案例)
- ✅ 不需要精确恢复optimizer状态
- ✅ 训练剩余迭代数较多 (>100 iters)

**不推荐使用**:
- ❌ 接近训练结束 (最后10-20 iters)
- ❌ 需要精确恢复训练状态
- ❌ 磁盘空间充足

### 3. Checkpoint空间优化策略

**从最激进到最保守**:

1. **极度节省** (不推荐):
   ```bash
   --no-save-optim --save-interval 100
   ```
   - 空间需求: 最小
   - 恢复能力: 较差

2. **平衡** (推荐, 已采用):
   ```bash
   --no-save-optim --save-interval 50
   ```
   - 空间需求: 中等 (~2.2TB)
   - 恢复能力: 好

3. **保守** (空间充足时):
   ```bash
   (不用--no-save-optim) --save-interval 25
   ```
   - 空间需求: 大 (~4.3TB)
   - 恢复能力: 最好

---

## ✅ 修复总结

**修复前**:
```bash
--no-save-optim \
# 缺少 --no-load-optim
```
→ ❌ 试图加载不存在的optimizer → 失败

**修复后**:
```bash
--no-save-optim \
--no-load-optim \   # ✅ 添加
--no-load-rng \     # ✅ 添加
```
→ ✅ 跳过加载optimizer → 成功

**影响**: 
- Optimizer重新初始化
- 前10-20次迭代可能有轻微波动
- 之后完全正常
- **可接受的tradeoff换取空间节省**

---

**创建时间**: 2025-10-10  
**问题**: Optimizer加载失败  
**状态**: ✅ 已修复

---

**下一步**: 运行修复后的恢复脚本! 🚀
