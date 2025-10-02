# Llama8b 训练恢复指南

## 问题：学习率不匹配

当从 checkpoint 恢复训练并修改学习率时，会遇到错误：
```
AssertionError: OptimizerParamScheduler: class input value 1e-05 and 
checkpoint value 5e-05 for learning rate do not match
```

## 解决方案

### **方案 1: 不加载优化器状态（推荐 - 用于调整超参数）**

**适用场景**：
- 你想使用新的学习率/超参数
- 你想重置 Adam 动量
- 从 NaN 错误恢复后想更保守地训练

**脚本**：`llama8b_presparse_training_tp8_stable.sh`

**命令**：
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

**工作原理**：
- 加载模型权重（从 iteration 400）
- **不加载**优化器状态（`--no-load-optim`）
- **不加载**随机数状态（`--no-load-rng`）
- 使用新的学习率：1e-5（降低5倍）

**优点**：
- ✅ 可以使用任意新的学习率
- ✅ 优化器状态重置，可能更稳定
- ✅ 适合从 NaN 恢复

**缺点**：
- ⚠️ 失去 Adam 动量信息
- ⚠️ 可能需要几次迭代重新适应

---

### **方案 2: 完全恢复原始状态（不推荐 - 可能再次 NaN）**

**适用场景**：
- 训练意外中断（不是因为 NaN）
- 你想完全继续之前的训练状态

**脚本**：`llama8b_presparse_training_tp8.sh`（原始脚本）

**命令**：
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

**工作原理**：
- 加载模型权重
- 加载优化器状态（包括 Adam 动量）
- 加载随机数状态
- 使用 checkpoint 中保存的学习率：5e-5

**优点**：
- ✅ 完全恢复训练状态
- ✅ 保留优化器动量

**缺点**：
- ❌ 必须使用原始学习率
- ❌ 可能在 iteration 496 附近再次遇到 NaN

---

### **方案 3: 从预稀疏 checkpoint 重新开始（最保守）**

**适用场景**：
- 你想用新参数完全重新训练
- iteration 400 的进度可以放弃

**命令**：
```bash
# 删除或备份旧的训练 checkpoint
mv output/checkpoints/llama8b_maskllm_training_tp8 \
   output/checkpoints/llama8b_maskllm_training_tp8.backup

# 从预稀疏模型开始（参数 0）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 0
```

**优点**：
- ✅ 干净的起点
- ✅ 完全使用新参数

**缺点**：
- ❌ 失去 iteration 1-400 的训练进度（~27小时）

---

## 推荐决策树

```
是否愿意放弃优化器状态？
├─ 是 → 方案 1（推荐）
│   └─ 使用 llama8b_presparse_training_tp8_stable.sh
│      ├─ 新学习率：1e-5（更稳定）
│      ├─ 从 iteration 400 继续
│      └─ 失去 Adam 动量（影响不大）
│
└─ 否 → 
    ├─ 方案 2：使用原始学习率
    │   └─ 风险：可能再次 NaN
    │
    └─ 方案 3：完全重新开始
        └─ 损失：400次迭代（~27小时）
```

## 执行步骤（方案 1 - 推荐）

### 1. 进入容器
```bash
ssh zdhs0054@hd02-gpu1-0056
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_apptainer_extended_libs.sh
```

### 2. 恢复训练（使用稳定参数）
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

### 3. 监控训练（另一个终端）
```bash
# 实时监控
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh tail"
```

### 4. 观察关键指标

**前几次迭代可能会有小波动（正常）**：
- Iteration 400-410: 适应新学习率
- Loss 可能短暂升高后下降
- Grad norm 应该保持稳定

**如果看到以下情况，说明恢复成功**：
- ✅ Loss 在 10-20 iterations 后开始下降
- ✅ Grad norm 保持在 0.01-0.1
- ✅ 没有 NaN 或 Inf

---

## 参数对比

### 原始脚本 vs 稳定脚本

| 参数 | 原始值 | 稳定值 | 说明 |
|------|--------|--------|------|
| `--lr` | 5e-5 | **1e-5** | 学习率降低5倍 |
| `--min-lr` | 5e-6 | **1e-6** | 最小学习率降低5倍 |
| `--lr-mult` | 10 | **5** | Mask学习率倍数减半 |
| `--weight-reg` | 1e-5 | **5e-6** | 正则化强度减半 |
| `--gumbel-temperature-range` | 4 0.05 | **4 0.5** | 最低温度提高10倍 |
| `--clip-grad` | 1.0 | **1.0** | 梯度裁剪（新增） |
| `SAVE_INTERVAL` | 100 | **50** | 保存频率加倍 |

---

## 常见问题

### Q1: 不加载优化器会影响训练质量吗？
**A**: 影响很小。前几次迭代可能需要重新建立动量，但很快会恢复。对于从 NaN 恢复的场景，重置优化器反而更安全。

### Q2: 为什么不能直接修改 checkpoint 中的学习率？
**A**: Checkpoint 文件是二进制格式，手动修改容易损坏。使用 `--no-load-optim` 是官方推荐的方式。

### Q3: 如果稳定版本还是遇到 NaN 怎么办？
**A**: 进一步降低学习率：
```bash
# 修改 llama8b_presparse_training_tp8_stable.sh
# 将 --lr 1e-5 改为 --lr 5e-6
# 将 --lr-mult 5 改为 --lr-mult 3
```

### Q4: 可以混合使用吗（先用稳定版，后面用原始版）？
**A**: 可以，但要注意：
- 稳定版训练几百次迭代后
- 如果状态良好，可以切回原始学习率
- 但需要使用 `--no-load-optim` 避免冲突

---

## 实际执行（立即可用）

```bash
# 在你的 Mac 上
ssh zdhs0054@hd02-gpu1-0056 'cd /data/home/zdhs0054/zzhou/MaskLLM && bash run_apptainer_extended_libs.sh'

# 进入容器后，执行（推荐方案 1）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

🎯 **这个命令会**：
- ✅ 从 iteration 400 继续训练
- ✅ 使用新的稳定学习率（1e-5）
- ✅ 不加载旧的优化器状态
- ✅ 每 50 次迭代保存 checkpoint
- ✅ 应用所有防 NaN 的优化
