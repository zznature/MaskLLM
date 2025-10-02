# Llama8b 训练 NaN 错误分析与恢复方案

## 🔴 错误分析

### 发生时间
- **Iteration**: 496 → 497
- **时间**: 2025-10-01 17:08

### 错误类型
```
AssertionError: Rank 0-7: found NaN in local grad norm in backward pass 
before data-parallel communication collective.
```

### 训练状态（出错前）
- ✅ **Loss 正常**: lm_loss ~2.5, reg_loss ~8400
- ✅ **Grad norm 正常**: 0.034-0.035
- ⚠️ **警告信号**:
  - `reg loss` 持续在 8400+ 高位（正则化损失很大）
  - `scale_multiplier` 持续增长: 198.4 → 199.0
  - `num zeros` 波动: 5178 → 5373
  - `loss scale: 1.0` （已降到最低，说明之前有数值问题）

### 根本原因推测

1. **稀疏化训练的数值不稳定**:
   - 正则化损失 (`reg_loss: 8.4k+`) 远大于语言模型损失 (`lm_loss: 2.5`)
   - `weight-reg 1e-5` 可能导致正则项过大
   - Mask 更新过程中可能出现极端值

2. **Gumbel-Softmax 温度问题**:
   - `temperature: 3.02` 在逐渐下降
   - 温度过低可能导致 softmax 输出极端值，进而产生梯度爆炸

3. **混合精度训练累积误差**:
   - `loss scale: 1.0` 说明自动缩放已失效
   - 可能在多次迭代中累积了数值误差

## ✅ 恢复方案

### 方案 A: 从最近的 checkpoint 恢复（推荐）

#### 1. 检查可用的 checkpoint

```bash
# 检查训练checkpoint目录
ls -ltr output/checkpoints/llama8b_maskllm_training_tp8/

# 查看最近的 checkpoint iteration
cat output/checkpoints/llama8b_maskllm_training_tp8/latest_checkpointed_iteration.txt
```

**预期结果**:
- 如果有 `iter_0000400/` 或 `iter_0000100/` → 可以恢复
- 如果只有 `iter_0000001/` → 需要从预稀疏模型重新开始

#### 2. 从 checkpoint 恢复训练

```bash
# 恢复训练（参数1=1表示从训练checkpoint恢复）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

**关键**: 脚本会自动：
- 从 `output/checkpoints/llama8b_maskllm_training_tp8/` 加载最新 checkpoint
- 恢复优化器状态、RNG状态
- 从保存的 iteration 继续训练

### 方案 B: 调整超参数后重新训练

如果没有可用的 checkpoint，或者想避免再次遇到 NaN，需要调整超参数。

#### 调整策略

1. **降低学习率**（最重要）:
```bash
# 当前: --lr 5.0e-4 --min-lr 5.0e-5
# 建议: --lr 1.0e-4 --min-lr 1.0e-5  （降低5倍）
```

2. **降低 mask 学习率倍数**:
```bash
# 当前: --lr-mult 10
# 建议: --lr-mult 5  （从10降到5）
```

3. **调整正则化权重**:
```bash
# 当前: --weight-reg 1e-5
# 建议: --weight-reg 5e-6  （降低一半）
```

4. **调整 Gumbel 温度范围**:
```bash
# 当前: --gumbel-temperature-range 4 0.05
# 建议: --gumbel-temperature-range 4 0.5  （最低温度从0.05提高到0.5）
```

5. **启用梯度裁剪**:
```bash
# 添加: --clip-grad 1.0
```

#### 修改后的训练脚本

创建调整版本：
```bash
cp llama8b_scripts/llama8b_presparse_training_tp8.sh \
   llama8b_scripts/llama8b_presparse_training_tp8_stable.sh
```

修改以下行（约106行）:
```bash
# 原来:
TASK_CMD="--gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5"

# 修改为（更稳定的配置）:
TASK_CMD="--gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 5 --weight-reg 5e-6 --clip-grad 1.0"
```

修改学习率（约120行）:
```bash
# 原来:
--lr 5.0e-4 \
--min-lr 5.0e-5 \

# 修改为:
--lr 1.0e-4 \
--min-lr 1.0e-5 \
```

运行修改后的脚本:
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 0
```

### 方案 C: 增加 checkpoint 保存频率

为了防止长时间训练后再次遇到 NaN 损失过多进度：

修改保存间隔（约91行）:
```bash
# 原来:
SAVE_INTERVAL=100

# 修改为:
SAVE_INTERVAL=50  # 每50次迭代保存一次
```

## 🔍 监控建议

### 1. 实时监控关键指标

```bash
# 监控 grad norm（应该稳定在 0.01-0.1 范围）
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh search 'grad norm'"

# 监控 loss scale（应该保持 > 1.0）
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh search 'loss scale'"

# 监控 reg loss（不应该持续增长）
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh search 'reg loss'"
```

### 2. 警告信号

如果出现以下情况，考虑提前停止并调整参数：
- ⚠️ `loss scale` 降到 1.0
- ⚠️ `grad norm` 突然增大（> 1.0）
- ⚠️ `reg loss` 持续大于 `lm loss` 的 1000 倍
- ⚠️ `num zeros` 剧烈波动（±500 以上）

## 📊 预期结果

### 成功的训练应该看到：
- ✅ `lm loss` 逐渐下降（2.5 → 2.0 → 1.8...）
- ✅ `reg loss` 稳定或缓慢下降
- ✅ `grad norm` 稳定在 0.01-0.1
- ✅ `loss scale` 保持在 > 1.0
- ✅ `num zeros` 逐渐稳定在目标稀疏度

### 训练收敛指标：
- 目标 iterations: 2000
- 预期时间: ~110小时（219秒/iter × 1800 iters）
- 最终 lm loss: < 2.0（参考 pre-sparse ppl 8.0）

## 🚀 立即执行步骤

### 步骤 1: 检查 checkpoint
```bash
ssh zdhs0054@hd02-gpu1-0056 'ls -ltr /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8/ && cat /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8/latest_checkpointed_iteration.txt'
```

### 步骤 2: 决策
- **如果有 iter_0000400 或更早的 checkpoint** → 执行方案 A（直接恢复）
- **如果只有很早期的 checkpoint** → 执行方案 B（调整参数重新训练）

### 步骤 3: 执行恢复
```bash
# 方案 A: 直接恢复
ssh zdhs0054@hd02-gpu1-0056 'cd /data/home/zdhs0054/zzhou/MaskLLM && bash run_apptainer_extended_libs.sh'
# 进入容器后：
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1

# 或方案 B: 调整参数（需要先修改脚本）
# 见上面的修改说明
```

## 📝 总结

**NaN 错误是稀疏化训练中常见的数值不稳定问题**，主要由于：
1. 正则化项过大
2. Gumbel-Softmax 温度过低
3. 学习率过高

**最佳恢复路径**：
1. 尝试从最近的 checkpoint 恢复（如果有）
2. 如果再次遇到 NaN，降低学习率和调整超参数
3. 增加 checkpoint 保存频率以减少潜在损失

需要我帮你生成调整后的训练脚本吗？

