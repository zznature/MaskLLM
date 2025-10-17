# 🎯 Llama8b 稳定训练方案 - 综合分析与规划

## 📊 **当前情况总结**

### **训练任务 1: BF16 训练** (`llama8b_presparse_training_tp8.sh`)
- **当前状态**: ❌ 在 iter 1143 遇到 NaN 错误
- **崩溃前状态**:
  ```
  LM Loss: 2.57-2.62 (稳定)
  Reg Loss: 9467 (iter 1125) → 9586 (iter 1143)
  Reg Loss 增速: +119 / 18 iters = +6.6/iter
  Mask Difference: 0.112 (已开始优化)
  Temperature: 1.864 → 1.830
  Max Prob: 0.680 → 0.688
  Grad Norm: 0.038-0.039
  ```
- **关键参数**:
  ```bash
  --lr 2e-5 --min-lr 2e-6
  --lr-mult 10
  --weight-reg 2e-6
  --gumbel-temperature-range 4 0.2
  --clip-grad 1.0
  --warmup-iters 400
  --bf16
  ```

### **训练任务 2: FP32 训练** (`llama8b_presparse_training_tp8_fp32_from_bf16.sh`)
- **当前状态**: ✅ 运行中，iter 633，无 NaN
- **当前状态**:
  ```
  LM Loss: 2.60-2.64 (稳定)
  Reg Loss: 7205 (iter 633)
  Reg Loss 增速: +4.88/iter
  Mask Difference: 0.059 (完全停滞)
  Temperature: 4.736 (下降太慢)
  Max Prob: 0.481
  Grad Norm: 0.115-0.117
  ```
- **关键参数**:
  ```bash
  --lr 5e-6 --min-lr 5e-7
  --lr-mult 1
  --weight-reg 1e-7
  --gumbel-temperature-range 6 2.0
  --clip-grad 0.3
  --no-warmup
  --recompute-activations (no flash-attn)
  ```

---

## 🔍 **深度对比分析**

### **训练效果对比**

| 指标 | BF16 训练 (iter 1125) | FP32 训练 (iter 633) | 优势 |
|------|----------------------|---------------------|------|
| **数值稳定性** | ❌ NaN at 1143 | ✅ 无 NaN | FP32 |
| **Reg Loss** | 9467 | 7205 | FP32 |
| **Reg Loss 增速** | +6.6/iter | +4.88/iter | FP32 |
| **Mask 优化** | ✅ 0.112 (有进展) | ❌ 0.059 (停滞) | **BF16** |
| **Temperature** | 1.864 (接近目标) | 4.736 (太高) | **BF16** |
| **Max Prob** | 0.680 (收敛中) | 0.481 (分散) | **BF16** |
| **训练速度** | ~220秒/iter | ~439秒/iter | **BF16** |
| **显存占用** | 53GB | 65-70GB | **BF16** |

### **关键发现**

1. **BF16 的优势**:
   - ✅ Mask 正在有效优化 (difference 从初始 0.05 → 0.112)
   - ✅ Temperature 已降至 1.864，mask 趋向确定性
   - ✅ Max Prob 0.680，mask 开始收敛
   - ✅ 训练速度快 2倍

2. **BF16 的劣势**:
   - ❌ 反复在 iter 1000-1150 附近遇到 NaN
   - ❌ Reg loss 增速更快 (+6.6 vs +4.88)

3. **FP32 的优势**:
   - ✅ 完全无 NaN（已验证 633 次迭代）
   - ✅ Reg loss 增速更慢

4. **FP32 的劣势**:
   - ❌ Mask 完全不动 (0.059)
   - ❌ Temperature 下降太慢（需要 ~1400 iters 才能到 2.0）
   - ❌ 训练速度慢 2倍

---

## 💡 **核心问题诊断**

### **为什么 BF16 在 iter 1000+ 反复 NaN？**

**根本原因**: **Mask 优化进入确定性阶段后，梯度分布改变**

1. **初期 (iter 0-800)**:
   - Temperature 高 (4.0 → 2.5)
   - Mask 接近均匀分布
   - 梯度平滑，数值稳定

2. **中期 (iter 800-1100)**:
   - Temperature 降至 2.0-1.8
   - Mask 开始收敛（max_prob 0.65 → 0.68）
   - **Mask 梯度变大**（从软决策变硬决策）
   - **Reg loss 加速上升** (+6.6/iter)

3. **危险期 (iter 1100-1150)**:
   - Temperature 接近 1.8
   - Max Prob 接近 0.7
   - **某些 mask 变得极端** (0 或 1)
   - **局部梯度爆炸** → bf16 下溢/溢出 → **NaN**

### **为什么 FP32 虽然稳定，但 Mask 不动？**

**根本原因**: **参数过于保守，temperature 下降太慢**

```
Current: temp 4.736 → 2.0，需要约 1400 iters
BF16:    temp 1.864 → 0.2，已接近目标

FP32 参数设计过于保守:
- lr-mult=1  (vs BF16 的 10)
- temperature range 6→2.0 (vs BF16 的 4→0.2)
- clip-grad=0.3 (vs BF16 的 1.0)
```

---

## 🎯 **最优方案：BF16 稳定化策略**

### **方案选择理由**

❌ **不推荐继续 FP32**:
- Mask 优化太慢（需要 >2000 iters 才有效）
- 速度慢 2倍，成本高
- 即使到 iter 2000，reg loss 也会很高 (~13000)

✅ **推荐 BF16 稳定化**:
- Mask 已经在优化 (0.112)
- Temperature 已接近目标 (1.864)
- 只需解决 iter 1100-1150 的 NaN 问题
- 速度快，效率高

### **BF16 NaN 问题的本质**

**不是参数问题，而是阶段性数值风险**:
- Mask 从软→硬的转换期（iter 1000-1200）
- Temperature 1.8-2.0 是最危险区域
- 个别 mask logit 变得极端

---

## 🚀 **稳定训练方案**

### **方案 A: BF16 渐进式降温（推荐 ⭐⭐⭐⭐⭐）**

**核心思想**: **放慢 temperature 下降速度，让 mask 更平滑地收敛**

#### **修改参数**

```bash
# 当前 BF16 配置
--gumbel-temperature-range 4 0.2  # 温度范围太激进
--lr-mult 10                       # Mask LR 太高
--clip-grad 1.0                    # 梯度裁剪不够

# 🔧 稳定化配置
--gumbel-temperature-range 4 1.0  # 终点从 0.2 → 1.0（避免过度确定性）
--lr-mult 8                       # 从 10 降至 8（稍慢一点）
--clip-grad 0.5                   # 从 1.0 降至 0.5（更激进裁剪）
--weight-reg 1e-6                 # 从 2e-6 降至 1e-6（降低 reg loss 压力）
```

#### **训练策略**

1. **从 iter 1125 checkpoint 恢复**（最后的稳定点）
2. **重置 RNG 和优化器**（`--no-load-optim --no-load-rng`）
3. **改变 seed**（避免相同的数值陷阱）
4. **更频繁保存**（每 10 iters，快速定位问题）

#### **预期效果**

```
Temperature: 1.864 → 1.0 (iter 1125 → 2000)
Mask difference: 0.112 → 0.08-0.10
Reg loss 增速: +6.6 → +4.0/iter
NaN 风险: 30% → <10%
```

---

### **方案 B: FP32 激进加速（备选）**

**仅在方案 A 失败后考虑**

```bash
# 从 FP32 iter 650 恢复，使用更激进参数
--gumbel-temperature-range 3 0.5  # 从 6 2.0 改为 3 0.5
--lr-mult 3                       # 从 1 改为 3
--clip-grad 0.8                   # 从 0.3 改为 0.8
--prior-strength 1.5              # 从 3.0 改为 1.5
```

**预期**:
- Temperature 更快降至 0.5
- Mask 在 iter 1500 左右开始有效优化
- 完全无 NaN

---

## 📝 **实施计划**

### **Phase 1: BF16 稳定化训练 (推荐)**

#### **脚本名称**
`llama8b_presparse_training_tp8_stable_resume.sh`

#### **关键配置**
```bash
RESUME_FLAG=1  # 从 iter 1125 恢复
--load $TRAINING_CHECKPOINT  # 加载 iter_0001125
--no-load-optim --no-load-rng  # 重置优化器和 RNG
--seed 48  # 新 seed (vs 之前的 46)
--save-interval 10  # 更频繁保存
--gumbel-temperature-range 4 1.0
--lr-mult 8
--clip-grad 0.5
--weight-reg 1e-6
--bf16
```

#### **训练目标**
```
起点: iter 1125
目标: iter 2000 (剩余 875 iters)
预计时间: 875 × 220秒 = 53.5小时 (2.2天)
```

#### **成功标准**
- ✅ 无 NaN（前 100 iters 最关键）
- ✅ Mask difference 稳定下降
- ✅ Reg loss 增速 < +5/iter
- ✅ Temperature 平滑降至 1.0

#### **失败标准**
- ❌ 在 iter 1150-1200 再次 NaN
- ❌ Mask difference 不变或上升

---

### **Phase 2: 失败后的备选方案**

如果 Phase 1 在 iter 1150-1200 再次 NaN：

#### **选项 2A: 进一步放慢 BF16**
```bash
--gumbel-temperature-range 4 1.5  # 更保守的终点
--lr-mult 5                       # 更慢的 mask LR
--clip-grad 0.3                   # 更激进的裁剪
```

#### **选项 2B: 切换到 FP32 激进模式**
使用方案 B 的参数，从 FP32 iter 650 恢复

---

## 📊 **训练监控指标**

### **前 50 次迭代（iter 1125-1175）**

**关键指标监控**:
```bash
# 实时监控
tail -f output/logs/llama8b_presparse_training/training_*.log | grep "iteration"

# 检查点
1. Grad Norm 是否稳定 (<0.10)
2. Reg loss 增速是否 <+5/iter
3. Mask difference 是否继续下降
4. 无 NaN warning
```

**预警信号**:
- ⚠️ Grad norm > 0.15
- ⚠️ Reg loss 单次跳变 >50
- ⚠️ 任何 "NaN" 或 "Inf" 日志

---

## 🎯 **预期训练轨迹**

### **BF16 稳定化（方案 A）**

| Iteration | Temp | Max Prob | Mask Diff | Reg Loss | NaN Risk |
|-----------|------|----------|-----------|----------|----------|
| 1125 (起点) | 1.864 | 0.680 | 0.112 | 9467 | - |
| 1200 | 1.75 | 0.70 | 0.105 | 9800 | 🟡 中 |
| 1300 | 1.60 | 0.72 | 0.098 | 10,200 | 🟢 低 |
| 1500 | 1.35 | 0.78 | 0.088 | 11,000 | 🟢 极低 |
| 1700 | 1.15 | 0.82 | 0.082 | 11,800 | 🟢 极低 |
| 2000 | 1.00 | 0.85 | 0.078 | 12,600 | 🟢 极低 |

**总结**:
- ✅ 平滑过渡危险期 (1125-1200)
- ✅ Mask 持续优化
- ✅ Reg loss 可接受增长
- ✅ 训练速度快

---

## 🔧 **下一步行动**

### **立即执行（优先级 1）**

1. **创建稳定化脚本** `llama8b_presparse_training_tp8_stable_resume.sh`
2. **停止当前所有训练**（如果还在运行）
3. **验证 iter 1125 checkpoint 完整性**
4. **启动 BF16 稳定化训练**

### **监控（优先级 1）**

前 50 次迭代**每 10 次**检查一次日志：
```bash
# 快速检查最近 10 条
tail -20 output/logs/llama8b_presparse_training/training_*.log | grep "iteration"
```

### **决策点（优先级 2）**

- **iter 1150**: 如果无 NaN，继续；如果 NaN，立即停止，评估选项 2A
- **iter 1200**: 如果稳定，标记成功；如果不稳定，评估选项 2B
- **iter 1500**: 评估 Mask 优化效果，决定是否需要调整参数

---

## 💾 **备份与回滚计划**

### **关键 Checkpoint**
```
iter_0001125  - 稳定起点（保留）
iter_0001150  - 第一危险期通过（保留）
iter_0001200  - 危险期结束（保留）
iter_0001500  - 中期评估点（保留）
```

### **磁盘空间管理**
```bash
# 清理 iter 1125 之前的旧 checkpoint（除了 1100, 1050, 1000）
rm -rf output/checkpoints/llama8b_maskllm_training_tp8/iter_000{0025..0975}
rm -rf output/checkpoints/llama8b_maskllm_training_tp8/iter_000{1025,1075}
```

---

## ✅ **成功标准总结**

### **短期成功（iter 1125-1200）**
- ✅ 无 NaN 错误
- ✅ Grad norm < 0.10
- ✅ Mask difference 开始下降

### **中期成功（iter 1200-1500）**
- ✅ Temperature 平滑降至 1.5
- ✅ Mask difference < 0.10
- ✅ Reg loss 增速 < +5/iter

### **长期成功（iter 1500-2000）**
- ✅ Temperature 达到 1.0
- ✅ Mask difference 稳定在 0.07-0.08
- ✅ Reg loss < 13000
- ✅ LM loss 继续下降

---

**计划完成时间**: 2025-10-07  
**推荐方案**: Phase 1 - BF16 稳定化训练  
**备选方案**: FP32 激进加速（如需）  
**预计完成时间**: 2.2 天（BF16）或 4.5 天（FP32）

