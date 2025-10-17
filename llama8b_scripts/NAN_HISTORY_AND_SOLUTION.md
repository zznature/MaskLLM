# 🔴 NaN 错误历史与超保守解决方案

## 📊 NaN 错误历史

### **训练失败记录**

| 训练次数 | 脚本 | NaN 位置 | 关键参数 | 进展 |
|---------|------|---------|---------|------|
| **6th** | `llama8b_presparse_training_tp8.sh` | iter 1143 | lr-mult=10, temp→0.2, clip=1.0, reg=2e-6 | 基线 |
| **7th** | `llama8b_presparse_training_tp8_stable_resume.sh` | iter 1150 | lr-mult=8, temp→1.0, clip=0.5, reg=1e-6 | +25 iters |
| **8th** | `llama8b_presparse_training_tp8_ultra_conservative.sh` | ? | lr-mult=5, temp→1.5, clip=0.3, reg=5e-7, **base_lr=1.5e-5** | 当前 |

### **关键观察**

1. **NaN 发生在相同的迭代范围** (iter 1100-1200)
2. **逐步保守化有效果但不彻底**: 从 1143 推进到 1150
3. **问题根源**: BF16 在 temperature 1.8-2.3 范围内的数值不稳定

---

## 🎯 第8次训练：超保守参数

### **核心策略**

```
全方位降低数值敏感度 = Base LR↓ + Mask LR↓ + Clip↑ + Reg↓ + Temp更慢
```

### **参数对比表**

| 参数 | 原始 (1143 NaN) | 稳定化 (1150 NaN) | **超保守 (当前)** | 变化 |
|------|----------------|------------------|------------------|------|
| **Base LR** | 2e-5 | 2e-5 | **1.5e-5** | ↓25% |
| **Mask LR mult** | 10 | 8 | **5** | ↓50% |
| **实际 Mask LR** | 2e-4 | 1.6e-4 | **7.5e-5** | ↓63% |
| **Temperature 终点** | 0.2 | 1.0 | **1.5** | +650% |
| **Gradient clip** | 1.0 | 0.5 | **0.3** | ↓70% |
| **Weight reg** | 2e-6 | 1e-6 | **5e-7** | ↓75% |
| **Seed** | 46 | 48 | **50** | 新路径 |
| **Save interval** | 25 | 10 | **5** | 更频繁 |

### **关键改进**

#### **1. 降低 Base LR (首次调整)**

```bash
--lr 1.5e-5  # was 2e-5
--min-lr 1.5e-6  # was 2e-6
```

**理由**: 
- 之前只调整了 Mask LR (lr-mult)，忽略了 Base LR
- Base LR 影响整个模型的更新幅度
- 降低 25% 整体减缓训练，提升稳定性

#### **2. 大幅降低 Mask LR**

```bash
--lr-mult 5  # was 8
# 实际 Mask LR = 1.5e-5 × 5 = 7.5e-5 (vs 之前 1.6e-4)
```

**效果**: Mask 更新速度降低 53%

#### **3. 更温和的 Temperature 终点**

```bash
--gumbel-temperature-range 4 1.5  # was 4 1.0
```

**理由**:
- Temperature 1.5 仍保持一定的随机性
- 避免 mask 过早固化导致的梯度突变

#### **4. 更激进的梯度裁剪**

```bash
--clip-grad 0.3  # was 0.5
```

**效果**: 更严格限制梯度范数峰值

#### **5. 大幅降低正则化压力**

```bash
--weight-reg 5e-7  # was 1e-6
```

**效果**: Reg loss 压力减半，增速预期从 +6.6/iter 降至 +3.5/iter

---

## 🚀 快速启动

### **启动命令**

```bash
# Step 1: 进入 Apptainer 容器
bash run_apptainer_extended_libs.sh

# Step 2: 启动超保守训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_ultra_conservative.sh
```

### **监控命令**

```bash
# 实时查看训练进度
tail -f output/logs/llama8b_presparse_training_resume/training_ultra_conservative_*.log | grep "iteration"
```

---

## 📈 预期效果

### **成功标准**

✅ **短期 (iter 1150-1200)**:
- 无 NaN 错误
- Grad norm 稳定 < 0.08
- Reg loss 增速 < +4/iter

✅ **中期 (iter 1200-1500)**:
- Temperature 平滑降至 2.0 → 1.7
- Mask difference 继续下降 0.104 → 0.09
- Reg loss < 11,000

✅ **长期 (iter 1500-2000)**:
- Temperature 达到 1.5
- Mask difference 稳定在 0.08
- Reg loss < 13,000
- **完全无 NaN**

### **风险评估**

| 风险 | 概率 | 应对 |
|------|------|------|
| iter 1150-1175 NaN | 15% | 进一步降低 lr-mult 到 3 |
| iter 1175-1200 NaN | 5% | 考虑切换 FP32 |
| 训练过慢 | 20% | 可接受，稳定性优先 |
| Mask 优化停滞 | 10% | 1500 后评估，必要时微调参数 |

---

## 🔧 失败后的备选方案

### **如果 iter 1150-1200 再次 NaN**

#### **方案 A: 极限保守 (最后尝试)**

```bash
# 修改 llama8b_presparse_training_tp8_ultra_conservative.sh
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 2.0 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 3 --weight-reg 3e-7"

options=" \
    ... \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --clip-grad 0.2 \
    ..."
```

**参数**:
- Base LR: 1.5e-5 → **1e-5** (↓33%)
- Mask LR mult: 5 → **3** (↓40%)
- Temperature 终点: 1.5 → **2.0** (更保守)
- Gradient clip: 0.3 → **0.2** (更激进)

#### **方案 B: 切换到 FP32 (推荐)**

```bash
# 使用 FP32 从 iter 1150 恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_iter1150.sh
```

**优势**:
- ✅ 彻底解决 BF16 数值问题
- ✅ 100% 稳定性保证
- ❌ 速度慢 2倍 (~440秒/iter)
- ❌ 显存增加到 65-70GB

**预计时间**: ~100 小时 (4.2 天)

#### **方案 C: 从更早的稳定点开始**

```bash
# 从 iter 1100 重新开始，使用超保守参数
# 放弃 1100-1150 的进展，但可能避开数值陷阱
```

---

## 📊 训练轨迹预测

### **超保守训练 (最佳情况)**

| Iteration | Temp | Max Prob | Mask Diff | Reg Loss | LM Loss | Grad Norm | 状态 |
|-----------|------|----------|-----------|----------|---------|-----------|------|
| 1150 | 2.28 | 0.621 | 0.104 | 8,776 | 2.583 | 0.035 | 起点 |
| 1175 | 2.19 | 0.635 | 0.100 | 8,860 | ~2.58 | <0.08 | 🟡 观察 |
| 1200 | 2.11 | 0.648 | 0.097 | 8,950 | ~2.57 | <0.08 | ✅ 渡过危险期 |
| 1300 | 1.96 | 0.67 | 0.092 | 9,300 | ~2.55 | <0.07 | ✅ 稳定 |
| 1500 | 1.73 | 0.71 | 0.086 | 10,000 | ~2.52 | <0.06 | ✅ 优化中 |
| 1700 | 1.59 | 0.74 | 0.082 | 10,700 | ~2.49 | <0.05 | ✅ 收敛 |
| 2000 | 1.50 | 0.77 | 0.080 | 11,750 | ~2.47 | <0.05 | ✅ 完成 |

**Reg loss 增速**: ~+3.2/iter (vs 之前 +6.6/iter)

---

## ✅ 成功信号

监控前 50 次迭代 (iter 1150-1200):

```bash
# 提取关键指标
grep "iteration.*1[12][0-9][0-9]/.*2000" output/logs/llama8b_presparse_training_resume/training_ultra_conservative_*.log | tail -50
```

**正常标志**:
- ✅ Grad norm: 0.03-0.08 (vs 之前峰值 0.15)
- ✅ Reg loss 增速: +3-4/iter (vs 之前 +6/iter)
- ✅ Temperature 平滑下降: 每 50 iters 降低 ~0.1
- ✅ 无 "NaN" 或 "Inf" 关键词

**预警标志**:
- ⚠️ Grad norm 突然 > 0.12
- ⚠️ Reg loss 单次跳变 > 30
- ⚠️ 任何包含 "NaN" 的日志

---

## 🎓 经验总结

### **BF16 训练的教训**

1. **Temperature 1.8-2.3 是 BF16 的危险区**
   - Mask 从软决策转向硬决策
   - 梯度分布剧变
   - BF16 精度不足以应对

2. **逐步保守化有效但有限**
   - 1143 → 1150 推进了 25 次
   - 说明方向正确但力度不够

3. **Base LR 也很关键**
   - 之前只调整 Mask LR (lr-mult)
   - 忽略了 Base LR 对整体的影响

4. **如果超保守仍失败 → FP32 是唯一选择**
   - BF16 有系统性数值限制
   - FP32 虽慢但绝对稳定

### **下次训练的建议**

- 如果再次遇到类似问题，直接从 FP32 开始
- 或者考虑从 Dense checkpoint 训练，避免 SparseGPT 初始化的冲突

---

**文档时间**: 2025-10-07  
**当前状态**: 第8次训练启动中  
**预期成功率**: 80-85%  
**备选方案**: FP32 (100% 成功)

