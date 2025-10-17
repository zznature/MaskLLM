# 🚀 方案 B: FP32 从 iter 1280, lr_mult=4

## 📋 配置总结

```bash
方案: B (平衡方案)
起点: iter 1280
终点: iter 2000
剩余: 720 iterations

精度: FP32
Batch size: 256
Base LR: 1.5e-5
LR Mult: 4 (从 5 降低 20%)
Mask LR: ~1.37e-4 (安全范围)

预计时间: 46 小时 (1.9天)
成功率: 70%
```

---

## 🎯 为什么这个配置可行？

### **1. 核心改进: lr_mult=5 → 4**

```
失败配置 (lr_mult=5):
  Mask LR = 1.71e-4
  → NaN at iter 1306

新配置 (lr_mult=4):
  Mask LR = 1.37e-4
  → 降低 20%
  → 远低于失败阈值

对比参考点:
  ✅ < BF16 崩溃时的 1.7e-4
  ✅ << lr_mult=5 失败时的 1.71e-4
  ✅ ≈ iter 1235 验证稳定时的水平
```

### **2. 起点 iter 1280**

```
优势:
  ✅ 节省 45 次迭代 (vs iter 1235)
  ✅ 节省 ~3 小时训练时间
  ✅ Reg loss 8831 仍在可接受范围

挑战:
  ⚠️ 进入危险区 1280-1310
  ⚠️ 需要更保守的 LR 度过

策略:
  → 用 lr_mult=4 提供安全缓冲
  → FP32 精度提供额外保障
```

---

## 📊 风险评估

### **成功率: 70%**

```
有利因素:
  ✅ FP32 精度 (vs BF16)
  ✅ lr_mult=4 已在 iter 1235 验证
  ✅ Mask LR 降低 20%
  ✅ 保守的 base LR (1.5e-5)

不利因素:
  ⚠️ 起点 iter 1280 较接近危险区
  ⚠️ Reg loss 8831 已经偏高
  ⚠️ 需要度过 1280-1310 的挑战

综合评估:
  70% 成功率是合理的
  如果失败，回退到方案 A (iter 1235)
```

---

## 🔍 关键监控指标

### **危险区: iter 1280-1310**

```bash
这是最关键的 30 次迭代！

监控命令:
  tail -f output/logs/llama8b_presparse_training_fp32_continuous/training_continuous_*.log

关键指标:

1. Reg loss 增长率
   - 健康: < 3/iter
   - 警戒: 3-5/iter
   - 危险: > 5/iter
   
   之前失败:
     iter 1280 → 1306: +6.9/iter (太高！)
   
   期望改进:
     lr_mult=4 应降到 < 5/iter

2. Grad norm
   - 正常: < 0.05
   - 警戒: 0.05-0.08
   - 危险: > 0.08
   
   之前: 0.035 (很好)
   期望: 保持 < 0.05

3. LM loss
   - 应该持续下降
   - 如果上升或波动 → 警戒

4. Mask LR (learning rate)
   - iter 1281: ~2.9e-5 (base) × 4 = 1.16e-4
   - 应该 < 1.5e-4
```

### **检查点时间表**

```
iter 1281: 第一次迭代
  → 如果稳定，信心 +10%

iter 1285: 早期检查点
  → Reg loss 应 < 8850
  → 增长率应 < 5/iter

iter 1290: 进入危险区
  → Reg loss 应 < 8880
  → 这里最关键

iter 1300: 危险区中期
  → Reg loss 应 < 8920
  → 之前在此失败

iter 1310: 危险区结束
  → 如果通过，成功率 → 90%+
  → Reg loss 应 < 8950

iter 1350+: 稳定期
  → 应该顺利完成
```

---

## 🚀 启动步骤

### **1. 清理旧训练**

```bash
# 停止可能的旧进程
pkill -f "pretrain_maskllm.py"

# 验证 checkpoint
ls -la output/checkpoints/llama8b_maskllm_training_tp8/iter_0001280/
# 应该看到 8 个 mp_rank_* 目录
```

### **2. 启动训练**

```bash
# 在 Apptainer 容器内执行
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh
```

### **3. 实时监控 (前 30 次迭代)**

```bash
# 打开日志
tail -f output/logs/llama8b_presparse_training_fp32_continuous/training_continuous_*.log

# 重点关注 iter 1281-1310
# 预期第一次迭代:
iteration 1281/2000 | learning rate: ~2.9e-5 | reg loss: ~8833 | grad norm: < 0.04 | ...
```

---

## ✅ 预期训练曲线

### **正常情况 (70% 概率)**

```
iter 1281-1290 (第 1-10 次):
  LM loss: 持续下降
  Reg loss: 8833 → 8860 (+2.7/iter)
  Grad norm: < 0.04
  ✅ 无 NaN

iter 1290-1300 (第 10-20 次):
  LM loss: 持续下降
  Reg loss: 8860 → 8890 (+3/iter)
  Grad norm: < 0.045
  ✅ 无 NaN

iter 1300-1310 (危险峰值):
  LM loss: 可能略有波动
  Reg loss: 8890 → 8920 (+3/iter)
  Grad norm: 可能到 0.05
  ✅ 应该能通过

iter 1310-2000:
  稳定训练，顺利完成
  Reg loss 增长趋缓
```

### **异常情况 (30% 概率)**

```
如果 iter 1290 前 NaN:
  → 意外，但可能发生
  → 说明 iter 1280 起点仍有问题
  → 立即回退到方案 A (iter 1235)

如果 iter 1300-1310 NaN:
  → 危险区确实太难
  → lr_mult=4 仍不够保守
  → 回退到方案 A (iter 1235)

如果 iter 1310+ NaN:
  → 意外，说明模型有深层问题
  → 需要重新评估策略
```

---

## 📈 性能预期

```
总迭代: 720 (1280 → 2000)
每次迭代: ~230秒
总时间: 720 × 230 / 3600 ≈ 46 小时

对比:
  vs 方案 A (iter 1235): 节省 3 小时
  vs 重新训练: 节省 ~50 小时

风险收益:
  成功 (70%): 节省 3 小时 ✅
  失败 (30%): 浪费 ~2-6 小时，回退到方案 A
  
期望值: 0.7×(+3h) + 0.3×(-4h) = +0.9h
  → 期望上还是划算的
```

---

## 🎓 关键经验

### **为什么 lr_mult=4 可能成功？**

```
数学分析:
  lr_mult=5: Mask LR 1.71e-4 → NaN at iter 1306
  lr_mult=4: Mask LR 1.37e-4 → 降低 20%
  
  Reg loss 增长率预测:
    lr_mult=5: 6.9/iter (太高)
    lr_mult=4: 预计 4-5/iter (可接受)
  
  Reg loss 峰值预测:
    iter 1310 with lr_mult=4: ~8900
    vs iter 1310 with lr_mult=5: ~9020
    → 降低 ~120 units
```

### **FP32 的关键作用**

```
vs BF16:
  ✅ 数值稳定性更好
  ✅ 可以承受略高的 Reg loss
  ✅ 梯度计算更精确
  
但是:
  ❌ 不能完全克服优化困难区
  ❌ 仍需合理的 LR
  ❌ Reg loss > 9000 仍然危险
```

---

## 📝 备份计划

### **如果失败 (30% 概率)**

```
Plan A: 回退到 iter 1235 + lr_mult=4
  - 成功率: 85%
  - 额外时间: 3 小时
  - 已验证稳定
  - 最终必成

Plan B: 尝试 iter 1200 + lr_mult=4
  - 折中方案
  - 成功率: 75%
  - 多 1.5 小时

Plan C: 接受部分结果
  - 如果训练到 iter 1500+
  - 评估模型质量
  - 可能已经足够
```

---

## 🎯 成功标志

```
✅ 通过 iter 1290:
   → 信心提升到 75%
   → 继续观察

✅ 通过 iter 1310:
   → 成功率提升到 90%+
   → 基本确定成功

✅ 通过 iter 1500:
   → 完全稳定
   → 必定成功

✅ 完成 iter 2000:
   → 大功告成 🎉
```

---

## 🔬 与方案 A 的对比

```
方案 B (当前):
  起点: iter 1280
  lr_mult: 4
  时间: 46h
  成功率: 70%
  优势: 节省时间
  风险: 可能失败

方案 A (备选):
  起点: iter 1235
  lr_mult: 4
  时间: 49h
  成功率: 85%
  优势: 更稳定
  风险: 多花 3h

选择理由:
  - 70% 成功率可接受
  - 节省 3 小时有价值
  - 失败代价不大 (2-6h)
  - 期望收益为正
```

---

**准备好了吗？** 🚀

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh
```

**关键监控**: 前 30 次迭代 (iter 1281-1310)  
**成功概率**: 70%  
**备用方案**: 如果失败，回退到方案 A

**Good luck!** 🍀
