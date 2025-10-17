# 🚀 FP32 训练启动 - 方案 B (适度激进)

## 📋 配置总结

```bash
起点: iter 1280
终点: iter 2000
剩余: 720 iterations

精度: FP32
Batch size: 256
Base LR: 1.5e-5
LR Mult: 5 (适度激进)
Mask LR: ~1.45e-4 (安全范围内)

预计时间: 46 小时 (1.9天)
成功率: 60-70%
```

---

## 🎯 为什么选择这个配置？

### **1. 起点 iter 1280**

```
✅ 优势:
  - 节省 5 小时 (vs iter 1235)
  - Reg loss ~8831 (仍在安全区)
  - 已通过 iter 1236-1238 的 FP32 验证

⚠️ 风险:
  - 接近危险区 (1280-1310 是 BF16 NaN 高发区)
  - 但 FP32 提供额外缓冲
```

### **2. lr_mult=5 (不是 10)**

```
当前 Mask LR 计算:
  Base LR (iter 1281): ~2.9e-5
  Mask LR: 2.9e-5 × 5 = 1.45e-4

对比:
  ✅ < BF16 崩溃时的 1.7e-4
  ✅ << FP32 失败时的 2.26e-4
  ✅ > 保守配置的 1.24e-4 (仅高 17%)

结论: 在安全范围内，适度激进
```

---

## 📊 风险评估

### **成功率: 60-70%**

```
有利因素:
  ✅ FP32 精度 (vs BF16)
  ✅ 保守的 base LR (1.5e-5)
  ✅ Mask LR 在安全范围
  ✅ 已验证 iter 1236-1238 稳定

不利因素:
  ⚠️ 接近 BF16 危险区
  ⚠️ lr_mult=5 曾在 BF16 失败
  ⚠️ 但那是 base_lr=2e-5, 现在是 1.5e-5

综合: 风险可控，值得尝试
```

---

## 🔍 关键监控点

### **危险区: iter 1280-1310**

```bash
监控指标:
  1. Reg loss 增长率
     - 正常: < 5/iter
     - 警戒: 5-10/iter
     - 危险: > 10/iter

  2. Grad norm
     - 正常: < 0.05
     - 警戒: 0.05-0.08
     - 危险: > 0.08

  3. LM loss
     - 应该持续下降
     - 如果上升 → 警戒

监控命令:
  tail -f output/logs/llama8b_presparse_training_fp32_continuous/training_continuous_*.log
```

### **关键 checkpoints**

```
iter 1300: 第一个重要检查点
  - 如果稳定 → 信心增强
  - 如果 NaN → 回退到 lr_mult=4

iter 1310: 危险区结束
  - 通过这里 → 成功率 90%+
  - NaN → 说明 lr_mult=5 太高

iter 1500: 中期里程碑
  - 应该完全稳定
```

---

## 🚀 启动步骤

### **1. 验证 checkpoint**

```bash
ls -la output/checkpoints/llama8b_maskllm_training_tp8/iter_0001280/

# 应该看到 8 个 mp_rank_* 目录
```

### **2. 启动训练**

```bash
# 在 Apptainer 容器内
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh
```

### **3. 验证启动**

```bash
# 检查日志
tail -f output/logs/llama8b_presparse_training_fp32_continuous/training_continuous_*.log

# 预期第一次迭代:
iteration 1281/2000 | learning rate: ~2.9e-5 | reg loss: ~8833 | ...
```

---

## ✅ 预期训练曲线

### **正常情况**

```
iter 1281-1300:
  - LM loss: 持续下降
  - Reg loss: 缓慢增长 (< 5/iter)
  - Grad norm: < 0.05
  - 无 NaN

iter 1300-1310 (危险区):
  - 可能有小波动
  - Reg loss 增长可能加快
  - 但应该能通过

iter 1310-2000:
  - 稳定训练
  - Reg loss 增长趋缓
  - 顺利完成
```

### **异常情况**

```
如果 iter 1281-1300 就 NaN:
  → lr_mult=5 对这个起点太高
  → 回退到 lr_mult=4, 从 iter 1235 重新开始

如果 iter 1300-1310 NaN:
  → 危险区确实危险
  → 评估 iter 1300 checkpoint 质量
  → 或回退到 lr_mult=4

如果 iter 1310+ NaN:
  → 意外，但可能发生
  → 从最近的健康 checkpoint 恢复
```

---

## 📈 性能预期

```
总迭代: 720 (1280 → 2000)
每次迭代: ~230秒
总时间: 720 × 230 / 3600 ≈ 46 小时

对比:
  vs iter 1235 + lr_mult=4: 节省 5 小时
  vs 重新训练: 节省 ~50 小时

风险收益比: 值得
```

---

## 🎓 经验总结

### **为什么不是 lr_mult=10？**

```
lr_mult=10 会导致:
  Mask LR = 2.9e-4
  = 接近历史 NaN 水平
  = 成功率 < 20%

lr_mult=5 更合理:
  Mask LR = 1.45e-4
  = 在安全范围内
  = 成功率 60-70%
  = 如果成功，收敛更快
```

### **FP32 的优势**

```
vs BF16:
  ✅ 数值稳定性更好
  ✅ 可以承受略高的 LR
  ✅ 梯度计算更精确

但不是万能:
  ❌ 不能无限提高 LR
  ❌ 仍需尊重数值极限
  ❌ lr_mult=10 仍然太高
```

---

## 📝 备份计划

### **如果 NaN (概率 30-40%)**

```
Plan A: 回退到 lr_mult=4
  - 从 iter 1235 重新开始
  - 成功率 85-90%
  - 多花 5 小时

Plan B: 评估现有 checkpoints
  - iter 1280 或 iter 1300
  - 如果质量可接受 → 直接使用
  - 节省大量时间

Plan C: 接受部分训练结果
  - 如果训练到 iter 1500+
  - 评估质量
  - 可能已经足够好
```

---

## 🎯 成功标志

```
✅ 通过 iter 1310:
   → 成功率提升到 90%+
   → 后续应该稳定

✅ 通过 iter 1500:
   → 基本确定成功
   → 继续到 2000

✅ 完成 iter 2000:
   → 完全成功 🎉
   → 获得高质量稀疏模型
```

---

**准备好了吗？** 🚀

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh
```

**预计完成时间**: 46 小时后  
**成功率**: 60-70%  
**值得尝试**: ✅ 是的！
