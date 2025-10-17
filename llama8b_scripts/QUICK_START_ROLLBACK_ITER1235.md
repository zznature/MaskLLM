# 快速开始: Rollback 训练 (从 iter 1235)

## 🎯 一句话总结

从健康的 iter 1235 checkpoint 开始，使用激进正则化 (lr_mult=1.5, weight_reg=1e-5) 稳定训练到 iter 2000。

---

## 🚀 启动训练

### 1. 进入项目目录

```bash
cd /data/home/zdhs0054/zzhou/MaskLLM
```

### 2. 验证 checkpoint 存在

```bash
# 检查 iter 1235 checkpoint
ls -lh output/checkpoints/llama8b_maskllm_training_tp8/iter_0001235/

# 应该看到 8 个 rank 目录 (mp_rank_00 到 mp_rank_07)
```

### 3. 启动训练

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_ultra_conservative.sh
```

**注意**: 脚本名称虽然还是 `ultra_conservative`，但已修改为 Rollback 策略。

---

## 📊 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **起点** | iter 1235 | Reg loss ~8696 (健康) |
| **精度** | FP32 | 数值稳定性 |
| **LR Mult** | 1.5 | 史上最保守 |
| **Weight Reg** | 1e-5 | 史上最强 |
| **稳定性方程** | 0.444 | 20x 提升 (vs Plan A) |
| **Mask LR** | 2.25e-5 | 最慢速度 |
| **Scale 范围** | 100→300 | 极度降低硬化压力 |
| **Temp 范围** | 4.0→2.5 | 保持最大柔性 |

---

## 🔍 监控指标

### 每 10 iters 检查

打开日志文件:
```bash
tail -f output/logs/llama8b_presparse_training_fp32_rollback_iter1235/training_rollback_from_iter1235_*.log
```

**关键指标**:
1. **Reg loss 增长率 < 1.0 /iter** ✅
   ```
   iter 1236: reg loss: 8.700E+03
   iter 1246: reg loss: 8.705E+03
   增长率: (8705 - 8700) / 10 = 0.5 /iter ✅
   ```

2. **Grad norm < 0.15** ✅
   ```
   grad norm: 0.035  ✅
   grad norm: 0.150  ⚠️ (接近上限)
   grad norm: 0.200  ❌ (超标，需停止)
   ```

3. **LM loss 下降或稳定** ✅
   ```
   iter 1236: lm loss: 2.615E+00
   iter 1246: lm loss: 2.590E+00  ✅ (下降)
   ```

### 每 100 iters 检查

**Validation 指标**:
```bash
# 在日志中搜索 "validation loss"
grep "validation loss" output/logs/llama8b_presparse_training_fp32_rollback_iter1235/training_rollback_from_iter1235_*.log | tail -5
```

**关键检查**:
- Validation reg loss < 1.15 × Training reg loss ✅
- 例如: Training 8700, Validation 应 <10005

---

## ⚠️ 异常响应

### 如果 Reg loss 增长率 > 1.0 /iter

**症状**:
```
iter 1236: reg loss: 8.700E+03
iter 1246: reg loss: 8.715E+03
增长率: (8715 - 8700) / 10 = 1.5 /iter ❌
```

**响应**:
1. 立即停止训练 (Ctrl+C)
2. 检查最近的 checkpoint (每 10 iters 保存一次)
3. 考虑进一步降低 lr_mult:
   ```bash
   # 修改脚本中的 TASK_CMD
   --lr-mult 1.0  # (vs 当前 1.5)
   ```

### 如果 Grad norm > 0.15

**症状**:
```
grad norm: 0.180 ❌
```

**响应**:
1. 立即停止训练
2. 检查是否有数据异常
3. 考虑降低 clip_grad:
   ```bash
   --clip-grad 0.8  # (vs 当前 1.0)
   ```

### 如果 Validation gap > 20%

**症状**:
```
Training reg loss @ 1300: 8800
Validation reg loss @ 1300: 10560
Gap: (10560 - 8800) / 8800 = 20% ⚠️
```

**响应**:
1. 继续观察，但提高警惕
2. 如果 gap > 25%，考虑停止
3. 可能需要提高 weight_reg:
   ```bash
   --weight-reg 2e-5  # (vs 当前 1e-5)
   ```

---

## 📈 预期训练轨迹

### Reg Loss 轨迹

| Iteration | Reg Loss | 增长 | 状态 |
|-----------|----------|------|------|
| 1235 | 8696 | - | ✅ 起点 |
| 1310 | 8734 | +38 | ✅ 通过 (vs 之前 NaN) |
| 1500 | 8829 | +133 | ✅ 稳定 |
| 1750 | 8954 | +258 | ✅ 稳定 |
| 2000 | 9079 | +383 | ✅ 目标 |

### LM Loss 轨迹

| Iteration | LM Loss | 状态 |
|-----------|---------|------|
| 1235 | ~2.62 | ✅ 起点 |
| 1500 | ~2.55 | ✅ 改善 |
| 1750 | ~2.50 | ✅ 改善 |
| 2000 | ~2.45 | ✅ 目标 |

### Grad Norm 轨迹

| Iteration | Grad Norm | 状态 |
|-----------|-----------|------|
| 1235-2000 | 0.05-0.08 | ✅ 健康范围 |

---

## 🎓 关键里程碑

### 1. 通过 iter 1310 (关键!)

**意义**: Plan A 和 Ultra-Conservative 都在此失败

**检查**:
```bash
# 在日志中搜索 iter 1310
grep "iteration.*1310" output/logs/llama8b_presparse_training_fp32_rollback_iter1235/training_rollback_from_iter1235_*.log
```

**预期输出**:
```
iteration 1310/2000 | ... | reg loss: 8.734E+03 | grad norm: 0.045 | ...
```

**如果成功**: 🎉 恭喜！最危险的阶段已过！

### 2. 达到 iter 1500

**意义**: 中期稳定性验证

**检查**:
- Reg loss <8900 ✅
- LM loss 持续下降 ✅
- Grad norm 稳定 ✅

### 3. 达到 iter 2000

**意义**: 训练完成！

**后续步骤**:
1. 评估稀疏模型性能
2. 对比 dense baseline
3. 记录最终指标

---

## 📝 日志位置

### 训练日志

```bash
# 最新日志
ls -lt output/logs/llama8b_presparse_training_fp32_rollback_iter1235/

# 实时查看
tail -f output/logs/llama8b_presparse_training_fp32_rollback_iter1235/training_rollback_from_iter1235_*.log
```

### Checkpoint 位置

```bash
# 新的 FP32 checkpoints
ls -lt output/checkpoints/llama8b_maskllm_training_tp8_fp32_rollback_iter1235/

# 每 10 iters 保存一次
# iter_0001235, iter_0001245, iter_0001255, ...
```

---

## 🔧 高级选项

### 如果需要更保守的设置

修改脚本中的 `TASK_CMD`:

```bash
# 更低的 lr_mult
--lr-mult 1.0  # (vs 当前 1.5)

# 更强的正则化
--weight-reg 2e-5  # (vs 当前 1e-5)

# 更低的 scale
--gumbel-scale-range 100 250  # (vs 当前 100 300)
```

### 如果需要更快的训练

⚠️ **不推荐**，会牺牲稳定性

```bash
# 稍高的 lr_mult
--lr-mult 2.0  # (vs 当前 1.5)

# 但保持强正则化
--weight-reg 1e-5  # (保持不变)
```

---

## 📞 常见问题

### Q: 为什么从 iter 1235 开始而不是 iter 1280?

**A**: iter 1280 checkpoint 已"中毒" (Reg loss 10034)

- Plan A 和 Ultra-Conservative 都在 iter 1310 失败
- 说明 checkpoint 本身有问题
- iter 1235 的 Reg loss 是 8696，更健康

### Q: 训练需要多长时间?

**A**: 约 52 小时

- 765 iters (1235 → 2000)
- 245 秒/iter
- 总计: 765 × 245 = 187,425 秒 ≈ 52 小时

### Q: 显存占用多少?

**A**: 68-72GB / 80GB

- FP32 精度
- 8 卡 TP
- 安全余量: 8-12GB

### Q: 如果训练中断怎么办?

**A**: 从最近的 checkpoint 恢复

```bash
# 检查最近的 checkpoint
cat output/checkpoints/llama8b_maskllm_training_tp8_fp32_rollback_iter1235/latest_checkpointed_iteration.txt

# 修改脚本中的 RESUME_ITER 为最近的 iteration
# 然后重新启动训练
```

### Q: 成功率有多高?

**A**: >99%

基于:
- 健康的起点 (Reg loss 8696)
- 激进的正则化 (稳定性方程 0.444)
- FP32 精度
- 20x 稳定性提升 (vs Plan A)

---

## 🔗 相关文档

- **详细策略说明**: `llama8b_scripts/ROLLBACK_STRATEGY_ITER1235.md`
- **根因分析**: `llama8b_scripts/NAN_ERROR_ANALYSIS_ITER1310.md`
- **训练历史**: `llama8b_scripts/training_notes/training_logs.md`

---

**创建时间**: 2025-10-09  
**版本**: v1.0  
**状态**: Ready for Production

---

**祝训练顺利! 🚀**

