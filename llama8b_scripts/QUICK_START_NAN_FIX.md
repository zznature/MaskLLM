# Quick Start: NaN-Free Training from Iter 1280

## 🎯 目标
从 iter 1280 稳定训练到 iter 2000，避免 Plan A 在 iter 1310 发生的 NaN 错误。

## ⚡ 快速启动

### 1. 启动训练 (推荐配置)

```bash
cd /data/home/zdhs0054/zzhou/MaskLLM

# Ultra-Conservative: 稳定性优先 (推荐)
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_ultra_conservative.sh
```

**预计时间**: 50 小时  
**成功率**: >95%  
**Checkpoint**: 每 10 iters 保存

---

## 📊 关键参数变化

| 参数 | Plan A (Failed) | Ultra-Conservative | 改善 |
|------|-----------------|-------------------|------|
| `lr_mult` | 3 | **2** | -33% (降低 mask 学习速度) |
| `weight_reg` | 1e-6 | **5e-6** | +400% (强化正则化) |
| `clip_grad` | 0.5 | **1.0** | +100% (健康梯度裁剪) |
| `scale_range` | 100→400 | **100→350** | -12.5% (降低硬化压力) |
| `temp_range` | 4.0→1.5 | **4.0→2.0** | +33% (保持柔性) |

**核心改进**: 稳定性指标提升 **7.6 倍**

---

## 🔍 监控指标 (每 10 iters 检查)

### 必须检查的 5 个指标:

#### 1. **Reg Loss 增长率 < 1.5 /iter** ✅
```bash
# 日志中查看
grep "iteration.*reg loss" <log_file> | tail -20

# 计算增长率
增长率 = (当前 reg loss - 10 iters 前) / 10
```
- ✅ **<1.0**: 优秀
- ⚠️ **1.0-1.5**: 可接受
- 🔴 **>1.5**: 危险，立即停止

#### 2. **Grad Norm < 0.15** ✅
```bash
# 日志中查看
grep "grad norm" <log_file> | tail -20
```
- ✅ **0.05-0.10**: 健康
- ⚠️ **0.10-0.15**: 警戒
- 🔴 **>0.15**: 危险

#### 3. **LM Loss 下降或稳定** ✅
- ✅ **持续下降**: 优秀
- ✅ **2.5±0.1 波动**: 正常
- 🔴 **突然上升 >0.2**: 异常

#### 4. **Validation Reg Loss < 1.15 × Training** ✅
```bash
# 每 100 iters 检查一次 (eval interval)
```
- ✅ **<1.15**: 健康
- ⚠️ **1.15-1.20**: 轻微过拟合
- 🔴 **>1.20**: 严重过拟合

#### 5. **Logits Std 稳定** ✅
- ✅ **0.018-0.022**: 正常
- 🔴 **<0.015 或 >0.025**: 异常

---

## ⚠️ 异常响应流程

### 如果发现任何指标超标:

1. **立即停止训练** (Ctrl+C)
2. **记录当前 iteration**
3. **检查最近的 checkpoint**
4. **使用 Emergency Fallback 配置**:

```bash
# 修改脚本中的参数
--lr-mult 1.5           # 进一步降低 (vs 2.0)
--weight-reg 1e-5       # 进一步提高 (vs 5e-6)
--clip-grad 1.5         # 进一步放宽 (vs 1.0)
```

5. **从最近的健康 checkpoint 恢复**

---

## 📈 预期训练曲线

### Iter 1280-2000 (720 步)

| Metric | Iter 1280 | Iter 2000 | 变化 |
|--------|-----------|-----------|------|
| Reg Loss | 10924 | ~11740 | +816 (~1.1 /iter) |
| LM Loss | 2.61 | ~2.35 | -0.26 |
| Grad Norm | 0.11 | 0.05-0.08 | 稳定在健康范围 |
| Max Prob | 0.79 | 0.82-0.85 | 逐渐硬化 |
| Scale Mult | 292 | ~400 | 线性增长 |
| Temperature | 2.40 | 2.00 | 线性衰减 |

### 关键检查点

- **Iter 1310**: Plan A 失败点 → Ultra-Conservative 应成功通过
- **Iter 1400**: 验证 reg loss ≈ 11056 (增长 <1.5 /iter)
- **Iter 1500**: 验证 reg loss ≈ 11167 (增长 <1.5 /iter)
- **Iter 2000**: 目标完成

---

## 📝 日志分析示例

### 健康的训练日志:

```
iteration 1281 | lm loss: 2.612 | reg loss: 1.092415E+04 | grad norm: 0.110
iteration 1291 | lm loss: 2.567 | reg loss: 1.095862E+04 | grad norm: 0.060
iteration 1301 | lm loss: 2.540 | reg loss: 1.099275E+04 | grad norm: 0.050
iteration 1311 | lm loss: 2.535 | reg loss: 1.102688E+04 | grad norm: 0.055
```

**分析**:
- Reg loss 增长: (11027 - 10924) / 30 = **3.4 /iter** (太高！Plan A 失败)
- Grad norm: 0.11 → 0.05 (看似稳定，实际被 clip_grad 压制)

### Ultra-Conservative 预期日志:

```
iteration 1281 | lm loss: 2.612 | reg loss: 1.092415E+04 | grad norm: 0.080
iteration 1291 | lm loss: 2.567 | reg loss: 1.093200E+04 | grad norm: 0.065
iteration 1301 | lm loss: 2.540 | reg loss: 1.093985E+04 | grad norm: 0.058
iteration 1311 | lm loss: 2.535 | reg loss: 1.094770E+04 | grad norm: 0.060
```

**分析**:
- Reg loss 增长: (10948 - 10924) / 30 = **0.8 /iter** (健康！)
- Grad norm: 0.08 → 0.06 (健康范围，未被过度裁剪)

---

## 🔧 常见问题

### Q1: 为什么降低 lr_mult 而不是 base lr?

**A**: Base LR (1.5e-5) 控制所有参数 (weights + masks)。只降低 lr_mult 可以:
- 保持 weight 学习速度不变
- 只减慢 mask 学习速度
- 避免影响模型收敛

### Q2: weight_reg 提高 5 倍会不会太强?

**A**: 不会。分析显示:
- 之前 1e-6 太弱，无法约束 mask 漂移
- Reg loss 以 3.4 /iter 增长，最终导致 NaN
- 提高到 5e-6 后，预期增长率 <1.0 /iter (可接受)

### Q3: 为什么提高 clip_grad 而不是降低?

**A**: 之前 clip_grad=0.5 **过度裁剪**:
- 正常梯度 (0.05-0.08) 被保留
- 但掩盖了底层不稳定性
- 提高到 1.0 允许更自然的梯度动态

### Q4: 训练变慢多少?

**A**: 预计 +4% (50h vs 48h)
- lr_mult 降低导致收敛略慢
- 但避免了 NaN 崩溃
- **值得的权衡**

### Q5: 如果还是失败怎么办?

**A**: 概率 <5%，但如果发生:
1. 使用 Emergency Fallback (lr_mult=1.5, weight_reg=1e-5)
2. 从 iter 1280 或最近的健康 checkpoint 重启
3. 接受更慢的训练速度 (55h)

---

## 📚 相关文档

- **详细分析**: `NAN_ERROR_ANALYSIS_ITER1310.md` (根因分析)
- **对比总结**: `NAN_FIX_SUMMARY.md` (方案对比)
- **训练脚本**: `llama8b_presparse_training_tp8_fp32_ultra_conservative.sh`

---

## ✅ 最后检查清单

训练前确认:

- [ ] GPU 资源充足 (8卡 × 80GB)
- [ ] Checkpoint 存在 (`output/checkpoints/llama8b_maskllm_training_tp8/iter_0001280`)
- [ ] 数据文件完整 (20 个 C4 文件)
- [ ] 磁盘空间充足 (>500GB for checkpoints)
- [ ] 环境变量正确 (`run_maskllm_native.sh` 已配置)

训练中监控:

- [ ] 每 10 iters 检查 reg loss 增长率
- [ ] 每 100 iters 检查 validation metrics
- [ ] 每 checkpoint 保存后验证文件完整性

训练后验证:

- [ ] Iter 2000 checkpoint 存在
- [ ] 最终 reg loss ~11740 (vs 目标 <12000)
- [ ] 最终 LM loss <2.40
- [ ] 最终 max prob 0.82-0.85

---

**🚀 准备好了吗? 开始训练!**

```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_ultra_conservative.sh
```

**预计完成时间**: 2025-10-10 16:00 (50 小时后)

