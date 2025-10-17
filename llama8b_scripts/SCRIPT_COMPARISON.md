# 训练脚本对比总览

## 📋 **可用脚本**

| 脚本 | 精度 | 状态 | 成功率 | 用途 |
|------|------|------|--------|------|
| `llama8b_presparse_training_tp8_stable.sh` | bf16 | ❌ 失败 | 0% | 已弃用（NaN） |
| `llama8b_presparse_training_tp8_fp32.sh` | fp32 | ✅ 推荐 | 75-85% | **主训练脚本** |

---

## 🔍 **详细对比**

### **llama8b_presparse_training_tp8_stable.sh**

**基本信息**:
```bash
精度: bf16
Checkpoint: output/checkpoints/llama8b_maskllm_training_tp8
日志: output/logs/llama8b_presparse_training
端口: 45532
```

**关键参数**:
```bash
--bf16
--lr 1e-5
--lr-mult 1
--weight-reg 1e-7
--gumbel-temperature-range 4 0.5
--clip-grad 0.5
--seed 43
--save-interval 50
```

**历史表现**:
```
尝试1: NaN at iter 496
尝试2: NaN at iter 496 (再次)
尝试3: NaN at iter 606 (seed=42)
尝试4: NaN at iter 606 (seed=43, 更保守参数)
```

**结论**: ❌ **bf16无法完成训练，已弃用**

---

### **llama8b_presparse_training_tp8_fp32.sh** ✅ **推荐**

**基本信息**:
```bash
精度: fp32
Checkpoint: output/checkpoints/llama8b_maskllm_training_tp8_fp32
日志: output/logs/llama8b_presparse_training_fp32
端口: 45533
```

**关键参数**:
```bash
--recompute-activations           # 显存优化
--recompute-granularity full      # 全层重计算
--recompute-method uniform        # 均匀分布
# 无 --bf16 (默认fp32)
--lr 5e-6                         # 更低LR
--lr-mult 0.5                     # 更保守mask更新
--weight-reg 1e-7
--gumbel-temperature-range 6 2.0  # 更保守温度
--clip-grad 0.3                   # 更激进裁剪
--seed 44
--save-interval 100               # 减少checkpoint频率
```

**预期表现**:
```
显存: 65-70GB / 80GB
速度: ~295秒/iter (+35% vs bf16)
完成时间: 6.9天
成功率: 75-85%
```

**优势**:
- ✅ 彻底解决bf16数值下溢/溢出
- ✅ 高成功率
- ✅ 显存可行（通过recomputation）
- ✅ 独立checkpoint，不影响bf16实验

**劣势**:
- ⚠️ 速度慢35%
- ⚠️ Checkpoint大2倍 (96GB vs 48GB)

**结论**: ✅ **主训练脚本，强烈推荐**

---

## 🚀 **使用指南**

### **首次训练（从预稀疏checkpoint开始）**

```bash
# 使用FP32（推荐）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0

# bf16（不推荐，仅供测试）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 0
```

### **从checkpoint恢复**

```bash
# FP32恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 1

# bf16恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

---

## 📊 **参数差异速查表**

| 参数 | bf16 (stable) | FP32 (推荐) | 说明 |
|------|--------------|------------|------|
| **数值精度** | bf16 | fp32 | 关键差异 |
| **显存优化** | 无 | recompute | fp32必需 |
| **学习率** | 1e-5 | 5e-6 | fp32更保守 |
| **Mask LR倍数** | 1 | 0.5 | fp32更慢 |
| **Gumbel温度** | 4→0.5 | 6→2.0 | fp32更宽松 |
| **梯度裁剪** | 0.5 | 0.3 | fp32更激进 |
| **Seed** | 43 | 44 | 不同数据顺序 |
| **保存间隔** | 50 | 100 | fp32减少频率 |
| **端口** | 45532 | 45533 | 避免冲突 |
| **Checkpoint路径** | `*_tp8` | `*_tp8_fp32` | 独立存储 |
| **日志路径** | `*_training` | `*_training_fp32` | 独立存储 |

---

## 🎯 **决策树**

```
需要训练Llama8b稀疏模型
          ↓
    选择精度？
          ↓
    ┌─────┴─────┐
    │           │
  bf16        fp32
 (stable)    (推荐)
    │           │
    ↓           ↓
NaN at iter  65-70GB
496/606      显存
    │           │
    ↓           ↓
  失败 ❌      成功 ✅
              6.9天完成
```

**建议**: 直接使用FP32，跳过bf16的失败循环。

---

## 📁 **文件结构**

```
llama8b_scripts/
├── llama8b_presparse_training_tp8_stable.sh  # bf16（已弃用）
├── llama8b_presparse_training_tp8_fp32.sh    # fp32（推荐）✅
├── START_FP32_TRAINING.md                    # 快速启动指南
└── fixing_notes/
    ├── fp32_nan_prevention_analysis.md       # FP32分析
    ├── bf16_vs_fp32_comparison.md            # 详细对比
    └── nan_error_deep_root_cause_analysis.md # NaN根因

output/
├── checkpoints/
│   ├── llama8b_maskllm_training_tp8/         # bf16 checkpoint
│   └── llama8b_maskllm_training_tp8_fp32/    # fp32 checkpoint ✅
└── logs/
    ├── llama8b_presparse_training/           # bf16日志
    └── llama8b_presparse_training_fp32/      # fp32日志 ✅
```

---

## 🔗 **快速链接**

- **立即开始**: `cat START_FP32_TRAINING.md`
- **FP32分析**: `cat fixing_notes/fp32_nan_prevention_analysis.md`
- **详细对比**: `cat fixing_notes/bf16_vs_fp32_comparison.md`
- **NaN根因**: `cat fixing_notes/nan_error_deep_root_cause_analysis.md`

---

## ✅ **推荐行动**

1. **阅读**: `START_FP32_TRAINING.md`
2. **启动**: `bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0`
3. **监控**: 前100次迭代
4. **等待**: 6.9天完成

**关键**: 使用FP32，一次性完成训练！
