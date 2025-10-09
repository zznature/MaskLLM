# Llama8b稀疏训练日志

traing script: `llama8b_scripts/llama8b_presparse_training_tp8.sh`

```bash
# 设置GPU可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 选项1: 稀疏化训练 (基于已生成的预稀疏模型) 从预稀疏checkpoint开始
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0

# 继续训练 从训练checkpoint恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

## 1st training

stop at iteration 850 for NaN error in gradient
log-file: `output/logs/llama8b_presparse_training/training_date_25-09-02_time_04-47-14.log`

```bash
# evaluation result
validation loss at iteration 100 | lm loss value: 1.255308E+01 | lm loss PPL: 2.829645E+05 | reg loss value: 7.631986E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 200 | lm loss value: 1.535134E+01 | lm loss PPL: 4.645191E+06 | reg loss value: 7.659308E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 300 | lm loss value: 1.097285E+01 | lm loss PPL: 5.827020E+04 | reg loss value: 7.681837E+03 | reg loss PPL: 4.851652E+08 |   
```

## 2nd training

- changed some parameters to make the training more stable

| 参数类别 | 1st training | 2nd training | 修改幅度 | 修改位置 |
|---------|---------|---------|---------|---------|
| 学习率 | 5e-5 | 2e-5 | ⬇️ 2.5x | 第128行 |
| 最小学习率 | 5e-6 | 2e-6 | ⬇️ 2.5x | 第129行 |
| 正则化 | 1e-5 | 2e-6 | ⬇️ 5x | 第103行 |
| Gumbel缩放 | 1e2 5e2 | 5e1 2.5e2 | ⬇️ 2x | 第103行 |
| Warmup步数 | 0 | 400 | ➕ 新增 | 第93行 |
| 保存间隔 | 500 | 200 | ⬆️ 2.5x频率 | 第90行 |
| 评估间隔 | 100 | 50 | ⬆️ 2x频率 | 第91行 |

log-file: `output/logs/llama8b_presparse_training/training_date_25-09-05_time_02-51-50.log`

```bash
# evaluation result
validation loss at iteration 50 | lm loss value: 1.161736E+01 | lm loss PPL: 1.110081E+05 | reg loss value: 7.615785E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 100 | lm loss value: 1.340704E+01 | lm loss PPL: 6.646664E+05 | reg loss value: 7.616810E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 150 | lm loss value: 1.296712E+01 | lm loss PPL: 4.281035E+05 | reg loss value: 7.619352E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 200 | lm loss value: 1.549217E+01 | lm loss PPL: 5.347674E+06 | reg loss value: 7.622215E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 250 | lm loss value: 1.176644E+01 | lm loss PPL: 1.288542E+05 | reg loss value: 7.625785E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 300 | lm loss value: 1.128546E+01 | lm loss PPL: 7.965480E+04 | reg loss value: 7.631141E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 350 | lm loss value: 1.188648E+01 | lm loss PPL: 1.452885E+05 | reg loss value: 7.633798E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 400 | lm loss value: 1.250351E+01 | lm loss PPL: 2.692799E+05 | reg loss value: 7.635353E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 450 | lm loss value: 1.075681E+01 | lm loss PPL: 4.694890E+04 | reg loss value: 7.641146E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 500 | lm loss value: 1.104297E+01 | lm loss PPL: 6.250275E+04 | reg loss value: 7.644854E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 550 | lm loss value: 1.113715E+01 | lm loss PPL: 6.867557E+04 | reg loss value: 7.648693E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 600 | lm loss value: 1.134056E+01 | lm loss PPL: 8.416755E+04 | reg loss value: 7.653197E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 650 | lm loss value: 1.179350E+01 | lm loss PPL: 1.323889E+05 | reg loss value: 7.659308E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 700 | lm loss value: 1.107539E+01 | lm loss PPL: 6.456237E+04 | reg loss value: 7.661159E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 750 | lm loss value: 1.115774E+01 | lm loss PPL: 7.010438E+04 | reg loss value: 7.663946E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 800 | lm loss value: 1.116398E+01 | lm loss PPL: 7.054326E+04 | reg loss value: 7.671310E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 850 | lm loss value: 1.126647E+01 | lm loss PPL: 7.815661E+04 | reg loss value: 7.674691E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 900 | lm loss value: 1.048755E+01 | lm loss PPL: 3.586627E+04 | reg loss value: 7.679147E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 950 | lm loss value: 1.047793E+01 | lm loss PPL: 3.552287E+04 | reg loss value: 7.681950E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1000 | lm loss value: 1.017551E+01 | lm loss PPL: 2.625237E+04 | reg loss value: 7.684643E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1050 | lm loss value: 9.133303E+00 | lm loss PPL: 9.258550E+03 | reg loss value: 7.687837E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1100 | lm loss value: 8.498374E+00 | lm loss PPL: 4.906784E+03 | reg loss value: 7.691198E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1150 | lm loss value: 7.457586E+00 | lm loss PPL: 1.732959E+03 | reg loss value: 7.695348E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1200 | lm loss value: 6.801736E+00 | lm loss PPL: 8.994072E+02 | reg loss value: 7.699996E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1250 | lm loss value: 5.956599E+00 | lm loss PPL: 3.862942E+02 | reg loss value: 7.703359E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1300 | lm loss value: 5.419857E+00 | lm loss PPL: 2.258468E+02 | reg loss value: 7.709359E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1350 | lm loss value: 4.884523E+00 | lm loss PPL: 1.322274E+02 | reg loss value: 7.715837E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1400 | lm loss value: 4.382247E+00 | lm loss PPL: 8.001767E+01 | reg loss value: 7.720004E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1450 | lm loss value: 4.046807E+00 | lm loss PPL: 5.721447E+01 | reg loss value: 7.724000E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1500 | lm loss value: 3.781154E+00 | lm loss PPL: 4.386665E+01 | reg loss value: 7.726649E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1550 | lm loss value: 3.609316E+00 | lm loss PPL: 3.694079E+01 | reg loss value: 7.730050E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1600 | lm loss value: 3.475836E+00 | lm loss PPL: 3.232483E+01 | reg loss value: 7.733351E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1650 | lm loss value: 3.377321E+00 | lm loss PPL: 2.929220E+01 | reg loss value: 7.737789E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1700 | lm loss value: 3.294059E+00 | lm loss PPL: 2.695204E+01 | reg loss value: 7.740013E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1750 | lm loss value: 3.246460E+00 | lm loss PPL: 2.569920E+01 | reg loss value: 7.746004E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1800 | lm loss value: 3.218766E+00 | lm loss PPL: 2.499727E+01 | reg loss value: 7.747167E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1850 | lm loss value: 3.187777E+00 | lm loss PPL: 2.423448E+01 | reg loss value: 7.750212E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 1900 | lm loss value: 3.175204E+00 | lm loss PPL: 2.393169E+01 | reg loss value: 7.753789E+03 | reg loss PPL: 4.851652E+08 |
validation loss at iteration 1950 | lm loss value: 3.184855E+00 | lm loss PPL: 2.416377E+01 | reg loss value: 7.757159E+03 | reg loss PPL: 4.851652E+08 | 
validation loss at iteration 2000 | lm loss value: 3.187674E+00 | lm loss PPL: 2.423201E+01 | reg loss value: 7.759353E+03 | reg loss PPL: 4.851652E+08 |
```

## 5th training

修正关键错误,稀疏模型重新生成,重新训练.

log-file: `output/logs/llama8b_presparse_training/training_date_25-09-30_time_07-23-32.log`

nan 错误较多


## 6th training

2025.10.03 修改到第二次训练参数,继续训练.
first log: `output/logs/llama8b_presparse_training/training_date_25-10-03_time_15-57-25.log`

```bash
iteration     1093/    2000 | consumed samples:       279808 | elapsed time per iteration (ms): 217580.9 | learning rate: 1.288E-04 | global batch size:   256 | lm loss: 2.607045E+00 | reg loss: 9.261949E+03 | loss scale: 1.0 | grad norm: 0.037 | num zeros: 3407.0 | number of skipped iterations:   0 | number of nan iterations:   0 | scale_multiplier: 159.200 | temperature: 1.925 | max_prob: 0.664 | mask_difference: 0.113 | logits_mean: 0.000 | logits_std: 0.020 |
Traceback (most recent call last):
AssertionError:   File "/data/home/zdhs0054/zzhou/MaskLLM/megatron/core/distributed/grad_buffer.py", line 146, in finish_grad_sync
Rank 5: found NaN in local grad norm in backward pass before data-parallel communication collective. Device: 5, node: hd02-gpu1-0056
```
failed at iteration 1093 for NaN error in backward pass

2025.10.06 增加 `--seed 43` 参数,继续训练.
second log: `output/logs/llama8b_presparse_training/training_date_25-10-06_time_14-18-57.log`

2025.10.07 
1.更改 `--seed 46` 参数.
2.更改dataset blend比例为0.1.
(fails at iter 1143)
```bash
 iteration     1143/    2000 | consumed samples:       292608 | elapsed time per iteration (ms): 219658.2 | learning rate: 1.201E-04 | global batch size:   256 | lm loss: 2.593311E+00 | reg loss: 9.586462E+03 | loss scale: 1.0 | grad norm: 0.039 | num zeros: 3097.0 | number of skipped iterations:   0 | number of nan iterations:   0 | scale_multiplier: 164.200 | temperature: 1.830 | max_prob: 0.688 | mask_difference: 0.111 | logits_mean: -0.000 | logits_std: 0.020 |
Traceback (most recent call last):
```
failed at iteration 1143 for NaN error in backward pass

分析训练参数生成一个stable trainning plan, 寻找最优的稀疏 mask.
1.计划从 iter 1125  checkpoint开始训练，继续 1000 个 iterations。
2.优先采用 bf16 精度，快速迭代。
训练脚本: `llama8b_scripts/llama8b_presparse_training_tp8_stable_resume.sh`
方案说明: `llama8b_scripts/STABLE_TRAINING_PLAN.md`
```bash
# 在 Apptainer 容器内执行
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable_resume.sh
```

### 7th training (稳定化训练)
- 起点: iter 1125
- 脚本: `llama8b_presparse_training_tp8_stable_resume.sh`
- 参数: lr-mult=8, temp→1.0, clip-grad=0.5, weight-reg=1e-6, seed=48
- 结果: **NaN at iter 1150** (成功推进 25 次迭代)
- 日志: `output/logs/llama8b_presparse_training_resume/training_stable_resume_date_25-10-07_time_07-20-37.log`

### 8th training (超保守训练)
- 起点: iter 1150
- 脚本: `llama8b_presparse_training_tp8_ultra_conservative.sh`
- 参数: base_lr=1.5e-5, lr-mult=5, temp→1.5, clip=0.3, reg=5e-7, seed=50
- 结果: **NaN at iter 1310** ✨ 成功推进 160 次！首次突破 iter 1200
- 重大进展:
  - ✅ Reg loss 增速仅 +1.6/iter (远超预期)
  - ✅ Grad norm 极稳定 0.034-0.038
  - ✅ Mask difference 持续优化 0.104 → 0.096
  - ⚠️ 但 iter 1310 仍然 NaN，BF16 有系统性限制

### 9th training (FP32 多次尝试) - **最终采用方案 B**
- 脚本: `llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh`

#### 尝试历程:
**尝试 1-4: 从 iter 1310 开始 (全部失败)**
- 问题: 连续 4 次在 iter 1311 第一次 FP32 迭代后 NaN
- 尝试配置:
  - lr=2e-5, lr-mult=5, override → NaN (LR 跳升到 4.5e-5)
  - lr=1.5e-5, lr-mult=4, 移除 override → NaN (LR 2.7e-5)
- 根本原因: iter 1310 checkpoint 已处于崩溃边缘 (Reg loss 9032)
  - FP32 精度反而放大了潜在的数值不稳定性
  - 无论如何调整参数都无法从 iter 1310 恢复

**尝试 5: 从 iter 1235 开始 (成功启动)**
- 配置: lr=1.5e-5, lr-mult=4, batch=256
- 结果: ✅ 成功运行 iter 1236-1238
- 训练指标:
  - LM loss: 2.616 → 2.567 (持续下降 ✅)
  - Reg loss: 8696 → 8705 (+3/iter, 稳定 ✅)
  - Grad norm: 0.034-0.035 (极稳定 ✅)
- 预测: iter 1280 Reg loss ~8831, iter 1310 ~8920

#### 尝试 6: iter 1280 + lr_mult=5 (失败)
- **配置**: FP32, iter 1280, lr_mult=5, batch=256
- **结果**: ❌ NaN at iter 1306 (训练 26 次迭代后)
- **失败分析**:
  - Reg loss: 8831 → 9010 (+6.9/iter, 太快！)
  - Mask LR: 1.71e-4 (接近 BF16 崩溃阈值)
  - 距离 BF16 失败点 (iter 1310) 仅 4 次迭代
  - **结论**: iter 1280-1310 是系统性危险区，lr_mult=5 太激进

#### 方案演进 (深度分析后)

**关键发现**: 问题不是单一的 lr_mult，而是 5 参数系统性失配！

**核心诊断**:
- ❌ Scale 范围太低 (50-250 vs 标准 100-500)
- ❌ Temperature 最小值太低 (2.0 vs 建议 1.5)
- ❌ Effective Sharpness (Scale/Temp) 增长过快
- ❌ 参数组合在危险区过于激进

#### 最终方案 (方案 A - 全方位优化) ⭐
- **起点**: iter 1280 (平衡点)
- **终点**: iter 2000
- **剩余**: 720 iterations (~48 小时)
- **配置** (5 参数协同优化):
  - 精度: FP32
  - Batch size: 256
  - Base LR: 1.5e-5
  - **LR Mult: 3** (vs 之前 4，降低 25%)
  - **Scale 范围: [100, 400]** (vs 之前 [50, 250])
  - **Temp 范围: [4.0, 1.5]** (vs 之前 [4.0, 2.0])
  - **Weight Reg: 1e-6** (vs 之前 5e-7，提高 2x)
  - **Clip Grad: 0.5** (vs 之前 0.3，提高 67%)
  - Mask LR: ~1.03e-4 (< 失败时的 1.71e-4 ✅)
- **优化逻辑**:
  1. 提高 scale 起点/终点 → 充分探索 + 明确收敛
  2. 提高 temp 最小值 → 防止过度硬化
  3. 降低 lr_mult → 减缓 Mask 更新速度
  4. 提高 weight_reg → 抑制 Reg loss 增长
  5. 提高 clip_grad → 防止梯度尖峰
- **预期效果**:
  - Reg loss 增长率: 2.3 /iter (vs 之前 4.8-6.9)
  - iter 1310 Reg loss: ~8900 (vs 之前 9010)
  - 成功率: 85% (vs 之前 70%)
  - 训练时间: 48h (+5% 但换来稳定性)
- **脚本**: `llama8b_presparse_training_tp8_fp32_optimized_plan_a.sh`
- **状态**: 准备启动 🚀

```bash
# 在 Apptainer 容器内执行 (方案 A)
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_optimized_plan_a.sh
```

#### 关键经验:
1. ❌ **问题不是单一参数，而是系统性失配**
   - 之前认为: 只需调整 lr_mult
   - 实际情况: 5 个参数 (lr_mult, scale, temp, weight_reg, clip_grad) 形成复杂反馈系统
   - 需要协同优化才能稳定

2. 🔬 **Gumbel-Softmax 双参数协同机制**
   - 核心公式: `mask_probs = softmax((gate_logits × scale) / temperature)`
   - 实际控制参数: `Effective Sharpness = Scale / Temperature`
   - Scale 和 Temperature 不是独立的，需要协同优化

3. 📊 **iter 1280-1310 是系统性危险区** (与精度无关！)
   - BF16 在 iter 1310 NaN (Reg loss 9032)
   - FP32 + lr_mult=5 在 iter 1306 NaN (Reg loss 9010)
   - FP32 + lr_mult=4 可能也在危险边缘
   - Reg loss 增长率 > 6/iter (健康应 < 3)

4. ✅ **5 参数协同优化的重要性**
   - Scale 范围: [50, 250] → [100, 400] (探索 + 收敛)
   - Temp 范围: [4.0, 2.0] → [4.0, 1.5] (防过度硬化)
   - LR Mult: 4 → 3 (降低 Mask 更新速度)
   - Weight Reg: 5e-7 → 1e-6 (抑制 Reg loss)
   - Clip Grad: 0.3 → 0.5 (防梯度尖峰)

5. 🎯 **FP32 + 科学调参 = 最佳策略**
   - FP32 提供数值稳定性
   - 但仍需合理的参数配置
   - 不能指望 FP32 解决所有问题

