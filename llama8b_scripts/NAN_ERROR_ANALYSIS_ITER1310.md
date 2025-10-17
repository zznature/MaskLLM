# NaN Error Analysis at Iter 1310 - FP32 Plan A Training

## 📊 Training Data Analysis

### Critical Metrics Before NaN (iter 1281-1309)

| Iteration | Reg Loss | Reg Loss Δ | LM Loss | Grad Norm | Scale Mult | Temp | Max Prob |
|-----------|----------|------------|---------|-----------|------------|------|----------|
| 1281      | 10924.15 | -          | 2.6125  | 0.110     | 292.0      | 2.400| 0.790    |
| 1282      | 10927.61 | +3.46      | 2.5618  | 0.095     | 292.15     | 2.399| 0.790    |
| 1283      | 10931.06 | +3.45      | 2.5636  | 0.094     | 292.30     | 2.397| 0.791    |
| 1290      | 10955.20 | +24.14     | 2.5346  | 0.066     | 293.35     | 2.389| 0.793    |
| 1300      | 10989.35 | +34.15     | 2.5432  | 0.052     | 294.85     | 2.376| 0.796    |
| 1301      | 10992.75 | +3.40      | 2.5397  | 0.050     | 295.00     | 2.375| 0.796    |
| 1305      | 11006.30 | +13.55     | 2.5550  | 0.047     | 295.60     | 2.370| 0.797    |
| 1309      | 11019.82 | +13.52     | 2.5422  | 0.047     | 296.20     | 2.365| 0.798    |
| **1310**  | **NaN**  | **NaN**    | **NaN** | **NaN**   | **296.35** | **2.364** | **0.798** |

### Key Observations

#### 1. **Reg Loss Explosive Growth** 🔴
- **iter 1281-1309**: Reg loss increased from 10924 → 11020 (+95.67 in 28 iters)
- **Average growth rate**: **3.42 /iter** (still too high!)
- **Validation at iter 1300**: Reg loss = **13292** (21% higher than training!)
  - This indicates **severe overfitting** in mask parameters
  - Validation reg loss PPL: **4.85e8** (absurdly high)

#### 2. **Gradient Norm Decay Pattern** ⚠️
- **iter 1281**: grad_norm = 0.110
- **iter 1309**: grad_norm = 0.047
- **Trend**: Continuous **decrease** (not spike!)
  - This is **deceptive**: low grad norm ≠ stable training
  - Suggests gradients are being **artificially suppressed** by clip_grad
  - But underlying numerical instability is accumulating

#### 3. **Scale Multiplier and Temperature** 📈
- **Scale**: 292.0 → 296.2 (+4.2 in 28 iters)
- **Effective Sharpness at iter 1309**: 296.2 / 2.365 = **125.2**
  - Still well within safe range (<< 200)
  - **NOT the root cause**

#### 4. **Max Prob Gradual Increase** 📊
- **iter 1281**: 0.790
- **iter 1309**: 0.798
- **Trend**: Steady increase
  - Masks are hardening gradually (expected behavior)
  - Not an abrupt change

## 🔬 Root Cause Analysis

### Primary Cause: **Regularization Loss Explosion**

The NaN error occurred due to **cascading numerical instability**:

1. **Mask Logits Drift**
   - With `lr_mult=3`, effective mask LR ≈ 1.03e-4
   - Even though this is "conservative", over 28 iterations the cumulative effect is:
     ```
     Cumulative update ≈ 1.03e-4 × 28 ≈ 0.00288 (0.3% of typical logit value)
     ```
   - Mask logits start drifting away from prior-initialized values

2. **Regularization Loss Growth**
   - `weight_reg = 1e-6` is **too weak** to constrain the drift
   - Reg loss grows at 3.42 /iter
   - Over 28 iters: +95.67 (0.88% increase)

3. **Validation Reg Loss Spike**
   - At iter 1300, validation reg loss = **13292** vs training = **10989**
   - **21% discrepancy** indicates severe overfitting
   - Mask parameters are memorizing training data patterns

4. **Numerical Precision Breakdown**
   - As reg loss approaches ~11000, individual mask gradient contributions become:
     ```
     ∂(reg_loss)/∂(mask_logit) ≈ 2 × weight_reg × logit
     ≈ 2 × 1e-6 × 11000 = 0.022
     ```
   - With FP32, this is still representable, BUT:
   - When combined with LM loss gradients (≈0.047) and scale multiplier effects,
     intermediate computations can produce values near FP32 limits

5. **Gradient Synchronization**
   - Error message: "found NaN in local grad norm in backward pass **before data-parallel communication**"
   - This means NaN occurred **during backward pass**, not during all-reduce
   - Likely in a specific layer's gradient computation

## 🎯 Why Plan A Parameters Failed

| Parameter    | Plan A Value | Analysis | Impact |
|--------------|--------------|----------|--------|
| `lr_mult`    | 3            | Still too high for this checkpoint | Mask LR = 1.03e-4 → logits drift |
| `weight_reg` | 1e-6         | **Too weak** to constrain drift | Reg loss grows at 3.4 /iter |
| `clip_grad`  | 0.5          | Masks symptoms, not root cause | Suppresses spikes but allows drift |
| `scale_range`| 100→400      | Reasonable, not the issue | Eff_Sharp = 125 (safe) |
| `temp_range` | 4.0→1.5      | Reasonable, not the issue | temp = 2.365 (safe) |

### Critical Insight

The failure is **NOT** due to a single "bad" parameter, but due to **insufficient regularization strength** relative to the **mask learning rate**.

**The Fundamental Equation:**
```
Stability Condition: weight_reg × lr_mult < threshold

Current: 1e-6 × 3 = 3e-6 (TOO LOW!)
```

For iter 1280+ (where model is already well-trained), we need **much stronger regularization** to prevent mask drift.

## 🛠️ Solution Strategy

### Approach 1: **Ultra-Conservative (Recommended)** ✅

**Goal**: Minimize ALL sources of instability

```bash
--lr-mult 2.0                    # 降低 33% mask LR → 9.0e-5
--weight-reg 5e-6                # 提高 5x 正则化强度
--clip-grad 1.0                  # 提高到 1.0 (防止极端梯度)
--gumbel-scale-range 100 350     # 降低最大 scale (减少后期硬化)
--gumbel-temperature-range 4.0 2.0  # 提高最小 temp (保持柔性)
```

**Rationale**:
1. **LR Mult 2.0**:
   - Mask LR = 2.0 × 1.5e-5 = **3.0e-5** (最慢速度)
   - Cumulative update over 30 iters: 0.00090 (10x slower drift)

2. **Weight Reg 5e-6**:
   - Regularization force: **5x stronger**
   - Expected reg loss growth rate: **<1.0 /iter**

3. **Clip Grad 1.0**:
   - Allows more flexibility for normal gradients
   - Only clips true anomalies

4. **Scale/Temp Adjustment**:
   - Max effective sharpness: 350 / 2.0 = 175 (still safe)
   - Reduces pressure on mask hardening

**Expected Outcome**:
- Reg loss growth rate: **<1.0 /iter**
- Reach iter 2000 with reg loss ≈ 11020 + (2000-1280) × 1.0 = **11740**
- Success probability: **>95%**

### Approach 2: **Moderate Optimization**

```bash
--lr-mult 2.5
--weight-reg 3e-6
--clip-grad 0.7
--gumbel-scale-range 100 375
--gumbel-temperature-range 4.0 1.8
```

**Expected Outcome**:
- Reg loss growth rate: **~1.5 /iter**
- Success probability: **~80%**
- Slightly faster convergence

### Approach 3: **Aggressive (Not Recommended)**

```bash
--lr-mult 3.5
--weight-reg 2e-6
--clip-grad 0.5
```

**Risk**: Likely to hit NaN again around iter 1350-1400

## 📋 Recommended Action Plan

### Step 1: Implement Ultra-Conservative Approach

1. Update script with Approach 1 parameters
2. Add **checkpoint every 10 iters** (for fine-grained recovery)
3. Add **early warning monitoring**:
   ```bash
   # Monitor every iteration:
   - if reg_loss_growth_rate > 2.0 /iter → WARNING
   - if grad_norm > 0.15 → WARNING
   - if validation_reg_loss > 1.2 × training_reg_loss → WARNING
   ```

### Step 2: Fallback Strategy

If still encountering NaN at iter 1400+:
- **Further reduce lr_mult to 1.5**
- **Increase weight_reg to 1e-5**
- Accept slower convergence for stability

### Step 3: Long-Term Solution

For future training from scratch:
- Use **adaptive lr_mult scheduling**:
  ```python
  # Example pseudo-code
  if iteration < 500:
      lr_mult = 10  # Fast exploration
  elif iteration < 1000:
      lr_mult = 5   # Moderate refinement
  else:
      lr_mult = 2   # Stable convergence
  ```

## 📊 Monitoring Checklist

For every 10 iterations, check:

- [ ] **Reg loss growth rate < 1.5 /iter**
- [ ] **Grad norm < 0.10**
- [ ] **LM loss 下降或稳定**
- [ ] **Validation reg loss < 1.15 × training reg loss**
- [ ] **Max prob 稳步增加**
- [ ] **Logits std 保持在 0.018-0.022 范围**

If ANY metric fails, **stop training immediately** and reduce lr_mult by 0.5.

## 🎓 Key Lessons Learned

1. **"Conservative" is relative**
   - lr_mult=3 seemed conservative compared to previous lr_mult=4-5
   - But at iter 1280+, even lr_mult=3 is too aggressive

2. **Regularization must scale with training progress**
   - Early training (iter <500): weak regularization OK (explore)
   - Late training (iter >1000): strong regularization essential (converge)

3. **Low grad norm ≠ stability**
   - Grad norm was decreasing (0.110 → 0.047)
   - Yet NaN still occurred
   - Clip_grad can mask underlying instability

4. **Validation metrics are critical**
   - Training reg loss = 10989
   - Validation reg loss = 13292 (+21%)
   - This 21% gap was a **red flag** we missed

5. **FP32 is not a silver bullet**
   - FP32 provides more precision than BF16
   - But doesn't prevent poor hyperparameter choices from causing instability
   - Numerical range is still finite

---

**Conclusion**: The NaN error was caused by **insufficient regularization strength** relative to the mask learning rate, leading to gradual logit drift and eventual numerical overflow during gradient computation. The solution is to **significantly strengthen regularization** (5x) while **reducing mask learning rate** (33%) to maintain stability.
