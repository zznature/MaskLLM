# Llama8b 稀疏化问题诊断与修复

## 问题现象
稀疏化后的模型加载时所有层的稀疏度显示为 0.00（稠密），表明稀疏化过程未真正执行。
稀疏模型和稠密模型的困惑度完全相同（8.0137），进一步证实稀疏化没有发生。

## 根本原因（已全部修复）✅

### ⚠️ 问题 1：参数名称映射错误（关键问题）

**现象**：即使添加了 `--update-weight`，稀疏化仍然不执行，困惑度与稠密模型完全一致。

**根本原因**：

脚本传递了 `--sparse-method SparseGPT`，但代码中没有处理这个值！

查看 `tasks/pruning/prune_main_llama.py:463-471`：
```python
if sparse_method == "hessian":
    print("Run Hessian pruning...")
    prune_hessian(calibration_dataloader, model[0], eval_metric)
elif sparse_method == "magnitude":
    print("Run magnitude pruning...")
    prune_magnitude(model)
elif sparse_method == "wanda":
    print("Run wanda pruning...")
    prune_wanda(calibration_dataloader, model[0], eval_metric)
# ❌ 没有 "SparseGPT" 分支！如果传入 "SparseGPT"，所有分支都被跳过
```

**代码期望的值**：
- `"hessian"` → SparseGPT 方法（基于 Hessian 的剪枝）
- `"magnitude"` → 幅度剪枝
- `"wanda"` → Wanda 方法

**脚本实际传入的值**：`"SparseGPT"` ❌

**结果**：所有 `if/elif` 条件都不满足，直接跳到 `save_fn()` 和 `evaluate()`，**完全没有执行任何稀疏化操作**！

**修复方案**：✅
在脚本中添加参数名称映射逻辑：
```bash
case "${SPARSEMETHOD}" in
    SparseGPT|sparsegpt|SPARSEGPT)
        INTERNAL_METHOD="hessian"  # SparseGPT 对应内部的 "hessian" 方法
        ;;
    Magnitude|magnitude|MAGNITUDE)
        INTERNAL_METHOD="magnitude"
        ;;
    Wanda|wanda|WANDA)
        INTERNAL_METHOD="wanda"
        ;;
esac

# 然后传递: --sparse-method ${INTERNAL_METHOD}
```

### 问题 2：缺少关键参数 `--update-weight`

**代码分析**（`tasks/pruning/sparsity/core.py`）：

```python
class HessianPruner():
    def prune(self, sparse_pattern='nmprune', ...):
        W = self.layer.weight.data.clone()
        # ... 计算 Hessian, 生成 mask ...
        
        # ⚠️ 关键：只有 update_weight=True 时才真正修改权重
        if self.args.update_weight:
            W[W_mask] = 0  # 将选中的权重置零
            self.layer.weight.data = W.reshape(...)
        
        # mask 总是会被设置，但权重不一定被修改
        self.layer.mask = M.to(dtype=self.layer.weight.dtype)
```

**参数定义**（`megatron/arguments.py:1523`）：
```python
group.add_argument("--update-weight", action="store_true")
# action="store_true" 意味着默认值是 False
# 只有显式传递该标志时才为 True
```

### 2. 稀疏化流程分析

#### 当前流程（缺少 `--update-weight`）：
1. ✅ 计算 Hessian 矩阵（`--hessian-compute`）
2. ✅ 生成稀疏 mask（标记哪些权重该被剪掉）
3. ❌ **不修改权重本身**（因为 `update_weight=False`）
4. ❌ 保存的 checkpoint 包含：
   - 完整的稠密权重（所有值都非零）
   - 稀疏 mask（理论上的稀疏模式）
5. ❌ 评测时读取权重，发现全是非零值，稀疏度 = 0

#### 正确流程（添加 `--update-weight`）：
1. ✅ 计算 Hessian 矩阵
2. ✅ 生成稀疏 mask
3. ✅ **将 mask 标记的权重置零**（因为 `update_weight=True`）
4. ✅ 保存的 checkpoint 包含：
   - 真正的稀疏权重（50% 权重为零）
   - 对应的 mask
5. ✅ 评测时读取权重，稀疏度 = 50%

## 已应用的修复方案 ✅

### 修复 1：参数名称映射（关键修复）✅

已在 `run_llama8b_prune_tp8.sh` 中添加：

```bash
# Map user-friendly names to internal method names
# SparseGPT is implemented as "hessian" method in the code
INTERNAL_METHOD=""
case "${SPARSEMETHOD}" in
    SparseGPT|sparsegpt|SPARSEGPT)
        INTERNAL_METHOD="hessian"
        echo "✅ 稀疏化方法: SparseGPT (内部方法: hessian)"
        ;;
    Magnitude|magnitude|MAGNITUDE)
        INTERNAL_METHOD="magnitude"
        echo "✅ 稀疏化方法: Magnitude"
        ;;
    Wanda|wanda|WANDA)
        INTERNAL_METHOD="wanda"
        echo "✅ 稀疏化方法: Wanda"
        ;;
    *)
        echo "❌ 错误: 未知的稀疏化方法 '$SPARSEMETHOD'"
        echo "支持的方法: SparseGPT, Magnitude, Wanda"
        exit 1
        ;;
esac

# 使用内部方法名
--sparse-method ${INTERNAL_METHOD}  # 而非 ${SPARSEMETHOD}
```

### 修复 2：添加 `--update-weight` 标志 ✅

已在 `options` 变量中添加：
```bash
--update-weight \
```

## 验证步骤

### 1. 重新运行稀疏化
```bash
bash run_maskllm_native.sh \
    llama8b_scripts/run_llama8b_prune_tp8.sh \
    SparseGPT \
    "--update-weight"
```

### 2. 检查稀疏化日志

**关键指标 - 必须看到这些输出**：

1. **方法映射确认**（脚本开始时）：
```
✅ 稀疏化方法: SparseGPT (内部方法: hessian)
```

2. **稀疏化执行确认**（Hessian 计算后）：
```
Run Hessian pruning...
calculating layerwise Hessian
token 0 / 128
token 1 / 128
...
Pruning module.language_model.encoder.layers.0.self_attention.query_key_value.
Pruning module.language_model.encoder.layers.0.self_attention.dense.
Pruning module.language_model.encoder.layers.0.mlp.dense_h_to_4h.
Pruning module.language_model.encoder.layers.0.mlp.dense_4h_to_h.
...
```

3. **权重保存确认**：
```
saving checkpoint at iteration 1 to .../llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight
successfully saved checkpoint at iteration 1 to ...
```

**如果没有看到 "Run Hessian pruning..." 和 "Pruning ..." 输出，说明稀疏化仍未执行！**

### 3. 运行评测验证
```bash
bash run_maskllm_native.sh \
    llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 \
    8 \
    sparse
```

期待的稀疏度日志：
```
[sparsity] module.language_model.encoder.layers.0.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight: 0.50
...
```

## 为什么这个问题这么隐蔽？

1. **参数名称误导**：
   - 脚本注释说 "# SparseGPT, Magnitude, Wanda"
   - 但代码内部使用的是 "hessian, magnitude, wanda"
   - 这种不一致导致用户按照注释使用 "SparseGPT" 时，代码却不认识

2. **静默失败**：
   - 当 `sparse_method` 不匹配任何分支时，代码不报错
   - 直接跳过稀疏化，继续执行保存和评测
   - 用户看到 "稀疏化完成" 的成功消息，但实际上什么都没做

3. **误导性的成功输出**：
   - 即使没有稀疏化，也会保存 checkpoint
   - 也会打印 "ppl: 8.0137"
   - 只有仔细比较稠密和稀疏模型的困惑度，或检查实际稀疏度，才能发现问题

## Llama2 参考脚本的问题

Llama2 脚本 (`scripts/oneshot/run_llama2_7b_prune_tp8.sh`) 也有同样的问题：
- 注释说支持 "SparseGPT"
- 但实际传递什么参数值需要进一步检查
- **建议**：如果要使用 Llama2 脚本，也需要应用相同的修复

## 总结

**核心问题**：
1. ⚠️ **参数名称不匹配**（主要问题）：脚本传递 `"SparseGPT"`，代码期望 `"hessian"`，导致稀疏化分支完全未执行
2. 缺少 `--update-weight`：即使稀疏化分支执行，也不会真正修改权重

**解决方案**（均已应用）：
1. ✅ 添加参数名称映射：`SparseGPT` → `hessian`
2. ✅ 添加 `--update-weight` 标志

**预期效果**：
- 线性层（attention, MLP）的稀疏度应为 0.50（2:4 结构化）
- 归一化层的稀疏度应为 0.00（不稀疏）
- Embedding 和 output_layer 的稀疏度取决于 `--exclude-layers-from-prune` 设置
