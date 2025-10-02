# 🚨 CRITICAL BUG FIX: Llama8b 稀疏化完全未执行

## TL;DR

**症状**：稀疏模型和稠密模型困惑度完全相同（8.0137）

**根本原因**：参数名称不匹配导致稀疏化代码分支完全被跳过

**状态**：✅ 已修复，需要重新运行稀疏化

---

## 问题详情

### 🔴 关键发现

您运行的命令传递了 `--sparse-method SparseGPT`，但代码中**完全没有处理这个值**！

查看 `tasks/pruning/prune_main_llama.py:463-471`：

```python
if sparse_method == "hessian":
    print("Run Hessian pruning...")
    prune_hessian(...)
elif sparse_method == "magnitude":
    print("Run magnitude pruning...")
    prune_magnitude(...)
elif sparse_method == "wanda":
    print("Run wanda pruning...")
    prune_wanda(...)

# ❌ 当 sparse_method="SparseGPT" 时：
#    - 所有 if/elif 条件都不满足
#    - 直接跳到 save_fn() 和 evaluate()
#    - 完全没有执行任何稀疏化操作！
```

### 为什么这么隐蔽？

1. **代码不报错**：当参数不匹配时，静默跳过稀疏化分支
2. **照常保存 checkpoint**：即使没有稀疏化，也会保存权重
3. **打印成功消息**：脚本末尾打印 "🎉 Llama8b稀疏化完成!"
4. **困惑度看起来正常**：8.0137 是合理的值，但其实和稠密模型一模一样

### 证据

您的运行日志显示：
- ✅ 加载了模型
- ✅ 计算了困惑度：8.0137
- ✅ 保存了 checkpoint
- ❌ **但完全没有 "Run Hessian pruning..." 或 "Pruning ..." 输出**
- ❌ 稀疏模型和稠密模型困惑度完全相同

---

## 修复方案（已应用）✅

### 修复 1：参数名称映射

在 `run_llama8b_prune_tp8.sh` 中添加：

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
esac

# 使用内部方法名，而不是用户友好的名称
--sparse-method ${INTERNAL_METHOD}  # 而非 ${SPARSEMETHOD}
```

### 修复 2：添加 `--update-weight`

确保在 `options` 中包含：
```bash
--update-weight \
```

---

## 立即行动 🚀

### 1. 重新运行稀疏化（必须）

```bash
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT
```

### 2. 验证关键输出（必看）

**必须在日志中看到以下内容，否则稀疏化仍未执行**：

```
✅ 稀疏化方法: SparseGPT (内部方法: hessian)
...
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
saving checkpoint at iteration 1 to .../llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight
```

### 3. 验证稀疏度

运行评测：
```bash
bash run_maskllm_native.sh \
    llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight \
    8 \
    sparse
```

期待的稀疏度日志：
```
[sparsity] module.language_model.encoder.layers.0.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight: 0.50
```

期待的困惑度：
- **应该高于稠密模型**（8.0137）
- 典型范围：10-15（取决于稀疏化质量）
- 如果仍然是 8.0137，说明稀疏化仍未成功

---

## 为什么之前的 checkpoint 无效？

之前生成的所有 checkpoint（没有 `.update_weight` 后缀的）都是**稠密权重**：
- `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0` ❌

只有重新运行后生成的才是真正的稀疏权重：
- `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight` ✅

---

## 相关文档

- 详细诊断：`llama8b_scripts/diagnose_and_fix_pruning.md`
- 评测方案：`llama8b_scripts/llama8b_inference/llama8b_sparse_eval.md`

---

## 经验教训

1. **参数命名一致性**：脚本注释和代码实现的参数名必须一致
2. **早期验证**：应该在稀疏化开始时就检查日志，确认 "Run Hessian pruning..." 出现
3. **对比基线**：稀疏模型和稠密模型困惑度相同是重大红旗
4. **代码审查**：if/elif 分支应该有 else 报错，而不是静默跳过
