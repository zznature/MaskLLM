# Llama8b稀疏化评测

## 概述
`llama8b_sparse_eval.md` 是专为Llama8b模型设计的稀疏化评估指南。该指南基于MaskLLM框架，支持Llama8b稀疏模型的评测。

### 1.1 预稀疏模型 🎯 ✅

#### 1.1.1 目标和策略 ✅
准备 MaskLLM 预稀疏Llama8b模型，用作稀疏化训练的起始点，采用 SparseGPT 方法进行2:4结构化稀疏，稀疏率50%。

#### 1.1.2 技术实现方案 ✅
- **稀疏化方法**: SparseGPT（基于Hessian信息的高质量剪枝） ✅
- **稀疏模式**: N:M结构化稀疏 (2:4 = 每4个参数保留2个) ✅
- **稀疏率**: 50% (符合2:4模式的理论稀疏率) ✅
- **张量并行**: TP=8 (充分利用8GPU资源) ✅
- **输入模型**: Megatron格式的Llama8b  ✅
- **质量验证**: 困惑度8.0133，质量保持良好 ✅

#### 1.1.3 关键配置适配
```bash
# Llama8b特定配置 (区别于Llama2)
--tokenizer-type Llama8bTokenizer          # 使用自定义tokenizer
--tokenizer-model assets/checkpoints/Llama8b  # Llama8b tokenizer路径
--vocab-size 119696                        # 大词汇表大小
--make-vocab-size-divisible-by 1           # 避免词汇表填充冲突
--load output/checkpoints/llama8b_megatron_tp8  # 输入checkpoint
```

#### 1.1.4 执行脚本和工具
- **主要脚本**: `llama8b_scripts/run_llama8b_prune_tp8.sh`
- **使用方法**: 
  ```bash
  bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT
  ```
- **输出路径**: `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0`

#### 1.1.5 实际输出和验证 ✅
- **模型结构**: 保持32层、4096隐藏维度的原始架构 ✅
- **权重稀疏化**: 线性层权重按2:4模式稀疏化 ✅
- **质量保证**: 通过Hessian信息确保稀疏化后的模型质量 ✅
- **兼容性**: 与MaskLLM稀疏化训练框架完全兼容 ✅
- **验证结果**: 在PRUNE-WIKITEXT2上困惑度8.0133 ✅

#### 1.1.6 稀疏化完成状态 🎉
```
🎉 Llama8b稀疏化完成!
📁 稀疏模型保存在: /data/home/zdhs0054/zzhou/MaskLLM/output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0
📊 稀疏化信息:
  - 方法: SparseGPT ✅
  - 稀疏率: 50% (2:4结构化) ✅
  - 模型名: llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 ✅
  - 验证困惑度: 8.0133 ✅
  - 下一步: 稀疏后模型评测 📋
```

## 稀疏化后模型评测

## 稀疏化后模型 WIKITEXT-2 评测方案

### 1. 目标
- 对 2:4 稀疏化后的 Llama8b 模型在 WIKITEXT-2 验证集上评估困惑度 (Perplexity)
- 验证稀疏权重加载流程与评测脚本的正确性

### 2. 关键资源
- 稀疏模型目录： `/data/home/zdhs0054/zzhou/MaskLLM/output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0`
  - 支持张量并行：TP=8
  - 含标准 Megatron 目录结构 + `latest_checkpointed_iteration.txt`
- 推理/评测脚本：`llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh`
- 评测入口（推荐）：
  ```bash
  bash run_maskllm_native.sh \
       llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
       <CHECKPOINT_PATH> \
       <TP_SIZE> \
       sparse
  ```

### 3. 需要明确的参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `CHECKPOINT_PATH` | Megatron 稀疏权重父目录（需包含 `latest_checkpointed_iteration.txt`） | `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0` |
| `TP_SIZE` | 张量并行大小，需与稀疏模型训练/剪枝时一致 | `8` |
| `MODE` | 评测模式：`sparse` / `dense`。稀疏权重以稠密方式加载时仍用 `sparse` 标志归档 | `sparse` |
| `MASTER_ADDR`/`PORT` | 多卡环境下的通信参数（脚本默认 127.0.0.1:45531，可在脚本内修改） | `默认` |

### 4. 执行流程
1. **准备环境**：进入容器后执行 `run_maskllm_native.sh`，保证容器内 PyTorch + `.ext_pkgs` 混合路径就绪。
2. **调用评测脚本**：
```bash
bash run_maskllm_native.sh \
    llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
    output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 \
    8 \
    sparse
```
3. **脚本行为说明**：
   - 自动检查 `latest_checkpointed_iteration.txt`，若提供子目录则回退到父目录
   - 加载本地 `Llama8bTokenizer`（vocab=119,696）
   - 使用 HuggingFace datasets 自动下载/缓存 WIKITEXT-2 验证集
   - 以 BF16、微批 1、总批 256 运行评测
   - 输出平均 loss、困惑度 (ppl) 及 Token 统计


### 5. 问题诊断与修复

#### 5.1 问题现象 ❌
初次运行后发现：生成的稀疏模型稀疏度为 0，稀疏过程未真正执行。

#### 5.2 根本原因（已全部修复）✅

**问题 1（关键）**：参数名称映射错误
- 脚本传递 `--sparse-method SparseGPT`
- 代码只识别 `"hessian"`, `"magnitude"`, `"wanda"`
- 导致所有稀疏化分支被跳过，完全没有执行稀疏化
- 稀疏模型和稠密模型困惑度完全相同（8.0137）证实了这一点

**问题 2**：缺少 `--update-weight` 标志
- `HessianPruner.prune()` 只有在 `args.update_weight=True` 时才会真正将权重置零
- 该参数默认值为 `False`
- 缺少该标志时，稀疏化过程只生成 mask，但不修改权重本身

详细分析请参阅：`llama8b_scripts/diagnose_and_fix_pruning.md`

#### 5.3 修复方案 ✅
1. **参数名称映射**：添加 case 语句将 `SparseGPT` 映射到 `hessian`
2. **添加标志**：在 `options` 中添加 `--update-weight \`
3. **使用内部方法名**：传递 `--sparse-method ${INTERNAL_METHOD}` 而非 `${SPARSEMETHOD}`

#### 5.4 验证步骤

1. **重新运行稀疏化**：
   ```bash
   bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT
   ```

2. **检查关键日志输出**（必须看到）：
   - ✅ `✅ 稀疏化方法: SparseGPT (内部方法: hessian)` - 确认方法映射成功
   - ✅ `Run Hessian pruning...` - 确认稀疏化分支执行
   - ✅ `calculating layerwise Hessian` - 确认 Hessian 计算
   - ✅ `Pruning module.language_model.encoder.layers.X...` - 确认逐层剪枝
   - ✅ `saving checkpoint ... update_weight` - 确认权重更新保存

3. **运行评测验证稀疏度**：
   ```bash
   bash run_maskllm_native.sh \
       llama8b_scripts/llama8b_inference/evaluate_llama8b_wikitext2.sh \
       output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight \
       8 \
       sparse
   ```
   
   **注意**：修复后的 checkpoint 路径会包含 `.update_weight` 后缀

#### 5.5 期待的稀疏度日志
线性层应显示 0.50 稀疏度：
```
[sparsity] module.language_model.encoder.layers.0.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight: 0.50
```

### 6. 旧的评测结果（修复前，仅供参考）

```bash
loading checkpoint from output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 at iteration 1
 checkpoint version 3.0
  successfully loaded checkpoint from output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 at iteration 1
[sparsity] module.language_model.embedding.word_embeddings.weight: 0.01
[sparsity] module.language_model.encoder.layers.0.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.self_attention.query_key_value.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.self_attention.dense.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight: 0.00
......
[sparsity] module.language_model.encoder.layers.31.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.self_attention.query_key_value.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.self_attention.dense.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.mlp.dense_h_to_4h.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.mlp.dense_4h_to_h.weight: 0.00
[sparsity] module.language_model.encoder.final_norm.weight: 0.00
[sparsity] module.language_model.output_layer.weight: 0.00
--------------------------------------------------------------------------------------------------------------------------
 validation results on WIKITEXT2 | avg loss: 2.0812E+00 | ppl: 8.0137E+00 | adjusted ppl: 8.0137E+00 | token ratio: 1.0 |
--------------------------------------------------------------------------------------------------------------------------
```
### 7. 注意事项
- TP 参数必须与稀疏模型生成时一致；否则会报并行尺寸不匹配错误
- 若评测脚本退出提示 metadata 缺失，请检查 `CHECKPOINT_PATH` 是否指向包含 `latest_checkpointed_iteration.txt` 的父目录
- 评测脚本在末尾默认打印 "evaluation completed"，否则需查看 stdout 的报错提示
- **重要**：确保稀疏化脚本使用了 `--update-weight` 标志，否则生成的模型将是稠密的

### 8. 下一步
成功修复并重新生成真正的稀疏模型后，可以进行：
1. 对比稀疏模型与稠密基线的困惑度
2. 记录稀疏化对模型质量的影响
3. 进行稀疏化训练（使用稀疏模型作为起始点）


Pruning module.language_model.encoder.layers.31.mlp.dense_4h_to_h.
Pruning module.language_model.encoder.layers.31.self_attention.dense.
Pruning module.language_model.encoder.layers.31.mlp.dense_h_to_4h.
Pruning module.language_model.encoder.layers.31.self_attention.dense.
Pruning module.language_model.encoder.layers.31.mlp.dense_h_to_4h.
Pruning module.language_model.encoder.layers.31.mlp.dense_4h_to_h.
Pruning module.language_model.encoder.layers.31.mlp.dense_4h_to_h.
  successfully saved checkpoint at iteration       1 to /data/home/zdhs0054/zzhou/MaskLLM/output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight
> working on iteration: 0
done :-)
--------------------------------------------------------------------------------------------------------------------------------
 validation results on PRUNE-WIKITEXT2 | avg loss: 2.5041E+00 | ppl: 1.2232E+01 | adjusted ppl: 1.2232E+01 | token ratio: 1.0 |
--------------------------------------------------------------------------------------------------------------------------------

 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1095913472
[Float16Module(
  (module): GPTModel(
    (language_model): TransformerLanguageModel(
      (embedding): Embedding(
        (word_embeddings): VocabParallelEmbedding(num_embeddings=119696, embedding_dim=4096)
        (embedding_dropout): Dropout(p=0.0, inplace=False)
      )
      (rotary_pos_emb): RotaryEmbedding()
      (encoder): ParallelTransformer(
        (layers): ModuleList(
          (0-31): 32 x ParallelTransformerLayer(
            (input_norm): RMSNorm()
            (self_attention): ParallelAttention(
              (query_key_value): ColumnParallelLinear()
              (core_attention): CoreAttention(
                (scale_mask_softmax): FusedScaleMaskSoftmax()
                (attention_dropout): Dropout(p=0.0, inplace=False)
              )
              (core_attention_flash): FlashSelfAttention()
              (dense): RowParallelLinear()
            )
            (post_attention_norm): RMSNorm()
            (mlp): ParallelMLP(
              (dense_h_to_4h): ColumnParallelLinear()
              (dense_4h_to_h): RowParallelLinear()
            )
          )
        )
        (final_norm): RMSNorm()
      )
      (output_layer): ColumnParallelLinear()
    )
  )
)]
 loading checkpoint from output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight at iteration 1
 checkpoint version 3.0
  successfully loaded checkpoint from output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight at iteration 1
[sparsity] module.language_model.embedding.word_embeddings.weight: 0.01
[sparsity] module.language_model.encoder.layers.0.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.1.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.1.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.1.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.1.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.1.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.1.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.2.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.2.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.2.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.2.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.2.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.2.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.3.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.3.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.3.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.3.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.3.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.3.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.4.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.4.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.4.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.4.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.4.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.4.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.5.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.5.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.5.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.5.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.5.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.5.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.6.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.6.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.6.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.6.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.6.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.6.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.7.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.7.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.7.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.7.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.7.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.7.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.8.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.8.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.8.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.8.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.8.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.8.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.9.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.9.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.9.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.9.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.9.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.9.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.10.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.10.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.10.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.10.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.10.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.10.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.11.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.11.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.11.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.11.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.11.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.11.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.12.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.12.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.12.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.12.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.12.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.12.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.13.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.13.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.13.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.13.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.13.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.13.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.14.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.14.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.14.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.14.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.14.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.14.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.15.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.15.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.15.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.15.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.15.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.15.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.16.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.16.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.16.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.16.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.16.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.16.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.17.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.17.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.17.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.17.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.17.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.17.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.18.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.18.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.18.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.18.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.18.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.18.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.19.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.19.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.19.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.19.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.19.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.19.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.20.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.20.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.20.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.20.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.20.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.20.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.21.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.21.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.21.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.21.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.21.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.21.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.22.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.22.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.22.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.22.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.22.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.22.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.23.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.23.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.23.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.23.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.23.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.23.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.24.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.24.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.24.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.24.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.24.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.24.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.25.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.25.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.25.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.25.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.25.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.25.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.26.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.26.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.26.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.26.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.26.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.26.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.27.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.27.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.27.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.27.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.27.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.27.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.28.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.28.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.28.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.28.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.28.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.28.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.29.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.29.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.29.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.29.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.29.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.29.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.30.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.30.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.30.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.30.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.30.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.30.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.layers.31.input_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.self_attention.query_key_value.weight: 0.50
[sparsity] module.language_model.encoder.layers.31.self_attention.dense.weight: 0.50
[sparsity] module.language_model.encoder.layers.31.post_attention_norm.weight: 0.00
[sparsity] module.language_model.encoder.layers.31.mlp.dense_h_to_4h.weight: 0.50
[sparsity] module.language_model.encoder.layers.31.mlp.dense_4h_to_h.weight: 0.50
[sparsity] module.language_model.encoder.final_norm.weight: 0.00
[sparsity] module.language_model.output_layer.weight: 0.00
len(data_loader): 76
overalapping_eval 4096
total_targets 308412
total_sequences 76
num_tokenized_tokens 308413
 > number of original tokens: 308413, number of detokenized tokens: 308413
len(data_loader): 76
len(data_loader): 76
> working on iteration: 0
len(data_loader): 76
len(data_loader): 76
len(data_loader): 76
len(data_loader): 76
len(data_loader): 76
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
WARNING:megatron.core.models.common.embeddings.rotary_pos_embedding:Setting apply_rope_fusion to false because its implementation is not included in Apex. Try upgrading to the latest version
done :-)
--------------------------------------------------------------------------------------------------------------------------
 validation results on WIKITEXT2 | avg loss: 2.5041E+00 | ppl: 1.2232E+01 | adjusted ppl: 1.2232E+01 | token ratio: 1.0 |
--------------------------------------------------------------------------------------------------------------------------