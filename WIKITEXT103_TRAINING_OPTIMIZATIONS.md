# WikiText-103 Training Optimizations

## Updated Script: `scripts/incremental/llama2_7b_mask_only_wikitext103_tp4.sh`

### Key Changes Made:

### 1. **Dataset Path Update**
```bash
# OLD (broken dataset)
WIKI_HOME=assets/data/wikitext-103/pretokenized
DATA_BLEND="1.0 ${WIKI_HOME}/wiki.train.llama2-7b_text_document"

# NEW (C4-format dataset) 
WIKI_HOME=assets/data/wikitext103_llama2_pretokenized
DATA_BLEND="1.0 ${WIKI_HOME}/wiki_train_llama2_text_document"
```

### 2. **Sequence Length Optimization**
```bash
# OLD: SEQ_LENGTH=4096
# NEW: SEQ_LENGTH=2048
```

**Rationale**: 
- WikiText-103 articles are typically 200-800 tokens
- 2048 is optimal to capture full articles without excessive padding
- Reduces memory usage and training time
- Better gradient updates for shorter sequences

### 3. **Training Iterations Adjustment**
```bash
# OLD: TRAIN_ITERS=400, WARMUP_ITERS=40
# NEW: TRAIN_ITERS=800, WARMUP_ITERS=80  
```

**Rationale**:
- WikiText-103 is smaller than C4, needs more epochs
- Longer training for better convergence on language modeling task
- More warmup for stability

### 4. **Batch Size Optimization**  
```bash
# OLD: GLOBAL_BATCH_SIZE=128
# NEW: GLOBAL_BATCH_SIZE=64
```

**Rationale**:
- Smaller dataset → smaller batch size to avoid overfitting
- Better gradient signal for incremental learning
- Matches dataset scale

### 5. **Updated Tag**
```bash
TAG="llama2-7b-tp4-mask-only-wikitext103-c4format"
```

**Rationale**: Clear distinction from old broken dataset runs

## Dataset Statistics:
- **Format**: Same as C4 (MMapIndexedDataset .bin/.idx)
- **Train Size**: 262MB (.bin) + 5.1MB (.idx) 
- **Documents**: ~264,000 documents
- **Status**: ✅ Ready for training (no offset errors)

## Training Command:
```bash
bash run_maskllm_native.sh scripts/incremental/llama2_7b_mask_only_wikitext103_tp4.sh 0
```

## Expected Results:
- **Memory**: ~50% less than 4096 seq_len (due to quadratic attention)
- **Speed**: ~30-40% faster training iterations  
- **Quality**: Better perplexity on WikiText-2 evaluation
- **Convergence**: More stable due to proper dataset format

## Comparison with C4 Training:
| Aspect | C4 Training | WikiText-103 Training |
|--------|-------------|----------------------|
| Seq Length | 4096 | 2048 (optimized) |
| Batch Size | 128 | 64 (reduced) |
| Iterations | 2000 | 800 (increased) |
| Dataset Size | ~7.4GB | ~268MB |
| Focus | General | Language modeling | 