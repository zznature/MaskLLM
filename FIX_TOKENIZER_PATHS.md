# Tokenizer路径修复记录

## 问题描述
评估脚本运行时出现错误：
```
OSError: Not found: "/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/llama2_7b_megatron_tp8/tokenizer.model": No such file or directory
```

## 根本原因
脚本中硬编码了错误的tokenizer路径，指向不存在的`llama2_7b_megatron_tp8`目录，而实际应该使用`llama2_7b_hf`目录。

## 修复的文件

### 1. scripts/ppl/evaluate_llama2_wikitext2.sh
- **修复前**: `TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/llama2_7b_megatron_tp8/tokenizer.model"`
- **修复后**: `TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/llama2_7b_hf/tokenizer.model"`

### 2. scripts/oneshot/run_llama2_7b_prune_tp8.sh
- **修复前**: `CHECKPOINT_LOAD_DIR="$PROJECT_DIR/assets/checkpoints/llama2_7b_megatron_tp8"`
- **修复后**: `CHECKPOINT_LOAD_DIR="$PROJECT_DIR/assets/checkpoints/llama2_7b_hf"`

### 3. scripts/oneshot/run_llama2_7b_prune_tp4.sh
- **修复前**: `CHECKPOINT_LOAD_DIR="$PROJECT_DIR/assets/checkpoints/llama2_7b_megatron_tp4"`
- **修复后**: `CHECKPOINT_LOAD_DIR="$PROJECT_DIR/assets/checkpoints/llama2_7b_hf"`

## 验证
确认`assets/checkpoints/llama2_7b_hf/`目录包含所需文件：
- ✅ tokenizer.model (488KB)
- ✅ tokenizer.json (1.8MB)
- ✅ tokenizer_config.json (918B)
- ✅ special_tokens_map.json (414B)
- ✅ config.json (684B)
- ✅ model.safetensors.index.json (23KB)
- ✅ model-00001-of-00003.safetensors (4.6GB)
- ✅ model-00002-of-00003.safetensors (4.6GB)
- ✅ model-00003-of-00003.safetensors (3.3GB)

## 使用方法
现在可以正常运行评估脚本：
```bash
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000 7b 4 sparse
```

## 注意事项
- 转换脚本（`convert_llama2_7b_hf_to_megatron*.sh`）保持原样，因为它们的作用是将HF格式转换为Megatron格式
- 所有训练和评估脚本现在都使用统一的`llama2_7b_hf`目录 