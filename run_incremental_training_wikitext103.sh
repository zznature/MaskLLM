#!/bin/bash
set -e

echo "=== MaskLLM Incremental Training Workflow ==="
echo "This script will run the complete workflow to improve wikitext2 perplexity"
echo "using incremental training on wikitext-103-v1 dataset with LEARNABLE SPARSITY."

# Create necessary directories
mkdir -p scripts/incremental
mkdir -p assets/data/wikitext-103
mkdir -p output/checkpoints/llama2-7b-tp4-finetune-wikitext103
mkdir -p output/tensorboard/llama2-7b-tp4-finetune-wikitext103
mkdir -p output/logs/llama2-7b-tp4-finetune-wikitext103

# Make scripts executable
chmod +x scripts/data/pretokenize_wikitext-103_llama2-7b.sh
chmod +x scripts/incremental/llama2_7b_finetune_wikitext103_tp4.sh

# Step 1: Download and prepare wikitext-103-v1 dataset
echo -e "\n=== Step 1: Downloading wikitext-103-v1 dataset ==="
echo "Setting HF_ENDPOINT to use mirror..."
export HF_ENDPOINT=https://hf-mirror.com

wget -nc -q --show-progress https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O assets/data/wikitext-103/wikitext-103-v1.zip
unzip -o assets/data/wikitext-103/wikitext-103-v1.zip -d assets/data/wikitext-103/v1

# Step 2: Pretokenize the dataset with Llama2-7b tokenizer
echo -e "\n=== Step 2: Pretokenizing dataset with Llama2-7b tokenizer (seq_length=4096) ==="
bash scripts/data/pretokenize_wikitext-103_llama2-7b.sh

# Step 3: Run incremental training with learnable sparsity
echo -e "\n=== Step 3: Running incremental training with LEARNABLE SPARSITY (total_iters=100) ==="
echo "Using checkpoint: output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000"
echo "Continuing training with learnable sparsity (not frozen pattern)"

# Start container and run training
bash run_maskllm_native.sh scripts/incremental/llama2_7b_finetune_wikitext103_tp4.sh output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000

# Step 4: Evaluate the fine-tuned model on wikitext2
echo -e "\n=== Step 4: Evaluating fine-tuned model on wikitext2 ==="
# Find the latest checkpoint
LATEST_CHECKPOINT=$(find output/checkpoints/llama2-7b-tp4-finetune-wikitext103 -name "iter_*" -type d | sort -V | tail -n 1)
echo "Using latest checkpoint: $LATEST_CHECKPOINT"

# Run evaluation
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh $LATEST_CHECKPOINT 7b 4 sparse

# Step 5: Compare results
echo -e "\n=== Step 5: Comparing perplexity results ==="
echo "Original model (iter_0002000):"
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000 7b 4 sparse

echo -e "\nFine-tuned model with learnable sparsity:"
cat output/logs/llama2-7b-tp4-finetune-wikitext103/wikitext2_ppl.log 2>/dev/null || echo "Fine-tuned model evaluation log not found"

echo -e "\n=== Workflow Complete ==="
echo "The wikitext-103-v1 incremental training with LEARNABLE SPARSITY, seq_length=4096 and total_iters=100 has completed."
echo "Check the logs to see the perplexity improvement on wikitext2." 