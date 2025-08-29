#!/bin/bash
# Sync essential files to H100_Llama8b branch

set -euo pipefail

echo "=== Git Sync for H100_Llama8b Branch ==="

# Ensure we're on the right branch
git checkout H100_Llama8b

echo "1. Adding core project files (modified)..."
git add Project_SelfDefinedModels.md
git add Project_Tasks.md  
git add run_maskllm_native.sh
git add megatron/arguments.py
git add tools/preprocess_data.py

echo "2. Adding essential Llama8b inference scripts..."
git add llama8b_scripts/llama8b_infer.py
git add llama8b_scripts/run_llama8b_infer.sh
git add llama8b_scripts/verify_tokenizer_match.py
git add llama8b_scripts/fix_inference_env_commands.sh
git add llama8b_scripts/fix_inference_env_with_accelerate.sh
git add llama8b_scripts/create_overlay_img.sh
git add llama8b_scripts/run_with_overlay_example.sh
git add llama8b_scripts/llama8b_inference_guide.md

echo "3. Adding progress documentation..."
git add llama8b_scripts/PHASE3_PROGRESS_SUMMARY.md
git add llama8b_scripts/PHASE1_COMPLETION_SUMMARY.md
git add llama8b_scripts/PHASE2_COMPLETION_SUMMARY.md

echo "4. Adding essential training/testing scripts..."
git add llama8b_scripts/llama8b_mask_only_tp4_c4.sh
git add llama8b_scripts/llama8b_mask_only_tp8_c4.sh
git add llama8b_scripts/run_llama8b_sparse_training_test.sh

echo "5. Check git status before commit..."
git status

echo ""
echo "=== Ready to commit ==="
echo "Files staged for commit. You can now run:"
echo "  git commit -m 'Add Llama8b inference and training infrastructure'"
echo "  git push origin H100_Llama8b"
echo ""
echo "To clean up redundant files first, run:"
echo "  bash cleanup_llama8b_scripts.sh"
