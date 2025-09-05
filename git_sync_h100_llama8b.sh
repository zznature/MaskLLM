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

echo "5. Adding new untracked files from current work..."
git add llama8b_scripts/ERROR_ANALYSIS_SUMMARY.md
git add llama8b_scripts/PHASE3_5_SPARSE_WORKFLOW_GUIDE.md
git add llama8b_scripts/README_SPARSE_SOLUTION.md
git add llama8b_scripts/diagnose_gpu_issue.sh
git add llama8b_scripts/fix_datasets_import_error.sh
git add llama8b_scripts/install_missing_packages.sh
git add llama8b_scripts/library_management_strategy.md
git add llama8b_scripts/llama8b_presparse_training_tp8.sh
git add llama8b_scripts/run_complete_sparse_workflow.sh
git add llama8b_scripts/run_llama8b_prune_tp8.sh
git add llama8b_scripts/run_llama8b_prune_tp8_fixed.sh
git add llama8b_scripts/verify_import_method.sh

echo "6. Adding modified files..."
git add .gitignore
git add Project_SelfDefinedModels.md
git add Project_Tasks.md

echo "7. Check git status before commit..."
git status

echo ""
echo "=== Committing and Pushing ==="

# Create commit message
COMMIT_MSG="Phase 3.5 Complete: Llama8b SparseGPT Pruning Success

✅ Major Achievements:
- Pre-sparse model generation: SparseGPT 50% sparsity achieved
- Quality validation: PPL 8.0133 on WIKITEXT2
- Environment setup: All 13 Python packages installed in .ext_pkgs/
- Documentation: Updated workflow guide and project status

📁 New Files Added:
- ERROR_ANALYSIS_SUMMARY.md
- PHASE3_5_SPARSE_WORKFLOW_GUIDE.md
- install_missing_packages.sh
- verify_import_method.sh
- library_management_strategy.md

🔧 Core Updates:
- Project_SelfDefinedModels.md: Updated with Phase 3.5 completion
- Project_Tasks.md: Progress tracking updated
- All sparse training and pruning scripts

🎯 Next: Phase 3.6 Sparse Training with MaskLLM"

echo "Commit message:"
echo "$COMMIT_MSG"
echo ""

# Perform commit and push
git commit -m "$COMMIT_MSG"
echo "✅ Commit successful"

echo "Pushing to origin H100_Llama8b..."
git push origin H100_Llama8b
echo "✅ Push successful"

echo ""
echo "=== Git Sync Complete ==="
echo "All changes have been committed and pushed to H100_Llama8b branch"
echo ""

echo "To clean up redundant files, run:"
echo "  bash cleanup_llama8b_scripts.sh"
