#!/bin/bash
# Cleanup redundant experimental scripts in llama8b_scripts/

set -euo pipefail

echo "=== Cleaning up redundant llama8b_scripts files ==="

# Remove experimental diagnostic scripts (multiple similar ones)
rm -f llama8b_scripts/alternative_assessment.sh
rm -f llama8b_scripts/alternative_solutions_analysis.sh
rm -f llama8b_scripts/check_current_status.sh
rm -f llama8b_scripts/complete_fix_solution.sh
rm -f llama8b_scripts/complete_ucc_compat.sh
rm -f llama8b_scripts/comprehensive_8gpu_analysis.sh
rm -f llama8b_scripts/comprehensive_fix.sh
rm -f llama8b_scripts/comprehensive_fix_solution.sh
rm -f llama8b_scripts/container_environment_diagnosis.sh
rm -f llama8b_scripts/container_pytorch_diagnosis.sh
rm -f llama8b_scripts/create_8gpu_distributed_env.sh
rm -f llama8b_scripts/create_minimal_hijack.sh
rm -f llama8b_scripts/create_single_gpu_env.sh
rm -f llama8b_scripts/deep_hpcx_conflict_fix.sh
rm -f llama8b_scripts/deep_pytorch_diagnosis.sh
rm -f llama8b_scripts/diagnostic_environment_issues.sh
rm -f llama8b_scripts/dual_solution_approach.sh
rm -f llama8b_scripts/environment_change_diagnosis.sh
rm -f llama8b_scripts/final_verification.sh
rm -f llama8b_scripts/find_container_conda.sh
rm -f llama8b_scripts/fix_container_cuda_libs.sh
rm -f llama8b_scripts/fix_container_cuda_libs_direct.sh
rm -f llama8b_scripts/fix_container_libs_no_symlink.sh
rm -f llama8b_scripts/fix_hpcx_library_conflict.sh
rm -f llama8b_scripts/fix_preprocess_data_tokenizer.sh
rm -f llama8b_scripts/fix_tokenizer_final.sh
rm -f llama8b_scripts/force_ext_torch_environment.sh
rm -f llama8b_scripts/force_fix_tokenizer.sh
rm -f llama8b_scripts/investigate_preprocess_args.sh
rm -f llama8b_scripts/llama8b_presparse_training_tp8.sh
rm -f llama8b_scripts/optimal_final_solution.sh
rm -f llama8b_scripts/optimal_strategy_executor.sh
rm -f llama8b_scripts/optimized_prepare_c4_megatron_llama8b.sh
rm -f llama8b_scripts/persistent_environment_fix.sh
rm -f llama8b_scripts/post_fix_analysis.sh
rm -f llama8b_scripts/precise_environment_repair.sh
rm -f llama8b_scripts/prepare_c4_megatron_llama8b.sh
rm -f llama8b_scripts/quick_apptainer_analysis.sh
rm -f llama8b_scripts/quick_env_check.sh
rm -f llama8b_scripts/quick_fix_test.sh
rm -f llama8b_scripts/radical_efficient_solution.sh
rm -f llama8b_scripts/radical_library_isolation.sh
rm -f llama8b_scripts/root_cause_analysis_and_fix.sh
rm -f llama8b_scripts/run_c4_preprocessing_fixed.sh
rm -f llama8b_scripts/run_llama8b_prune_tp8.sh
rm -f llama8b_scripts/run_maskllm_ultimate.sh
rm -f llama8b_scripts/smart_environment_wrapper.sh
rm -f llama8b_scripts/smart_ucc_compat.sh
rm -f llama8b_scripts/system_level_fix.sh
rm -f llama8b_scripts/test_intelligent_environment.sh
rm -f llama8b_scripts/test_llama2tokenizer_fix.sh
rm -f llama8b_scripts/test_llama8b_tokenizer_correct.sh
rm -f llama8b_scripts/test_multiple_tokenizer_configs.sh
rm -f llama8b_scripts/test_nulltokenizer_config.py
rm -f llama8b_scripts/test_pytorch_import.py
rm -f llama8b_scripts/test_run_maskllm_native_fix.py
rm -f llama8b_scripts/test_single_file_preprocessing.sh
rm -f llama8b_scripts/tokenizer_diagnosis_and_fix.sh
rm -f llama8b_scripts/torch_isolation_wrapper.py
rm -f llama8b_scripts/ucc_ee_patch.sh
rm -f llama8b_scripts/ucs_missing_symbols_patch.sh
rm -f llama8b_scripts/ultimate_environment_fix.sh
rm -f llama8b_scripts/ultimate_pytorch_fix.sh
rm -f llama8b_scripts/verify_tokenizer_fix.sh

# Remove backup files
rm -f run_maskllm_native.sh.backup_20250826_154307
rm -f tools/preprocess_data.py.backup
rm -f test_integration.py

# Remove temporary directories
rm -rf comprehensive_fix/
rm -rf efficient_solution/
rm -rf environment_diagnosis_20250826_153612/
rm -rf final_solution/
rm -f container_environment_setup.sh

# Remove redundant markdown files
rm -f llama8b_scripts/llama8b_tokenizer_final_solution.md
rm -f llama8b_scripts/smart_hijack_work_plan.md

echo "✅ Cleanup completed!"
echo "Run 'git status' to see remaining files"
