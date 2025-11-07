#!/bin/bash
# Verification script for NaN Recovery implementation

echo "=========================================="
echo "NaN Recovery Implementation Verification"
echo "=========================================="
echo ""

PROJECT_DIR=$(pwd)
SUCCESS=0
FAIL=0

# Check 1: NaN Recovery Manager exists
echo "✓ Checking NaN Recovery Manager..."
if [ -f "$PROJECT_DIR/megatron/core/utils/nan_recovery_manager.py" ]; then
    echo "  ✅ File exists: nan_recovery_manager.py"
    ((SUCCESS++))
else
    echo "  ❌ Missing: nan_recovery_manager.py"
    ((FAIL++))
fi

# Check 2: Key methods in NaN Recovery Manager
echo ""
echo "✓ Checking NaN Recovery Manager methods..."
REQUIRED_METHODS=("record_nan" "should_adjust_hyperparams" "get_adjusted_params" "get_top_hotspots")
for method in "${REQUIRED_METHODS[@]}"; do
    if grep -q "def $method" "$PROJECT_DIR/megatron/core/utils/nan_recovery_manager.py" 2>/dev/null; then
        echo "  ✅ Method found: $method()"
        ((SUCCESS++))
    else
        echo "  ❌ Missing method: $method()"
        ((FAIL++))
    fi
done

# Check 3: grad_buffer.py modifications
echo ""
echo "✓ Checking grad_buffer.py modifications..."
if grep -q "set_nan_manager" "$PROJECT_DIR/megatron/core/distributed/grad_buffer.py" 2>/dev/null; then
    echo "  ✅ Method added: set_nan_manager()"
    ((SUCCESS++))
else
    echo "  ❌ Missing: set_nan_manager() in grad_buffer.py"
    ((FAIL++))
fi

if grep -q "extreme_grad_threshold" "$PROJECT_DIR/megatron/core/distributed/grad_buffer.py" 2>/dev/null; then
    echo "  ✅ Attribute added: extreme_grad_threshold"
    ((SUCCESS++))
else
    echo "  ❌ Missing: extreme_grad_threshold in grad_buffer.py"
    ((FAIL++))
fi

if grep -q "NaN Prevention" "$PROJECT_DIR/megatron/core/distributed/grad_buffer.py" 2>/dev/null; then
    echo "  ✅ Preventive clipping implemented"
    ((SUCCESS++))
else
    echo "  ❌ Missing: Preventive clipping logic"
    ((FAIL++))
fi

if grep -q "NaN Recovery.*Rank" "$PROJECT_DIR/megatron/core/distributed/grad_buffer.py" 2>/dev/null; then
    echo "  ✅ Soft recovery implemented"
    ((SUCCESS++))
else
    echo "  ❌ Missing: Soft recovery logic"
    ((FAIL++))
fi

# Check 4: training.py modifications
echo ""
echo "✓ Checking training.py modifications..."
if grep -q "from megatron.core.utils.nan_recovery_manager import NaNRecoveryManager" "$PROJECT_DIR/megatron/training.py" 2>/dev/null; then
    echo "  ✅ NaN manager imported in training.py"
    ((SUCCESS++))
else
    echo "  ❌ Missing: NaN manager import in training.py"
    ((FAIL++))
fi

if grep -q "nan_manager = NaNRecoveryManager" "$PROJECT_DIR/megatron/training.py" 2>/dev/null; then
    echo "  ✅ NaN manager initialization"
    ((SUCCESS++))
else
    echo "  ❌ Missing: NaN manager initialization"
    ((FAIL++))
fi

if grep -q "bucket.set_nan_manager" "$PROJECT_DIR/megatron/training.py" 2>/dev/null; then
    echo "  ✅ NaN manager injection to buckets"
    ((SUCCESS++))
else
    echo "  ❌ Missing: NaN manager injection"
    ((FAIL++))
fi

if grep -q "Auto-Recovery.*Consecutive NaN count" "$PROJECT_DIR/megatron/training.py" 2>/dev/null; then
    echo "  ✅ Auto-recovery logic implemented"
    ((SUCCESS++))
else
    echo "  ❌ Missing: Auto-recovery logic"
    ((FAIL++))
fi

if grep -q "nan_manager=nan_manager" "$PROJECT_DIR/megatron/training.py" 2>/dev/null; then
    echo "  ✅ NaN manager passed to training_log()"
    ((SUCCESS++))
else
    echo "  ❌ Missing: NaN manager in training_log() call"
    ((FAIL++))
fi

# Check 5: arguments.py modifications
echo ""
echo "✓ Checking arguments.py modifications..."
REQUIRED_ARGS=("nan-recovery-mode" "extreme-grad-threshold" "nan-adjust-threshold" "nan-rollback-threshold")
for arg in "${REQUIRED_ARGS[@]}"; do
    if grep -q "$arg" "$PROJECT_DIR/megatron/arguments.py" 2>/dev/null; then
        echo "  ✅ Argument added: --$arg"
        ((SUCCESS++))
    else
        echo "  ❌ Missing argument: --$arg"
        ((FAIL++))
    fi
done

# Check 6: Training script modifications
echo ""
echo "✓ Checking training script modifications..."
SCRIPT_PATH="$PROJECT_DIR/llama8b_scripts/llama8b_presparse_training_tp8_update_prior.sh"
if grep -q "nan-recovery-mode" "$SCRIPT_PATH" 2>/dev/null; then
    echo "  ✅ Training script updated with --nan-recovery-mode"
    ((SUCCESS++))
else
    echo "  ❌ Missing: --nan-recovery-mode in training script"
    ((FAIL++))
fi

if grep -q "extreme-grad-threshold" "$SCRIPT_PATH" 2>/dev/null; then
    echo "  ✅ Training script updated with --extreme-grad-threshold"
    ((SUCCESS++))
else
    echo "  ❌ Missing: --extreme-grad-threshold in training script"
    ((FAIL++))
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "✅ Successful checks: $SUCCESS"
echo "❌ Failed checks: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎉 All checks passed! NaN recovery is properly implemented."
    echo ""
    echo "Next steps:"
    echo "1. Resume training: bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_update_prior.sh 1"
    echo "2. Monitor logs for [NaN Recovery] and [NaN Prevention] messages"
    echo "3. Check for [Auto-Recovery] adjustments if NaNs occur"
    echo ""
    exit 0
else
    echo "⚠️  Some checks failed. Please review the implementation."
    exit 1
fi

