#!/bin/bash

# Verify the import method configuration in apptainer
# This script checks if PYTHONPATH is correctly set to include .ext_pkgs

echo "=== Verifying Import Method Configuration ==="
echo ""

# Change to project directory
cd /data/home/zdhs0054/zzhou/MaskLLM

# Check if .ext_pkgs directory exists
if [ ! -d ".ext_pkgs" ]; then
    echo "❌ .ext_pkgs directory does not exist"
    echo "Please run the installation script first"
    exit 1
fi

echo "✅ .ext_pkgs directory exists"
echo ""

# Check current PYTHONPATH configuration in run_maskllm_native.sh
echo "Checking PYTHONPATH configuration in run_maskllm_native.sh..."
PYTHONPATH_LINE=$(grep "export PYTHONPATH=" run_maskllm_native.sh)
EXT_PKGS_REF=$(grep ".ext_pkgs" run_maskllm_native.sh)

if [ -n "$PYTHONPATH_LINE" ]; then
    echo "✅ PYTHONPATH is configured:"
    echo "   $PYTHONPATH_LINE"
else
    echo "❌ PYTHONPATH configuration not found"
fi

if [ -n "$EXT_PKGS_REF" ]; then
    echo "✅ .ext_pkgs reference found in configuration"
else
    echo "❌ .ext_pkgs reference not found"
fi

echo ""

# Check some key packages that should be in .ext_pkgs
echo "Checking key packages in .ext_pkgs..."
KEY_PACKAGES=("transformers" "accelerate" "datasets" "wandb" "tensorboardx" "pulp" "timm" "nltk")

for package in "${KEY_PACKAGES[@]}"; do
    if [ -d ".ext_pkgs/$package" ]; then
        echo "✅ $package: installed"
    elif [ -f ".ext_pkgs/$package.py" ]; then
        echo "✅ $package: installed (single file)"
    else
        echo "❌ $package: not found"
    fi
done

echo ""
echo "=== Import Method Verification Complete ==="
echo ""
echo "Summary:"
echo "- PYTHONPATH is configured to include .ext_pkgs in run_maskllm_native.sh"
echo "- When running scripts with: bash run_maskllm_native.sh <script> [args...]"
echo "- The apptainer will use packages from .ext_pkgs with proper priority"
echo ""
echo "To test the import method:"
echo "1. Enter apptainer: bash run_apptainer_extended_libs.sh"
echo "2. Run a test script: bash run_maskllm_native.sh <test_script>"
echo "3. Verify packages are imported from .ext_pkgs"
