#!/bin/bash

# Install missing Python packages in .ext_pkgs/ directory for MaskLLM
# This script installs the required packages that are not already present

echo "=== Installing Missing Python Packages in .ext_pkgs/ ==="
echo "Target directory: /data/home/zdhs0054/zzhou/MaskLLM/.ext_pkgs"
echo ""

# Set pip configuration for Tsinghua mirror
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "Using pip mirror: $PIP_INDEX_URL"
echo ""

# Change to the project directory
cd /data/home/zdhs0054/zzhou/MaskLLM

# Create .ext_pkgs directory if it doesn't exist
if [ ! -d ".ext_pkgs" ]; then
    mkdir -p .ext_pkgs
    echo "Created .ext_pkgs directory"
fi

# Install missing packages
echo "Installing missing packages..."
echo ""

# datasets - HuggingFace datasets library
echo "📦 Installing datasets..."
pip install --target=./.ext_pkgs datasets

# wandb - Weights & Biases for experiment tracking
echo "📦 Installing wandb..."
pip install --target=./.ext_pkgs wandb

# ninja - Small build system
echo "📦 Installing ninja..."
pip install --target=./.ext_pkgs ninja

# tensorboardx==2.6 - TensorBoard extension
echo "📦 Installing tensorboardx==2.6..."
pip install --target=./.ext_pkgs tensorboardx==2.6

# pulp - Linear programming solver
echo "📦 Installing pulp..."
pip install --target=./.ext_pkgs pulp

# timm - PyTorch Image Models
echo "📦 Installing timm..."
pip install --target=./.ext_pkgs timm

# nltk - Natural Language Toolkit
echo "📦 Installing nltk..."
pip install --target=./.ext_pkgs nltk

# pydantic==1.10.8 - Data validation
echo "📦 Installing pydantic==1.10.8..."
pip install --target=./.ext_pkgs pydantic==1.10.8

echo ""
echo "=== Package Installation Complete ==="
echo ""

# Verify installations
echo "Verifying installed packages..."
python3 -c "
import sys
sys.path.insert(0, './.ext_pkgs')

packages_to_check = [
    'datasets',
    'wandb',
    'ninja',
    'tensorboardx',
    'pulp',
    'timm',
    'nltk',
    'pydantic'
]

for package in packages_to_check:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {package}: {version}')
    except ImportError as e:
        print(f'❌ {package}: Import failed - {e}')
"

echo ""
echo "=== Installation Summary ==="
echo "All required packages have been installed in .ext_pkgs/"
echo "The import method is already configured in run_maskllm_native.sh"
echo "PYTHONPATH includes: \${CURRENT_DIR}/.ext_pkgs:\${CURRENT_DIR}"
echo ""
echo "To use these packages in apptainer, run:"
echo "bash run_maskllm_native.sh <your_script> [args...]"
echo ""
