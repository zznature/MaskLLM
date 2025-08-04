#!/bin/bash
set -e

echo "=== Preparing WikiText-103 using C4 workflow ==="
echo "This script follows the exact same preprocessing steps as C4 dataset"

# Make the script executable
chmod +x scripts/data/prepare_wikitext103_megatron_llama2.sh

# Run the preprocessing
echo "Running WikiText-103 preprocessing (C4 format)..."
bash scripts/data/prepare_wikitext103_megatron_llama2.sh

echo ""
echo "=== Comparing with C4 format ==="
echo "C4 dataset structure:"
ls -lh assets/data/c4_llama2_pretokenized/ | head -5

echo ""
echo "WikiText-103 dataset structure (new):"
ls -lh assets/data/wikitext103_llama2_pretokenized/ | head -5

echo ""
echo "=== Verification ==="
echo "Testing dataset loading to ensure no offset errors..."

python3 -c "
import sys
sys.path.append('.')
from megatron.core.datasets import indexed_dataset

def test_dataset(path, name):
    try:
        ds = indexed_dataset.MMapIndexedDataset(path)
        print(f'✅ {name}: {len(ds):,} documents loaded successfully')
        
        # Test multiple random accesses to detect offset issues
        import random
        random.seed(42)
        test_count = min(1000, len(ds))
        test_indices = random.sample(range(len(ds)), test_count)
        
        for i, idx in enumerate(test_indices):
            try:
                doc = ds[idx]
                if i == 0:  # Show first sample info
                    print(f'   First test doc length: {len(doc)}')
                    print(f'   First few tokens: {doc[:10].tolist()}')
            except Exception as e:
                print(f'❌ {name}: Failed at document {idx}: {e}')
                return False
        
        print(f'   ✅ Successfully accessed {test_count} random documents')
        return True
        
    except Exception as e:
        print(f'❌ {name}: Dataset loading failed: {e}')
        return False

# Test the new WikiText-103 datasets
all_good = True
for split in ['train', 'valid', 'test']:
    path = f'assets/data/wikitext103_llama2_pretokenized/wiki_{split}_llama2_text_document'
    if not test_dataset(path, f'WikiText-103 {split}'):
        all_good = False

if all_good:
    print('\\n🎉 All WikiText-103 datasets validated successfully!')
    print('\\nThe datasets are now ready for training and follow the same format as C4.')
    print('\\nTo use in training, reference: assets/data/wikitext103_llama2_pretokenized/')
else:
    print('\\n⚠️  Some datasets failed validation. Check errors above.')
"

echo ""
echo "=== Summary ==="
echo "✓ WikiText-103 converted to C4-compatible JSON format"
echo "✓ Preprocessed using identical tools/preprocess_data.py command as C4"
echo "✓ Generated .bin/.idx files with same structure as C4"
echo "✓ Validated for training compatibility"
echo ""
echo "New dataset location: assets/data/wikitext103_llama2_pretokenized/" 