#!/bin/bash
set -e

echo "=== Preparing WikiText-103 using C4 workflow with MaskLLM environment ==="
echo "This script uses run_maskllm_native.sh to ensure proper CUDA environment"

# First, prepare the JSON files (this doesn't need CUDA)
echo "Step 1: Converting WikiText-103 to C4-compatible JSON format..."

WIKITEXT_DIR="assets/data/wikitext-103"
WIKITEXT_RAW_DIR="${WIKITEXT_DIR}/wikitext-103"

# Ensure the source files exist
if [ ! -f "${WIKITEXT_RAW_DIR}/wiki.train.tokens" ]; then
    echo "ERROR: WikiText-103 source files not found!"
    echo "Please run the download portion first or check if files exist in ${WIKITEXT_RAW_DIR}/"
    exit 1
fi

# Create output directory
mkdir -p assets/data/wikitext103_llama2_pretokenized

# Convert to C4 format (no CUDA needed)
for split in train valid test; do
    input_file="${WIKITEXT_RAW_DIR}/wiki.${split}.tokens"
    output_file="${WIKITEXT_DIR}/wiki-${split}.json"
    
    echo "Converting ${input_file} to ${output_file}"
    
    python3 -c "
import json

# Read the WikiText file
with open('${input_file}', 'r', encoding='utf-8') as f:
    content = f.read()

# Split into articles/documents like C4 format
documents = []
lines = content.split('\\n')
current_article = []

for line in lines:
    line = line.strip()
    
    if not line:  # Empty line
        continue
    elif line.startswith('=') and line.endswith('='):  # Wikipedia section header
        # Save current article if substantial
        if current_article:
            article_text = '\\n'.join(current_article).strip()
            if len(article_text) > 100:  # Only keep substantial articles
                documents.append({'text': article_text})
        # Start new article
        current_article = [line]
    else:
        current_article.append(line)

# Add the last article
if current_article:
    article_text = '\\n'.join(current_article).strip()
    if len(article_text) > 100:
        documents.append({'text': article_text})

print(f'Created {len(documents)} documents for ${split}')

# Write in C4 format (one JSON object per line)
with open('${output_file}', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=False) + '\\n')

print(f'Saved to ${output_file}')
"
done

echo ""
echo "Step 2: Running Megatron preprocessing with proper CUDA environment..."
echo "Using run_maskllm_native.sh to ensure CUDA libraries are available"

# Create a temporary script for the preprocessing commands
cat > temp_preprocess_commands.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Megatron preprocessing in CUDA environment..."

# Process each split using the exact same command as C4
for split in train valid test; do
    input_file="assets/data/wikitext-103/wiki-${split}.json"
    
    echo "Processing ${input_file}"
    python tools/preprocess_data.py \
           --input "${input_file}" \
           --output-prefix assets/data/wikitext103_llama2_pretokenized/wiki_${split}_llama2 \
           --vocab-file ./assets/checkpoints/llama2_7b_hf/tokenizer.json \
           --tokenizer-type Llama2Tokenizer \
           --tokenizer-model ./assets/checkpoints/llama2_7b_hf/tokenizer.model \
           --append-eod \
           --workers 8
done

echo "Preprocessing complete!"
EOF

chmod +x temp_preprocess_commands.sh

# Run the preprocessing using MaskLLM native environment
echo "Executing preprocessing with CUDA environment..."
bash run_maskllm_native.sh temp_preprocess_commands.sh

echo ""
echo "Step 3: Validation and cleanup..."

# Clean up temporary script
rm temp_preprocess_commands.sh

# Validate the results
echo "Validating processed datasets..."
python3 -c "
import sys
import os
sys.path.append('.')

# Only validate if the files exist
dataset_dir = 'assets/data/wikitext103_llama2_pretokenized'
if os.path.exists(dataset_dir):
    from megatron.core.datasets import indexed_dataset
    
    for split in ['train', 'valid', 'test']:
        dataset_path = f'{dataset_dir}/wiki_{split}_llama2_text_document'
        
        if os.path.exists(f'{dataset_path}.bin') and os.path.exists(f'{dataset_path}.idx'):
            try:
                ds = indexed_dataset.MMapIndexedDataset(dataset_path)
                print(f'✅ {split}: {len(ds):,} documents successfully loaded')
                
                # Test a few random accesses
                import random
                random.seed(42)
                test_indices = random.sample(range(len(ds)), min(10, len(ds)))
                for idx in test_indices:
                    _ = ds[idx]  # Should not raise an error
                    
                print(f'   Sample lengths: {[len(ds[i]) for i in range(min(3, len(ds)))]}')
                
            except Exception as e:
                print(f'❌ {split}: Validation failed - {e}')
        else:
            print(f'❌ {split}: Dataset files not found')
else:
    print('❌ Dataset directory not found - preprocessing may have failed')
"

echo ""
echo "=== Results ==="
ls -lh assets/data/wikitext103_llama2_pretokenized/ 2>/dev/null || echo "No output files found"

echo ""
echo "=== Summary ==="
echo "✓ WikiText-103 converted to C4-compatible JSON format"
echo "✓ Preprocessed using MaskLLM environment with CUDA support"
echo "✓ Same tools/preprocess_data.py command as C4 dataset"
echo ""
echo "Dataset location: assets/data/wikitext103_llama2_pretokenized/"
echo "Use this path in your training configuration instead of the broken dataset." 