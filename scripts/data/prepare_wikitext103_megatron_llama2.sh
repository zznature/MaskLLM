#!/bin/bash
# WikiText-103 preprocessing following the exact C4 workflow
# Based on prepare_c4_megatron_llama2.sh

set -e

echo "=== Preparing WikiText-103 dataset following C4 workflow ==="

# Create output directory
mkdir -p assets/data/wikitext103_llama2_pretokenized

# Set HF_ENDPOINT to use mirror (if needed)
export HF_ENDPOINT=https://hf-mirror.com

# Download and prepare WikiText-103 dataset if not already present
WIKITEXT_DIR="assets/data/wikitext-103"
WIKITEXT_RAW_DIR="${WIKITEXT_DIR}/wikitext-103"

if [ ! -f "${WIKITEXT_RAW_DIR}/wiki.train.tokens" ]; then
    echo "Downloading wikitext-103 dataset..."
    mkdir -p ${WIKITEXT_RAW_DIR}
    
    # Download from Hugging Face datasets
    python3 -c "
import os
from datasets import load_dataset

print('Downloading wikitext-103-v1 dataset...')
dataset = load_dataset('wikitext', 'wikitext-103-v1')
os.makedirs('${WIKITEXT_RAW_DIR}', exist_ok=True)

# Save each split to a file
for split in ['train', 'validation', 'test']:
    output_file = '${WIKITEXT_RAW_DIR}/wiki.' + ('valid' if split == 'validation' else split) + '.tokens'
    print(f'Saving {split} split to {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset[split]:
            f.write(item['text'] + '\\n')
"
    echo "Dataset downloaded and extracted to ${WIKITEXT_RAW_DIR}"
fi

# Convert WikiText-103 to C4-like JSON format
echo "Converting WikiText-103 to C4-compatible JSON format..."

for split in train valid test; do
    input_file="${WIKITEXT_RAW_DIR}/wiki.${split}.tokens"
    output_file="${WIKITEXT_DIR}/wiki-${split}.json"
    
    echo "Converting ${input_file} to ${output_file}"
    
    python3 -c "
import json
import re

# Read the WikiText file
with open('${input_file}', 'r', encoding='utf-8') as f:
    content = f.read()

# Split into articles/documents like C4 format
# Each line in C4 JSON is a separate document with 'text' field
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

# Now preprocess each split using the exact same command as C4
for split in train valid test; do
    input_file="${WIKITEXT_DIR}/wiki-${split}.json"
    
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

echo "Checking results..."
ls -la assets/data/wikitext103_llama2_pretokenized/

# Validate the datasets
echo "Validating datasets..."
python3 -c "
import sys
sys.path.append('.')
from megatron.core.datasets import indexed_dataset

for split in ['train', 'valid', 'test']:
    try:
        ds = indexed_dataset.MMapIndexedDataset(f'assets/data/wikitext103_llama2_pretokenized/wiki_{split}_llama2_text_document')
        print(f'✅ {split}: {len(ds):,} documents')
        
        # Test random access to ensure no offset errors
        import random
        random.seed(42)
        test_indices = random.sample(range(len(ds)), min(100, len(ds)))
        for idx in test_indices:
            _ = ds[idx]  # This should not throw an error
            
        print(f'   Sample lengths: {[len(ds[i]) for i in range(min(5, len(ds)))]}')
        
    except Exception as e:
        print(f'❌ {split}: Error - {e}')
"

echo "=== WikiText-103 preprocessing complete ==="
echo "Tokenized data is saved in: assets/data/wikitext103_llama2_pretokenized/"
echo "Format matches C4 dataset structure for training compatibility" 