#!/bin/bash
set -e

echo "=== Pretokenizing wikitext-103-v1 dataset with Llama2-7b tokenizer ==="

# Create output directory
mkdir -p assets/data/wikitext-103/pretokenized

# Set HF_ENDPOINT to use mirror
echo "Setting HF_ENDPOINT to use mirror..."
export HF_ENDPOINT=https://hf-mirror.com

# Download wikitext-103-v1 dataset if not already present
WIKITEXT_DIR="assets/data/wikitext-103/wikitext-103"
if [ ! -f "${WIKITEXT_DIR}/wiki.train.tokens" ]; then
    echo "Downloading wikitext-103 dataset from official source..."
    mkdir -p assets/data/wikitext-103
    
    # Download from Hugging Face datasets
    python -c "
import os
from datasets import load_dataset

print('Downloading wikitext-103-v1 dataset...')
dataset = load_dataset('wikitext', 'wikitext-103-v1')
os.makedirs('${WIKITEXT_DIR}', exist_ok=True)

# Save each split to a file
for split in ['train', 'validation', 'test']:
    output_file = '${WIKITEXT_DIR}/wiki.' + ('valid' if split == 'validation' else split) + '.tokens'
    print(f'Saving {split} split to {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset[split]:
            f.write(item['text'] + '\\n')
"
    echo "Dataset downloaded and extracted to ${WIKITEXT_DIR}"
fi

# Process each split (train, valid, test) using Megatron's preprocess_data.py
for split in train valid test; do
    echo "Processing ${WIKITEXT_DIR}/wiki.${split}.tokens"
    
    # Check if file exists
    if [ ! -f "${WIKITEXT_DIR}/wiki.${split}.tokens" ]; then
        echo "ERROR: File ${WIKITEXT_DIR}/wiki.${split}.tokens not found!"
        exit 1
    fi
    
    # Pre-chunk the text into multiple documents before tokenization
    python -c "
import json
import re

# Read the entire file
with open('${WIKITEXT_DIR}/wiki.${split}.tokens', 'r', encoding='utf-8') as f:
    text = f.read()

# Split text into paragraphs and articles
# Remove empty lines and split by double newlines or article boundaries
articles = []
current_article = []

lines = text.split('\\n')
for line in lines:
    line = line.strip()
    if not line:  # Empty line
        if current_article:
            # End current article
            article_text = '\\n'.join(current_article).strip()
            if len(article_text) > 100:  # Only keep substantial articles
                articles.append(article_text)
            current_article = []
    elif line.startswith('=') and line.endswith('='):  # Wikipedia section header
        if current_article:
            # End current article
            article_text = '\\n'.join(current_article).strip()
            if len(article_text) > 100:
                articles.append(article_text)
            current_article = []
        current_article.append(line)
    else:
        current_article.append(line)

# Add the last article
if current_article:
    article_text = '\\n'.join(current_article).strip()
    if len(article_text) > 100:
        articles.append(article_text)

print(f'Split into {len(articles)} articles for ${split}')

# Write articles as separate JSON documents
with open('${WIKITEXT_DIR}/wiki.${split}.chunked.json', 'w', encoding='utf-8') as f:
    for i, article in enumerate(articles):
        # Each line in the JSON file should be a separate document
        f.write(json.dumps({'text': article}) + '\\n')
"
    
    # Process with Megatron's preprocess_data.py using the chunked data
    python tools/preprocess_data.py \
        --input "${WIKITEXT_DIR}/wiki.${split}.chunked.json" \
        --output-prefix "assets/data/wikitext-103/pretokenized/wiki.${split}.llama2-7b" \
        --vocab-file ./assets/checkpoints/llama2_7b_hf/tokenizer.json \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model ./assets/checkpoints/llama2_7b_hf/tokenizer.model \
        --append-eod \
        --workers 32
    
    # Verify the indexed dataset was created properly
    python -c "
import json
import numpy as np
from megatron.core.datasets import indexed_dataset

# Load the indexed dataset
ds = indexed_dataset.MMapIndexedDataset('assets/data/wikitext-103/pretokenized/wiki.${split}.llama2-7b_text_document')
print(f'Dataset length (number of documents): {len(ds)}')

# Check sample lengths
if len(ds) > 0:
    sample_lengths = []
    for i in range(min(10, len(ds))):
        sample_lengths.append(len(ds[i]))
    print(f'Sample document lengths: {sample_lengths}')
    print(f'First few tokens of document 0: {ds[0][:20].tolist()}')

# Write to jsonl format for compatibility (optional, mainly for verification)
with open('assets/data/wikitext-103/pretokenized/wiki.${split}.llama2-7b.jsonl', 'w') as f:
    for i in range(len(ds)):
        tokens = ds[i].tolist()
        f.write(json.dumps({'tokens': tokens}) + '\\n')
"
done

echo "Checking results..."
ls -la assets/data/wikitext-103/pretokenized/
wc -l assets/data/wikitext-103/pretokenized/*.jsonl

echo "=== Pretokenization complete ==="
echo "Tokenized data is saved in: assets/data/wikitext-103/pretokenized/"
echo "You can now use this data for incremental training to improve wikitext2 perplexity" 