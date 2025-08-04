#!/bin/bash
set -e

echo "=== Comparing Wikitext-103 Datasets for Training ==="
echo "This script will download and analyze both wikitext-103-raw-v1 and wikitext-103-v1 datasets"
echo "to determine which is better for improving wikitext2 perplexity scores."

# Create directories
mkdir -p assets/data/wikitext-103/raw-v1
mkdir -p assets/data/wikitext-103/v1
mkdir -p assets/data/wikitext-103/analysis

# Download datasets
echo "=== Downloading datasets ==="
wget -nc -q --show-progress https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -O assets/data/wikitext-103/wikitext-103-raw-v1.zip
wget -nc -q --show-progress https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O assets/data/wikitext-103/wikitext-103-v1.zip

# Extract datasets
echo "=== Extracting datasets ==="
unzip -o assets/data/wikitext-103/wikitext-103-raw-v1.zip -d assets/data/wikitext-103/raw-v1
unzip -o assets/data/wikitext-103/wikitext-103-v1.zip -d assets/data/wikitext-103/v1

# Analyze datasets
echo "=== Analyzing datasets ==="

# Count tokens and analyze vocabulary overlap with wikitext-2
echo "Counting lines and words in wikitext-103-raw-v1 train set..."
wc -l assets/data/wikitext-103/raw-v1/wikitext-103-raw/wiki.train.raw > assets/data/wikitext-103/analysis/raw-v1-stats.txt
wc -w assets/data/wikitext-103/raw-v1/wikitext-103-raw/wiki.train.raw >> assets/data/wikitext-103/analysis/raw-v1-stats.txt

echo "Counting lines and words in wikitext-103-v1 train set..."
wc -l assets/data/wikitext-103/v1/wikitext-103/wiki.train.tokens > assets/data/wikitext-103/analysis/v1-stats.txt
wc -w assets/data/wikitext-103/v1/wikitext-103/wiki.train.tokens >> assets/data/wikitext-103/analysis/v1-stats.txt

# Compare preprocessing between datasets
echo "=== Sample comparison between datasets ==="
echo "First 10 lines of wikitext-103-raw-v1:"
head -n 10 assets/data/wikitext-103/raw-v1/wikitext-103-raw/wiki.train.raw > assets/data/wikitext-103/analysis/raw-v1-sample.txt
cat assets/data/wikitext-103/analysis/raw-v1-sample.txt

echo -e "\nFirst 10 lines of wikitext-103-v1:"
head -n 10 assets/data/wikitext-103/v1/wikitext-103/wiki.train.tokens > assets/data/wikitext-103/analysis/v1-sample.txt
cat assets/data/wikitext-103/analysis/v1-sample.txt

# Check if wikitext-2 test set is available for comparison
if [ -f "assets/data/wikitext-2/wikitext-2/wiki.test.tokens" ]; then
  echo -e "\nComparing with wikitext-2 test set (first 5 lines):"
  head -n 5 assets/data/wikitext-2/wikitext-2/wiki.test.tokens
fi

echo -e "\n=== Analysis Results ==="
echo "1. wikitext-103-v1: Pre-tokenized dataset with special tokens and formatting"
echo "2. wikitext-103-raw-v1: Raw text without preprocessing"
echo ""
echo "Recommendation:"
echo "For improving wikitext-2 perplexity scores, wikitext-103-v1 is recommended because:"
echo "- It has the same preprocessing/tokenization format as wikitext-2 test set"
echo "- It contains special tokens that match the evaluation dataset"
echo "- The tokenization consistency will help the model better adapt to the test set format"
echo ""
echo "Next steps:"
echo "1. Use wikitext-103-v1 for continued training"
echo "2. Run the pretokenization script for the Llama2 tokenizer"
echo "3. Configure the training parameters for the incremental training run" 