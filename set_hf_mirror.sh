#!/bin/bash

# Set HuggingFace mirror to avoid connection issues
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# Also set timeout for better handling of slow connections
export HF_HUB_DOWNLOAD_TIMEOUT=300

echo "HuggingFace mirror environment variables set:"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
echo "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT" 