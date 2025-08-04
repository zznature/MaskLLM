#!/bin/bash

# Script to fix git authentication issues
# Exit on error
set -e

echo "=== Git Authentication Fix Script ==="

# 1. Check current git remote configuration
echo -e "\n=== Current Git Remote Configuration ==="
git remote -v

# 2. Check current git user configuration
echo -e "\n=== Current Git User Configuration ==="
git config --get user.name
git config --get user.email

# 3. Options to fix authentication issues
echo -e "\n=== Options to Fix Authentication Issues ==="
echo "1. Update remote URL to use SSH instead of HTTPS (if you have SSH keys set up)"
echo "2. Configure git credentials for HTTPS"
echo "3. Set up correct git user information"

echo -e "\n=== Instructions ==="
echo "Based on the output above, you can fix the authentication issue with one of these methods:"
echo ""
echo "Option 1: Update remote URL to SSH (if you have SSH keys configured)"
echo "  git remote set-url origin git@github.com:YOUR_USERNAME/MaskLLM.git"
echo ""
echo "Option 2: Configure git credentials for HTTPS"
echo "  git config --global credential.helper store"
echo "  # Then perform a git operation and enter your credentials when prompted"
echo ""
echo "Option 3: Set correct git user information"
echo "  git config --global user.name \"Your Name\""
echo "  git config --global user.email \"your.email@example.com\""
echo ""
echo "After fixing the authentication, you can retry pushing with:"
echo "  git push origin \$(git rev-parse --abbrev-ref HEAD)" 