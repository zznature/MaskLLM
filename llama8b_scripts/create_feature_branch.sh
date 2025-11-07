#!/bin/bash

# Script to create a new feature branch for NaN recovery system
# This keeps all current uncommitted changes in the new branch

set -e  # Exit on any error

echo "================================================"
echo "Creating Feature Branch: nan-recovery-system"
echo "================================================"
echo ""

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Verify we're on the correct branch
if [ "$CURRENT_BRANCH" != "H100_Llama8b" ]; then
    echo "ERROR: Not on H100_Llama8b branch!"
    exit 1
fi

# Show current status
echo "Current git status:"
echo "-------------------"
git status --short
echo ""

# Create and checkout new feature branch
NEW_BRANCH="feature/nan-recovery-system"
echo "Creating new branch: $NEW_BRANCH"
git checkout -b $NEW_BRANCH

echo "✓ Successfully created and switched to branch: $NEW_BRANCH"
echo ""

# Verify the switch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch is now: $CURRENT_BRANCH"
echo ""

# Push the new branch to remote (personal)
echo "Pushing new branch to remote 'personal'..."
echo "-------------------------------------------"
git push -u personal $NEW_BRANCH

echo ""
echo "================================================"
echo "✓ Feature branch created successfully!"
echo "================================================"
echo ""
echo "Summary:"
echo "  - New branch: $NEW_BRANCH"
echo "  - All uncommitted changes preserved in new branch"
echo "  - Pushed to: personal/$NEW_BRANCH"
echo "  - Ready to implement NaN recovery system"
echo ""
echo "To switch back to main branch: git checkout H100_Llama8b"
echo "To see all branches: git branch -a"

