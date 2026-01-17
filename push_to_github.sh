#!/bin/bash
# ============================================================
# Soccer Film Analysis - Push to GitHub Script
# ============================================================
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_NAME="soccer_film_analysis"

echo ""
echo "============================================================"
echo "   PUSH TO GITHUB"
echo "============================================================"
echo ""

# Check if username provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage:${NC} ./push_to_github.sh YOUR_GITHUB_USERNAME"
    echo ""
    echo "Example: ./push_to_github.sh johnsmith"
    echo ""
    echo "Make sure you've created the repository on GitHub first:"
    echo "  https://github.com/new"
    echo "  Repository name: $REPO_NAME"
    echo "  Do NOT initialize with README"
    exit 1
fi

GITHUB_USER=$1
REMOTE_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "GitHub Username: $GITHUB_USER"
echo "Repository: $REPO_NAME"
echo "Remote URL: $REMOTE_URL"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${BLUE}[1/4]${NC} Initializing git repository..."
    git init
else
    echo -e "${BLUE}[1/4]${NC} Git repository already initialized"
fi

# Add all files
echo -e "${BLUE}[2/4]${NC} Staging files..."
git add .

# Commit
echo -e "${BLUE}[3/4]${NC} Creating commit..."
git commit -m "Initial commit: Soccer film analysis application

Features:
- Player detection with Roboflow Sports models
- Team classification by jersey color (K-means)
- Real-time visualization during analysis
- PostgreSQL database for persistence
- PyQt6 GUI with video player
- CLI tool for batch processing
- Configurable analysis depth (quick/standard/deep)
- Comprehensive logging and error handling" 2>/dev/null || echo "   No changes to commit"

# Set up remote and push
echo -e "${BLUE}[4/4]${NC} Pushing to GitHub..."

# Check if remote exists
if git remote | grep -q "origin"; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

# Push
git branch -M main
git push -u origin main

echo ""
echo "============================================================"
echo -e "   ${GREEN}PUSHED TO GITHUB${NC}"
echo "============================================================"
echo ""
echo "Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "Next: Use Claude Code to test and improve the application"
echo ""
