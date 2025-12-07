#!/bin/bash
# Setup script for GitHub ML Project Repository

echo "🚀 Setting up GitHub repository for ML project..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M main
else
    echo "✅ Git repository already initialized"
fi

# Verify .gitignore exists
if [ ! -f .gitignore ]; then
    echo "❌ Error: .gitignore not found!"
    exit 1
fi

echo "📋 Checking what will be tracked..."
echo ""
echo "Files that WILL be tracked:"
git status --short --ignored | grep -v "^!!" | head -20

echo ""
echo "Files that WILL be ignored (venv, models, etc.):"
git status --short --ignored | grep "^!!" | head -10

echo ""
read -p "Continue with setup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Add files (respecting .gitignore)
echo "📝 Adding files to git..."
git add .gitignore
git add .dockerignore
git add requirements.txt
git add render.yaml
git add *.py
git add README.md 2>/dev/null || echo "⚠️  No README.md found (optional)"

# Check if CSV should be added (might be large)
if [ -f "Realistic_LJP_Facts.csv" ]; then
    CSV_SIZE=$(du -h "Realistic_LJP_Facts.csv" | cut -f1)
    echo ""
    read -p "Add Realistic_LJP_Facts.csv? (Size: $CSV_SIZE) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add Realistic_LJP_Facts.csv
    else
        echo "⏭️  Skipping CSV file (add manually if needed)"
    fi
fi

echo ""
echo "📊 Repository status:"
git status --short

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a repository on GitHub (https://github.com/new)"
echo "2. Run these commands:"
echo "   git commit -m 'Initial commit: Legal Case Predictor ML project'"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "⚠️  IMPORTANT: Model files are excluded. For deployment:"
echo "   - Use Git LFS for models (git lfs track '*.safetensors')"
echo "   - Or upload to cloud storage (S3/GCS) and download at runtime"
echo "   - Or use Hugging Face Hub"

