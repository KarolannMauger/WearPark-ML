#!/bin/bash
# ============================================================
# WearPark ML — Hugging Face Spaces deployment script
# Usage: bash deploy-hf.sh <HF_TOKEN>
# ============================================================
set -e

HF_TOKEN=$1
HF_USER="KarolannMauger"
HF_SPACE="wearpark-ml"
REPO_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="/tmp/wearpark-hf-deploy"

echo "==> Cleaning previous deploy directory..."
rm -rf "$DEPLOY_DIR"

echo "==> Cloning HF Space..."
git clone "$REPO_URL" "$DEPLOY_DIR"

echo "==> Copying files..."

# Source code
mkdir -p "$DEPLOY_DIR/src"
cp "$SCRIPT_DIR/src/api.py"     "$DEPLOY_DIR/src/"
cp "$SCRIPT_DIR/src/model.py"   "$DEPLOY_DIR/src/"
cp "$SCRIPT_DIR/src/predict.py" "$DEPLOY_DIR/src/"

# Model artifacts
mkdir -p "$DEPLOY_DIR/models"
cp "$SCRIPT_DIR/models/wearpark_cnn1d_best.pt" "$DEPLOY_DIR/models/"
cp "$SCRIPT_DIR/models/norm_mean.npy"           "$DEPLOY_DIR/models/"
cp "$SCRIPT_DIR/models/norm_std.npy"            "$DEPLOY_DIR/models/"

# Results (metrics.json needed for optimal threshold)
mkdir -p "$DEPLOY_DIR/results"
cp "$SCRIPT_DIR/results/metrics.json" "$DEPLOY_DIR/results/"

# Dockerfile and dependencies
cp "$SCRIPT_DIR/Dockerfile"               "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/requirements-inference.txt" "$DEPLOY_DIR/"

# HF Space README (with required YAML frontmatter)
cp "$SCRIPT_DIR/hf-readme.md" "$DEPLOY_DIR/README.md"

# Minimal .gitignore (models/ must NOT be ignored in the Space repo)
cat > "$DEPLOY_DIR/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.venv/
venv/
.env
EOF

echo "==> Committing..."
cd "$DEPLOY_DIR"
git config user.name  "Karolann Mauger"
git config user.email "karolann.mauger@gmail.com"
git add -A
git commit -m "Deploy WearPark ML inference API"

echo "==> Pushing to Hugging Face..."
git push

echo ""
echo "✓ Deployed! Space building at:"
echo "  https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
echo ""
echo "Check build logs in the Space → Logs tab (takes ~3-5 min)"
