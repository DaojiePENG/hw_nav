#!/bin/bash
set -e

echo "Packaging hw_nav environment..."

conda activate hw_nav

# 确保 torch 是真实目录（非软链接）
if [ -L "$CONDA_PREFIX/lib/python3.10/site-packages/torch" ]; then
    echo "❌ Error: torch is still a symlink!"
    exit 1
fi

cd ~
tar -czvf hw_nav_standalone.tar.gz -C miniconda3/envs hw_nav

echo "✅ Package created: ~/hw_nav_standalone.tar.gz"
echo "Ready to deploy to another Jetson (same JetPack)!"