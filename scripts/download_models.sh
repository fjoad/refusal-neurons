#!/bin/bash
# Run on the panther login node (downloads are allowed there; compute is not).
# Usage: bash scripts/download_models.sh Qwen/Qwen3-1.7B Qwen/Qwen3-8B
set -e
source ~/anaconda3/bin/activate refneu
for m in "$@"; do
  huggingface-cli download "$m" --exclude "*.pth" "original/*"
done
df -h /export/home/fjoad | tail -1
