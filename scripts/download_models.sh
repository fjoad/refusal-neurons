#!/bin/bash
#SBATCH -J refneu_download
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out

# The login node kills large downloads, so run this as a job.
# Usage: sbatch scripts/download_models.sh Qwen/Qwen3-1.7B Qwen/Qwen3-8B
source ~/anaconda3/bin/activate "${ENV:-refneu}"
for m in "$@"; do
  huggingface-cli download "$m" --exclude "*.pth" "original/*" > /dev/null && echo "done $m" || echo "FAILED $m"
done
df -h /export/home/fjoad | tail -1
