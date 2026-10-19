#!/bin/bash
#SBATCH -J neuron_search
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH --exclude=crimv3mgpu018
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out

# Usage: sbatch replication/neuron_search.sh Qwen/Qwen3-1.7B [float16|bfloat16|float32] [extra args...]
#   e.g. ... Qwen/Qwen3-8B float16 --bs 1 --tag train
# Full-phrase forward/backward search needs more time than neuron_check.
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== $(date '+%F %T') on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python -u replication/neuron_search.py --model "$1" --dtype "${2:-float16}" "${@:3}"
