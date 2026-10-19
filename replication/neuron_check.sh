#!/bin/bash
#SBATCH -J neuron_check
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH --exclude=crimv3mgpu018
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out

# Usage: sbatch replication/neuron_check.sh Qwen/Qwen3-1.7B 13 3270 [float16|float32]
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== $(date '+%F %T') on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python -u replication/neuron_check.py --model "$1" --layer "$2" --neuron "$3" --dtype "${4:-float16}"
