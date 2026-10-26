#!/bin/bash
# Per-token activations (paper 3.1) and base models (3.2). Run with tag 'instruct' for the chat models and 'base' for Qwen3-*-Base.
# Results log: 'Per-token firing (3.1) and base models (3.2)'.
#SBATCH -J token_acts
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/token_acts.sh <tag> "<model> <layer> <neuron>" ...
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
tag=$1; shift
for spec in "$@"; do
  set -- $spec
  echo "##### $1 L$2:F$3"
  python -u replication/token_acts.py --model $1 --layer $2 --neuron $3 --splits jailbreakbench harmless_heldout xstest_safe --n 100 --tag $tag
done
