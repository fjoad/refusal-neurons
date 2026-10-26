#!/bin/bash
# Section 3.3 residual-stream baseline: residual search, rerank (both signs), judge.
# Results log: 'Section 3.3 residual-stream baseline'.
#SBATCH -J residual
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/residual.sh <model> <watch L:F>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/neuron_search.py --model "$1" --space residual --tag arditi --watch "$2"
python -u replication/rerank.py --search "outputs/neuron_search/${short}_float16_arditi_residual.json" --m 2 4 8 15 30 60 --tag arditi
python -u replication/judge.py "outputs/rerank/${short}_float16_arditi_residual.json"
