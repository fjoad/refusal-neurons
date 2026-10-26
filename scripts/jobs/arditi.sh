#!/bin/bash
# Arditi refusal direction: selection, App. G alignment, ablation on JailbreakBench, then the judge.
# Results log: 'Appendix G alignment and the Arditi ablation baseline'.
#SBATCH -J arditi
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arditi.sh <model> <watch L:F>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u replication/arditi.py --model "$1" --watch "$2" --tag jbb
python -u replication/judge.py "outputs/evaluate/$(basename $1)_arditi_jbb.json"
