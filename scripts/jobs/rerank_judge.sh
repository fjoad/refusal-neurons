#!/bin/bash
# M2 rerank (m sweep over the top-5 candidates on HarmBench), then the judge.
# Results log: 'Milestone 2 rerank'.
#SBATCH -J rerank
#SBATCH -p gpu-all
#SBATCH --gres=gpu:P100_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/rerank_judge.sh <search json> <model short name>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u replication/rerank.py --search "$1" --tag arditi
python -u replication/judge.py "outputs/rerank/$2_float16_arditi.json"
