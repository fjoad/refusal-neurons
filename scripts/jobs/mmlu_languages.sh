#!/bin/bash
# 17 November meeting, point 30: MMLU accuracy per language with no intervention (mmlu_languages.py), the check
# behind dropping the languages where the model fails. The same 1000 questions in every language.
# Results log: 'Follow-ups to the 17 November meeting', point 30.
#SBATCH -J mmlu_lang
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/mmlu_languages.sh <model> [languages ...]
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
model=$1
shift
python -u replication/mmlu_languages.py --model "$model" ${@:+--langs "$@"}
