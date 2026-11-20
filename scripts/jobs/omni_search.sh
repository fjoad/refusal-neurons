#!/bin/bash
# Images, step 2 (omni.py search, report): Apple's search on Qwen2.5-Omni-7B with the English search prompts
# (Arditi's 128 harmful and 128 harmless) as typed text and as pictures of the typed text, same refusal phrases
# for both; then how far the two rankings overlap.
# Results log: '2026-11-20: Images, typed-out pictures on Qwen2.5-Omni-7B'.
#SBATCH -J omni_search
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_search.sh [<forms, default "text image">]
#   The watched neurons are the two Asim reported.
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
source ~/envs/refomni/bin/activate
forms=${1:-text image}
python -u replication/omni.py search --forms $forms --watch 13:4787 14:13014
python -u replication/omni.py report --forms $forms
