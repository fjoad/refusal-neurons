#!/bin/bash
# English gap: the capability cost (MMLU, GSM8K) of Arditi's ablation, with the English direction
# from scripts/jobs/arditi.sh. Results log: 'Arditi's capability cost'.
#SBATCH -J cap_arditi
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/cap_arditi.sh <model> <layer> <neuron> <pin>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/capability.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
  --search outputs/neuron_search/${short}_float16_arditi.json --modes arditi \
  --arditi outputs/arditi/${short}_jbb.pt --tag arditi
