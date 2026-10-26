#!/bin/bash
# M3 capability only (MMLU/GSM8K) for one neuron; scripts/m3.sh runs attack + judge + capability together.
# Results log: 'Milestone 3'.
#SBATCH -J cap
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/cap.sh <model> <layer> <neuron> <m*>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u replication/capability.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
  --search "outputs/neuron_search/$(basename $1)_float16_arditi.json"
