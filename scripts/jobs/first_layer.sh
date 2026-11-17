#!/bin/bash
# 5 November follow-up, note 6 (second half): the layer where harmful and harmless prompts first separate, in every
# language set. Forward passes only. Results log: 'Follow-ups to the 5 November meeting', note 6.
#SBATCH -J first_layer
#SBATCH -t 02:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/first_layer.sh <model>     (Qwen/Qwen3-8B | meta-llama/Llama-3.1-8B-Instruct | Qwen/Qwen3-1.7B)
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u replication/first_layer.py --model "$1"
