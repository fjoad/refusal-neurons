#!/bin/bash
# Quick causal check, the other direction: pin each refusal neuron TOWARD its harmful value while the model
# answers harmless prompts (100 Alpaca prompts, 64 tokens, English string-match refusal check). If the neuron causes refusal, harmless
# prompts should start being refused. Results log: 'Refusal induced on harmless prompts'.
#SBATCH -J induce_refusal
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
run() { python -u replication/neuron_check.py --model $1 --layer $2 --neuron $3 --m 0 $4 --n_gen 100 --max_new 64 \
  --harmful $5 --harmless $6 --gen_split $6 --tag induce_$7; }
run Qwen/Qwen3-1.7B 13 3270 "-10 -20 -40" jailbreakbench harmless_heldout en
run meta-llama/Llama-3.1-8B-Instruct 11 4258 "-1 -2 -4" jailbreakbench harmless_heldout en
run Qwen/Qwen3-8B 14 7924 "-4 -10 -20" jailbreakbench harmless_heldout en
