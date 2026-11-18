#!/bin/bash
# 17 November meeting, point 25b: Apple's pin grid on the neurons the blind reverse route finds. The top 5 neurons
# by share of their own layer's MLP direction (mlp_every_layer.py --blind_only; nothing about Apple's neuron is
# used to choose them) go through Apple's rerank: each pinned over the grid on the 191 HarmBench prompts, 128
# tokens; then Llama-Guard-3 and the K.1 judge. Results log: 'Follow-ups to the 17 November meeting'.
#SBATCH -J blind_rerank
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/blind_rerank.sh <model> <layer> <neuron> <pin>   (Apple's neuron, only for the report)
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/mlp_every_layer.py --model "$1" --layer $2 --neuron $3 --pin $4 --blind_only
python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_blind.json --tag blind
f=outputs/rerank/${short}_float16_blind.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
