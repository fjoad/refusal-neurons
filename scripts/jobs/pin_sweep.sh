#!/bin/bash
# 12 November follow-up: are the answers gibberish when the neuron is pinned at a high value? One neuron, a long
# pin sweep (from small values to far beyond the attack pin) on the 191 HarmBench prompts, 128 tokens; then
# Llama-Guard-3 and the K.1 judge, which reports the share of answers that are coherent at each pin.
# Results log: 'Follow-ups to the 12 November meeting'.
#SBATCH -J pin_sweep
#SBATCH -t 12:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/pin_sweep.sh <model> <L:F> "<pin magnitudes>"
#   Qwen/Qwen3-8B 14:7924 "2 5 10 20 40 80 120 200" | meta-llama/Llama-3.1-8B-Instruct 11:4258 "1 2 4 8 15 30 60 100" | Qwen/Qwen3-1.7B 13:3270 "5 10 20 40 80 120 200"
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_arditi.json --only $2 --m $3 --tag pinsweep
f=outputs/rerank/${short}_float16_pinsweep.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
