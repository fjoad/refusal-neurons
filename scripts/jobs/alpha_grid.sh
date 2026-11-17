#!/bin/bash
# 17 November meeting, point 25: a grid search over the steering strength alpha for the MLP direction, as Apple
# did for the pin and as our EMNLP paper did for Nanda's direction (mlp_direction.py --steer_only). alpha counts
# harmful-minus-harmless gaps; the log prints the direction's length so it can be converted to unit-vector
# strengths. HarmBench 191, 128 tokens; Llama-Guard-3 and the K.1 judge (coherence).
# Results log: 'Follow-ups to the 17 November meeting'.
#SBATCH -J alpha_grid
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/alpha_grid.sh <model> <layer> <neuron> <pin>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/mlp_direction.py --model "$1" --layer $2 --neuron $3 --pin $4 --bs 16 --tag en_alpha \
  --steer_only --alphas 0.5 1 1.5 2 2.5 3 4 6 8 12 16 24 32
f=outputs/evaluate/${short}_L$2_mlpdir_en_alpha.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
