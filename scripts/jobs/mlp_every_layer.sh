#!/bin/bash
# 17 November meeting, notes 9 and 11: the MLP refusal direction removed at every layer (one direction in every
# MLP; each MLP's own direction; Nanda's direction from MLP outputs only, from attention outputs only), and the
# reverse route run blind (mlp_every_layer.py). HarmBench 191, 128 tokens; Llama-Guard-3 and the K.1 judge.
# Results log: 'Follow-ups to the 17 November meeting'.
#SBATCH -J mlp_every_layer
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/mlp_every_layer.sh <model> <layer> <neuron> <pin> [extra arguments, e.g. --bs 8]
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/mlp_every_layer.py --model "$1" --layer $2 --neuron $3 --pin $4 "${@:5}"
f=outputs/evaluate/${short}_L$2_mlpall_en.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
