#!/bin/bash
# 12 November follow-ups, points 14-17: the refusal neuron against the refusal direction (direction_variants.py):
# the neuron's write vector used as the direction, the direction without the neuron's part, the top-5 candidates
# all off, the direction added back while the neuron is pinned, and the direction recomputed with the neuron off.
# HarmBench 191, 128 tokens; Llama-Guard-3 and the K.1 judge. Results log: 'Follow-ups to the 12 November meeting'.
#SBATCH -J direction_variants
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/direction_variants.sh <model> <layer> <neuron> <pin>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
python -u replication/direction_variants.py --model "$1" --layer $2 --neuron $3 --pin $4 \
  --search outputs/neuron_search/${short}_float16_arditi.json --arditi outputs/arditi/${short}_jbb.pt
f=outputs/evaluate/${short}_L$2_dirvar_en.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
