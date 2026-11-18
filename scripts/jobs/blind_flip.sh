#!/bin/bash
# 17 November meeting, point 25b, follow-up: a blind-route candidate whose harmless mean has the same sign as its
# harmful mean, pinned towards and past the harmless side (rerank.py --flip), which Apple's sign rule does not
# try. 191 HarmBench prompts, 128 tokens; Llama-Guard-3 and the K.1 judge.
# Results log: 'Follow-ups to the 17 November meeting'.
#SBATCH -J blind_flip
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/blind_flip.sh <model> <L:F> "<pin magnitudes>"
#   Qwen/Qwen3-8B 29:1450 "10 16 20 30 40 60" | Qwen/Qwen3-8B 3:1803 "0.5 1 2 4 6 10"
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
tag=blind_flip_L${2/:/F}
python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_blind.json --only $2 --m $3 --flip --tag $tag
f=outputs/rerank/${short}_float16_$tag.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
