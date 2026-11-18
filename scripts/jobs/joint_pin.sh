#!/bin/bash
# 17 November meeting, point 26b: the top-5 candidates pinned together (joint_pin.py). Each candidate at its own
# best pin from the judged rerank, then the best 2, 3, 4, 5 together at full and half pins, on the rerank's
# prompts; judged like the rerank, plus the K.1 judge; then the synergy table (--report).
# Results log: 'Follow-ups to the 17 November meeting'.
#SBATCH -J joint_pin
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/joint_pin.sh <model> <track: en | poly | mj> ["<scales>" <tag suffix> <judge key>]
#   Pins chosen under K.1 (the rerank file must be K.1-judged): ... Qwen/Qwen3-8B mj "1 0.75 0.5 0.25" _k1 unsafe_k1
#   Finer strengths: ... Qwen/Qwen3-8B mj "0.25 0.375 0.625 0.75" _fine
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
scales=${3:-1 0.5}
sfx=${4:-}
key=${5:+--judge_key $5}
if [ "$2" = en ]; then
  python -u replication/joint_pin.py --search outputs/neuron_search/${short}_float16_arditi.json \
    --rerank outputs/rerank/${short}_float16_arditi.json --split harmbench_val --tag en$sfx --scales $scales $key
  f=outputs/joint/${short}_en$sfx.json
  python -u replication/judge.py $f
else
  python -u replication/joint_pin.py --search outputs/neuron_search/${short}_float16_$2_ar_full.json \
    --rerank outputs/rerank/${short}_float16_$2_ar.json --split $2_ar_harmful_rerank --tag $2_ar$sfx --scales $scales $key
  f=outputs/joint/${short}_$2_ar$sfx.json
  python -u replication/judge.py $f --judge qwen3guard
  python -u replication/judge.py $f --translate --src_lang ar --prompts_en $2_en_harmful_rerank
fi
python -u replication/judge.py $f --judge k1 --bs 8
python -u replication/joint_pin.py --report $f
