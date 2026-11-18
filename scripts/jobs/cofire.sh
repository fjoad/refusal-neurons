#!/bin/bash
# 17 November meeting, point 27: do the top-5 neurons fire together, and does holding the others back cancel a
# pin? (cofire.py). English HarmBench: the search's top 5, each at its own best pin from the judged rerank.
# Part (a) reads their natural values per prompt; part (b) pins the best one with the other four held or pushed
# the other way. Judged with Llama-Guard and the K.1 judge, then the tables (--report).
# Results log: 'Follow-ups to the 17 November meeting', point 27.
#SBATCH -J cofire
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/cofire.sh <model> [<search/rerank tag, default arditi> [<attackers, default 1>]]
#   The blind route's top 5 instead of Apple's search: ... Qwen/Qwen3-8B blind
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
src=${2:-arditi}
rerank=outputs/rerank/${short}_float16_$src.json
# The pins must be chosen where the answers are coherent: K.1-judge the rerank first if it has not been.
grep -q '"unsafe_k1"' $rerank || python -u replication/judge.py $rerank --judge k1 --bs 8
python -u replication/cofire.py --search outputs/neuron_search/${short}_float16_$src.json --rerank $rerank \
  --tag en_$src --attackers ${3:-1}
f=outputs/cofire/${short}_en_$src.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
python -u replication/cofire.py --report $f
