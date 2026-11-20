#!/bin/bash
# Images, step 3 (omni.py rerank): Apple's pin sweep on Qwen2.5-Omni-7B for one search's top 5, on the 191
# HarmBench prompts given in one form (typed text or pictures of the typed text), judged with Llama-Guard and the
# K.1 judge. Picks each form's own best neuron and pin, as the English replication did for text.
# Results log: '2026-11-20: Images, typed-out pictures on Qwen2.5-Omni-7B'.
#SBATCH -J omni_rerank
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_rerank.sh <form the search ran on> [<form of the HarmBench prompts, default the same>]
#   e.g. ... image          the picture search's top 5, pinned on pictures
#        ... text image     the text search's top 5, pinned on pictures
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/envs/refomni/bin/activate
on=${2:-$1}
python -u replication/omni.py rerank --search outputs/neuron_search/Qwen2.5-Omni-7B_float16_omni_$1.json --form $on
deactivate
source ~/anaconda3/bin/activate refneu
f=outputs/rerank/Qwen2.5-Omni-7B_float16_omni_$1_on_$on.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
