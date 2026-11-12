#!/bin/bash
# Arabic Experiment 6 (paper 3.3): residual-stream baseline in Arabic, both tracks: residual search on the
# Arabic search sets with the model's Arabic refusal phrases, rerank both signs on the Arabic rerank
# prompts, both Arabic judges and Apple's K.1 criterion. Results log: 'Arabic Experiment 6'.
#SBATCH -J arabic_exp6
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --gres=gpu:v100_16GB:2 scripts/jobs/arabic_exp6.sh <model> <English residual top L:F> [tracks]
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
tracks="${@:3}"  # every argument after the residual L:F is a track
for track in ${tracks:-poly mj}; do
  python -u replication/neuron_search.py --model "$1" --space residual --watch "$2" --tag ${track}_ar \
    --harmful ${track}_ar_harmful_train --harmless ${track}_ar_harmless_train \
    --phrases_json outputs/refusal_phrases/${short}_ar.json --phrases_key ${track}_ar_harmful_train
  python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_${track}_ar_residual.json \
    --m 2 4 8 15 30 60 --split ${track}_ar_harmful_rerank --tag ${track}_ar
  f=outputs/rerank/${short}_float16_${track}_ar_residual.json
  python -u replication/judge.py $f --judge qwen3guard
  python -u replication/judge.py $f --translate --src_lang ar --prompts_en ${track}_en_harmful_rerank
  python -u replication/judge.py $f --judge k1 --bs 8   # the guards reward garbled residual pins; K.1 checks coherence
  python -u replication/judge_agreement.py $f
done
