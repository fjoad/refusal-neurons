#!/bin/bash
# Arabic Experiment 1 (Apple's method, from scratch, in Arabic) on both tracks:
#   1. the model's own Arabic refusal sentences (PolyRefuse and MultiJail harmful train)
#   2. the gradient x activation search on Arabic prompts (PolyRefuse: same 128 harmful prompts as the
#      English replication; MultiJail: human-translated), watching the English refusal neuron
#   3. the rerank: pin each Arabic top-5 candidate over the pin grid on held-out Arabic prompts
#      (PolyRefuse 191, MultiJail 87). Judging follows in a separate step (scripts/jobs/arabic_judge.sh).
# Results log: "Arabic Experiment 1".
#SBATCH -J arabic_exp1
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arabic_exp1.sh <model> <English refusal neuron L:F>
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
P=outputs/refusal_phrases/${short}_ar.json
python -u replication/refusal_phrases.py --model "$1" --splits poly_ar_harmful_train mj_ar_harmful_train --tag ar
for track in poly mj; do
  python -u replication/neuron_search.py --model "$1" --watch "$2" --tag ${track}_ar_full \
    --harmful ${track}_ar_harmful_train --harmless ${track}_ar_harmless_train \
    --phrases_json $P --phrases_key ${track}_ar_harmful_train
  python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_${track}_ar_full.json \
    --split ${track}_ar_harmful_rerank --tag ${track}_ar
done
