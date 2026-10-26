#!/bin/bash
# Refusal phrases per language: each model's most common refusal openings on every multilingual harmful
# train split (PolyRefuse 14 + MultiJail 10). Feeds neuron_search.py --phrases_json. Results log: "Refusal phrases".
#SBATCH -J refusal_phrases
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/refusal_phrases.sh <model>
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
S=""
for l in en ar de es fr it ja ko nl pl ru th yo zh; do S="$S poly_${l}_harmful_train"; done
for l in en ar zh it vi ko th bn sw jv; do S="$S mj_${l}_harmful_train"; done
python -u replication/refusal_phrases.py --model "$1" --splits $S --tag multilingual
