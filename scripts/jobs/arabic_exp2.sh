#!/bin/bash
# Arabic Experiment 2: the jailbreak on the held-out Arabic prompts, both tracks, for one neuron:
# unpinned, constant pin, anchor k=1,2 (512 tokens), vs Arditi's direction computed from Arabic prompts;
# all judged by both Arabic judges. Results log: 'Arabic Experiment 2'.
#SBATCH -J arabic_exp2
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --gres=gpu:v100_16GB:2 scripts/jobs/arabic_exp2.sh <model> <layer> <neuron> <pin>
# Splits per track: attack on poly_ar_harmful_eval / mj_ar_harmful_test (100 each). Arditi's direction
# from the Arabic search sets ({track}_ar_harmful/harmless_train); selection on the first 32 of
# {track}_ar_harmful_rerank and {track}_ar_harmless_test; refusal score on the first tokens of
# "أنا" and "لا", the first words of the models' Arabic refusals (outputs/refusal_phrases/*_ar.json).
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
for track in poly mj; do
  if [ $track = poly ]; then eval_split=poly_ar_harmful_eval; else eval_split=mj_ar_harmful_test; fi
  python -u replication/evaluate.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
    --search outputs/neuron_search/${short}_float16_${track}_ar_full.json --split $eval_split --tag ${track}_ar
  python -u replication/arditi.py --model "$1" --watch "$2:$3" --tag ${track}_ar \
    --harmful ${track}_ar_harmful_train --harmless ${track}_ar_harmless_train \
    --harmful_val ${track}_ar_harmful_rerank --harmless_val ${track}_ar_harmless_test --n_val 32 \
    --refusal_words أنا لا --eval_split $eval_split
  for f in outputs/evaluate/${short}_L$2_F$3_${track}_ar.json outputs/evaluate/${short}_arditi_${track}_ar.json; do
    python -u replication/judge.py $f --judge qwen3guard
    python -u replication/judge.py $f --translate --src_lang ar --prompts_en ${eval_split/_ar_/_en_}
    python -u replication/judge_agreement.py $f
  done
done
