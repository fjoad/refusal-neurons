#!/bin/bash
# Part 2, every language (Exp. 1 search + Exp. 7 detector), one model: refusal phrases with a Qwen3Guard refusal
# check on all 24 harmful train sets (PolyRefuse 14 + MultiJail 10; also each language's unpinned refusal rate),
# then Apple's search in each language (watching Apple's neuron) and the neuron's AUROC on each language's
# test sets. Arabic is skipped (done in Part 1). Results log: 'Part 2: search in every language'.
#SBATCH -J part2_search
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/part2_search.sh <model> <Apple's neuron L:F> [resume_from, e.g. mj_bn]
# With resume_from, the phrase step is skipped (its file exists) and languages before it are skipped.
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1"); L=${2%:*}; F=${2#*:}
POLY="en de es fr it ja ko nl pl ru th yo zh"; MJ="en zh it vi ko th bn sw jv"
S=""; for l in $POLY ar; do S="$S poly_${l}_harmful_train"; done; for l in $MJ ar; do S="$S mj_${l}_harmful_train"; done
P=outputs/refusal_phrases/${short}_multilingual.json
[ -n "$3" ] || python -u replication/refusal_phrases.py --model "$1" --splits $S --tag multilingual --refusal_check
skip=${3:+1}
for track in poly mj; do
  if [ $track = poly ]; then langs=$POLY; else langs=$MJ; fi
  for l in $langs; do
    [ "${track}_${l}" = "$3" ] && skip=""
    [ -n "$skip" ] && continue
    echo "##### $track $l"
    python -u replication/neuron_search.py --model "$1" --watch "$2" --tag ${track}_${l} \
      --harmful ${track}_${l}_harmful_train --harmless ${track}_${l}_harmless_train \
      --phrases_json $P --phrases_key ${track}_${l}_harmful_train --phrases_fallback
    python -u replication/neuron_check.py --model "$1" --layer $L --neuron $F --m 0 --n_gen 0 \
      --harmful ${track}_${l}_harmful_test --harmless ${track}_${l}_harmless_test --tag ${track}_${l}_detect
  done
done
