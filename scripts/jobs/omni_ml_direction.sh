#!/bin/bash
# Languages on the omni model, the refusal direction (omni_direction.py; RQ1: is the direction shared?). For each
# language of a track: the difference-of-means direction on that language's typed prompts, Arditi's selection, then
# answers on the 100 held-out harmful prompts with nothing changed, with the language's own direction removed, and
# with the English direction removed. Set up as arditi.py was for Arabic on the text models (validation: the first
# 32 rerank and harmless-test prompts; 512 new tokens), judged as Part 2. Run English first (it is the reference).
# Results log: '2026-11-26: The refusal direction across languages (Qwen2.5-Omni-7B)'.
#SBATCH -J omni_ml_direction
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_ml_direction.sh <poly|mj> <language> [<language> ...]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
track=$1; shift
if [ $track = poly ]; then split=harmful_eval; else split=harmful_test; fi
M=Qwen2.5-Omni-7B
for l in "$@"; do
  echo "##### $track $l"
  source ~/envs/refomni/bin/activate
  also=""; [ $l = en ] || also="--also en=outputs/arditi/${M}_omni_text_${track}_en.pt"
  python -u replication/omni_direction.py run --tag ${track}_$l --lang $l $also \
    --harmful ${track}_${l}_harmful_train --harmless ${track}_${l}_harmless_train \
    --harmful_val ${track}_${l}_harmful_rerank --harmless_val ${track}_${l}_harmless_test --n_val 32 \
    --eval_split ${track}_${l}_${split} || echo "FAILED direction $track $l"
  deactivate
  source ~/anaconda3/bin/activate refneu
  f=outputs/evaluate/${M}_arditi_omni_text_${track}_${l}.json
  if [ -f $f ]; then
    J="python -u replication/judge.py $f --skip_done"
    $J --judge qwen3guard --bs 8
    if [ $l = en ]; then $J --bs 8; else $J --translate --src_lang $l --prompts_en ${track}_en_${split} --bs 8; fi
    $J --judge k1 --bs 4
  fi
  conda deactivate
done
