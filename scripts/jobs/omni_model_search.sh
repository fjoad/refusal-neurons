#!/bin/bash
# A second omni model, the languages: for one omni model (any Qwen2.5-Omni checkpoint the thinker class loads) and
# each language of a track, the model's own refusal phrases (Qwen3Guard-confirmed) and then Apple's search on that
# language's 128 harmful and 128 harmless prompts as typed text (omni.py phrases, search with --model). Then the
# shared-neuron scores across the languages (omni_champion.py --model). Same steps as omni_ml_search.sh, with the
# model as an argument and typed text only. The model must be downloaded first (scripts/download_models.sh).
# Results log: '2026-11-28: A second omni model, the languages'.
#SBATCH -J omni_model_search
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_model_search.sh <Hugging Face model> <poly|mj> [languages ...]
#   e.g. sbatch scripts/jobs/omni_model_search.sh Qwen/Qwen2.5-Omni-3B poly
#   Options, as environment variables in front of sbatch:
#     OMNI_ENV=refomni5     a conda env in place of the refomni venv (Gemma 4 needs transformers 5: omni_env5.sh)
#     DTYPE=bfloat16        the precision (default float16; Gemma's weights are bfloat16)
#   e.g. OMNI_ENV=refomni5 DTYPE=bfloat16 sbatch scripts/jobs/omni_model_search.sh google/gemma-4-E4B-it poly
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
MODEL=$1; track=$2; shift 2
short=${MODEL##*/}
if [ $# -gt 0 ]; then langs="$*"
elif [ $track = poly ]; then langs="en ar de es fr it ja ko nl pl ru th yo zh"
else langs="en ar zh it vi ko th bn sw jv"; fi
DTYPE=${DTYPE:-float16}
if [ -n "${OMNI_ENV:-}" ]; then source ~/anaconda3/bin/activate $OMNI_ENV; else source ~/envs/refomni/bin/activate; fi
python -u replication/omni.py phrases --model $MODEL --dtype $DTYPE --splits $(for l in $langs; do echo ${track}_${l}_harmful_train; done) \
  || { echo "FAILED phrases"; exit 1; }
for l in $langs; do
  echo "##### $short $track $l"
  python -u replication/omni.py search --model $MODEL --dtype $DTYPE --lang $l --tag ${track}_$l --forms text \
    --harmful ${track}_${l}_harmful_train --harmless ${track}_${l}_harmless_train \
    --phrases_json outputs/refusal_phrases/${short}_omni.json || echo "FAILED search $track $l"
done
if [ -n "${OMNI_ENV:-}" ]; then conda deactivate; else deactivate; fi
source ~/anaconda3/bin/activate refneu
python -u replication/omni_champion.py --model $short --dtype $DTYPE
