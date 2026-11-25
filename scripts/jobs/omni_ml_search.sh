#!/bin/bash
# Images in every language, search (omni.py phrases, search, grid): on Qwen2.5-Omni-7B, for each language of a
# track, the model's own refusal phrases (Qwen3Guard-confirmed), then Apple's search on that language's 128
# harmful and 128 harmless prompts as typed text and as pictures of the typed text, then the table across
# languages (where the English neurons rank, overlap with the English search).
# Results log: '2026-11-22: Images in every language (Qwen2.5-Omni-7B)'.
#SBATCH -J omni_ml_search
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_ml_search.sh <poly|mj> [languages ...]
#   FORMS="text" in front of sbatch runs typed prompts only (default "text image").
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/envs/refomni/bin/activate
track=$1; shift
if [ $# -gt 0 ]; then langs="$*"
elif [ $track = poly ]; then langs="en ar de es fr it ja ko nl pl ru th yo zh"
else langs="en ar zh it vi ko th bn sw jv"; fi
W="14:13014 16:1870 14:13435 16:11620 17:5484 13:4787"
FORMS=${FORMS:-text image}
python -u replication/omni.py phrases --splits $(for l in $langs; do echo ${track}_${l}_harmful_train; done)
for l in $langs; do
  echo "##### $track $l"
  python -u replication/omni.py search --lang $l --tag ${track}_$l --forms $FORMS --watch $W \
    --harmful ${track}_${l}_harmful_train --harmless ${track}_${l}_harmless_train \
    --phrases_json outputs/refusal_phrases/Qwen2.5-Omni-7B_omni.json || echo "FAILED search $track $l"
done
python -u replication/omni.py grid --tags $(for l in $langs; do echo ${track}_$l; done) --watch $W --forms $FORMS
