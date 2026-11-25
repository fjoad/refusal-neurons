#!/bin/bash
# Languages on the omni model, each language's own pick (omni.py rerank): for one language, the pin sweep of that
# language's text search top 5 on its rerank prompts (typed), judged by Qwen3Guard on the original text,
# Llama-Guard-3 on an NLLB-200 English translation (English: on the text itself) and the K.1 judge. The pick is
# then pinned in the other languages (omni_ml_attack.sh with PIN=...) to test whether a neuron found in one
# language works in another. Needs omni_ml_search.sh first.
# Results log: '2026-11-25: Languages on the omni model, one neuron for all'.
#SBATCH -J omni_ml_rerank
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_ml_rerank.sh <poly|mj> <language> [<language> ...]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
track=$1; shift
M=Qwen2.5-Omni-7B
for l in "$@"; do
  echo "##### $track $l"
  source ~/envs/refomni/bin/activate
  python -u replication/omni.py rerank --lang $l --form text --split ${track}_${l}_harmful_rerank \
    --search outputs/neuron_search/${M}_float16_omni_text_${track}_${l}.json || { echo "FAILED rerank $track $l"; deactivate; continue; }
  deactivate
  source ~/anaconda3/bin/activate refneu
  f=outputs/rerank/${M}_float16_omni_text_${track}_${l}_on_text.json
  J="python -u replication/judge.py $f --skip_done"
  $J --judge qwen3guard --bs 8
  if [ $l = en ]; then $J --bs 8; else $J --translate --src_lang $l --prompts_en ${track}_en_harmful_rerank --bs 8; fi
  $J --judge k1 --bs 4
  conda deactivate
done
