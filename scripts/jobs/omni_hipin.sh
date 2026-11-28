#!/bin/bash
# Does a larger pin switch off the refusals that are left, before the answers stop making sense? One neuron, pins
# from below its attack size to four times it, on one language's rerank prompts (typed; MultiJail 87, PolyRefuse
# 191), 128 new tokens (omni.py rerank --only). Judged by K.1 first (refusal and coherence, the two numbers the
# question needs), then Qwen3Guard on the original text and Llama-Guard-3 (on an NLLB-200 English translation
# outside English). Needs that language's search file (omni_ml_search.sh).
# Results log: '2026-11-28: Larger pins on MultiJail and PolyRefuse (Qwen2.5-Omni-7B)'.
#SBATCH -J omni_hipin
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_hipin.sh <poly|mj> <language> <L:F> [pin sizes ...]
#   e.g. sbatch scripts/jobs/omni_hipin.sh mj en 14:13014
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
track=$1; l=$2; N=$3; shift 3
PINS=${*:-10 20 30 40 50 60 80 100 120}
M=Qwen2.5-Omni-7B
echo "##### $track $l $N pins $PINS"
source ~/envs/refomni/bin/activate
python -u replication/omni.py rerank --lang $l --form text --split ${track}_${l}_harmful_rerank --only $N --m $PINS \
  --tag hipin --search outputs/neuron_search/${M}_float16_omni_text_${track}_${l}.json || { echo "FAILED rerank"; exit 1; }
deactivate
source ~/anaconda3/bin/activate refneu
f=outputs/rerank/${M}_float16_omni_text_${track}_${l}_on_text_L${N/:/F}_hipin.json
J="python -u replication/judge.py $f --skip_done"
$J --judge k1 --bs 4
$J --judge qwen3guard --bs 8
if [ $l = en ]; then $J --bs 8; else $J --translate --src_lang $l --prompts_en ${track}_en_harmful_rerank --bs 8; fi
conda deactivate
