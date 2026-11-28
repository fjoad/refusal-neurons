#!/bin/bash
# Another omni model, the pin sweep for one neuron: the neuron pinned over a grid of values on the first N of a
# language's sweep prompts (typed, 128 new tokens; omni.py rerank --only), judged by K.1 (refusal and coherence)
# and then Qwen3Guard. One neuron per job so that candidates run side by side. Needs that language's search file.
# Results log: '2026-11-28: A second omni family: Gemma 4 E4B'.
#SBATCH -J omni_model_rerank
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: OMNI_ENV=refomni5 DTYPE=bfloat16 [N=100] sbatch scripts/jobs/omni_model_rerank.sh <Hugging Face model> <poly|mj> <language> <L:F> [pin sizes ...]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
MODEL=$1; track=$2; l=$3; NEURON=$4; shift 4
PINS=${*:-5 10 15 20 30 40}
short=${MODEL##*/}; DTYPE=${DTYPE:-float16}
if [ -n "${OMNI_ENV:-}" ]; then source ~/anaconda3/bin/activate $OMNI_ENV; else source ~/envs/refomni/bin/activate; fi
echo "##### $short $track $l $NEURON pins $PINS"
python -u replication/omni.py rerank --dtype $DTYPE --lang $l --form text --split ${track}_${l}_harmful_rerank --n ${N:-100} \
  --only $NEURON --m $PINS --search outputs/neuron_search/${short}_${DTYPE}_omni_text_${track}_${l}.json || { echo "FAILED rerank"; exit 1; }
if [ -n "${OMNI_ENV:-}" ]; then conda deactivate; else deactivate; fi
source ~/anaconda3/bin/activate refneu
f=outputs/rerank/${short}_${DTYPE}_omni_text_${track}_${l}_on_text_L${NEURON/:/F}.json
J="python -u replication/judge.py $f --skip_done"
$J --judge k1 --bs 4
$J --judge qwen3guard --bs 8
