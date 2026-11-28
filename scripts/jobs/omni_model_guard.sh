#!/bin/bash
# Another omni model, the neuron as a guard: for each language of a track, the values of the given neurons on the
# 100 held-out harmful and 100 harmless prompts with nothing pinned (omni.py check --skip_answers: nothing is
# generated), then the guard table next to Llama-Guard-3 and Qwen3Guard (omni_guard.py --model). Typed text.
# Results log: '2026-11-28: A second omni family: Gemma 4 E4B'.
#SBATCH -J omni_model_guard
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: OMNI_ENV=refomni5 DTYPE=bfloat16 sbatch scripts/jobs/omni_model_guard.sh <Hugging Face model> <poly|mj> "<L:F> [<L:F> ...]"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
MODEL=$1; track=$2; W=$3
short=${MODEL##*/}; DTYPE=${DTYPE:-float16}
if [ $track = poly ]; then langs="en ar de es fr it ja ko nl pl ru th yo zh"; split=harmful_eval
else langs="en ar zh it vi ko th bn sw jv"; split=harmful_test; fi
if [ -n "${OMNI_ENV:-}" ]; then source ~/anaconda3/bin/activate $OMNI_ENV; else source ~/envs/refomni/bin/activate; fi
for l in $langs; do
  echo "##### $short $track $l"
  python -u replication/omni.py check --model $MODEL --dtype $DTYPE --lang $l --forms text --n 100 --skip_answers \
    --watch $W --harmful ${track}_${l}_${split} --harmless ${track}_${l}_harmless_test || echo "FAILED check $track $l"
done
if [ -n "${OMNI_ENV:-}" ]; then conda deactivate; else deactivate; fi
source ~/anaconda3/bin/activate refneu
python -u replication/omni_guard.py --track $track --model $short --neurons $W
