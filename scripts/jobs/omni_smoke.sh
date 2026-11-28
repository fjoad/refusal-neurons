#!/bin/bash
# A quick check that omni.py runs on a new omni model before the long jobs: loads it, prints the decoder layers and
# the template tokens it found, takes the model's own refusal openings from 32 English prompts (so generation is
# exercised) and runs the search on those 32 + 32 prompts. Typed text only. Not a result.
#SBATCH -J omni_smoke
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_smoke.sh <Hugging Face model> [dtype, default bfloat16] [conda env, default refomni5]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/anaconda3/bin/activate ${3:-refomni5}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u replication/omni.py search --model $1 --dtype ${2:-bfloat16} --n 32 --forms text --tag smoke \
  --harmful poly_en_harmful_train --harmless poly_en_harmless_train
