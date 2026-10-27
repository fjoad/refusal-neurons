#!/bin/bash
# Arabic Experiment 1 part 2: judge one model's Arabic rerank answers (both tracks) with both Arabic
# judges, Qwen3Guard on the Arabic and Llama-Guard on an NLLB-200 translation; each prints the 0.90 rule.
# Results log: 'Arabic Experiment 1'.
#SBATCH -J judge_ar
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/judge_ar_rerank.sh <model short name, e.g. Qwen3-8B> [tracks, default "poly mj"]
# On 2x P100 an 8B model's PolyRefuse half alone takes ~6.5 h (2x V100-16GB: ~1 h); ask for V100s with
# sbatch --gres=gpu:v100_16GB:2 ...
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for track in ${2:-poly mj}; do
  f=outputs/rerank/$1_float16_${track}_ar.json
  python -u replication/judge.py $f --judge qwen3guard
  python -u replication/judge.py $f --translate --src_lang ar --prompts_en ${track}_en_harmful_rerank
  python -u replication/judge_agreement.py $f
done
