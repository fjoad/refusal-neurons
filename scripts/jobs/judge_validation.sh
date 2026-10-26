#!/bin/bash
# Validates the Arabic judges. Qwen3-8B answers 100 held-out Arabic PolyRefuse prompts, unpinned and with
# the English refusal neuron pinned (L14:F7924 at +20). The answers are judged four ways (Llama-Guard on the
# Arabic, Llama-Guard on the NLLB English translation, Qwen3Guard on the Arabic, Qwen3Guard on the translation)
# and compared. English sanity check: Qwen3Guard vs Llama-Guard on the English M3 outputs.
# Results log: "Arabic judge validation".
#SBATCH -J judge_validation
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
AR=outputs/evaluate/Qwen3-8B_L14_F7924_ar_judgecheck.json
EN=outputs/evaluate/Qwen3-8B_L14_F7924_jbb.json
python -u replication/evaluate.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20 \
  --search outputs/neuron_search/Qwen3-8B_float16_arditi.json --split poly_ar_harmful_eval --k --tag ar_judgecheck
python -u replication/judge.py $AR
python -u replication/judge.py $AR --translate --src_lang ar --prompts_en poly_en_harmful_eval
python -u replication/judge.py $AR --judge qwen3guard
python -u replication/judge.py $AR --judge qwen3guard --translate --src_lang ar --prompts_en poly_en_harmful_eval
python -u replication/judge_agreement.py $AR
python -u replication/judge.py $EN --judge qwen3guard
python -u replication/judge_agreement.py $EN
