#!/bin/bash
# Arabic Experiments 4 and 5: per-token firing of the refusal neuron over Arabic prompts (both tracks),
# with the same PolyRefuse prompts in English for comparison, in the chat models and the Qwen3 base models.
# Results log: 'Arabic Experiments 4-5'.
#SBATCH -J arabic_exp45
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arabic_exp45.sh
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for spec in "Qwen/Qwen3-1.7B 13 3270" "Qwen/Qwen3-8B 14 7924" "meta-llama/Llama-3.1-8B-Instruct 11 4258" \
            "Qwen/Qwen3-1.7B-Base 13 3270" "Qwen/Qwen3-8B-Base 14 7924"; do
  set -- $spec
  for set in "poly_ar poly_ar_harmful_eval poly_ar_harmless_test" "mj_ar mj_ar_harmful_test mj_ar_harmless_test" \
             "poly_en poly_en_harmful_eval poly_en_harmless_test"; do
    read tag harmful harmless <<< "$set"
    echo "##### $1 L$2:F$3 $tag"
    python -u replication/token_acts.py --model $1 --layer $2 --neuron $3 --splits $harmful $harmless --n 100 --tag $tag
  done
done
