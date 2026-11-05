#!/bin/bash
# Arabic Experiment 7 (paper 3.4): the neuron as a harmful-prompt detector on PolyRefuse's test sets
# (572 harmful + 500 harmless), Arabic and the same prompts in English, for all 3 models; MultiJail for the
# two models not yet run; Llama-Guard-3 as the prompt-detector baseline. Results log: 'Arabic Experiment 7'.
#SBATCH -J arabic_exp7
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arabic_exp7.sh
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
C="python -u replication/neuron_check.py --m 0 --n_gen 0"
for spec in "Qwen/Qwen3-1.7B 13 3270" "Qwen/Qwen3-8B 14 7924" "meta-llama/Llama-3.1-8B-Instruct 11 4258"; do
  set -- $spec
  for lang in ar en; do
    echo "##### $1 L$2:F$3 PolyRefuse $lang"
    $C --model $1 --layer $2 --neuron $3 --harmful poly_${lang}_harmful_test --harmless poly_${lang}_harmless_test --tag poly_${lang}_detect
    if [ $1 != Qwen/Qwen3-8B ]; then
      echo "##### $1 L$2:F$3 MultiJail $lang"
      $C --model $1 --layer $2 --neuron $3 --harmful mj_${lang}_harmful_test --harmless mj_${lang}_harmless_test --tag mj_${lang}_detect
    fi
  done
done
echo "##### Llama-Guard-3 detector"
for lang in ar en; do
  python -u replication/guard_detect.py --harmful poly_${lang}_harmful_test --harmless poly_${lang}_harmless_test --tag poly_$lang
done
