#!/bin/bash
# Arabic Experiment 3: capability cost on Arabic MMLU (mmmlu_ar_1000 = the same 1000 questions as mmlu_1000,
# from MMMLU) under none, constant pin, anchor k=1,2, and Arditi's Arabic (PolyRefuse) direction when one was
# selected. No Arabic GSM8K exists. Results log: 'Arabic Experiment 3'.
#SBATCH -J arabic_exp3
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arabic_exp3.sh <model> <layer> <neuron> <pin>   (after arabic_exp2.sh)
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
A=outputs/arditi/${short}_poly_ar.pt
python -u replication/capability.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
  --search outputs/neuron_search/${short}_float16_poly_ar_full.json --tasks mmlu --mmlu mmmlu_ar_1000 \
  --mmlu_instruction "أجب بحرف الخيار الصحيح فقط." --tag mmmlu_ar
python -u replication/capability.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
  --search outputs/neuron_search/${short}_float16_poly_ar_full.json --tasks mmlu --mmlu mmmlu_ar_1000 \
  --mmlu_instruction "أجب بحرف الخيار الصحيح فقط." --tag mmmlu_ar_arditi --modes arditi --arditi $A \
  || echo "no Arabic Arditi direction in $A; skipping its capability cost"
