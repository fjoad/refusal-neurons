#!/bin/bash
#SBATCH -J m3
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/m3.sh <model> <layer> <neuron> <m*>
# M3 for one neuron: JailbreakBench attack (baseline, constant, anchor k=1,2; 512 tokens),
# judged by Llama-Guard-3, then the MMLU/GSM8K capability cost under the same interventions.
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
search="outputs/neuron_search/${short}_float16_arditi.json"
python -u replication/evaluate.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" --search "$search"
python -u replication/judge.py "outputs/evaluate/${short}_L$2_F$3_jbb.json"
python -u replication/capability.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" --search "$search"
