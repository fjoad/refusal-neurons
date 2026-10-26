#!/bin/bash
# Arabic search, part 1: unpinned en/ar runs (AUROC + refusal openings) and en/ar searches with the English phrases.
# Results log: 'Arabic search on all three models' and 'Detection'. Run for Qwen/Qwen3-1.7B 13 3270 and Llama 11 4258.
#SBATCH -J arabic
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/arabic.sh <model> <layer> <neuron>
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for lang in en ar; do
  echo "##### unpinned + AUROC, multijail_$lang"
  python -u replication/neuron_check.py --model "$1" --layer "$2" --neuron "$3" --m 0 --n_gen 100 \
    --harmful multijail_$lang --harmless aya_harmless_$lang --tag mj_${lang}_base
done
S="python -u replication/neuron_search.py --model $1 --dtype float16 --watch $2:$3"
echo "##### search en (paper phrases)"
$S --harmful multijail_train_en --harmless aya_train_en --tag mj_en
echo "##### search ar (English paper phrases)"
$S --harmful multijail_train_ar --harmless aya_train_ar --tag mj_ar_enphrases
