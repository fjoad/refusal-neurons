#!/bin/bash
# Arabic search on Qwen3-8B: English search, Arabic search with Qwen's Arabic refusal phrase, Arabic search with English phrases.
# Results log: 'Arabic search on Qwen3-8B'. Usage: sbatch scripts/jobs/search_ar.sh
#SBATCH -J search_ar
#SBATCH -p gpu-all
#SBATCH --gres=gpu:P100_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
S="python -u replication/neuron_search.py --model Qwen/Qwen3-8B --dtype float16 --watch 14:7924"
echo "##### en (paper phrases)"
$S --harmful multijail_train_en --harmless aya_train_en --tag mj_en
echo "##### ar (Arabic refusal phrase)"
$S --harmful multijail_train_ar --harmless aya_train_ar --tag mj_ar --refusal_phrases "أنا لا أستطيع مساعدتك"
echo "##### ar (English paper phrases)"
$S --harmful multijail_train_ar --harmless aya_train_ar --tag mj_ar_enphrases
