#!/bin/bash
# Arabic searches for Qwen3-1.7B and Llama with each model's own Arabic refusal phrases.
# Results log: 'Arabic search on all three models'. Usage: sbatch scripts/jobs/search_ar2.sh
#SBATCH -J search_ar2
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
A="--harmful multijail_train_ar --harmless aya_train_ar --tag mj_ar"
echo "##### Qwen3-1.7B ar (own Arabic refusal phrases)"
python -u replication/neuron_search.py --model Qwen/Qwen3-1.7B --watch 13:3270 12:4580 15:1109 $A \
  --refusal_phrases "لا أستطيع أن أقدم" "أنا لا أستطيع"
echo "##### Llama-3.1-8B ar (own Arabic refusal phrases)"
python -u replication/neuron_search.py --model meta-llama/Llama-3.1-8B-Instruct --watch 11:4258 $A \
  --refusal_phrases "لا أستطيع تقديم معلومات" "لا أستطيع مساعدتك في"
