#!/bin/bash
# Detection: neuron AUROC on XSTest for all 3 models, Qwen3-8B on MultiJail en/ar, and Llama-Guard-3 as a prompt detector.
# Results log: 'Detection, the neuron vs Llama-Guard-3'. Usage: sbatch scripts/jobs/detect.sh
#SBATCH -J detect
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:2
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
C="python -u replication/neuron_check.py --m 0 --n_gen 0"
for spec in "Qwen/Qwen3-1.7B 13 3270" "Qwen/Qwen3-8B 14 7924" "meta-llama/Llama-3.1-8B-Instruct 11 4258"; do
  set -- $spec
  echo "##### neuron $1 L$2:F$3 on XSTest"
  $C --model $1 --layer $2 --neuron $3 --harmful xstest_unsafe --harmless xstest_safe --tag xstest
done
for lang in en ar; do
  echo "##### neuron Qwen3-8B on MultiJail $lang"
  $C --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --harmful multijail_$lang --harmless aya_harmless_$lang --tag mj_${lang}_detect
done
echo "##### Llama-Guard-3 detector"
G="python -u replication/guard_detect.py"
$G --harmful xstest_unsafe --harmless xstest_safe --tag xstest
$G --harmful multijail_en --harmless aya_harmless_en --tag mj_en
$G --harmful multijail_ar --harmless aya_harmless_ar --tag mj_ar
