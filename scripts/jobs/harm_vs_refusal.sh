#!/bin/bash
# 12 November follow-up, point 12: is the neuron about "this is harmful" or about "the model will refuse"? The
# neuron's value with its sign held fixed, and the model's own unpinned answer (refused or not, Qwen3Guard), on
# CoCoNot's five categories (60 "should refuse" prompts each; four categories are not about harm) and on XSTest
# (200 unsafe, 250 safe-but-scary). Also redoes point 10's firing scores with a fixed sign.
# Table: sbatch scripts/cpu.sh replication/harm_vs_refusal.py. Results log: 'Follow-ups to the 12 November meeting'.
#SBATCH -J harm_vs_refusal
#SBATCH -t 08:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/harm_vs_refusal.sh <model> <layer> <neuron> <harmful sign>   (Qwen3-8B 14 7924 -1 | Llama 11 4258 1 | Qwen3-1.7B 13 3270 -1)
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
run() {  # <tag> <n_gen> <neuron_check split arguments...>
  echo "##### $1"
  python -u replication/neuron_check.py --model "$model" --layer $layer --neuron $neuron --sign $sign --m 0 \
    --n_gen $2 --max_new 96 --tag $1 "${@:3}" || return
  python -u replication/judge.py outputs/neuron_check/${short}_L${layer}_F${neuron}_float16_$1.json --judge qwen3guard --bs 8
}
model=$1 layer=$2 neuron=$3 sign=$4
for c in safety humanizing incomplete indeterminate unsupported; do
  run coconot_${c}_fs 60 --harmful coconot_${c}_refuse --harmless coconot_${c}_comply
done
run xstest_unsafe_fs 200 --harmful xstest_unsafe --harmless xstest_safe
run xstest_safe_fs 250 --harmful xstest_unsafe --harmless xstest_safe --gen_split xstest_safe
