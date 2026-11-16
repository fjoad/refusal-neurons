#!/bin/bash
# 12 November follow-up: does the refusal neuron also handle refusals that are not about harm? CoCoNot's five
# categories (data/prepare_coconot.py). Per category: the neuron's value on prompts the model should refuse against
# matched prompts it should answer (AUROC, per-token means), then the model's answers to 60 "refuse" prompts with no
# pin, with the attack pin, and with the same pin toward the harmful side (96 tokens), with Qwen3Guard's refusal label. Results log: 'Follow-ups to the 12 November meeting'.
#SBATCH -J coconot
#SBATCH -t 06:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/coconot.sh <model> <layer> <neuron> <pin magnitude>     (Qwen3-8B 14 7924 20 | Llama 11 4258 4 | Qwen3-1.7B 13 3270 40)
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
for c in safety humanizing incomplete indeterminate unsupported; do
  echo "##### $c"
  python -u replication/neuron_check.py --model "$1" --layer $2 --neuron $3 --m 0 $4 --n_gen 60 --max_new 96 \
    --harmful coconot_${c}_refuse --harmless coconot_${c}_comply --tag coconot_$c || continue
  python -u replication/judge.py outputs/neuron_check/${short}_L$2_F$3_float16_coconot_$c.json --judge qwen3guard --bs 8
done
