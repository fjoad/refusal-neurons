#!/bin/bash
# Follow-ups to the 5 November meeting notes, the short ones, for one model (English):
#  notes 1-2: why the constant pin costs ability and whether the anchor's zero cost means anything
#             (why_capability.py: how often the neuron fires on MMLU, dose-response, KL);
#  note 9:    how the residual-stream direction is made up (residual_share.py: over its coordinates, and
#             the refusal neuron's share of it);
#  note 4:    why Apple's neuron is not first in English on other data: the English search with one ingredient
#             changed at a time (the model's own refusal openings; PolyRefuse's or MultiJail's prompts with
#             Apple's Table 4 phrases).
# Results log: 'Follow-ups to the 5 November meeting'.
#SBATCH -J why_quick
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/why_quick.sh <model> <layer> <neuron> <pin> <detector cut-off>
#   Qwen/Qwen3-8B 14 7924 20 -2.80 | meta-llama/Llama-3.1-8B-Instruct 11 4258 -4 0.46 | Qwen/Qwen3-1.7B 13 3270 40 -7.7
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
echo "##### notes 1-2: ability cost"
python -u replication/why_capability.py --model "$1" --layer $2 --neuron $3 --pin $4 --cutoff $5 \
  --search outputs/neuron_search/${short}_float16_arditi.json --arditi outputs/arditi/${short}_jbb.pt
echo "##### note 9: residual direction"
python -u replication/residual_share.py --model "$1" --layer $2 --neuron $3 --arditi outputs/arditi/${short}_jbb.pt
S="python -u replication/neuron_search.py --model $1 --watch $2:$3"
P=outputs/refusal_phrases/${short}_multilingual.json
echo "##### note 4: Apple's prompts, the model's own refusal openings"
$S --tag why_own_phrases --phrases_json $P --phrases_key poly_en_harmful_train --phrases_fallback
echo "##### note 4: PolyRefuse English prompts, Apple's Table 4 phrases"
$S --tag why_poly_table4 --harmful poly_en_harmful_train --harmless poly_en_harmless_train
echo "##### note 4: MultiJail English prompts, Apple's Table 4 phrases"
$S --tag why_mj_table4 --harmful mj_en_harmful_train --harmless mj_en_harmless_train
