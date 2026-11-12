#!/bin/bash
# Part 2, the neuron as one guard across languages. (1) The baselines: Llama-Guard-3 and Qwen3Guard as
# prompt-only detectors on every language's test sets (PolyRefuse 572 harmful + 500 harmless, MultiJail
# 100 + 100). (2) Each model's refusal neuron as a filter with ONE cut-off, chosen on the English test set of
# the same dataset and applied unchanged to every language (detect_threshold.py --calibrate), from the
# detection scores part2_search.sh saved. Results log: 'Part 2: one guard across languages'.
#SBATCH -J part2_guard
#SBATCH -t 12:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/part2_guard.sh
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
POLY="en ar de es fr it ja ko nl pl ru th yo zh"; MJ="en ar zh it vi ko th bn sw jv"
H=""; S=""; T=""
for l in $POLY; do H="$H poly_${l}_harmful_test"; S="$S poly_${l}_harmless_test"; T="$T poly_$l"; done
for l in $MJ; do H="$H mj_${l}_harmful_test"; S="$S mj_${l}_harmless_test"; T="$T mj_$l"; done
for g in llamaguard qwen3guard; do
  echo "##### $g"
  python -u replication/guard_detect.py --guard $g --harmful $H --harmless $S --tag $T
done
for n in Qwen3-1.7B_L13_F3270 Qwen3-8B_L14_F7924 Llama-3.1-8B-Instruct_L11_F4258; do
  for track in poly mj; do
    echo "##### neuron $n $track (cut-off from English)"
    d=outputs/neuron_check/${n}_float16_${track}
    python -u replication/detect_threshold.py ${d}_*_detect.json --calibrate ${d}_en_detect.json
  done
done
