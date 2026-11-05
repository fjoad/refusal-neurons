#!/bin/bash
# Part 2, every language (Exp. 2): Apple's neuron at its English pin on each language's 100 held-out prompts
# (PolyRefuse harmful_eval / MultiJail harmful_test): unpinned, constant, anchor k=1,2, 512 tokens; judged by
# Qwen3Guard on the original text and Llama-Guard-3 on an NLLB-200 English translation (English: on the
# text itself). Needs part2_search.sh first (the anchor uses each language's search file). Arabic is skipped
# (done in Part 1). Results log: 'Part 2: the attack in every language'.
#SBATCH -J part2_attack
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/part2_attack.sh <model> <layer> <neuron> <pin> <poly|mj>
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1"); track=$5
if [ $track = poly ]; then langs="en de es fr it ja ko nl pl ru th yo zh"; split=harmful_eval
else langs="en zh it vi ko th bn sw jv"; split=harmful_test; fi
for l in $langs; do
  echo "##### $track $l"
  python -u replication/evaluate.py --model "$1" --layer "$2" --neuron "$3" --pin "$4" \
    --search outputs/neuron_search/${short}_float16_${track}_${l}.json --split ${track}_${l}_${split} --tag ${track}_${l} || continue
  f=outputs/evaluate/${short}_L$2_F$3_${track}_${l}.json
  python -u replication/judge.py $f --judge qwen3guard
  if [ $l = en ]; then python -u replication/judge.py $f
  else python -u replication/judge.py $f --translate --src_lang $l --prompts_en ${track}_en_${split}; fi
  python -u replication/judge_agreement.py $f
done
