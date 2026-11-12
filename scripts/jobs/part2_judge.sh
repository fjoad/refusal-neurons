#!/bin/bash
# Part 2, every language: finish judging part2_attack.sh's answers for one model and track. Every file gets
# Qwen3Guard on the original text, Llama-Guard-3 on an NLLB-200 English translation (English: on the text),
# and Apple's K.1 criterion (not refusing and coherent, gemma-2-9b-it). Yoruba, Swahili and Javanese also get
# Qwen3Guard and K.1 on the translation, because the judges read those languages poorly. Judges that already
# ran are skipped (--skip_done); Arabic (Part 1's files) is included so the table is complete.
# Results log: 'Part 2: the attack in every language'.
#SBATCH -J part2_judge
#SBATCH -t 48:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --gres=gpu:v100_16GB:2 scripts/jobs/part2_judge.sh <model> <layer> <neuron> <poly|mj>
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1"); track=$4
if [ $track = poly ]; then langs="en ar de es fr it ja ko nl pl ru th yo zh"; split=harmful_eval
else langs="en ar zh it vi ko th bn sw jv"; split=harmful_test; fi
for l in $langs; do
  f=outputs/evaluate/${short}_L$2_F$3_${track}_${l}.json
  [ -f $f ] || { echo "missing $f"; continue; }
  echo "##### $track $l"
  J="python -u replication/judge.py $f --skip_done"; TR="--translate --src_lang $l --prompts_en ${track}_en_${split}"
  $J --judge qwen3guard --bs 8
  if [ $l = en ]; then $J --bs 8; else $J $TR --bs 8; fi
  $J --judge k1 --bs 4
  case $l in yo|sw|jv) $J --judge qwen3guard $TR --bs 8; $J --judge k1 $TR --bs 4;; esac
  python -u replication/judge_agreement.py $f
done
