#!/bin/bash
# Part 2, the rerank in languages where the search does not put Apple's neuron in its top 5: pin each of that
# language's top-5 candidates, plus Apple's neuron (--add_watched), over the usual 9-value grid on the
# language's rerank prompts (PolyRefuse 191, MultiJail 87; 128 tokens), then both judges (Yoruba, Swahili,
# Javanese: also Qwen3Guard on the translation). Results log: 'Part 2: rerank where the top neuron differs'.
#SBATCH -J part2_rerank
#SBATCH -t 72:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/part2_rerank.sh <model> <track_lang> [<track_lang> ...]   e.g. Qwen/Qwen3-8B poly_ja poly_ru
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
for tl in "${@:2}"; do
  track=${tl%%_*}; l=${tl#*_}
  echo "##### $track $l"
  python -u replication/rerank.py --search outputs/neuron_search/${short}_float16_${tl}.json \
    --split ${tl}_harmful_rerank --tag $tl --add_watched || continue
  f=outputs/rerank/${short}_float16_${tl}.json
  TR="--translate --src_lang $l --prompts_en ${track}_en_harmful_rerank"
  python -u replication/judge.py $f --judge qwen3guard
  if [ $l = en ]; then python -u replication/judge.py $f; else python -u replication/judge.py $f $TR; fi
  case $l in yo|sw|jv) python -u replication/judge.py $f --judge qwen3guard $TR;; esac
  python -u replication/judge_agreement.py $f
done
