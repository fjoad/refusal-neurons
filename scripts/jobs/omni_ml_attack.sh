#!/bin/bash
# Images in every language, the attack, set up exactly as Part 2 on the text models (part2_attack.sh and
# part2_judge.sh): for each language, the 100 held-out harmful prompts (PolyRefuse harmful_eval / MultiJail
# harmful_test) with no pin, the constant pin and the anchor pin (k = 1, 2), 512 new tokens; judged by Qwen3Guard
# on the original text, Llama-Guard-3 on an NLLB-200 English translation (English: on the text itself) and the K.1
# judge; Yoruba, Swahili and Javanese also by Qwen3Guard and K.1 on the translation. The only differences from
# Part 2: the model (Qwen2.5-Omni-7B, the English pick L13:F4787 at +30) and the prompt being given twice, as
# typed text and as a picture of the typed text. Before the attack, each language also gets the reading check and
# the six English neurons' values at the template tokens (omni.py check --skip_answers).
# Needs omni_ml_search.sh first (the anchor uses each language's own search file, per form).
# Results log: '2026-11-22: Images in every language (Qwen2.5-Omni-7B)'.
#SBATCH -J omni_ml_attack
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 72:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_ml_attack.sh <poly|mj> <language> [<language> ...]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
track=$1; shift
if [ $track = poly ]; then split=harmful_eval; else split=harmful_test; fi
M=Qwen2.5-Omni-7B
for l in "$@"; do
  echo "##### $track $l"
  source ~/envs/refomni/bin/activate
  python -u replication/omni.py check --lang $l --forms text image --n 100 --skip_answers \
    --watch 14:13014 16:1870 14:13435 16:11620 17:5484 13:4787 \
    --harmful ${track}_${l}_${split} --harmless ${track}_${l}_harmless_test || echo "FAILED check $track $l"
  for form in text image; do
    python -u replication/omni.py evaluate --lang $l --form $form --layer 13 --neuron 4787 --pin 30 \
      --search outputs/neuron_search/${M}_float16_omni_${form}_${track}_${l}.json \
      --split ${track}_${l}_${split} --tag ${track}_${l} || echo "FAILED evaluate $track $l $form"
  done
  deactivate
  source ~/anaconda3/bin/activate refneu
  for form in text image; do
    f=outputs/evaluate/${M}_L13_F4787_omni_${form}_${track}_${l}.json
    [ -f $f ] || { echo "missing $f"; continue; }
    J="python -u replication/judge.py $f --skip_done"; TR="--translate --src_lang $l --prompts_en ${track}_en_${split}"
    $J --judge qwen3guard --bs 8
    if [ $l = en ]; then $J --bs 8; else $J $TR --bs 8; fi
    $J --judge k1 --bs 4
    case $l in yo|sw|jv) $J --judge qwen3guard $TR --bs 8; $J --judge k1 $TR --bs 4;; esac
    python -u replication/judge_agreement.py $f
  done
  conda deactivate
done
