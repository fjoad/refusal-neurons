#!/bin/bash
# Images in every language, attack (omni.py check): on Qwen2.5-Omni-7B, for each language, the 100 held-out
# harmful prompts (PolyRefuse harmful_eval / MultiJail harmful_test) and 100 harmless ones as typed text and as
# pictures of the typed text: reading check, the six English neurons' values, and answers with no pin and with
# the English pick (L13:F4787 at +30) pinned. Judged like Part 2: Qwen3Guard on the original text, Llama-Guard
# on an NLLB-200 English translation (English: on the text itself), and the K.1 judge.
# Results log: '2026-11-22: Images in every language (Qwen2.5-Omni-7B)'.
#SBATCH -J omni_ml_attack
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_ml_attack.sh <poly|mj> <language> [<language> ...]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
track=$1; shift
if [ $track = poly ]; then split=harmful_eval; else split=harmful_test; fi
for l in "$@"; do
  echo "##### $track $l"
  source ~/envs/refomni/bin/activate
  python -u replication/omni.py check --lang $l --forms text image --n 100 --pins 13:4787:30 \
    --watch 14:13014 16:1870 14:13435 16:11620 17:5484 13:4787 \
    --harmful ${track}_${l}_${split} --harmless ${track}_${l}_harmless_test || { echo "FAILED check $track $l"; deactivate; continue; }
  deactivate
  source ~/anaconda3/bin/activate refneu
  f=outputs/omni/Qwen2.5-Omni-7B_check_${track}_${l}_${split}.json
  python -u replication/judge.py $f --judge qwen3guard
  if [ $l = en ]; then python -u replication/judge.py $f
  else python -u replication/judge.py $f --translate --src_lang $l --prompts_en ${track}_en_${split}; fi
  python -u replication/judge.py $f --judge k1 --bs 8
  conda deactivate
done
