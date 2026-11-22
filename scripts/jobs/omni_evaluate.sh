#!/bin/bash
# Images, the English test set up exactly as Milestone 3 on the text models (evaluate.py): the 100 JailbreakBench
# prompts with no pin, the constant pin and the anchor pin (k = 1, 2), 512 new tokens, judged by Llama-Guard and
# the K.1 judge. On Qwen2.5-Omni-7B, once with typed prompts and once with pictures of the typed prompts; d comes
# from that form's own search.
# Results log: '2026-11-20: Images, typed-out pictures on Qwen2.5-Omni-7B'.
#SBATCH -J omni_evaluate
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_evaluate.sh [<layer> <neuron> <pin>]     default: 13 4787 30
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
M=Qwen2.5-Omni-7B; L=${1:-13}; F=${2:-4787}; PIN=${3:-30}
source ~/envs/refomni/bin/activate
for form in text image; do
  python -u replication/omni.py evaluate --form $form --layer $L --neuron $F --pin $PIN \
    --search outputs/neuron_search/${M}_float16_omni_${form}.json || echo "FAILED evaluate $form"
done
deactivate
source ~/anaconda3/bin/activate refneu
for form in text image; do
  f=outputs/evaluate/${M}_L${L}_F${F}_omni_${form}.json
  [ -f $f ] || { echo "missing $f"; continue; }
  python -u replication/judge.py $f --bs 8
  python -u replication/judge.py $f --judge k1 --bs 4
done
