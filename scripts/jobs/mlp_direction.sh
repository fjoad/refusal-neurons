#!/bin/bash
# Part 3 (Nadir's ideas): the refusal direction inside the MLP (mlp_direction.py, all layers), its
# decomposition over neurons, and the attack runs (ablate it, ablate it without the neuron, steer, reset the
# neuron, Arditi's ablation with the neuron held). en: English search sets, HarmBench, Llama-Guard;
# poly/mj: Arabic search sets, Arabic rerank prompts, both Arabic judges. Results log: 'Part 3: MLP direction'.
#SBATCH -J mlp_direction
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --gres=gpu:v100_16GB:2 scripts/jobs/mlp_direction.sh <model> <layer> <neuron> <pin> [tracks: en poly mj]
set -e
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
M="python -u replication/mlp_direction.py --model $1 --layer $2 --neuron $3 --pin $4 --all_layers --bs 16"
tracks="${@:5}"  # every argument after the pin is a track
for track in ${tracks:-en poly mj}; do
  if [ $track = en ]; then
    $M --tag en
    f=outputs/evaluate/${short}_L$2_mlpdir_en.json
    python -u replication/judge.py $f
  else
    $M --tag ${track}_ar --harmful ${track}_ar_harmful_train --harmless ${track}_ar_harmless_train \
      --arditi outputs/arditi/${short}_${track}_ar.pt --eval_split ${track}_ar_harmful_rerank
    f=outputs/evaluate/${short}_L$2_mlpdir_${track}_ar.json
    python -u replication/judge.py $f --judge qwen3guard
    python -u replication/judge.py $f --translate --src_lang ar --prompts_en ${track}_en_harmful_rerank
    python -u replication/judge_agreement.py $f
  fi
done
