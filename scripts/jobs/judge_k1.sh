#!/bin/bash
# Apple's main judge criterion (App. K.1: not a refusal AND coherent) on a local model (judge.py --judge k1,
# gemma-2-9b-it), then agreement with the other judges, for each file given. Results log: 'K.1 judge'.
#SBATCH -J judge_k1
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch --gres=gpu:v100_16GB:2 scripts/jobs/judge_k1.sh <output JSON> ...   (globs allowed, quoted)
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for f in $@; do
  [ -f $f ] || { echo "missing $f"; continue; }
  echo "##### $f"
  python -u replication/judge.py $f --judge k1 && python -u replication/judge_agreement.py $f
done
