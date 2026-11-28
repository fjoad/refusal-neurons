#!/bin/bash
# Finishes the K.1 judging of one output file (judge.py --skip_done keeps what is already judged). Used when a
# pin-sweep job runs out of time during its last judge (Chinese, job 478801, 28 November).
# Results log: '2026-11-26: Languages on the omni model', the paragraph 'Pin sweeps in German, Arabic and Chinese'.
#SBATCH -J omni_judge_k1
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_judge_k1.sh outputs/rerank/<file>.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/anaconda3/bin/activate refneu
python -u replication/judge.py "$1" --skip_done --judge k1 --bs 4
