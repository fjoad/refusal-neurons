#!/bin/bash
#SBATCH -J cpu
#SBATCH -p cpu-all
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out

# Runs one repo script on a CPU node, for quick analyses of outputs/.
# Usage: sbatch scripts/cpu.sh replication/summarize.py outputs/rerank/<file>.json
source ~/anaconda3/bin/activate refneu
python -u "$@"
