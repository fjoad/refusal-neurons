#!/bin/bash
# How far each refusal neuron's write vector points along Nanda's direction on the omni model (Kazemi et al.
# Appendix G; omni_direction.py align). A test of the hypothesis "one neuron reads the request, others decide":
# the decision neurons should write along the direction, L13:F4787 less so. Needs the English direction
# (omni_ml_direction.sh poly en) and the English text search.
# Results log: '2026-11-26: Languages on the omni model', the paragraph 'The neurons against the direction'.
#SBATCH -J omni_align
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/envs/refomni/bin/activate
python -u replication/omni_direction.py align --tag ${1:-poly_en}
