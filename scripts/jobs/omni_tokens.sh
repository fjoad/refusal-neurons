#!/bin/bash
# Where in the prompt each refusal neuron fires, and what pinning one neuron does to the others
# (omni_tokens.py; two tests of the hypothesis "one neuron reads the request, others decide"). English, typed
# prompts, PolyRefuse and MultiJail test prompts. No answers are generated.
# Results log: '2026-11-26: Languages on the omni model', the paragraph 'Where the neurons fire, and the chain'.
#SBATCH -J omni_tokens
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/envs/refomni/bin/activate
# With arguments, they are passed to omni_tokens.py as they are (e.g. other pin values):
#   sbatch scripts/jobs/omni_tokens.sh --tag poly_en_off --pins 13:4787:-0.63 14:13014:-0.04 16:1870:-0.01
if [ $# -gt 0 ]; then python -u replication/omni_tokens.py "$@"; exit; fi
python -u replication/omni_tokens.py --harmful poly_en_harmful_eval --harmless poly_en_harmless_test --tag poly_en || echo "FAILED poly"
python -u replication/omni_tokens.py --harmful mj_en_harmful_test --harmless mj_en_harmless_test --tag mj_en || echo "FAILED mj"
