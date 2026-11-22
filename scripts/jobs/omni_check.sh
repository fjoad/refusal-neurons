#!/bin/bash
# Images, step 1 (omni.py check): Qwen2.5-Omni-7B on the English test prompts (JailbreakBench, held-out harmless)
# as typed text, as a picture of the typed text, and as that picture plus a "follow the picture" sentence.
# Reading check, the text-found neurons' values and AUROC per form, and answers with no pin and with each neuron
# pinned, judged with Llama-Guard and the K.1 judge.
# Results log: '2026-11-20: Images, typed-out pictures on Qwen2.5-Omni-7B'.
#SBATCH -J omni_check
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/omni_check.sh [<pins, default "13:4787:30 14:13014:30"> [<tag> [<neurons to read, "L:F ...">]]]
#   Our own run's neurons (both searches' top 5, and the two the picture pin sweep picked):
#   ... "13:4787:30 16:1870:-40" ours "14:13014 16:1870 14:13435 16:11620 17:5484 13:4787"
#   The default pins are the two neurons Asim reported (text-found and speech-found, both at +30).
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
source ~/envs/refomni/bin/activate
python -u replication/omni.py check --pins ${1:-13:4787:30 14:13014:30} ${2:+--tag $2} ${3:+--watch $3}
deactivate
source ~/anaconda3/bin/activate refneu
f=outputs/omni/Qwen2.5-Omni-7B_check_jailbreakbench${2:+_$2}.json
python -u replication/judge.py $f
python -u replication/judge.py $f --judge k1 --bs 8
