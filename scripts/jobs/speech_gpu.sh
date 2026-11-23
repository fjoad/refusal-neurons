#!/bin/bash
# Speech track: run one script from replication/speech/ (or its tts/) on one H200.
# Results log: '2026-11-19: Speech, first results' (search, readout, audio) and '2026-11-23: Speech rerun with
# the stop token (v2)' (the pin runs are in speech_v2.sh).
# Usage (from the repo root):
#   sbatch scripts/jobs/speech_gpu.sh replication/speech/search.py --modality text --phrases ...
#   ENV=xtts sbatch scripts/jobs/speech_gpu.sh replication/speech/tts/make_audio.py
#   A100 instead: sbatch -p gpu-A100 -A a100 -q a100_qos scripts/jobs/speech_gpu.sh ...
# Partition: Asim's H200 allocation (gpu-H200 / h200 / h200_qos); on gpu-all ask for an 80 GB card or
# --gres=gpu:2 (the K.1 judge alone needs two 16 GB cards). Node crirdchpxd006 once had a full /tmp and
# hung a job: add --exclude=crirdchpxd006 if it shows up again.
#SBATCH -J speech
#SBATCH -p gpu-H200
#SBATCH -A h200
#SBATCH -q h200_qos
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false COQUI_TOS_AGREED=1
source "$HOME/envs/${ENV:-refneu-omni}/bin/activate"
SCRIPT=$1; shift
cd "$(dirname "$SCRIPT")"   # the scripts import their siblings (core.py)
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  $(date)"
# Node-local /tmp can be full: keep temp files in home, removed at exit.
export TMPDIR=$HOME/.cache/jobtmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$HOME/.cache/triton
python "$(basename "$SCRIPT")" "$@"
echo "DONE $(date)"
