#!/bin/bash
# Speech track: download the models it uses into the default HF cache (the login node kills large downloads).
# Uses the hf CLI of the refneu-omni venv (scripts/jobs/speech_env.sh). Llama-Guard-3-8B is gated: log in first.
# Usage (from the repo root): sbatch scripts/jobs/speech_download.sh [repo ...]
#SBATCH -J speech_dl
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
HF=${HF:-$HOME/envs/refneu-omni/bin/hf}
REPOS=("$@")
[ ${#REPOS[@]} -gt 0 ] || REPOS=(Qwen/Qwen2.5-Omni-7B coqui/XTTS-v2 openai/whisper-large-v3 meta-llama/Llama-Guard-3-8B google/gemma-2-9b-it)
for r in "${REPOS[@]}"; do
  echo "== $r  $(date)"
  $HF download "$r" --exclude "original/*" > /dev/null
  du -sh "$HOME/.cache/huggingface/hub/models--${r//\//--}"
done
echo DONE
