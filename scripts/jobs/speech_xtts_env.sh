#!/bin/bash
# Speech track: separate venv for XTTS-v2 text-to-speech (coqui-tts pins its own torch/transformers).
# Results log: '2026-11-19: Speech, first results' (the spoken prompts).
# Usage (from the repo root): sbatch scripts/jobs/speech_xtts_env.sh   (UV / PY as in speech_env.sh)
#SBATCH -J speech_xtts_env
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
UV=${UV:-$(command -v uv || echo "$HOME/tools/miniconda3/bin/uv")}
PY=${PY:-$HOME/tools/miniconda3/bin/python3.12}
ENV=$HOME/envs/xtts
export UV_CACHE_DIR=$HOME/.cache/uv UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
[ -d "$ENV" ] || $UV venv "$ENV" --python "$PY"
source "$ENV/bin/activate"
# torch < 2.9: newer torchaudio needs torchcodec + system FFmpeg for audio IO
$UV pip install "torch==2.8.0" "torchaudio==2.8.0" "coqui-tts" soundfile
python -c "import TTS, torch, transformers; print('coqui-tts', TTS.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
echo DONE
