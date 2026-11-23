#!/bin/bash
# Speech track (replication/speech/): build the refneu-omni venv (torch, transformers, audio libs) on a CPU node.
# Results log: '2026-11-19: Speech, first results' and '2026-11-23: Speech rerun with the stop token (v2)'.
# Separate from refomni (the image work's venv): this one uses newer torch/transformers and the audio stack.
# Usage (from the repo root): sbatch scripts/jobs/speech_env.sh
#   UV=/path/to/uv PY=/path/to/python3.12 ENVNAME=refneu-omni sbatch scripts/jobs/speech_env.sh
#SBATCH -J speech_env
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
UV=${UV:-$(command -v uv || echo "$HOME/tools/miniconda3/bin/uv")}
PY=${PY:-$HOME/tools/miniconda3/bin/python3.12}
ENV=$HOME/envs/${ENVNAME:-refneu-omni}
export UV_CACHE_DIR=$HOME/.cache/uv UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
[ -d "$ENV" ] || $UV venv "$ENV" --python "$PY"
source "$ENV/bin/activate"
$UV pip install "torch==${TORCH:-2.14.0}" torchvision "transformers==5.17.0" accelerate soundfile librosa \
    scikit-learn pandas numpy qwen-omni-utils "huggingface_hub[hf_xet]" jiwer
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
python -c "from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor; print('qwen2.5-omni classes ok')"
echo DONE
