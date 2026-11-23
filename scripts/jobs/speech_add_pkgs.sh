#!/bin/bash
# Speech track: add packages to a venv without touching its torch.
# Usage (from the repo root): sbatch scripts/jobs/speech_add_pkgs.sh <env name under ~/envs> <pkg> [pkg...]
#SBATCH -J speech_add_pkgs
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
UV=${UV:-$(command -v uv || echo "$HOME/tools/miniconda3/bin/uv")}
ENV=$HOME/envs/$1; shift
export UV_CACHE_DIR=$HOME/.cache/uv UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
TORCH=$($ENV/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
$UV pip install --python $ENV/bin/python "torch==$TORCH" "$@"
echo DONE
