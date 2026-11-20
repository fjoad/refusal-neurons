#!/bin/bash
# Builds the 'refomni' environment for replication/omni.py (the omni model needs transformers >= 4.52 and Pillow;
# refneu has transformers 4.51.3 and must stay as it is, since every text result was run in it). It is a venv on
# top of refneu's Python: it reuses refneu's torch and adds only the newer transformers and Pillow.
# Run once: sbatch scripts/jobs/omni_env.sh     Then in a job: source ~/envs/refomni/bin/activate
#SBATCH -J omni_env
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
set -e
~/anaconda3/envs/refneu/bin/python -m venv --system-site-packages ~/envs/refomni
source ~/envs/refomni/bin/activate
pip install --no-cache-dir "transformers==4.52.3" pillow
python -c "import transformers, PIL, torch; print('transformers', transformers.__version__, 'Pillow', PIL.__version__, 'torch', torch.__version__); from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor; print('omni classes import')"
