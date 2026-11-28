#!/bin/bash
# Builds the environment for omni models that need transformers 5 (Gemma 4 needs 5.5 or later; refneu and refomni
# stay on transformers 4, since every earlier result was run in them). A separate conda env, `refomni5`:
# Python 3.11, torch 2.6.0 from PyPI (transformers 5 asks for torch 2.6 or later; its cu124 build still runs on
# V100 and T4 cards), the newest transformers, and what the project's scripts import.
# Run once: sbatch scripts/jobs/omni_env5.sh     Then in a job: source ~/anaconda3/bin/activate refomni5
#SBATCH -J omni_env5
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
set -e
source ~/anaconda3/bin/activate
conda env list | grep -q "^refomni5 " || conda create -y -n refomni5 python=3.11
conda activate refomni5
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0   # PyPI CDN: download.pytorch.org stalls from panther
pip install --no-cache-dir -U transformers accelerate pillow scikit-learn "numpy<2" tqdm sentencepiece protobuf timm
python -c "import torch, transformers, torchvision; print('torch', torch.__version__, 'transformers', transformers.__version__, 'torchvision', torchvision.__version__); from transformers import AutoModelForMultimodalLM, AutoProcessor; print('multimodal classes import')"
