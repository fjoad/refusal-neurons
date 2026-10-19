#!/bin/bash
#SBATCH -J refneu_env
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out

# Creates the `refneu` conda env. torch 2.5.1 (PyPI, cu124) still supports V100 (sm_70).
set -e
source ~/anaconda3/bin/activate
conda create -y -n refneu python=3.11
conda activate refneu
pip install --no-cache-dir torch==2.5.1  # PyPI CDN: download.pytorch.org stalls from panther
pip install --no-cache-dir -r env/requirements.txt
conda clean -y --all
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
