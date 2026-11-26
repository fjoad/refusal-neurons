#!/bin/bash
# Control for the hypothesis "one neuron reads the request, others decide" (RESEARCH_DIRECTIONS.md): harmless
# English prompts with L13:F4787 pinned at +30 and with L14:F13014 pinned at +40 (and nothing pinned), typed,
# 512 new tokens, set up as the attack. If the pin on L13:F4787 simply damages reading, harmless prompts also get
# "I'm not sure what you are asking" replies; if it removes the model's reading of harm, they do not.
# No judge: omni_noengage.py counts the replies' openings (sbatch scripts/cpu.sh replication/omni_noengage.py).
# Results log: '2026-11-26: Languages on the omni model', the paragraph 'Harmless prompts with the neurons pinned'.
#SBATCH -J omni_harmless_pin
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
M=Qwen2.5-Omni-7B
source ~/envs/refomni/bin/activate
for track in poly mj; do
  for pin in "13 4787 30" "14 13014 40"; do
    read -r PL PF PV <<< "$pin"
    python -u replication/omni.py evaluate --lang en --form text --layer $PL --neuron $PF --pin $PV --k \
      --search outputs/neuron_search/${M}_float16_omni_text_${track}_en.json \
      --split ${track}_en_harmless_test --n 100 --tag harmless_${track}_en || echo "FAILED $track $pin"
  done
done
