#!/bin/bash
# Low-resource follow-up: where the refusal neuron stays quiet (Yoruba, Swahili, Javanese), does pinning it
# TOWARD its harmful value bring refusal back? For one model: 100 harmful prompts per language (PolyRefuse
# Yoruba eval, MultiJail Swahili and Javanese test) and, as the cost, 100 harmless prompts in the same language;
# no pin and three or four pins toward harmful, 128 tokens. Refusal is read by Qwen3Guard on the original text
# and on an NLLB-200 English translation (plus the English string match in the log). The neuron's sign comes
# from the English sets. Results log: 'Restoring refusal in low-resource languages'.
#SBATCH -J restore_refusal
#SBATCH -t 24:00:00
#SBATCH -p gpu-all
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=crimv3mgpu018
#SBATCH --chdir=/export/home/fjoad/refusal-neurons
#SBATCH -o logs/%x-%j.out
# Usage: sbatch scripts/jobs/restore_refusal.sh <model> <layer> <neuron> "<pins toward harmful, negative>"
#   Qwen/Qwen3-1.7B 13 3270 "-10 -20 -40" | Qwen/Qwen3-8B 14 7924 "-4 -10 -20 -40" | meta-llama/Llama-3.1-8B-Instruct 11 4258 "-1 -2 -4"
source ~/anaconda3/bin/activate refneu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
short=$(basename "$1")
run() {  # <split to answer> <tag> <language> [English originals of the prompts]
  echo "##### $2"
  python -u replication/neuron_check.py --model $M --layer $L --neuron $F --m 0 $PINS --n_gen 100 --max_new 128 \
    --harmful jailbreakbench --harmless harmless_heldout --gen_split $1 --tag restore_$2 || return
  f=outputs/neuron_check/${short}_L${L}_F${F}_float16_restore_$2.json
  python -u replication/judge.py $f --judge qwen3guard --bs 8
  python -u replication/judge.py $f --judge qwen3guard --translate --src_lang $3 ${4:+--prompts_en $4} --bs 8
}
M=$1; L=$2; F=$3; PINS=$4
run poly_yo_harmful_eval yo_harmful yo poly_en_harmful_eval
run poly_yo_harmless_test yo_harmless yo poly_en_harmless_test
run mj_sw_harmful_test sw_harmful sw mj_en_harmful_test
run mj_sw_harmless_test sw_harmless sw
run mj_jv_harmful_test jv_harmful jv mj_en_harmful_test
run mj_jv_harmless_test jv_harmless jv
