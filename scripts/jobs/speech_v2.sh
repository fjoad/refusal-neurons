#!/bin/bash
# Speech track, pick and test stages with the stop token (v2). Results log: '2026-11-23: Speech rerun with the
# stop token (v2)'. Every answer stops at <|im_end|> (v1 answers ran on into invented "Human:" turns). The
# searches generate nothing, so they are not rerun. Outputs: outputs/speech/pin_v2/ (+ pin_v2_bs1/).
# Usage (from the repo root; run in this order):
#   sbatch scripts/jobs/speech_v2.sh rerank A            # text rerank (A), Llama-Guard + K.1 -> pick, pick_k1
#   sbatch scripts/jobs/speech_v2.sh rerank B            # voice-1 rerank on the speech search (B)
#   sbatch scripts/jobs/speech_v2.sh attack A text       # JailbreakBench at A's K.1 pick (text | v1 | v2), padding check
#   sbatch scripts/jobs/speech_v2.sh attack B v1
#   sbatch scripts/jobs/speech_v2.sh fixed 14:13014 60 text Bp60    # a given neuron and pin (here B at +60)
#   sbatch scripts/jobs/speech_v2.sh fixed 15:7277 -40 v1 BLG       # B's Llama-Guard pick
#   sbatch scripts/jobs/speech_v2.sh bs1                 # judge the one-at-a-time answers (after all attacks)
#   python3 replication/speech/summarize.py              # the results-log tables
# Each file is judged by one job only, so the two judges never write the same file at once.
#SBATCH -J speech_v2
#SBATCH -p gpu-H200
#SBATCH -A h200
#SBATCH -q h200_qos
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source "$HOME/envs/${ENV:-refneu-omni}/bin/activate"
export TMPDIR=$HOME/.cache/jobtmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$HOME/.cache/triton
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  $(date)  args $*"
cd replication/speech
O=../../outputs/speech/pin_v2
form() { case "$1" in text) echo "--modality text";; v1|v2) echo "--modality speech --voice $1";; esac; }
fpart() { case "$1" in text) echo text;; *) echo "speech_$1";; esac; }
judge() { python judge.py "$1" && python judge_k1.py "$1"; }
case "$1" in
  rerank)
    if [ "$2" = A ]; then
      python pin_run.py rerank --search text.json --split harmbench_val --modality text --max_new 128 --bs 64 --out_dir pin_v2
      judge $O/rerank_harmbench_val_text.json
    else
      python pin_run.py rerank --search speech_v1.json --split harmbench_val --modality speech --voice v1 \
        --max_new 128 --bs 32 --tag fromspeech --out_dir pin_v2
      judge $O/rerank_harmbench_val_speech_v1_fromspeech.json
    fi ;;
  attack)
    S=$2; F=$3
    if [ "$S" = A ]; then PICK=rerank_harmbench_val_text.json; else PICK=rerank_harmbench_val_speech_v1_fromspeech.json; fi
    BS=25; [ "$F" = text ] && BS=50
    python pin_run.py attack --pick $PICK --pick_key pick_k1 --split jailbreakbench $(form $F) \
      --max_new 256 --bs $BS --tag $S --out_dir pin_v2 --padcheck
    judge $O/attack_jailbreakbench_$(fpart $F)_$S.json ;;
  fixed)
    N=$2; V=$3; F=$4; T=$5
    BS=25; [ "$F" = text ] && BS=50
    python pin_run.py attack --neuron $N --value $V --split jailbreakbench $(form $F) \
      --max_new 256 --bs $BS --tag $T --out_dir pin_v2 --padcheck
    judge $O/attack_jailbreakbench_$(fpart $F)_$T.json ;;
  bs1)  # padding check: copies of every attack file with answers = the one-at-a-time answers, then both judges
    mkdir -p ${O}_bs1
    python - <<'PY'
import glob, json, os
for f in glob.glob("../../outputs/speech/pin_v2/attack_*.json"):
    d = json.load(open(f))
    for r in d["runs"]:
        r["answers"] = r.pop("answers_bs1")
        for k in [k for k in r if k.startswith(("unsafe", "asr_", "refusal_k1", "coherent_k1", "helpful_k1", "k1_"))]:
            del r[k]
    d["bs1_of"] = f
    json.dump(d, open("../../outputs/speech/pin_v2_bs1/" + os.path.basename(f), "w"), indent=1, ensure_ascii=False)
PY
    F=$(ls ${O}_bs1/attack_*.json)
    python judge.py $F && python judge_k1.py $F ;;
  *) echo "unknown mode $1"; exit 2 ;;
esac
echo "DONE $(date)"
