"""Check every clip is intelligible: Whisper large-v3 transcribes it, word error rate vs the text.
A clip is kept when WER <= --max_wer. Adds asr, wer, keep to each manifest entry.
Runs in the refneu-omni venv.

Usage: python asr_check.py --splits arditi_harmful_train arditi_harmless_train jailbreakbench
"""
import argparse
import json
import re
from pathlib import Path

import jiwer
import librosa
import torch
from transformers import pipeline

REPO = Path(__file__).resolve().parents[3]   # repository root
AUDIO = REPO / "data/audio_speech"


def clip(m):
    p = m["path"]  # relative to AUDIO; older manifests start with data/audio/
    return AUDIO / (p[len("data/audio/"):] if p.startswith("data/audio/") else p)


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["arditi_harmful_train", "arditi_harmless_train", "jailbreakbench"])
    ap.add_argument("--max_wer", type=float, default=0.3)
    args = ap.parse_args()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", dtype=torch.bfloat16,
                   device="cuda")  # offline via HF_HUB_OFFLINE=1 in the job script
    for split in args.splits:
        path = AUDIO / split / "manifest.json"
        man = json.loads(path.read_text())
        wavs = [librosa.load(clip(m), sr=16000)[0] for m in man]
        # The cached whisper-large-v3 generation config predates the `language` kwarg; English
        # clips are auto-detected, and any misdetection shows up as a high WER (clip dropped).
        outs = asr(wavs, batch_size=16, chunk_length_s=30)
        for m, o in zip(man, outs):
            m["asr"] = o["text"].strip()
            ref, hyp = norm(m["text"]), norm(m["asr"])
            m["wer"] = round(jiwer.wer(ref, hyp), 3) if ref else 0.0
            m["keep"] = m["wer"] <= args.max_wer
        path.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        for v in sorted({m["voice"] for m in man}):
            mv = [m for m in man if m["voice"] == v]
            kept = sum(m["keep"] for m in mv)
            mean = sum(m["wer"] for m in mv) / len(mv)
            print(f"{split} {v}: kept {kept}/{len(mv)}  mean WER {mean:.3f}", flush=True)
        for m in sorted(man, key=lambda m: -m["wer"])[:3]:
            print(f"  worst {m['voice']} #{m['idx']} WER {m['wer']}: {m['text'][:70]!r} -> {m['asr'][:70]!r}")


if __name__ == "__main__":
    main()
