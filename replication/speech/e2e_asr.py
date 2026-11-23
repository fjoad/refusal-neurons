"""Whisper transcript of each spoken reply in outputs/speech/e2e/ (audio loaded with librosa; no ffmpeg needed).
Adds reply_asr to outputs/speech/e2e/manifest.json."""
import difflib
import json

import librosa
import torch
from transformers import pipeline

from core import OUT

out = OUT / "e2e"
rows = json.loads((out / "manifest.json").read_text())
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", dtype=torch.bfloat16, device="cuda")
for r in rows:
    wav, _ = librosa.load(out / r["reply_audio"], sr=16000)
    r["reply_asr"] = asr(wav, chunk_length_s=30)["text"].strip()
    match = difflib.SequenceMatcher(None, r["reply_text"].lower().split(), r["reply_asr"].lower().split()).ratio()
    print(f"#{r['idx']} {r['cond']}: spoken reply matches the text {match:.0%}", flush=True)  # transcripts go to the JSON only
(out / "manifest.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
print("saved outputs/speech/e2e/manifest.json")
