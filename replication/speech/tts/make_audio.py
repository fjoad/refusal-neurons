"""Speak the prompt splits in data/splits/ with XTTS-v2 (runs in the `xtts` venv).

Two built-in voices (one female, one male) so results are not tied to one speaker. Writes
data/audio_speech/<split>/<voice>/<idx>.wav (24 kHz) and data/audio_speech/<split>/manifest.json
(paths in the manifest are relative to data/audio_speech/).
Voices are named v1 / v2 in file names; the manifest keeps the XTTS speaker name.

Usage: python make_audio.py --splits arditi_harmful_train arditi_harmless_train jailbreakbench
"""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]   # repository root
AUDIO = REPO / "data/audio_speech"
VOICES = {"v1": "Claribel Dervla", "v2": "Damien Black"}


def load_xtts():
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    d = Path(snapshot_download("coqui/XTTS-v2", local_files_only=True))
    import importlib
    config = XttsConfig()
    config.load_json(str(d / "config.json"))
    model = Xtts.init_from_config(config)
    # torch >= 2.6 loads with weights_only=True; the official XTTS-v2 checkpoint also stores its
    # config objects. List the classes it references (inspection only, nothing executed) and
    # allow exactly those, provided every one comes from the TTS package (torch's recommended fix).
    names = torch.serialization.get_unsafe_globals_in_checkpoint(str(d / "model.pth"))
    print("checkpoint references:", names, flush=True)
    if any(not n.startswith("TTS.") for n in names):
        raise RuntimeError(f"unexpected non-TTS globals in checkpoint: {names}")
    # (class, stored name) pairs: some classes are re-exported, so their __module__ differs from
    # the path the checkpoint stored, and torch matches by name.
    allowed = [(getattr(importlib.import_module(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]), n) for n in names]
    with torch.serialization.safe_globals(allowed):
        model.load_checkpoint(config, checkpoint_dir=str(d), use_deepspeed=False)
    model.cuda()
    model.eval()  # XTTS overrides eval() and returns None, so no chaining
    return model, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["arditi_harmful_train", "arditi_harmless_train", "jailbreakbench"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--voices", nargs="+", default=list(VOICES), help="subset of v1 v2")
    args = ap.parse_args()
    model, config = load_xtts()
    speakers = model.speaker_manager.speakers
    print(f"{len(speakers)} built-in speakers; using {VOICES}", flush=True)
    for split in args.splits:
        texts = json.loads((REPO / "data/splits" / f"{split}.json").read_text())
        if args.n:
            texts = texts[:args.n]
        manifest, t0 = [], time.time()
        for vid, name in ((v, VOICES[v]) for v in args.voices):
            latent = speakers[name]["gpt_cond_latent"].cuda()
            emb = speakers[name]["speaker_embedding"].cuda()
            out_dir = AUDIO / split / vid
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, text in enumerate(texts):
                path = out_dir / f"{i:04d}.wav"
                if not path.exists():
                    torch.manual_seed(i)
                    with torch.no_grad():
                        wav = model.inference(text, args.lang, latent, emb, temperature=0.65,
                                              enable_text_splitting=True)["wav"]
                    sf.write(path, wav.cpu().numpy() if torch.is_tensor(wav) else wav, 24000)
                info = sf.info(path)
                manifest.append({"idx": i, "voice": vid, "speaker": name, "text": text,
                                 "path": str(path.relative_to(AUDIO)), "seconds": round(info.duration, 2)})
                if i % 50 == 0:
                    print(f"  {split} {vid} {i}/{len(texts)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        old = AUDIO / split / "manifest.json"
        if old.exists():  # keep ASR results from an earlier check
            prev = {(m["voice"], m["idx"]): m for m in json.loads(old.read_text())}
            manifest = [{**prev.get((m["voice"], m["idx"]), {}), **m} for m in manifest]
        old.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        secs = sum(m["seconds"] for m in manifest)
        print(f"{split}: {len(manifest)} clips, {secs / 60:.1f} min of audio", flush=True)


if __name__ == "__main__":
    main()
