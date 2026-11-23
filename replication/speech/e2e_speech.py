"""End to end, speech in -> speech out, with the FULL Qwen2.5-Omni (thinker + talker, voice output on).

For each spoken JailbreakBench request (voice 1 clip), three runs with the same input:
untouched, A pinned (L13:4787 at +30), B pinned (L14:13014 at +60). The pin sits on the
thinker's MLP (as in all other experiments); the talker voices what the thinker produces.
Saves the input clip, each spoken reply (24 kHz wav) and its text, plus Whisper's transcript of
the spoken reply (to check the voice says what the thinker wrote), to outputs/speech/e2e/.

Usage: python e2e_speech.py --idx 16 49
"""
import argparse
import difflib
import json
import shutil

import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, pipeline

from core import AUDIO, MODEL_ID, OUT, SYSTEM, PinHook, audio_file, load_wav

CONDS = [("untouched", None), ("A", (13, 4787, 30.0)), ("B", (14, 13014, 60.0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", default=[16, 49])
    ap.add_argument("--voice", default="v1")
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--speaker", default="Chelsie")
    ap.add_argument("--conds", nargs="+", default=None, metavar="NAME[:L:F:V]",
                    help="conditions to run, e.g. B30:14:13014:30 (default: untouched, A, B)")
    args = ap.parse_args()
    conds = CONDS if not args.conds else [
        (c.split(":")[0], None if ":" not in c else (int(c.split(":")[1]), int(c.split(":")[2]), float(c.split(":")[3])))
        for c in args.conds]
    out = OUT / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    man = {m["idx"]: m for m in json.loads((AUDIO / "jailbreakbench/manifest.json").read_text()) if m["voice"] == args.voice}

    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model.eval()
    layers = model.thinker.model.layers
    print("loaded full model; thinker layers:", len(layers), flush=True)

    mpath = out / "manifest.json"
    results = json.loads(mpath.read_text()) if mpath.exists() else []   # append; never drop earlier runs
    done = {(r["idx"], r["cond"]) for r in results}
    for idx in args.idx:
        m = man[idx]
        wav_in = audio_file(m)
        in_name = f"in_{idx:03d}.wav"
        shutil.copyfile(wav_in, out / in_name)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "audio", "audio": str(wav_in)}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], audio=[load_wav(wav_in)], return_tensors="pt", padding=True).to("cuda")
        n_in = inputs["input_ids"].shape[1]
        for cond, pin in conds:
            if (idx, cond) in done:
                print(f"#{idx} {cond}: already in manifest, skipped", flush=True)
                continue
            handle = None
            if pin:
                hook = PinHook(pin[1]); hook.value = pin[2]
                handle = layers[pin[0]].mlp.down_proj.register_forward_pre_hook(hook)
            torch.manual_seed(0)
            try:
                with torch.no_grad():
                    text_ids, audio = model.generate(**inputs, speaker=args.speaker, return_audio=True, use_audio_in_video=False,
                                                     thinker_max_new_tokens=args.max_new, thinker_do_sample=False)
            finally:
                if handle:
                    handle.remove()
            ids = text_ids[0, n_in:] if text_ids.shape[1] > n_in else text_ids[0]
            reply = proc.decode(ids, skip_special_tokens=True).strip()
            name = f"out_{idx:03d}_{cond}.wav"
            wav = audio.reshape(-1).float().cpu().numpy()
            sf.write(out / name, wav, 24000)
            results.append({"idx": idx, "request": m["text"], "input": in_name, "cond": cond,
                            "pin": None if not pin else {"layer": pin[0], "neuron": pin[1], "value": pin[2]},
                            "reply_text": reply, "reply_audio": name, "seconds": round(len(wav) / 24000, 1)})
            print(f"#{idx} {cond}: {len(wav) / 24000:.1f}s, {len(reply.split())} words", flush=True)  # text goes to the JSON only
            (out / "manifest.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

    del model
    torch.cuda.empty_cache()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", dtype=torch.bfloat16, device="cuda")
    import librosa
    for r in results:
        if r.get("reply_asr"):
            continue
        wav16, _ = librosa.load(out / r["reply_audio"], sr=16000)
        r["reply_asr"] = asr(wav16, chunk_length_s=30)["text"].strip()
        match = difflib.SequenceMatcher(None, r["reply_text"].lower().split(), r["reply_asr"].lower().split()).ratio()
        print(f"#{r['idx']} {r['cond']}: spoken reply matches the text {match:.0%}", flush=True)
    (out / "manifest.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("saved outputs/speech/e2e/manifest.json")


if __name__ == "__main__":
    main()
