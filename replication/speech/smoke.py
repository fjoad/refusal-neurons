"""Smoke test: model loads, hooks sit on the language model's MLPs, template suffix is as
expected, and the model answers text (and optionally audio). Also shows how this model opens
its refusals, which the search uses as its target phrases.

Usage: python omni/smoke.py [--audio a.wav b.wav] [--n 64]
"""
import argparse
import collections
import json
import time

import torch

from core import (OUT, SCORE_TOKENS, Record, encode, generate, is_refusal, load_model, load_split,
                  suffix_tokens, text_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="harmful prompts to answer")
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--audio_text", nargs="*", default=[])
    args = ap.parse_args()
    t0 = time.time()
    model, proc = load_model()
    layers = text_layers(model)
    mlp = layers[0].mlp
    print(f"model class {type(model).__name__}; {len(layers)} decoder layers; "
          f"down_proj in_features {mlp.down_proj.in_features}; dtype {next(model.parameters()).dtype}; "
          f"load {time.time() - t0:.0f}s", flush=True)
    print("named path of layer 0 down_proj:",
          [n for n, m in model.named_modules() if m is mlp.down_proj][0], flush=True)

    harmful, harmless = load_split("arditi_harmful_train"), load_split("arditi_harmless_train")
    items = [{"text": t} for t in harmful[:3] + harmless[:3]]
    batch = encode(proc, items)
    suf = suffix_tokens(proc, batch)
    print("suffix tokens:", suf, "OK" if suf == SCORE_TOKENS else "MISMATCH", flush=True)
    print("prompt 0 as the model sees it:\n" + proc.tokenizer.decode(batch["input_ids"][0][batch["attention_mask"][0].bool()]))

    # hook check: one Record hook on a middle layer sees [batch, n_post, k]
    rec = Record([0, 1, 2], len(SCORE_TOKENS))
    h = layers[len(layers) // 2].mlp.down_proj.register_forward_pre_hook(rec)
    with torch.no_grad():
        model(**batch, use_cache=False)
    h.remove()
    print("hook record shape:", tuple(rec.out[0].shape), flush=True)

    res = {}
    for name, prompts in (("harmful", harmful[:args.n]), ("harmless", harmless[:16])):
        t1 = time.time()
        answers = generate(model, proc, [{"text": t} for t in prompts], bs=32, max_new=96)
        refused = [is_refusal(a) for a in answers]
        print(f"\n{name}: {sum(refused)}/{len(answers)} refused by string match ({time.time() - t1:.0f}s)")
        res[name] = [{"prompt": p, "answer": a, "refused": r} for p, a, r in zip(prompts, answers, refused)]
    openings = collections.Counter(" ".join(a.split()[:6]) for a in (x["answer"] for x in res["harmful"]))
    print("\nmost common openings on harmful prompts:")
    for o, c in openings.most_common(12):
        print(f"  {c:3d}  {o!r}")

    if args.audio:
        items = [{"audio": a} for a in args.audio]
        batch = encode(proc, items)
        print("\naudio prompt suffix:", suffix_tokens(proc, batch), "length", batch["input_ids"].shape[1])
        answers = generate(model, proc, items, bs=8, max_new=96)
        res["audio"] = []
        for i, (a, ans) in enumerate(zip(args.audio, answers)):
            said = args.audio_text[i] if i < len(args.audio_text) else ""
            print(f"  audio {a}: refused by string match {is_refusal(ans)}")  # answers go to the JSON only
            res["audio"].append({"audio": a, "text": said, "answer": ans})
    out = OUT / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\nsaved outputs/speech/smoke/smoke.json ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
