"""Question 1: does the text-found refusal neuron fire the same way on spoken prompts?

Forward passes only. For the candidates of a search file (default: the text search's top 20),
record each prompt's value at the five post-instruction tokens, for typed and/or spoken prompts.
Per neuron, at the token the search chose: mean on harmful, mean on harmless, gap, AUROC
(harmful vs harmless, sign-corrected so 1.0 = perfect separation in the expected direction),
and for speech the per-prompt correlation with the same prompt typed.

Usage: python readout.py --search text.json --conds text speech:v1 speech:v2
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from core import OUT, SCORE_TOKENS, Record, encode, forward_logits, load_model, speech_items, text_items, text_layers

SPLITS = {"harmful": "arditi_harmful_train", "harmless": "arditi_harmless_train", "jbb": "jailbreakbench"}


@torch.no_grad()
def record(model, proc, layers, neurons, items, bs):
    """values[prompt, neuron, token] for the given (layer, neuron) list."""
    by_layer = {}
    for j, (l, f) in enumerate(neurons):
        by_layer.setdefault(l, []).append((j, f))
    hooks = {l: Record([f for _, f in v], len(SCORE_TOKENS)) for l, v in by_layer.items()}
    handles = [layers[l].mlp.down_proj.register_forward_pre_hook(h) for l, h in hooks.items()]
    out = np.zeros((len(items), len(neurons), len(SCORE_TOKENS)), dtype=np.float32)
    try:
        for i in range(0, len(items), bs):
            batch = encode(proc, items[i:i + bs])
            for h in hooks.values():
                h.out.clear()
            forward_logits(model, batch)
            for l, h in hooks.items():
                vals = h.out[0].numpy()  # [batch, tokens, k]
                for k, (j, _) in enumerate(by_layer[l]):
                    out[i:i + bs, j] = vals[:, :, k]
    finally:
        for h in handles:
            h.remove()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", default="text.json")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--conds", nargs="+", default=["text", "speech:v1", "speech:v2"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="", help="output suffix, e.g. speechsearch")
    args = ap.parse_args()
    search = json.loads((OUT / "search" / args.search).read_text())
    cands = search["candidates"][:args.k]
    neurons = [(c["layer"], c["neuron"]) for c in cands]
    tok_of = [c["post_token_index"] for c in cands]
    model, proc = load_model()
    layers = text_layers(model)
    data = {}
    for cond in args.conds:
        mod, _, voice = cond.partition(":")
        data[cond] = {}
        for key, split in SPLITS.items():
            items = speech_items(split, voice) if mod == "speech" else text_items(split)
            vals = record(model, proc, layers, neurons, items, args.bs)
            data[cond][key] = {"idx": [it["idx"] for it in items], "values": vals}
            print(f"{cond} {key}: {len(items)} prompts", flush=True)

    summary = []
    for j, ((l, f), t, c) in enumerate(zip(neurons, tok_of, cands)):
        row = {"rank": j + 1, "neuron": f"L{l}:F{f}", "token": SCORE_TOKENS[t], "score": c["score"]}
        sign = -1.0 if c["mean_act_harmful"] < c["mean_act_harmless"] else 1.0
        for cond, d in data.items():
            h, s = d["harmful"]["values"][:, j, t], d["harmless"]["values"][:, j, t]
            y = np.r_[np.ones(len(h)), np.zeros(len(s))]
            row[cond] = {"harmful": float(h.mean()), "harmless": float(s.mean()), "gap": float(h.mean() - s.mean()),
                         "auroc": float(roc_auc_score(y, sign * np.r_[h, s])),
                         "jbb": float(d["jbb"]["values"][:, j, t].mean())}
            if cond != "text" and "text" in data:  # same prompt typed vs spoken
                ti = {k: v for k, v in zip(data["text"]["harmful"]["idx"], data["text"]["harmful"]["values"][:, j, t])}
                pairs = [(ti[k], v) for k, v in zip(d["harmful"]["idx"], h) if k in ti]
                if len(pairs) > 2:
                    a, b = np.array(pairs).T
                    row[cond]["r_vs_text"] = float(np.corrcoef(a, b)[0, 1])
        summary.append(row)
    print("\nneuron           token        " + "".join(f"{c:>34s}" for c in data))
    print(" " * 30 + "".join(f"{'harmful / harmless / AUROC':>34s}" for _ in data))
    for r in summary[:10]:
        cells = "".join(f"{r[c]['harmful']:+9.2f} {r[c]['harmless']:+9.2f} {r[c]['auroc']:9.3f}    " for c in data)
        print(f"{r['rank']:2d} {r['neuron']:13s} {r['token']!r:12s} {cells}")
    out = OUT / "readout"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"values{'_' + args.tag if args.tag else ''}.npz", **{f"{c}|{k}": v["values"] for c, dd in data.items() for k, v in dd.items()})
    (out / f"summary{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps({"search": args.search, "neurons": [f"L{l}:F{f}" for l, f in neurons],
                                                  "idx": {c: {k: d[k]["idx"] for k in d} for c, d in data.items()},
                                                  "summary": summary}, indent=1))
    print(f"saved outputs/speech/readout/summary{'_' + args.tag if args.tag else ''}.json")


if __name__ == "__main__":
    main()
