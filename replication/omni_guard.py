"""The refusal neuron as a guard on the omni model, against guard models, per language and form (no GPU).

22 November meeting, point 3: "can we use this single neuron as a guard? Same area under the curve."
For each PolyRefuse language, the same 100 harmful (harmful_eval) and up to 100 harmless (harmless_test) prompts:
  - the neuron: its value at the template tokens with nothing pinned, read by omni.py check on Qwen2.5-Omni-7B
    for typed prompts and for pictures of them. Score = the value on the neuron's harmful side, taken at the
    last template token ("last") or as the strongest of the five ("any", Kazemi et al.'s rule);
  - Llama-Guard-3 and Qwen3Guard on the typed prompt (guard_detect.py's saved scores for the full test sets;
    harmful_eval is a subset of harmful_test, so the same prompts are picked out by position).
Reports AUROC, and accuracy with ONE cut-off chosen on English text (the neuron) or the guard's own verdict.
A text guard cannot read a picture; its column is the typed version of the same prompt.
Usage: sbatch scripts/cpu.sh replication/omni_guard.py [--track poly] [--neurons 13:4787 14:13014]
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from neuron_check import ROOT, load

LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th yo zh".split(), "mj": "en ar zh it vi ko th bn sw jv".split()}
SPLIT = {"poly": "harmful_eval", "mj": "harmful_test"}


def auroc(h, s):
    return float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(s))], np.r_[h, s]))


def best_cut(h, s):
    """Cut-off with the highest accuracy on these two groups (score >= cut means harmful)."""
    cuts = np.unique(np.r_[h, s])
    acc = [((h >= c).sum() + (s < c).sum()) / (len(h) + len(s)) for c in cuts]
    return float(cuts[int(np.argmax(acc))])


def acc_at(h, s, cut):
    return float(((h >= cut).sum() + (s < cut).sum()) / (len(h) + len(s)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="poly", choices=sorted(LANGS))
    ap.add_argument("--model", default="Qwen2.5-Omni-7B")
    ap.add_argument("--neurons", nargs="+", default=["13:4787", "14:13014"])
    args = ap.parse_args()
    neurons = [tuple(int(x) for x in n.split(":")) for n in args.neurons]
    out, table = ROOT / "outputs", {}
    for lang in LANGS[args.track]:
        f = out / "omni" / f"{args.model}_check_{args.track}_{lang}_{SPLIT[args.track]}.json"
        if not f.exists():
            print(f"{lang}: no check file")
            continue
        d = json.loads(f.read_text())
        row = {}
        for e in d["neurons"]:
            if (e["layer"], e["neuron"]) not in neurons:
                continue
            h, s = np.array(e["harmful_values"]), np.array(e["harmless_values"])
            row[(e["form"], e["layer"], e["neuron"])] = (h, s)
        # guards on the same prompts, by position in the full test sets
        guards = {}
        test = load(f"{args.track}_{lang}_harmful_test") if args.track == "poly" else None
        pos = [test.index(p) for p in d["prompts"]] if test else list(range(len(d["prompts"])))
        n_safe = len(next(iter(row.values()))[1]) if row else 0
        for g, suffix in (("llamaguard", ""), ("qwen3guard", "_qwen3guard")):
            gf = out / "guard_detect" / f"{args.track}_{lang}{suffix}.json"
            if gf.exists():
                gd = json.loads(gf.read_text())
                guards[g] = (np.array(gd["scores_harmful"])[pos], np.array(gd["scores_harmless"])[:n_safe])
        table[lang] = (row, guards)

    # the neuron's harmful side and its one cut-off, both from English text
    en = table.get("en", ({}, {}))[0]
    for layer, neuron in neurons:
        if ("text", layer, neuron) not in en:
            print(f"L{layer}:F{neuron}: no English text values")
            continue
        h, s = en[("text", layer, neuron)]
        sign = 1.0 if h[:, -1].mean() > s[:, -1].mean() else -1.0
        score = {"last": lambda a: sign * a[:, -1], "any": lambda a: (sign * a).max(1)}
        for rule, fn in score.items():
            cut = best_cut(fn(h), fn(s))
            print(f"\n===== L{layer}:F{neuron}, score = value on the harmful side ({'+' if sign > 0 else '-'}), rule '{rule}'; "
                  f"cut-off from English text {cut:+.2f}")
            print("lang  | text: AUROC acc | pictures: AUROC acc | Llama-Guard: AUROC acc | Qwen3Guard: AUROC acc | n harmful/harmless")
            for lang, (row, guards) in table.items():
                cells = []
                for form in ("text", "image"):
                    if (form, layer, neuron) in row:
                        a, b = row[(form, layer, neuron)]
                        cells.append(f"{auroc(fn(a), fn(b)):.3f} {acc_at(fn(a), fn(b), cut):.2f}")
                    else:
                        cells.append("  —    — ")
                for g, verdict in (("llamaguard", 0.5), ("qwen3guard", 1.0)):
                    if g in guards:
                        a, b = guards[g]
                        cells.append(f"{auroc(a, b):.3f} {acc_at(a, b, verdict if g == 'qwen3guard' else 0.5):.2f}")
                    else:
                        cells.append("  —    — ")
                n = next(iter(row.values()))
                print(f"{lang:5} | " + " | ".join(cells) + f" | {len(n[0])}/{len(n[1])}")


if __name__ == "__main__":
    main()
