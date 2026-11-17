"""5 November follow-up, note 3, second half: overlap of the search's top-k lists across languages for k above 5.

The search saves every neuron's mean activation and gradient (*_means.pt), so the full ranking can be rebuilt
without a GPU. For each model and k in 5, 10, 20, 50, 100:
  - how many of each language set's top k are also in the top k of our run of Apple's English setup (*_arditi);
  - which neurons are in every language set's top k (and with Yoruba and Swahili left out);
  - in how many language sets' top k Apple's neuron is, and its rank in each set.
Check: the rebuilt top 5 must equal the top 5 saved by the search.
Usage: sbatch scripts/cpu.sh replication/topk_overlap.py
"""
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neuron_search"
MODELS = [("Qwen3-8B", 14, 7924), ("Llama-3.1-8B-Instruct", 11, 4258), ("Qwen3-1.7B", 13, 3270)]
LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th yo zh".split(), "mj": "en ar zh it vi ko th bn sw jv".split()}
KS = [5, 10, 20, 50, 100]
WEAK = {"poly_yo", "mj_sw"}


def ranking(stem, kmax):
    """(top kmax (layer, neuron) pairs, {(layer, neuron): rank} lookup function) of one search, rebuilt from its saved
    means with the search's own rule: best token per neuron, magnitude filter at that token, global sort by score."""
    js, pt = OUT / f"{stem}.json", OUT / f"{stem}_means.pt"
    if not js.exists() or not pt.exists():
        return None
    d = json.loads(js.read_text())
    m = torch.load(pt, map_location="cpu", weights_only=True)
    n = d["args"].get("score_tokens") or len(m["post_tokens"])
    (act_h, grad_h), (act_s, grad_s) = m["harmful"], m["harmless"]
    scores, layers, neurons = [], [], []
    for layer, (ah, gh, ash, gs) in enumerate(zip(act_h, grad_h, act_s, grad_s)):
        ah, gh, ash, gs = (x[:n].float() for x in (ah, gh, ash, gs))
        best, tok = ((gh + gs) * (ah - ash)).min(dim=0)
        idx = torch.arange(best.shape[0])
        valid = ah[tok, idx].abs() > ash[tok, idx].abs()
        scores.append(best[valid]); neurons.append(idx[valid]); layers.append(torch.full((int(valid.sum()),), layer))
    scores, layers, neurons = torch.cat(scores), torch.cat(layers), torch.cat(neurons)
    order = torch.argsort(scores, stable=True)  # ties keep (layer, neuron) order, as in the search
    ranked = list(zip(layers[order].tolist(), neurons[order].tolist()))
    saved = [(c["layer"], c["neuron"]) for c in d["candidates"]]
    if ranked[:len(saved)] != saved:
        print(f"  WARNING {stem}: rebuilt top {len(saved)} differs from the saved one")
    return ranked


def main():
    kmax = None  # the full ranking of the neurons that pass the filter
    for short, layer, neuron in MODELS:
        ref = ranking(f"{short}_float16_arditi", kmax)
        if ref is None:
            print(f"\n{short}: reference search missing")
            continue
        sets = {}
        for track, langs in LANGS.items():
            for lang in langs:
                for stem in (f"{short}_float16_{track}_{lang}", f"{short}_float16_{track}_{lang}_full"):
                    r = ranking(stem, kmax)
                    if r is not None:
                        sets[f"{track}_{lang}"] = r
                        break
        apple = (layer, neuron)
        rank = lambda r: r.index(apple) + 1 if apple in r else None
        print(f"\n===== {short}, Apple's neuron L{layer}:F{neuron}; {len(sets)} language sets; reference = our run of Apple's English setup "
              f"(Apple's neuron is rank {rank(ref)} there)")
        print("  rank of Apple's neuron in each set: " + "  ".join(f"{name} {rank(r)}" for name, r in sets.items()))
        sets = {name: r[:max(KS)] for name, r in sets.items()}
        ref = ref[:max(KS)]
        for k in KS:
            refk = set(ref[:k])
            shared = {name: len(set(r[:k]) & refk) for name, r in sets.items()}
            counts = Counter(x for r in sets.values() for x in r[:k])
            inall = set.intersection(*(set(r[:k]) for r in sets.values()))
            strong = set.intersection(*(set(r[:k]) for name, r in sets.items() if name not in WEAK))
            fmt = lambda s: ", ".join(f"L{a}:F{b}" for a, b in sorted(s)) or "none"
            print(f"  k = {k}")
            print(f"    shared with the reference top {k}: " + "  ".join(f"{name} {v}" for name, v in shared.items()))
            print(f"    mean share shared: {sum(shared.values()) / (k * len(shared)):.0%}; lowest {min(shared, key=shared.get)} "
                  f"{min(shared.values())}, highest {max(shared, key=shared.get)} {max(shared.values())}")
            print(f"    in every set's top {k} ({len(inall)}): {fmt(inall) if len(inall) <= 12 else str(len(inall)) + ' neurons'}")
            print(f"    in every set's top {k} without Yoruba and Swahili ({len(strong)}): {fmt(strong) if len(strong) <= 12 else str(len(strong)) + ' neurons'}")
            print(f"    Apple's neuron is in {counts[apple]} of {len(sets)} top-{k} lists; the most widely shared: "
                  + ", ".join(f"L{a}:F{b} {c}" for (a, b), c in counts.most_common(5)))


if __name__ == "__main__":
    main()
