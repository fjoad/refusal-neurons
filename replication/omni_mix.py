"""Everything mixed together: one neuron as a switch and as a guard over every language and both datasets at once
(no GPU; counts only).

Guard, mixed: the neuron's value with nothing pinned (omni.py check's saved values; strongest of the five template
tokens, sign taken from English) for the harmful and harmless test prompts of every language, put in ONE pool, so
one cut-off has to serve every language and dataset: the pooled AUROC, and the accuracy at the single best pooled
cut-off. The same for Llama-Guard-3 and Qwen3Guard on the typed prompts. Pools: PolyRefuse, MultiJail, both; each
with and without the languages the model rarely refuses in (Yoruba, Swahili).
Switch, mixed: from the pinned runs already judged (omni.py evaluate files), K.1 success over all prompts of all
languages together, nothing pinned and pinned, per dataset and overall, for each neuron and pin that was run.
Usage: sbatch scripts/cpu.sh replication/omni_mix.py
Results log: '2026-11-28: ... everything mixed together'.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from neuron_check import ROOT, load

M = "Qwen2.5-Omni-7B"
LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th yo zh".split(), "mj": "en ar zh it vi ko th bn sw jv".split()}
SPLIT = {"poly": "harmful_eval", "mj": "harmful_test"}
WEAK = {("poly", "yo"), ("mj", "sw")}
NEURONS = [(14, 13014), (13, 4787), (16, 1870), (14, 13435), (16, 11620), (17, 5484)]
PINS = [(14, 13014), (13, 4787)]


def auroc(h, s):
    return float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(s))], np.r_[h, s]))


def best_acc(h, s):
    xs = np.r_[h, s]
    y = np.r_[np.ones(len(h)), np.zeros(len(s))]
    cuts = np.unique(xs)
    acc = [((xs >= c) == y).mean() for c in cuts]
    return float(max(acc))


def main():
    out = ROOT / "outputs"
    neuron, guard = {}, {}  # (track, lang) -> {(layer, neuron): (harmful scores, harmless scores)}, {name: (h, s)}
    for track, langs in LANGS.items():
        signs = {}
        for lang in langs:
            f = out / "omni" / f"{M}_check_{track}_{lang}_{SPLIT[track]}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            vals = {(e["layer"], e["neuron"]): (np.array(e["harmful_values"]), np.array(e["harmless_values"]))
                    for e in d["neurons"] if e["form"] == "text"}
            if lang == "en":
                signs = {k: 1.0 if h[:, -1].mean() > s[:, -1].mean() else -1.0 for k, (h, s) in vals.items()}
            neuron[(track, lang)] = {k: ((signs[k] * h).max(1), (signs[k] * s).max(1)) for k, (h, s) in vals.items() if k in signs}
            test = load(f"{track}_{lang}_harmful_test") if track == "poly" else None
            pos = [test.index(p) for p in d["prompts"]] if test else list(range(len(d["prompts"])))
            n_safe = len(next(iter(vals.values()))[1])
            g = {}
            for name, suffix in (("Llama-Guard", ""), ("Qwen3Guard", "_qwen3guard")):
                gf = out / "guard_detect" / f"{track}_{lang}{suffix}.json"
                if gf.exists():
                    gd = json.loads(gf.read_text())
                    g[name] = (np.array(gd["scores_harmful"])[pos], np.array(gd["scores_harmless"])[:n_safe])
            guard[(track, lang)] = g
    pools = {"PolyRefuse, 13 languages": [k for k in neuron if k[0] == "poly" and k not in WEAK],
             "MultiJail, 9 languages": [k for k in neuron if k[0] == "mj" and k not in WEAK],
             "both datasets, 22 language sets": [k for k in neuron if k not in WEAK],
             "both datasets, all 24 (Yoruba and Swahili kept in)": list(neuron)}
    print("===== GUARD, MIXED: every language's prompts in one pool (pooled AUROC; accuracy at the one best pooled cut-off)")
    for title, keys in pools.items():
        n_h = sum(len(neuron[k][NEURONS[0]][0]) for k in keys)
        n_s = sum(len(neuron[k][NEURONS[0]][1]) for k in keys)
        print(f"\n{title}: {n_h} harmful and {n_s} harmless prompts")
        for nk in NEURONS:
            if not all(nk in neuron[k] for k in keys):
                continue
            h = np.concatenate([neuron[k][nk][0] for k in keys])
            s = np.concatenate([neuron[k][nk][1] for k in keys])
            per = [auroc(*neuron[k][nk]) for k in keys]
            print(f"  L{nk[0]}:F{nk[1]:<6} pooled AUROC {auroc(h, s):.3f}  accuracy {best_acc(h, s):.3f}   (per language: lowest {min(per):.3f}, mean {np.mean(per):.3f})")
        for name in ("Llama-Guard", "Qwen3Guard"):
            if not all(name in guard[k] for k in keys):
                continue
            h = np.concatenate([guard[k][name][0] for k in keys])
            s = np.concatenate([guard[k][name][1] for k in keys])
            per = [auroc(*guard[k][name]) for k in keys]
            print(f"  {name:<12} pooled AUROC {auroc(h, s):.3f}  accuracy {best_acc(h, s):.3f}   (per language: lowest {min(per):.3f}, mean {np.mean(per):.3f})")

    print("\n===== SWITCH, MIXED: K.1 success over all prompts of all languages together (nothing pinned -> pinned)")
    for layer, nrn in PINS:
        rows = {}
        for track, langs in LANGS.items():
            for lang in langs:
                f = out / "evaluate" / f"{M}_L{layer}_F{nrn}_omni_text_{track}_{lang}.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                runs = {r["mode"]: r for r in d["runs"]}
                if "unsafe_k1" not in runs.get("constant", {}):
                    continue
                pin = runs["constant"].get("pin")
                rows[(track, lang)] = (np.array(runs["none"]["unsafe_k1"], dtype=float), np.array(runs["constant"]["unsafe_k1"], dtype=float), pin)
        if not rows:
            continue
        pin = next(iter(rows.values()))[2]
        print(f"\nL{layer}:F{nrn} pinned at {pin:+g}")
        for title, keys in (("PolyRefuse, languages the model refuses in", [k for k in rows if k[0] == "poly" and k not in WEAK]),
                            ("MultiJail, languages the model refuses in", [k for k in rows if k[0] == "mj" and k not in WEAK]),
                            ("both datasets, languages the model refuses in", [k for k in rows if k not in WEAK]),
                            ("both datasets, every language run", list(rows))):
            if not keys:
                continue
            a = np.concatenate([rows[k][0] for k in keys])
            b = np.concatenate([rows[k][1] for k in keys])
            per = [rows[k][1].mean() for k in keys]
            print(f"  {title}: {len(keys)} language sets, {len(b)} prompts: {a.mean():.3f} -> {b.mean():.3f}   (per language: lowest {min(per):.2f}, highest {max(per):.2f})")


if __name__ == "__main__":
    main()
