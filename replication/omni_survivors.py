"""Why do more refusals survive the pin on MultiJail than on PolyRefuse? (no GPU; counts only)

For the constant-pin run of omni.py evaluate (typed prompts), per dataset and language, splits the 100 harmful
prompts into those still refused under the pin (K.1's refusal label) and those not, and compares:
  1. the same prompt across languages: MultiJail and PolyRefuse are parallel, so a prompt that stays refused in
     most languages is about the request, not the language;
  2. the form of the request (English text of the prompt): a question, or an instruction; its length;
  3. the six neurons' natural values (nothing pinned) at the last template token: are the other neurons more
     active on the prompts whose refusal survives?
  4. how the surviving refusals open (first four words, only openings shared by at least 3 answers), against how
     refusals open with nothing pinned: the same stock refusal, or another style?
Prints counts and short refusal openings only, never full answers.
Usage: sbatch scripts/cpu.sh replication/omni_survivors.py
"""
import collections
import json

import numpy as np

from neuron_check import ROOT, load

M = "Qwen2.5-Omni-7B"
SETS = {"mj": ("harmful_test", "en ar zh it vi ko th bn sw jv".split()),
        "poly": ("harmful_eval", "en ar de es fr it ja ko nl pl ru th zh".split())}


def opening(text, n=4):
    w = text.split()
    return " ".join(w[:n]) if len(w) > 1 else ""


def main():
    for track, (split, langs) in SETS.items():
        refused, files = {}, {}
        for l in langs:
            f = ROOT / "outputs" / "evaluate" / f"{M}_L13_F4787_omni_text_{track}_{l}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            runs = {r["mode"]: r for r in d["runs"]}
            if "refusal_k1" not in runs.get("constant", {}):
                continue
            refused[l] = np.array([bool(x) for x in runs["constant"]["refusal_k1"]])
            files[l] = (d, runs)
        if not refused:
            print(f"\n===== {track}: no judged files")
            continue
        print(f"\n===== {track}: still refused under the pin (K.1), per language: "
              + "  ".join(f"{l} {int(v.sum())}" for l, v in refused.items()))
        # 1. the same prompt across languages
        ls = list(refused)
        n = min(len(v) for v in refused.values())
        counts = np.sum([refused[l][:n] for l in ls], axis=0)
        p = np.mean([refused[l][:n].mean() for l in ls])
        print(f"1. the same prompt across {len(ls)} languages: prompts still refused in k languages: "
              + "  ".join(f"{k}:{int((counts == k).sum())}" for k in range(len(ls) + 1)))
        exp_all = n * np.prod([refused[l][:n].mean() for l in ls])
        print(f"   refused in every language: {int((counts == len(ls)).sum())} (if languages were independent: {exp_all:.2f}); "
              f"never refused: {int((counts == 0).sum())} (independent: {n * np.prod([1 - refused[l][:n].mean() for l in ls]):.1f}); mean rate {p:.2f}")
        # 2-4 on English
        if "en" not in files:
            continue
        d, runs = files["en"]
        prompts, r = d["prompts"], refused["en"]
        q = np.array([p_.strip().endswith("?") for p_ in prompts])
        words = np.array([len(p_.split()) for p_ in prompts])
        print(f"2. English prompts: questions (end with '?') {int(q.sum())}/{len(q)}; still refused among questions "
              f"{r[q].mean() if q.any() else float('nan'):.2f}, among the rest {r[~q].mean() if (~q).any() else float('nan'):.2f}; "
              f"mean length in words: survivors {words[r].mean():.1f}, others {words[~r].mean():.1f}")
        check = ROOT / "outputs" / "omni" / f"{M}_check_{track}_en_{split}.json"
        if check.exists():
            c = json.loads(check.read_text())
            print("3. natural value at the last template token (nothing pinned), prompts whose refusal survives / the rest:")
            for e in c["neurons"]:
                if e["form"] != "text":
                    continue
                v = np.array(e["harmful_values"])[:, -1]
                first = np.array(e["harmful_values"])[:, 1]
                print(f"   L{e['layer']}:F{e['neuron']}: last newline {v[r].mean():+.2f} / {v[~r].mean():+.2f};  "
                      f"first newline {first[r].mean():+.2f} / {first[~r].mean():+.2f}")
        base = runs["none"]
        for name, run, mask in (("nothing pinned, all refusals", base, np.array([bool(x) for x in base["refusal_k1"]])),
                                ("pinned, surviving refusals", runs["constant"], r)):
            c = collections.Counter(opening(t.strip()) for t, m in zip(run["responses"], mask) if m)
            top = [(o, k) for o, k in c.most_common(8) if k >= 3 and o]
            covered = sum(k for _, k in top)
            print(f"4. openings, {name} ({int(mask.sum())}): " + "; ".join(f"{o!r} x{k}" for o, k in top)
                  + f"  [other openings: {int(mask.sum()) - covered}]")


if __name__ == "__main__":
    main()
