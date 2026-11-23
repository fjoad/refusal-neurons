"""Tables of the speech results log, counted from the per-answer records (never from summary fields).
Prints counts only, never answer text. CPU only.

Usage (from the repo root): python3 replication/speech/summarize.py [out_dir]   (default pin_v2)
Expects outputs/speech/<out_dir>/ from scripts/jobs/speech_v2.sh (and <out_dir>_bs1/ for the padding check).
"""
import collections
import json
import re
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = sys.argv[1] if len(sys.argv) > 1 else "pin_v2"
D = REPO / "outputs/speech" / RUN
BS1 = REPO / "outputs/speech" / f"{RUN}_bs1"
FORMS = {"text": "text", "v1": "speech_v1", "v2": "speech_v2"}
TAGS = {"A": "A, L13:F4787 (K.1 pick, text search)", "B": "B, K.1 pick (speech search)",
        "Bp60": "B neuron at +60", "BLG": "B, Llama-Guard pick"}
PINS = [2, 4, 6, 10, 15, 20, 30, 40, 60]


def loops(a):
    """Some 4-word sequence occurs 4 or more times."""
    w = re.findall(r"[a-z0-9']+", a.lower())
    g = collections.Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return bool(g) and max(g.values()) >= 4


def mcnemar(b, c):
    m = b + c
    return 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m)


def runs(path):
    d = json.loads(path.read_text())
    return next(r for r in d["runs"] if r["value"] is None), [r for r in d["runs"] if r["value"] is not None]


def picks():
    for name in ("rerank_harmbench_val_text", "rerank_harmbench_val_speech_v1_fromspeech"):
        f = D / f"{name}.json"
        if not f.exists():
            continue
        u, pinned = runs(f)
        n = len(u["answers"])
        print(f"\nPICK STAGE {name} ({n} prompts). K.1 success per pin (untouched {sum(u['unsafe_k1'])}, "
              f"Llama-Guard {sum(u['unsafe'])})")
        print("  neuron       " + " ".join(f"{p:>5d}" for p in PINS))
        by = collections.defaultdict(dict)
        for r in pinned:
            by[(r["layer"], r["neuron"])][abs(r["value"])] = r
        for (layer, neuron), rs in by.items():
            print(f"  L{layer}:F{neuron:<6d}" + " ".join(f"{sum(rs[p]['unsafe_k1']):5d}" if p in rs else "    -" for p in PINS))
        for key, field in (("Llama-Guard", "unsafe"), ("K.1", "unsafe_k1")):
            best = max(sum(r[field]) for r in pinned)
            pick = min((r for r in pinned if sum(r[field]) >= best - 0.02 * n), key=lambda r: abs(r["value"]))
            print(f"  pick by {key}: L{pick['layer']}:F{pick['neuron']} at {pick['value']:+g} "
                  f"({sum(pick[field])}/{n}; best {best})")


def tests():
    print("\nTEST SET (JailbreakBench). untouched -> pinned. K.1 = success; LG = Llama-Guard unsafe; loops")
    for tag, label in TAGS.items():
        for form, part in FORMS.items():
            f = D / f"attack_jailbreakbench_{part}_{tag}.json"
            if not f.exists():
                continue
            u, (p,) = runs(f)
            n = len(p["answers"])
            row = (f"  {label:38s} {form:4s} L{p['layer']}:F{p['neuron']} {p['value']:+g}  n={n}  "
                   f"K.1 {sum(u['unsafe_k1'])} -> {sum(p['unsafe_k1'])}  LG {sum(u['unsafe'])} -> {sum(p['unsafe'])}  "
                   f"loops {sum(map(loops, u['answers']))} -> {sum(map(loops, p['answers']))}")
            g = BS1 / f.name
            if g.exists():
                ub, (pb,) = runs(g)
                row += f"  | one at a time: K.1 {sum(ub['unsafe_k1'])} -> {sum(pb['unsafe_k1'])}, identical {u.get('same_as_bs1')}/{p.get('same_as_bs1')}"
            print(row)


def paired(a, b):
    print(f"\nK.1 success request by request, {a} vs {b}: only {a} / only {b}, exact McNemar p")
    for form, part in FORMS.items():
        fa, fb = (D / f"attack_jailbreakbench_{part}_{t}.json" for t in (a, b))
        if not (fa.exists() and fb.exists()):
            continue
        (_, (x,)), (_, (y,)) = runs(fa), runs(fb)
        assert x["idx"] == y["idx"]
        oa = sum(i and not j for i, j in zip(x["unsafe_k1"], y["unsafe_k1"]))
        ob = sum(j and not i for i, j in zip(x["unsafe_k1"], y["unsafe_k1"]))
        print(f"  {form:4s} {oa} / {ob}  p = {mcnemar(oa, ob):.3g}")


if __name__ == "__main__":
    picks()
    tests()
    paired("A", "B")
    paired("A", "Bp60")
