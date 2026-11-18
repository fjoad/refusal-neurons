"""What the other top-5 candidates do, from rerank files already judged (17 November meeting, point 26a). No GPU.

For each rerank output and each judge found in it, takes every candidate at its own best pin (Apple's rule: the
smallest pin reaching 0.90; if none does, the pin with the highest rate) and prints:
  - each candidate's rate alone;
  - "any of the best k": the share of prompts jailbroken by at least one of the k best candidates, each pinned
    alone (k = 1..5). This is NOT the neurons pinned together; it shows how much room one neuron leaves and
    whether the others cover different prompts;
  - the prompts the best candidate fails on: how many of them every other candidate also fails on, against
    what independent failures would give (the product of the failure rates).
Prints numbers only, never responses.
Usage: python replication/topk_union.py outputs/rerank/<file>.json [more files ...]
"""
import json
import math
import sys
from pathlib import Path

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    runs = d["runs"]
    keys = sorted({k for r in runs for k, v in r.items()
                   if (k == "unsafe" or k.startswith("unsafe_")) and isinstance(v, list)})
    print(f"# {Path(path).name}: {len(d['prompts'])} prompts")
    for key in keys:
        base = [r for r in runs if r.get("layer") is None and key in r]
        best = {}
        for r in runs:
            if r.get("layer") is None or key not in r:
                continue
            rate = sum(r[key]) / len(r[key])
            best.setdefault((r["layer"], r["neuron"]), []).append((abs(r["pin"]), rate, r))
        if not best:
            continue
        chosen = []
        for (layer, neuron), sweep in best.items():
            sweep.sort(key=lambda x: x[0])
            reach = [s for s in sweep if s[1] >= 0.90]
            m, rate, run = reach[0] if reach else max(sweep, key=lambda x: x[1])
            chosen.append((rate, layer, neuron, run["pin"], run[key]))
        chosen.sort(key=lambda x: -x[0])
        n = len(chosen[0][4])
        none = f"{sum(base[0][key]) / n:.2f}" if base else "?"
        print(f"  judge {key}: no pin {none}")
        print("    alone, at its own best pin: "
              + "; ".join(f"L{l}:F{f} {p:+g} {r:.2f}" for r, l, f, p, _ in chosen))
        union, line = [False] * n, []
        for k, (_, _, _, _, hits) in enumerate(chosen, 1):
            union = [a or bool(b) for a, b in zip(union, hits)]
            line.append(f"k={k} {sum(union) / n:.2f}")
        print("    any of the best k (each pinned alone): " + ", ".join(line))
        fails = [i for i in range(n) if not chosen[0][4][i]]
        all_fail = sum(all(not c[4][i] for c in chosen) for i in fails)
        independent = n * math.prod(1 - c[0] for c in chosen)
        print(f"    the best one fails on {len(fails)} of {n}; all {len(chosen)} fail on {all_fail} of those "
              f"(independent failures would give {independent:.1f})")
