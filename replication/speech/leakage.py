"""Leakage check between search (Arditi train), pick (HarmBench val) and test (JailbreakBench) prompts.
Prints near-duplicate prompt pairs (prompts from data/splits only, never model answers) and checks that the stored
runs used exactly the split files. CPU only. Usage (from the repo root): python3 replication/speech/leakage.py [out_dir]
(out_dir under outputs/speech/, default pin_v2)."""
import json, re, sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs/speech" / (sys.argv[1] if len(sys.argv) > 1 else "pin_v2")
S = {n: json.loads((REPO / "data/splits" / f"{n}.json").read_text())
     for n in ("arditi_harmful_train", "arditi_harmless_train", "harmbench_val", "jailbreakbench")}
for k, v in S.items(): print(k, len(v))
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())
N = {k: [norm(x) for x in v] for k, v in S.items()}

# do the stored run prompts equal the split files?
for f, split in (("attack_jailbreakbench_text_A.json", "jailbreakbench"), ("rerank_harmbench_val_text.json", "harmbench_val")):
    if not (RUNS / f).exists():
        print(f, "not found (skipped)")
        continue
    r = json.loads((RUNS / f).read_text())["runs"][0]
    print(f, "prompts == split by idx:", all(S[split][i] == p for i, p in zip(r["idx"], r["prompts"])), len(r["prompts"]))

def cmp(a, b, thr=0.8):
    hits = []
    for i, x in enumerate(N[a]):
        for j, y in enumerate(N[b]):
            if x == y: hits.append((i, j, 1.0)); continue
            r = SequenceMatcher(None, x, y).ratio()
            if r >= thr: hits.append((i, j, round(r, 3)))
    print(f"\n{a} vs {b}: {len(hits)} pairs with similarity >= {thr} (exact: {sum(h[2] == 1.0 for h in hits)})")
    for i, j, r in hits[:30]:
        print(f"  {r}  {a}[{i}] {S[a][i][:90]!r}\n        {b}[{j}] {S[b][j][:90]!r}")
cmp("jailbreakbench", "harmbench_val")
cmp("jailbreakbench", "arditi_harmful_train")
cmp("harmbench_val", "arditi_harmful_train")
cmp("jailbreakbench", "arditi_harmless_train", 0.9)
