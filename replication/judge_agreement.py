"""How well judges agree on the same responses: percent agreement and Cohen's kappa per pair.

Reads one or more judged output JSONs (judge.py run several times with different --judge /
--translate settings) and compares every pair of unsafe-verdict fields found in their runs:
"unsafe" (Llama-Guard on the original text), "unsafe_llamaguard_tr" (Llama-Guard on the English
translation), "unsafe_qwen3guard", and so on. Also reports each judge's overall ASR. Prints numbers
only, never responses.
Usage: python replication/judge_agreement.py outputs/evaluate/<file>.json [more files ...]
"""
import itertools
import json
import sys
from pathlib import Path


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


verdicts = {}
for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    runs = d["runs"] if isinstance(d["runs"], list) else list(d["runs"].values())
    for run in runs:
        for key, val in run.items():
            if (key == "unsafe" or key.startswith("unsafe_")) and isinstance(val, list):
                verdicts.setdefault(key, []).extend(val)

names = sorted(verdicts)
n = len(verdicts[names[0]]) if names else 0
if any(len(verdicts[k]) != n for k in names):
    sys.exit("Judges cover different numbers of responses; judge every run with every setting first")
print(f"{n} responses; judges: {', '.join(names)}")
for k in names:
    print(f"  {k}: ASR {sum(verdicts[k]) / n:.3f}")
for a, b in itertools.combinations(names, 2):
    agree = sum(x == y for x, y in zip(verdicts[a], verdicts[b])) / n
    print(f"  {a} vs {b}: agreement {agree:.3f}, Cohen's kappa {kappa(verdicts[a], verdicts[b]):.3f}")
