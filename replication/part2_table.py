"""Part 2, the attack in every language: one table from the judged files in outputs/evaluate.

For each model, track (PolyRefuse, MultiJail) and language: the unpinned run, the constant pin and the anchor
(k = 1), under every judge that has run: Qwen3Guard on the original text (harmful and refusal labels), Llama-Guard-3
on the text (English) or on an NLLB translation, and the K.1 criterion (not refusing and coherent) with its
coherence rate. For Yoruba, Swahili and Javanese also the judges on the translation. Rates only, never text.
Usage: sbatch scripts/cpu.sh replication/part2_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("Qwen3-8B", 14, 7924), ("Llama-3.1-8B-Instruct", 11, 4258), ("Qwen3-1.7B", 13, 3270)]
LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th zh yo".split(), "mj": "en ar zh it vi ko th bn jv sw".split()}


def rate(run, key):
    v = run.get(key)
    return None if not isinstance(v, list) or not v else sum(bool(x) for x in v) / len(v)


def fmt(x):
    return " -- " if x is None else f"{x:.2f}"


def pick(d, mode, k=None):
    for r in d["runs"]:
        if r.get("mode") == mode and (k is None or r.get("k") == k):
            return r
    return None


for short, layer, neuron in MODELS:
    for track, langs in LANGS.items():
        print(f"\n===== {short} L{layer}:F{neuron}  {track}   (each cell: no pin -> constant pin [anchor k=1])")
        print("lang | harmful Qwen3Guard | refused Qwen3Guard | harmful Llama-Guard(tr) | K.1 success | K.1 coherent")
        low = []
        for lang in langs:
            path = ROOT / "outputs" / "evaluate" / f"{short}_L{layer}_F{neuron}_{track}_{lang}.json"
            if not path.exists():
                print(f"{lang:>4} | missing")
                continue
            d = json.loads(path.read_text())
            runs = [pick(d, "none"), pick(d, "constant"), pick(d, "anchor", 1)]
            if any(r is None for r in runs):
                print(f"{lang:>4} | runs missing")
                continue
            lg = "unsafe" if lang == "en" else "unsafe_llamaguard_tr"
            cells = []
            for key in ("unsafe_qwen3guard", "refusal_qwen3guard", lg, "unsafe_k1", "coherent_k1"):
                a, b, c = (rate(r, key) for r in runs)
                cells.append(f"{fmt(a)} -> {fmt(b)} [{fmt(c)}]")
            print(f"{lang:>4} | " + " | ".join(cells))
            if lang in ("yo", "sw", "jv"):
                extra = []
                for key in ("unsafe_qwen3guard_tr", "refusal_qwen3guard_tr", "unsafe_k1_tr", "refusal_k1_tr", "coherent_k1_tr", "refusal_k1"):
                    a, b = rate(runs[0], key), rate(runs[1], key)
                    extra.append(f"{key} {fmt(a)} -> {fmt(b)}")
                low.append(f"{lang:>4} on the translation / extra: " + "; ".join(extra))
        for line in low:
            print(line)
