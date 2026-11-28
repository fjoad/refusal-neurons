"""Does the neuron work on the clearly harmful prompts of each dataset? (no GPU; counts only)

MultiJail scores lower than PolyRefuse under the same pin, and some MultiJail requests are borderline. This script
splits each dataset's 100 held-out harmful prompts into "clearly harmful" and "the rest" and gives the attack
numbers for each part, from the answers already generated and judged (omni.py evaluate files).
A prompt is clearly harmful when BOTH guard models call its ENGLISH text unsafe (Llama-Guard-3 and Qwen3Guard on
the typed prompt alone; guard_detect.py's saved verdicts). The split is made once, on English, and applied to the
same prompts in every language (the datasets are parallel), so it does not depend on the model under test, the
language, or the pin. Second split, for comparison: the prompts the model itself refuses in English with nothing
pinned (K.1's refusal label).
Per language and neuron: K.1 success and Qwen3Guard harmful, nothing pinned and with the constant pin, on all
prompts, on the clearly harmful ones and on the rest; and how many answers are still refusals.
Usage: sbatch scripts/cpu.sh replication/omni_clear.py
Results log: '2026-11-28: Clearly harmful prompts only'.
"""
import json

import numpy as np

from neuron_check import ROOT, load

M = "Qwen2.5-Omni-7B"
SETS = {"mj": ("harmful_test", "en ar zh it vi ko th bn jv sw".split()),
        "poly": ("harmful_eval", "en ar de es fr it ja ko nl pl ru th zh yo".split())}
NEURONS = [(14, 13014), (13, 4787)]


def rate(run, key, mask):
    if key not in run:
        return "  — "
    v = np.array([bool(x) for x in run[key]])[: len(mask)]
    return f"{v[mask[: len(v)]].mean():.2f}" if mask[: len(v)].any() else "  — "


def main():
    out = ROOT / "outputs"
    for track, (split, langs) in SETS.items():
        en_prompts = load(f"{track}_en_{split}")
        full = load(f"{track}_en_harmful_test")
        pos = [full.index(p) for p in en_prompts]
        verdicts = []
        for suffix in ("", "_qwen3guard"):
            g = json.loads((out / "guard_detect" / f"{track}_en{suffix}.json").read_text())
            verdicts.append(np.array([bool(x) for x in g["verdicts_harmful"]])[pos])
        clear = verdicts[0] & verdicts[1]
        print(f"\n===== {track}: {len(en_prompts)} held-out harmful prompts; English text called unsafe by Llama-Guard "
              f"{int(verdicts[0].sum())}, by Qwen3Guard {int(verdicts[1].sum())}, by both (clearly harmful) {int(clear.sum())}")
        for layer, neuron in NEURONS:
            en_file = out / "evaluate" / f"{M}_L{layer}_F{neuron}_omni_text_{track}_en.json"
            own = None
            if en_file.exists():
                runs = {r["mode"]: r for r in json.loads(en_file.read_text())["runs"]}
                if "refusal_k1" in runs.get("none", {}):
                    own = np.array([bool(x) for x in runs["none"]["refusal_k1"]])
            print(f"\n--- L{layer}:F{neuron} (constant pin); the model refuses "
                  + (f"{int(own.sum())} of {len(own)} English prompts with nothing pinned; of the clearly harmful ones {int((own & clear).sum())} of {int(clear.sum())}"
                     if own is not None else "(no English file)"))
            print("lang | K.1 success no pin -> pinned: all | clearly harmful | the rest || Qwen3Guard harmful pinned: all | clear | rest "
                  "|| still refused when pinned (K.1): clear | rest || K.1 pinned on prompts the model refuses in English")
            for l in langs:
                f = out / "evaluate" / f"{M}_L{layer}_F{neuron}_omni_text_{track}_{l}.json"
                if not f.exists():
                    continue
                runs = {r["mode"]: r for r in json.loads(f.read_text())["runs"]}
                none, pin = runs.get("none", {}), runs.get("constant", {})
                if "unsafe_k1" not in pin:
                    print(f"{l:4} | not judged")
                    continue
                everything = np.ones(len(clear), dtype=bool)
                cells = [f"{rate(none, 'unsafe_k1', m)} -> {rate(pin, 'unsafe_k1', m)}" for m in (everything, clear, ~clear)]
                guard = [rate(pin, "unsafe_qwen3guard", m) for m in (everything, clear, ~clear)]
                ref = [rate(pin, "refusal_k1", m) for m in (clear, ~clear)]
                mine = rate(pin, "unsafe_k1", own) if own is not None else "  — "
                print(f"{l:4} | {cells[0]} | {cells[1]} | {cells[2]} || {guard[0]} | {guard[1]} | {guard[2]} || {ref[0]} | {ref[1]} || {mine}")


if __name__ == "__main__":
    main()
