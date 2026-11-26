"""'Does not engage' replies under intervention, counted from answer openings (no GPU; counts only).

Follow-up to omni_survivors.py. An answer is counted as "does not engage" when its first 12 words contain
"not sure what", "don't understand", "do not understand", "not clear what" or "could you clarify" / "please clarify":
the model says it does not understand the request, which is neither a safety refusal nor an answer.
  dose:   for each pin value of a pin sweep (omni.py rerank files), answers K.1 calls refusals and, among all
          answers, the does-not-engage ones. Does it grow with the size of the pin?
  method: the same count for evaluate files (pin and anchor) and direction files (direction removed), so the pin
          and Nanda's ablation can be compared on the same prompts.
Usage: sbatch scripts/cpu.sh replication/omni_noengage.py
"""
import glob
import json
import re

from neuron_check import ROOT

PAT = re.compile(r"not sure what|don'?t understand|do not understand|not clear what|could you (please )?clarify|please clarify", re.I)


def noengage(text):
    return bool(PAT.search(" ".join(text.split()[:12])))


def line(run, label):
    n = len(run["responses"])
    ne = sum(noengage(t) for t in run["responses"])
    ref = sum(bool(x) for x in run.get("refusal_k1", [])) if "refusal_k1" in run else None
    both = sum(noengage(t) and bool(r) for t, r in zip(run["responses"], run.get("refusal_k1", []))) if ref is not None else None
    k1 = f"K.1 refusals {ref:3d}, of which does-not-engage {both:3d}" if ref is not None else "K.1 not run"
    print(f"  {label:34} n {n:3d}  does not engage {ne:3d}  |  {k1}")


def main():
    print("===== dose: pin sweeps (HarmBench, 191 prompts)")
    for f in sorted(glob.glob(str(ROOT / "outputs/rerank/Qwen2.5-Omni-7B_float16_omni_*_on_*.json"))):
        d = json.loads(open(f).read())
        print(f.split("/")[-1])
        for r in d["runs"]:
            if r.get("layer") is None:
                line(r, "no pin")
            elif (r["layer"], r["neuron"]) in ((13, 4787), (14, 13014), (16, 1870)):
                line(r, f"L{r['layer']}:F{r['neuron']} pin {r['pin']:+g}")
    print("\n===== method: test prompts, typed, per language set (pin, anchor, direction removed)")
    for tag in ("poly_en", "poly_de", "poly_ar", "poly_zh", "mj_en", "mj_ar", "mj_zh"):
        for name, f in (("L13:F4787 +30", f"outputs/evaluate/Qwen2.5-Omni-7B_L13_F4787_omni_text_{tag}.json"),
                        ("L14:F13014 +40", f"outputs/evaluate/Qwen2.5-Omni-7B_L14_F13014_omni_text_{tag}.json"),
                        ("direction", f"outputs/evaluate/Qwen2.5-Omni-7B_arditi_omni_text_{tag}.json")):
            if not (ROOT / f).exists():
                continue
            d = json.loads((ROOT / f).read_text())
            print(f"{tag}  [{name}]")
            for r in d["runs"]:
                line(r, r["mode"] + (f" k={r['k']:g}" if r.get("k") else ""))


if __name__ == "__main__":
    main()
