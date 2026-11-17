"""Point 12 of the 12 November meeting: is the neuron about "this is harmful" or about "the model will refuse"?

Reads the files scripts/jobs/harm_vs_refusal.sh makes (neuron value with a fixed sign + the model's own unpinned
answer, labelled refused or not by Qwen3Guard) and prints, per model:
  - per prompt set: how many prompts the model refuses, and the neuron's value on the refused and the answered ones;
  - the 2x2 (harmful or not x refused or not): prompts, mean neuron value, share where the neuron "fires";
  - AUROC of the neuron for refused-vs-answered among non-harmful prompts, and for harmful-vs-not among refused ones;
  - CoCoNot firing scores with the fixed sign ("should refuse" against "should answer"; redoes point 10).
"Fires" = beyond the 95th percentile of the neuron's value on harmless Alpaca prompts (the JailbreakBench / Alpaca
file of Milestone 1), one cut-off per model. Values are the neuron's strongest value over Table 3's tokens.
Reads activations and labels only; prompts and answers are never printed.
Usage: sbatch scripts/cpu.sh replication/harm_vs_refusal.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neuron_check"
MODELS = [("Qwen3-8B", 14, 7924), ("Llama-3.1-8B-Instruct", 11, 4258), ("Qwen3-1.7B", 13, 3270)]
SETS = [("CoCoNot safety (harmful)", "coconot_safety_fs", True), ("CoCoNot humanizing", "coconot_humanizing_fs", False),
        ("CoCoNot incomplete", "coconot_incomplete_fs", False), ("CoCoNot indeterminate", "coconot_indeterminate_fs", False),
        ("CoCoNot unsupported", "coconot_unsupported_fs", False), ("XSTest unsafe (harmful)", "xstest_unsafe_fs", True),
        ("XSTest safe but scary", "xstest_safe_fs", False)]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def auroc(pos, neg):
    """Chance that a prompt from pos has a larger value than one from neg (ties count half)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return "n/a (fewer than 5 in one group)"
    return f"{auroc(pos, neg):.3f}"


def cell(s, sign, cut):
    return f"n={len(s):3d}  mean value {np.mean(s) * sign:+6.2f}  fires {np.mean(s > cut):4.0%}" if len(s) else "n=  0"


def main():
    for short, layer, neuron in MODELS:
        stem = f"{short}_L{layer}_F{neuron}_float16"
        ref = read(OUT / f"{stem}.json")
        if ref is None:
            print(f"\n{short}: reference file missing")
            continue
        sign = ref["harmful_sign"]
        s, lab = np.array(ref["detection_scores"]), np.array(ref["detection_labels"])  # already x sign: larger = more harmful
        cut = np.percentile(s[lab == 0], 95)
        print(f"\n===== {short} L{layer}:F{neuron}; harmful sign {sign:+.0f}; fires = value beyond {cut * sign:+.2f} "
              f"(95% of {int((lab == 0).sum())} harmless Alpaca prompts are below it; {np.mean(s[lab == 1] > cut):.0%} of JailbreakBench fire)")
        pooled = {(h, r): [] for h in (True, False) for r in (True, False)}
        for label, tag, harmful in SETS:
            d = read(OUT / f"{stem}_{tag}.json")
            if d is None or "refusal_qwen3guard" not in d["runs"].get("None", {}):
                print(f"  {label}: file missing or not judged")
                continue
            if d["harmful_sign"] != sign:
                print(f"  {label}: sign mismatch, skipped")
                continue
            sc, lb = np.array(d["detection_scores"]), np.array(d["detection_labels"])
            refused = np.array(d["runs"]["None"]["refusal_qwen3guard"], dtype=bool)
            start = int((lb == 1).sum()) if d["args"].get("gen_split") == d["args"]["harmless"] else 0
            v = sc[start:start + len(refused)]
            pooled[(harmful, True)].append(v[refused])
            pooled[(harmful, False)].append(v[~refused])
            print(f"  {label}: {len(v)} prompts, model refuses {refused.mean():.0%}")
            print(f"      refused : {cell(v[refused], sign, cut)}")
            print(f"      answered: {cell(v[~refused], sign, cut)}")
            if tag.startswith("coconot"):
                print(f"      fixed-sign AUROC, all 'should refuse' against 'should answer' prompts ({int((lb == 1).sum())} / {int((lb == 0).sum())}): "
                      f"{auroc(sc[lb == 1], sc[lb == 0]):.3f}; mean value {sc[lb == 1].mean() * sign:+.2f} / {sc[lb == 0].mean() * sign:+.2f}")
        pooled = {k: np.concatenate(val) if val else np.array([]) for k, val in pooled.items()}
        print("  2x2, all sets pooled:")
        for h in (True, False):
            for r in (True, False):
                print(f"      {'harmful    ' if h else 'not harmful'} x {'refused ' if r else 'answered'}: {cell(pooled[(h, r)], sign, cut)}")
        print(f"  AUROC of the neuron for refused against answered, among not-harmful prompts: {auc(pooled[(False, True)], pooled[(False, False)])}")
        print(f"  AUROC of the neuron for refused against answered, among harmful prompts: {auc(pooled[(True, True)], pooled[(True, False)])}")
        print(f"  AUROC of the neuron for harmful against not harmful, among refused prompts: {auc(pooled[(True, True)], pooled[(False, True)])}")
        print(f"  AUROC of the neuron for harmful against not harmful, among answered prompts: {auc(pooled[(True, False)], pooled[(False, False)])}")


if __name__ == "__main__":
    main()
