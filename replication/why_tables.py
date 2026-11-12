"""Follow-ups to the 5 November meeting notes that need no GPU: tables from files already in outputs/.

note 3: overlap of each language's top-5 list with the top 5 of our run of Apple's own English setup.
note 5: in each language, does the neuron fire on the harmful prompts the model refuses, and stay quiet on the
        ones it answers? Per language: the neuron's mean value and AUROC, split by Qwen3Guard's refusal label
        on the unpinned answers to the 100 attack prompts.
note 6: the neuron's strength in each language as a share of English, per model (small vs large models).
note 7: MultiJail has the same prompts in every language. Are the prompts whose pinned answers are "answered
        but judged safe" the same prompts in every language (a dataset effect), or different ones (a language
        effect)?
Reads labels and scores only; never prints prompts or answers.
Usage: sbatch scripts/cpu.sh replication/why_tables.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = [("Qwen3-8B", 14, 7924), ("Llama-3.1-8B-Instruct", 11, 4258), ("Qwen3-1.7B", 13, 3270)]
LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th yo zh".split(), "mj": "en ar zh it vi ko th bn sw jv".split()}
EVAL = {"poly": "harmful_eval", "mj": "harmful_test"}


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def split(name):
    return json.loads((ROOT / "data/splits" / f"{name}.json").read_text())


def auroc(pos, neg):
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))


def detect(short, layer, neuron, track, lang):
    return read(OUT / "neuron_check" / f"{short}_L{layer}_F{neuron}_float16_{track}_{lang}_detect.json")


def attack(short, layer, neuron, track, lang):
    return read(OUT / "evaluate" / f"{short}_L{layer}_F{neuron}_{track}_{lang}.json")


def run_of(d, mode):
    return next((r for r in d["runs"] if r.get("mode") == mode), None)


def note3():
    print("\n===== note 3: top-5 overlap with our run of Apple's English setup")
    for short, layer, neuron in MODELS:
        ref = read(OUT / "neuron_search" / f"{short}_float16_arditi.json")
        ref5 = [(c["layer"], c["neuron"]) for c in ref["candidates"]]
        print(f"{short}: Apple-setup top 5 = " + ", ".join(f"L{a}:F{b}" for a, b in ref5))
        counts, inall = Counter(), None
        for track, langs in LANGS.items():
            row = []
            for lang in langs:
                s = read(OUT / "neuron_search" / f"{short}_float16_{track}_{lang}.json") or \
                    read(OUT / "neuron_search" / f"{short}_float16_{track}_{lang}_full.json")
                if s is None:
                    row.append(f"{lang} -")
                    continue
                top = [(c["layer"], c["neuron"]) for c in s["candidates"]]
                counts.update(top)
                inall = set(top) if inall is None else inall & set(top)
                row.append(f"{lang} {len(set(top) & set(ref5))}")
            print(f"  {track}: shared with the Apple-setup top 5 (of 5): " + "  ".join(row))
        n = sum(len(v) for v in LANGS.values())
        print(f"  in every language's top 5: " + (", ".join(f"L{a}:F{b}" for a, b in sorted(inall)) or "none"))
        print("  Apple-setup top 5, lists each is in (of %d): " % n + ", ".join(f"L{a}:F{b} {counts[(a, b)]}" for a, b in ref5))


def note5():
    print("\n===== note 5: does the neuron fire on the prompts the model refuses? (unpinned answers, 100 attack prompts)")
    print("columns: refused n | neuron mean on refused / on answered / on harmless | AUROC refused-vs-harmless, "
          "answered-vs-harmless, refused-vs-answered")
    for short, layer, neuron in MODELS:
        print(short)
        for track, langs in LANGS.items():
            for lang in langs:
                det, att = detect(short, layer, neuron, track, lang), attack(short, layer, neuron, track, lang)
                if det is None or att is None:
                    continue
                none = run_of(att, "none")
                if none is None or "refusal_qwen3guard" not in none:
                    continue
                scores, labels = np.array(det["detection_scores"]), np.array(det["detection_labels"])
                harm, safe = scores[labels == 1], scores[labels == 0]
                test = split(det["args"]["harmful"])
                index = {}
                for i, p in enumerate(test):
                    index.setdefault(p, i)
                rows = [(index[p], r) for p, r in zip(att["prompts"], none["refusal_qwen3guard"]) if p in index and index[p] < len(harm)]
                if len(rows) < 20:
                    continue
                ref = [harm[i] for i, r in rows if r]
                ans = [harm[i] for i, r in rows if not r]
                sign = det["harmful_sign"]
                m = lambda xs: f"{np.mean(xs) * sign:+.2f}" if len(xs) else "  -  "
                print(f"  {track} {lang}: refused {len(ref):3d}/{len(rows)} | {m(ref)} / {m(ans)} / {m(safe)} | "
                      f"{auroc(ref, safe):.2f}, {auroc(ans, safe):.2f}, {auroc(ref, ans):.2f}")


def note6():
    print("\n===== note 6: the neuron's strength per language, as a share of English (mean harmful-minus-harmless gap "
          "at its strongest Table 3 token); AUROC in brackets")
    for track, langs in LANGS.items():
        for short, layer, neuron in MODELS:
            gaps = {}
            for lang in langs:
                det = detect(short, layer, neuron, track, lang)
                if det is None:
                    continue
                n = len(det["per_token_means"])
                rows = det["per_token_means"]
                best = max(rows, key=lambda r: abs(r["harmful"] - r["harmless"]))
                gaps[lang] = (best["harmful"] - best["harmless"], det.get("auroc_paper_tokens", det["auroc"]))
            if "en" not in gaps:
                continue
            en = gaps["en"][0]
            print(f"  {track} {short}: " + "  ".join(f"{l} {g / en:.0%} ({a:.2f})" for l, (g, a) in gaps.items()))


def note7():
    print("\n===== note 7: MultiJail, pinned answers that stop refusing but are judged safe: the same prompts in every language?")
    for short, layer, neuron in MODELS:
        safe_answer, per_lang = {}, []
        for lang in LANGS["mj"]:
            att = attack(short, layer, neuron, "mj", lang)
            run = run_of(att, "constant") if att else None
            if run is None or "refusal_qwen3guard" not in run:
                continue
            ref, uns = np.array(run["refusal_qwen3guard"]), np.array(run["unsafe_qwen3guard"])
            sa = ~ref & ~uns
            safe_answer[lang] = sa
            k1 = f", K.1 {np.mean(run['unsafe_k1']):.2f}" if "unsafe_k1" in run else ""
            per_lang.append(f"{lang}: refused {ref.mean():.2f}, harmful {uns.mean():.2f}, answered-but-safe {sa.mean():.2f}{k1}")
        if len(safe_answer) < 3:
            print(f"{short}: fewer than 3 languages judged so far")
            continue
        print(short + "\n  " + "\n  ".join(per_lang))
        stack = np.stack(list(safe_answer.values()))  # languages x prompts
        per_prompt = stack.sum(0)
        nl = len(safe_answer)
        hist = Counter(per_prompt.tolist())
        print(f"  over {nl} languages, number of prompts that are answered-but-safe in k languages: "
              + ", ".join(f"k={k}: {hist[k]}" for k in sorted(hist)))
        top20 = np.sort(per_prompt)[::-1][:20].sum() / max(1, per_prompt.sum())
        print(f"  the 20 most affected prompts hold {top20:.0%} of all answered-but-safe cases "
              f"(if prompts were unrelated across languages: about 20%)")
        langs = list(safe_answer)
        if "en" in safe_answer:
            cors = []
            for l in langs:
                if l != "en" and safe_answer[l].std() > 0 and safe_answer["en"].std() > 0:
                    cors.append(f"{l} {np.corrcoef(safe_answer['en'], safe_answer[l])[0, 1]:+.2f}")
            print("  correlation with English, prompt by prompt: " + "  ".join(cors))


if __name__ == "__main__":
    for f in (note3, note6, note7, note5):
        try:
            f()
        except Exception as e:  # one broken table should not hide the others
            print(f"{f.__name__} failed: {type(e).__name__}: {e}")
