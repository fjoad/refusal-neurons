"""Follow-ups to the 12 November meeting that need no GPU: tables from files already in outputs/.

range:   the spread of the refusal neuron's value on harmful and on harmless prompts (the paper quotes only the
         averages, e.g. -4.41): minimum, percentiles, maximum, and how much the two groups overlap. From the
         detection scores neuron_check.py saves (the neuron's strongest value over Table 3's tokens).
base:    base model against chat model, prompt by prompt, on the same 100 harmful and 100 harmless prompts
         (token_acts.py files): when the chat model's neuron fires, does the base model's fire too? And the size
         of the harmful-minus-harmless gap in each, on the user's words and at each template token.
benign:  harmful prompts where the neuron fires at the decision point without firing on any single word of the
         request ("a sequence of benign words that leads to harmful").
Reads activations only; prompts are never printed.
Usage: sbatch scripts/cpu.sh replication/followup_tables.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHAT = [("Qwen3-8B", 14, 7924), ("Llama-3.1-8B-Instruct", 11, 4258), ("Qwen3-1.7B", 13, 3270)]
BASE = {"Qwen3-8B": "Qwen3-8B-Base", "Qwen3-1.7B": "Qwen3-1.7B-Base"}
PCT = [0, 5, 25, 50, 75, 95, 100]


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return "  ".join(f"{np.percentile(x, p):+7.2f}" for p in PCT)


def note_range():
    print("\n===== range: the neuron's strongest value at the template tokens, per prompt (raw activation)")
    print("percentiles: " + "  ".join(f"{p:>6}%" for p in PCT) + "   (0% = minimum, 100% = maximum)")
    sets = [("PolyRefuse English test, 572 harmful / 500 harmless", "poly_en_detect"), ("JailbreakBench / Alpaca, 100 / 100", ""),
            ("XSTest, 200 unsafe / 250 safe", "xstest")]
    for short, layer, neuron in CHAT:
        print(short)
        for label, tag in sets:
            d = read(OUT / "neuron_check" / f"{short}_L{layer}_F{neuron}_float16{'_' + tag if tag else ''}.json")
            if d is None or "detection_scores" not in d:
                continue
            s = np.array(d["detection_scores"]) * d["harmful_sign"]
            lab = np.array(d["detection_labels"])
            harm, safe = s[lab == 1], s[lab == 0]
            sign = d["harmful_sign"]
            cut = np.percentile(safe * sign, 95) * sign  # the harmless prompts' 95th percentile toward "harmful"
            beyond = np.mean(harm * sign > cut * sign)
            print(f"  {label}")
            print(f"    harmful : {pct(harm)}")
            print(f"    harmless: {pct(safe)}")
            print(f"    harmful prompts beyond 95% of the harmless ones ({cut:+.2f}): {beyond:.1%};  AUROC {d.get('auroc_paper_tokens', d['auroc']):.3f}")
            hist, edges = np.histogram(harm, bins=12)
            hs, _ = np.histogram(safe, bins=edges)
            print("    histogram bins (harmful/harmless counts): " + "  ".join(f"[{edges[i]:+.1f},{edges[i + 1]:+.1f}) {hist[i]}/{hs[i]}" for i in range(len(hist))))


def peaks(split, n_paper):
    """Per prompt: the neuron's strongest value over the user's words, and its value at each scored template token."""
    n_post, n_prefix = len(split["post_tokens"]), split["n_prefix"]
    content, post = [], []
    for a in split["acts"]:
        a = np.array(a)
        c = a[n_prefix:len(a) - n_post]
        content.append(c[np.abs(c).argmax()] if len(c) else 0.0)
        post.append(a[len(a) - n_post:][:n_paper])
    return np.array(content), np.array(post)


def note_base():
    print("\n===== base: base model against chat model, the same 100 harmful + 100 harmless PolyRefuse English prompts")
    for short, layer, neuron in CHAT:
        if short not in BASE:
            continue
        chat = read(OUT / "token_acts" / f"{short}_L{layer}_F{neuron}_poly_en.json")
        base = read(OUT / "token_acts" / f"{BASE[short]}_L{layer}_F{neuron}_poly_en.json")
        if chat is None or base is None:
            print(f"{short}: files missing")
            continue
        (hk, sk) = list(chat["splits"])[:2]
        n_paper = 5 if "qwen" in short.lower() else 4
        res = {}
        for name, d in (("chat", chat), ("base", base)):
            ch, ph = peaks(d["splits"][hk], n_paper)
            cs, ps = peaks(d["splits"][sk], n_paper)
            res[name] = (ch, ph, cs, ps)
        print(f"{short}  (scored template tokens: {chat['splits'][hk]['post_tokens'][:n_paper]})")
        for name in ("base", "chat"):
            ch, ph, cs, ps = res[name]
            gap_c = ch.mean() - cs.mean()
            gap_p = ph.mean(0) - ps.mean(0)
            print(f"  {name}: mean on harmful / harmless, strongest word {ch.mean():+.2f} / {cs.mean():+.2f} (gap {gap_c:+.2f}); "
                  f"gap at each template token: " + " ".join(f"{g:+.2f}" for g in gap_p))
        sign = -1.0 if res["chat"][1].mean() < res["chat"][3].mean() else 1.0
        strongest = lambda p: (p * sign).max(1) * sign
        chat_post, base_post = strongest(res["chat"][1]), strongest(res["base"][1])
        chat_cont, base_cont = res["chat"][0], res["base"][0]
        r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        print(f"  across the 100 harmful prompts, correlation of: chat decision-point value with base strongest-word value {r(chat_post, base_cont):+.2f}; "
              f"chat strongest-word with base strongest-word {r(chat_cont, base_cont):+.2f}; chat decision-point with base decision-point {r(chat_post, base_post):+.2f}")
        # "fires" = beyond the 95th percentile of that model's own harmless prompts, same quantity
        thr = lambda safe: np.percentile(safe * sign, 95)
        chat_fires = chat_post * sign > thr(strongest(res["chat"][3]))
        base_word = base_cont * sign > thr(res["base"][2])
        base_dec = base_post * sign > thr(strongest(res["base"][3]))
        n = int(chat_fires.sum())
        print(f"  chat neuron fires at the decision point on {n} of {len(chat_fires)} harmful prompts; of those, the base neuron fires on a "
              f"word in {int((chat_fires & base_word).sum())} and at its own decision point in {int((chat_fires & base_dec).sum())}")
        print(f"  base neuron fires on a word in {int(base_word.sum())} harmful prompts; of those the chat neuron fires at the decision point in "
              f"{int((base_word & chat_fires).sum())}")


def note_benign():
    print("\n===== benign: harmful prompts where the neuron fires at the decision point but on no single word")
    print("'fires' = beyond the 95th percentile of the same quantity on that model's harmless prompts")
    for short, layer, neuron in CHAT:
        n_paper = 5 if "qwen" in short.lower() else 4
        for tag in ("poly_en", "instruct"):
            d = read(OUT / "token_acts" / f"{short}_L{layer}_F{neuron}_{tag}.json")
            if d is None:
                continue
            keys = list(d["splits"])
            ch, ph = peaks(d["splits"][keys[0]], n_paper)
            cs, ps = peaks(d["splits"][keys[1]], n_paper)
            sign = -1.0 if ph.mean() < ps.mean() else 1.0
            dec = lambda p: (p * sign).max(1)
            fires_dec = dec(ph) > np.percentile(dec(ps), 95)
            fires_word = ch * sign > np.percentile(cs * sign, 95)
            n = len(ch)
            both, only_dec, only_word, neither = (int((fires_dec & fires_word).sum()), int((fires_dec & ~fires_word).sum()),
                                                  int((~fires_dec & fires_word).sum()), int((~fires_dec & ~fires_word).sum()))
            print(f"  {short} [{keys[0]}], {n} harmful prompts: fires on a word and at the decision point {both}; "
                  f"decision point only {only_dec}; a word only {only_word}; neither {neither}")
            if only_dec:
                idx = np.where(fires_dec & ~fires_word)[0]
                print(f"    the 'decision point only' prompts: mean decision-point value {np.mean(dec(ph)[idx]) * sign:+.2f}, mean strongest-word "
                      f"value {np.mean(ch[idx]):+.2f}; prompt indices {idx.tolist()[:20]}")


if __name__ == "__main__":
    for f in (note_range, note_base, note_benign):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__} failed: {type(e).__name__}: {e}")
