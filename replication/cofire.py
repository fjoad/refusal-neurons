"""Do the top-5 neurons fire together, and does holding the others back cancel a pin? (17 November meeting, point 27)

Takes a search file's candidates and the judged rerank of the same candidates (each candidate's own best pin,
chosen as in joint_pin.py). Two parts, on the rerank's harmful prompts:

(a) Co-firing, no intervention. Each candidate's natural value is read on every harmful and harmless prompt, at
    the template token the search chose for it (the mean over Table 3's tokens if the search file has none). A
    neuron counts as "on" when its value is on the harmful side of the midpoint between its harmful and harmless
    means. Printed: the share of prompts each neuron is on, how many of the five are on per prompt, "if this one
    is on, are the others on?", and the correlations between neurons.

(b) Cancel test, with generation. The best candidate (--attackers of them) is pinned at its own best pin while
    the other four are
      held      at their mean value on harmful prompts (where they sit when the model refuses), or
      opposed   at -s x their own best pin, s in --levels (pushed the opposite way from their own attack pin).
    Controls: the other four held or opposed with no attack pin. Pins overwrite the neuron at every token, as in
    Apple's rerank.

Responses go only to outputs/cofire/<model>_<tag>.json; the log shows string-match refusal rates. After judging
(judge.py), --report prints part (a) again and every run's rate under each judge. Numbers only.

Usage: python replication/cofire.py --search outputs/neuron_search/Qwen3-8B_float16_arditi.json \\
           --rerank outputs/rerank/Qwen3-8B_float16_arditi.json --tag en
       python replication/cofire.py --report outputs/cofire/Qwen3-8B_en.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from joint_pin import best_pins, pins


def cofire_table(d):
    names = [f"L{l}:F{f}" for l, f in d["neurons"]]
    harm, safe = np.array(d["acts"]["harmful"]), np.array(d["acts"]["harmless"])
    mh, ms = harm.mean(0), safe.mean(0)
    side = np.sign(mh - ms)
    on_h = (harm - (mh + ms) / 2) * side > 0  # on = on the harmful side of the midpoint
    on_s = (safe - (mh + ms) / 2) * side > 0
    k = len(names)
    print(f"(a) co-firing: {len(harm)} harmful and {len(safe)} harmless prompts; on = harmful side of the midpoint")
    for i, name in enumerate(names):
        print(f"  {name}: mean harmful {mh[i]:+.2f}, harmless {ms[i]:+.2f}; on for {on_h[:, i].mean():.2f} of "
              f"harmful, {on_s[:, i].mean():.2f} of harmless")
    for label, on in (("harmful", on_h), ("harmless", on_s)):
        counts = on.sum(1)
        print(f"  number on per {label} prompt: "
              + ", ".join(f"{c}: {(counts == c).mean():.2f}" for c in range(k + 1)))
    for i, name in enumerate(names):
        rows = on_h[on_h[:, i]]
        if len(rows):
            others = np.delete(rows, i, axis=1)
            print(f"  harmful prompts with {name} on ({len(rows)}): all the others on in {others.all(1).mean():.2f}; "
                  f"each other one on: " + " ".join(f"{x:.2f}" for x in others.mean(0)))
        rows = on_h[~on_h[:, i]]
        if len(rows):
            others = np.delete(rows, i, axis=1)
            print(f"  harmful prompts with {name} off ({len(rows)}): each other one on: "
                  + " ".join(f"{x:.2f}" for x in others.mean(0)))
    for label, x in (("harmful prompts only", harm), ("harmful and harmless together", np.concatenate([harm, safe]))):
        c = np.corrcoef(x.T)
        print(f"  correlation between neurons, {label} (sign-corrected so + means on together):")
        for i, name in enumerate(names):
            print(f"    {name:>12} " + " ".join(f"{c[i, j] * side[i] * side[j]:+.2f}" for j in range(k)))
    none = next((r for r in d["runs"] if r["mode"] == "none"), None)
    if none:
        refused = np.array(none["refused_stringmatch"])
        counts = on_h.sum(1)
        print(f"  no pin: refused (string match) {refused.mean():.2f}; number on, refused prompts "
              f"{counts[refused].mean():.2f}, not refused {counts[~refused].mean() if (~refused).any() else float('nan'):.2f}")
        print("  refusal rate by number on: " + ", ".join(
            f"{c}: {refused[counts == c].mean():.2f} (n={int((counts == c).sum())})"
            for c in range(k + 1) if (counts == c).any()))


def report(path):
    d = json.loads(Path(path).read_text())
    print(f"# {Path(path).name}: pins chosen with {d['args']['judge_key']}")
    cofire_table(d)
    runs = d["runs"]
    keys = sorted({k for r in runs for k, v in r.items()
                   if (k == "unsafe" or k.startswith("unsafe_")) and isinstance(v, list)})
    print("(b) cancel test: " + " | ".join(["string-match refusal"] + keys + ["coherent_k1"]))
    for r in runs:
        cells = [f"{r['refusal_rate_stringmatch']:.2f}"] + [f"{sum(r[k]) / len(r[k]):.2f}" if k in r else "-" for k in keys]
        cells.append(f"{sum(bool(c) for c in r['coherent_k1']) / len(r['coherent_k1']):.2f}" if "coherent_k1" in r else "-")
        print(f"  {r['label']}: " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", help="an outputs/cofire file: print the tables and exit")
    ap.add_argument("--search", help="neuron_search.py output (the model and the candidates)")
    ap.add_argument("--rerank", help="judged rerank.py output for the same candidates (their best pins)")
    ap.add_argument("--judge_key", default=None, help="verdict field for the best pins (default: unsafe_k1 if the "
                                                      "rerank has it, else unsafe_qwen3guard, else unsafe)")
    ap.add_argument("--split", default="harmbench_val")
    ap.add_argument("--harmless", default="harmless_heldout")
    ap.add_argument("--n_harmless", type=int, default=200)
    ap.add_argument("--attackers", type=int, default=1, help="how many of the best candidates take the attack pin in turn")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="en")
    args = ap.parse_args()
    if args.report:
        return report(args.report)

    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_check import ROOT, build_prompts, generate, is_refusal, load, paper_score_tokens

    t0 = time.time()
    search = json.loads((ROOT / args.search).read_text())
    rerank = json.loads((ROOT / args.rerank).read_text())
    if args.judge_key is None:
        args.judge_key = next(k for k in ("unsafe_k1", "unsafe_qwen3guard", "unsafe")
                              if any(k in r for r in rerank["runs"]))
    token_of = {(c["layer"], c["neuron"]): c.get("post_token_index") for c in search["candidates"]}
    chosen = [c for c in best_pins(rerank, args.judge_key) if (c[1], c[2]) in token_of]
    if len(chosen) < 2:
        raise SystemExit("Fewer than two candidates have judged rerank runs")
    model_name = search["args"]["model"]
    print(f"{model_name}: best pins by {args.judge_key}: "
          + "; ".join(f"L{l}:F{f} {p:+g} ({r:.2f})" for r, l, f, p in chosen), flush=True)

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    raw, raw_safe = load(args.split), load(args.harmless)[:args.n_harmless]
    prompts, n_post, post_tokens = build_prompts(tok, raw)
    n_score = paper_score_tokens(model_name, post_tokens)

    # (a) natural values of every candidate, no intervention
    by_layer, store = {}, {}
    for _, layer, neuron, _ in chosen:
        by_layer.setdefault(layer, []).append(neuron)

    def recorder(layer):
        def hook(module, hook_args):
            store[layer] = hook_args[0][:, -n_post:, by_layer[layer]].detach().float().cpu()  # left padding
        return hook

    handles = [model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(recorder(layer)) for layer in by_layer]

    @torch.no_grad()
    def values(texts):
        rows = []
        for i in range(0, len(texts), args.bs):
            batch = tok(texts[i:i + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**batch, logits_to_keep=1)
            cols = []
            for _, layer, neuron, _ in chosen:
                a = store[layer][:, :, by_layer[layer].index(neuron)]  # [batch, post tokens]
                t = token_of[(layer, neuron)]
                cols.append(a[:, t] if t is not None else a[:, :n_score].mean(1))
            rows.append(torch.stack(cols, 1))
        return torch.cat(rows).tolist()

    acts = {"harmful": values(prompts), "harmless": values(build_prompts(tok, raw_safe)[0])}
    for handle in handles:
        handle.remove()
    harm_mean = np.array(acts["harmful"]).mean(0)

    out = ROOT / "outputs" / "cofire" / f"{model_name.split('/')[-1]}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "model": model_name, "prompts": raw,
              "neurons": [[l, f] for _, l, f, _ in chosen], "acts": acts, "runs": []}

    def run(mode, label, items, **fields):
        with pins(model, items):
            texts = generate(model, tok, SimpleNamespace(records=[]), prompts, args.bs, args.max_new)
        flags = [is_refusal(t) for t in texts]
        result["runs"].append({"layer": None, "neuron": None, "pin": None, "mode": mode, "label": label, **fields,
                               "neurons": [list(i) for i in items], "refusal_rate_stringmatch": sum(flags) / len(flags),
                               "refused_stringmatch": flags, "responses": texts})
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{label}: string-match refusal {sum(flags) / len(flags):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    run("none", "no pin", [])
    cofire_table(result)

    # (b) one pinned to the attack side, the others held or pushed the other way
    for a in range(min(args.attackers, len(chosen))):
        _, al, af, ap_ = chosen[a]
        name, attack = f"L{al}:F{af} at {ap_:+g}", [(al, af, ap_)]
        others = [(i, c) for i, c in enumerate(chosen) if i != a]
        held = [(l, f, float(harm_mean[i])) for i, (_, l, f, _) in others]
        run("attack", f"{name} alone", attack)
        run("attack_held", f"{name}, others held at their harmful mean", attack + held)
        run("held_only", f"no attack pin (attacker L{al}:F{af} free), others held at their harmful mean", held)
        for s in args.levels:
            opposed = [(l, f, -s * p) for _, (_, l, f, p) in others]
            run("attack_opposed", f"{name}, others at -{s:g}x their own pin", attack + opposed, k=s)
            run("opposed_only", f"no attack pin (attacker L{al}:F{af} free), others at -{s:g}x their own pin", opposed, k=s)
    print(f"saved {out.relative_to(ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
