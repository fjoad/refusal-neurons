"""Where in the prompt each refusal neuron fires, and what pinning one neuron does to the others
(Qwen2.5-Omni-7B's thinker, typed prompts; no answers are generated).

Two tests of the hypothesis "one neuron reads the request, others decide" (RESEARCH_DIRECTIONS.md):
  1. where:  each watched neuron's value at every token of the user's request and at the five template tokens,
             on harmful and harmless prompts. A neuron that reads the request should already separate harmful
             from harmless inside the request's words; a neuron on the decision side only at the template tokens.
             Reported per neuron: its mean value on its harmful side and the AUROC (harmful against harmless),
             once from the request's tokens (the strongest token) and once from the template tokens (the
             strongest of the five, Kazemi et al.'s rule), and the share of harmful prompts whose strongest value
             in the whole prompt lies inside the request.
  2. chain:  with one neuron pinned (as in the attack, at every token), the other neurons' values at the template
             tokens on the harmful prompts: how much of each one's gap between harmful and harmless prompts the
             pin closes. Pinning a neuron that the others read from should silence them; pinning a neuron at
             the end of the chain should leave the earlier ones as they were (they are upstream by construction).
The harmful side of a neuron is the sign of its harmful-minus-harmless mean at the last template token.
Usage (refomni environment):
  python replication/omni_tokens.py --harmful poly_en_harmful_eval --harmless poly_en_harmless_test --tag poly_en
Logs carry numbers only.
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import omni
from intervene import Pin
from neuron_check import ROOT, load

N_POST = len(omni.SCORE_TOKENS)


@torch.no_grad()
def read(model, proc, pins, prompts, bs):
    """{neuron: (list of arrays over each request's tokens, array [prompts, 5] at the template tokens)} for every
    neuron whose Pin is not set (a set Pin overwrites its neuron and records nothing)."""
    tok = proc.tokenizer
    start_id = tok.convert_tokens_to_ids("<|im_start|>")
    user_id = tok.convert_tokens_to_ids("user")
    free = [k for k, p in pins.items() if p.values is None]
    req = {k: [None] * len(prompts) for k in free}
    tpl = {k: np.zeros((len(prompts), N_POST), dtype=np.float32) for k in free}
    for idx in omni.batches(proc, "text", prompts, min(bs, 8)):
        for p in pins.values():
            p.records.clear()
        batch = omni.unpadded(model, proc, "text", prompts, idx)
        model(**batch, use_cache=False)
        ids = batch["input_ids"]
        for row, i in enumerate(idx):
            starts = (ids[row] == start_id).nonzero().flatten().tolist()
            a = starts[-2] + 3  # <|im_start|> user \n, then the request; starts[-1] opens the assistant's turn
            b = ids.shape[1] - N_POST
            if int(ids[row, a - 2]) != user_id or b <= a:
                raise ValueError("could not locate the user's request in the prompt")
            for k in free:
                v = pins[k].records[0][row]
                req[k][i] = v[a:b].numpy()
                tpl[k][i] = v[-N_POST:].numpy()
    return {k: (req[k], tpl[k]) for k in free}


def auc(h, s):
    return float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(s))], np.r_[h, s]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=omni.MODEL)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--harmful", default="poly_en_harmful_eval")
    ap.add_argument("--harmless", default="poly_en_harmless_test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="poly_en")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--watch", nargs="+", default=["13:4787", "14:13014", "16:1870", "14:13435", "16:11620", "17:5484"])
    ap.add_argument("--pins", nargs="*", default=["13:4787:30", "14:13014:40", "16:1870:-40"], metavar="L:F:value")
    args = ap.parse_args()
    t0 = time.time()
    omni.LANG = args.lang
    watch = omni.parse_neurons(args.watch, 2)
    pinned = omni.parse_neurons(args.pins, 3)
    model, proc = omni.load_omni(args.model, args.dtype)
    pins = {}
    for layer, neuron in watch:
        pins[(layer, neuron)] = Pin(neuron)
        model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pins[(layer, neuron)])
    harmful, harmless = load(args.harmful)[:args.n or None], load(args.harmless)[:args.n or None]
    H, S = read(model, proc, pins, harmful, args.bs), read(model, proc, pins, harmless, args.bs)
    n_req = [len(x) for x in H[watch[0]][0]]
    print(f"{args.tag}: {len(harmful)} harmful ({args.harmful}; request length {np.mean(n_req):.1f} tokens on average), "
          f"{len(harmless)} harmless ({args.harmless})  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out = {"args": vars(args), "where": [], "chain": []}

    print("\n===== 1. where each neuron fires (values on the neuron's harmful side; AUROC harmful against harmless)")
    print("neuron      | request, strongest token: harmful / harmless  AUROC | request, mean token: AUROC | "
          "template, strongest of 5: harmful / harmless  AUROC | strongest value of the whole prompt is inside the request")
    sign = {}
    for k in watch:
        (hr, ht), (sr, st) = H[k], S[k]
        sign[k] = 1.0 if ht[:, -1].mean() > st[:, -1].mean() else -1.0
        g = sign[k]
        h_req, s_req = np.array([(g * x).max() for x in hr]), np.array([(g * x).max() for x in sr])
        h_mean, s_mean = np.array([(g * x).mean() for x in hr]), np.array([(g * x).mean() for x in sr])
        h_tpl, s_tpl = (g * ht).max(1), (g * st).max(1)
        inside = float(np.mean(h_req > h_tpl))
        where = float(np.mean([(g * x).argmax() / max(len(x) - 1, 1) for x in hr]))
        row = {"layer": k[0], "neuron": k[1], "harmful_side": g,
               "request_max": {"harmful": float(h_req.mean()), "harmless": float(s_req.mean()), "auroc": auc(h_req, s_req)},
               "request_mean": {"harmful": float(h_mean.mean()), "harmless": float(s_mean.mean()), "auroc": auc(h_mean, s_mean)},
               "template_max": {"harmful": float(h_tpl.mean()), "harmless": float(s_tpl.mean()), "auroc": auc(h_tpl, s_tpl)},
               "share_peak_inside_request": inside, "mean_relative_position_of_request_peak": where,
               "template_means_harmful": ht.mean(0).tolist(), "template_means_harmless": st.mean(0).tolist()}
        out["where"].append(row)
        print(f"L{k[0]}:F{k[1]:<6}| {h_req.mean():+6.2f} / {s_req.mean():+6.2f}  {row['request_max']['auroc']:.3f} | "
              f"{row['request_mean']['auroc']:.3f} | {h_tpl.mean():+6.2f} / {s_tpl.mean():+6.2f}  {row['template_max']['auroc']:.3f} | "
              f"{inside:.2f}  (peak at {where:.2f} of the way through the request)", flush=True)

    print("\n===== 2. chain: one neuron pinned at every token, the others read at the template tokens on the harmful prompts")
    print("value = on the neuron's harmful side, last template token (first newline in brackets); "
          "gap closed = (unpinned harmful - pinned harmful) / (unpinned harmful - harmless)")
    for layer, neuron, value in pinned:
        pins[(layer, neuron)].values = value
        P = read(model, proc, pins, harmful, args.bs)
        pins[(layer, neuron)].values = None
        print(f"pinned L{layer}:F{neuron} at {value:+g}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        for k in watch:
            if k == (layer, neuron):
                continue
            g, cells = sign[k], {}
            for name, t in (("last", -1), ("first_newline", 1)):
                h, s, p = g * H[k][1][:, t].mean(), g * S[k][1][:, t].mean(), g * P[k][1][:, t].mean()
                cells[name] = {"harmful": float(h), "harmless": float(s), "pinned": float(p),
                               "gap_closed": float((h - p) / (h - s)) if abs(h - s) > 1e-6 else None}
            out["chain"].append({"pinned": [layer, neuron, value], "layer": k[0], "neuron": k[1], **cells})
            c, f = cells["last"], cells["first_newline"]
            note = "  (earlier layer: cannot change)" if k[0] < layer else ""
            print(f"  L{k[0]}:F{k[1]:<6} harmful {c['harmful']:+6.2f}  harmless {c['harmless']:+6.2f}  pinned {c['pinned']:+6.2f}  "
                  f"gap closed {c['gap_closed']:+.2f}  ({f['gap_closed']:+.2f}){note}", flush=True)
    path = ROOT / "outputs" / "omni" / f"{args.model.split('/')[-1]}_tokens_{args.tag}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved outputs/omni/{path.name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
