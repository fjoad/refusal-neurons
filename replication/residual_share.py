"""How is the residual-stream refusal direction made up? (meeting note 9, two readings)

Reading A: spread the direction over its own coordinates, as mlp_direction.py does for the MLP direction.
  share of coordinate i = d_i^2 / sum_j d_j^2. If the largest share is tiny, no single residual-stream number
  carries the direction, which would explain why pinning one residual number fails (paper Section 3.3).
Reading B: how much of the direction does one MLP neuron account for? The residual stream at a layer's input is
  the sum of everything written before it, and an MLP layer writes sum_j h_j * w_j (w_j = neuron j's write
  vector, a column of down_proj). So with D = mean(harmful) - mean(harmless) at the direction's layer and
  token, and g_j the same difference for neuron j's value, neuron j contributes c_j = g_j * (w_j . D/|D|),
  and its share is c_j / |D|. The shares of all neurons, attention heads and the embedding add up to 1.
  A small share for the refusal neuron would explain why holding it "on" does not undo the ablation.
The direction's layer and token come from an arditi.py .pt; D is recomputed here on the unfiltered search
prompts, and its cosine with the saved direction is reported. Prompts only; nothing is generated.
Usage: python replication/residual_share.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --arditi outputs/arditi/Qwen3-8B_jbb.pt
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, build_prompts, load


def spread(v):
    s = (v.double() / v.double().norm()).square()
    order = s.argsort(descending=True)
    return {"n": len(v), "top1_index": int(order[0]), "top1_share": float(s[order[0]]),
            "top10_share": float(s[order[:10]].sum()), "top100_share": float(s[order[:100]].sum()),
            "participation_ratio": float(1 / s.square().sum())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="the refusal neuron's layer")
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--arditi", required=True, help="arditi.py .pt (direction and its selected layer and token)")
    ap.add_argument("--harmful", default="arditi_harmful_train")
    ap.add_argument("--harmless", default="arditi_harmless_train")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="en")
    args = ap.parse_args()

    t0 = time.time()
    saved = torch.load(ROOT / args.arditi, map_location="cpu", weights_only=True)
    sel_layer, pos = saved["selected"]["layer"], saved["selected"]["pos"]
    d_saved = saved["direction"].double()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    blocks = model.model.layers
    n_mlp = max(sel_layer, args.layer + 1)  # MLP layers whose neurons are read
    store = {}

    def neuron_hook(l):
        def hook(module, a):
            store[("h", l)] = store.get(("h", l), 0) + a[0][:, pos].detach().double().sum(0).cpu()
        return hook

    def resid_hook(module, a, kw):
        h = a[0] if a else kw["hidden_states"]
        store["resid"] = store.get("resid", 0) + h[:, pos].detach().double().sum(0).cpu()

    handles = [blocks[l].mlp.down_proj.register_forward_pre_hook(neuron_hook(l)) for l in range(n_mlp)]
    handles.append(blocks[sel_layer].register_forward_pre_hook(resid_hook, with_kwargs=True))
    means = {}
    for name, split in (("harmful", args.harmful), ("harmless", args.harmless)):
        prompts, _, _ = build_prompts(tok, load(split))
        store.clear()
        for i in range(0, len(prompts), args.bs):
            batch = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**batch, logits_to_keep=1)
        means[name] = {k: v / len(prompts) for k, v in store.items()}
    for h in handles:
        h.remove()

    D = means["harmful"]["resid"] - means["harmless"]["resid"]
    unit = D / D.norm()
    cos_saved = float(torch.dot(unit, d_saved / d_saved.norm()))
    print(f"{args.model}: residual direction at layer {sel_layer}, token {pos}; |D| = {float(D.norm()):.3f}; "
          f"cosine with the saved direction {cos_saved:.3f}")

    # Reading A: the direction over its own coordinates
    a = {"saved_direction": spread(d_saved), "recomputed": spread(D)}
    s = a["saved_direction"]
    print(f"A. residual direction over its {s['n']} coordinates: largest share {s['top1_share']:.3%} (coordinate "
          f"{s['top1_index']}), top 10 {s['top10_share']:.1%}, top 100 {s['top100_share']:.1%}, "
          f"participation ratio {s['participation_ratio']:.0f}")
    g_own = means["harmful"][("h", args.layer)] - means["harmless"][("h", args.layer)]
    m = spread(g_own)
    a["mlp_direction_same_token"] = m
    print(f"   for comparison, the MLP direction at layer {args.layer} (same token) over its {m['n']} neurons: largest "
          f"share {m['top1_share']:.1%} (neuron {m['top1_index']}), top 10 {m['top10_share']:.1%}, "
          f"participation ratio {m['participation_ratio']:.0f}")

    # Reading B: each neuron's share of the residual direction
    layers, allc = [], []
    for l in range(sel_layer):
        g = means["harmful"][("h", l)] - means["harmless"][("h", l)]
        w = blocks[l].mlp.down_proj.weight
        proj = (unit.to(w.device, torch.float32) @ w.float()).double().cpu()  # w_j . unit, for every neuron j
        c = g * proj / D.norm()
        top = int(c.abs().argmax())
        layers.append({"layer": l, "mlp_share": float(c.sum()), "top_neuron": top, "top_share": float(c[top])})
        allc.append(c)
        if l == args.layer:
            rank = int((c.abs() > c[args.neuron].abs()).sum()) + 1
            own = {"share": float(c[args.neuron]), "rank_in_layer": rank, "gap": float(g[args.neuron]),
                   "write_cosine": float(proj[args.neuron] / w[:, args.neuron].float().norm().cpu())}
    b = {"layers": layers}
    if args.layer < sel_layer:
        flat = torch.cat(allc)
        grank = int((flat.abs() > abs(own["share"])).sum()) + 1
        own["rank_all_layers"] = grank
        b["neuron"] = own
        print(f"B. L{args.layer}:F{args.neuron} accounts for {own['share']:+.2%} of the residual direction "
              f"(its gap {own['gap']:+.2f} x its write vector's cosine with the direction {own['write_cosine']:+.3f}); "
              f"rank {own['rank_in_layer']} in its layer, {grank} of {len(flat)} neurons in layers 0-{sel_layer - 1}")
        order = flat.abs().argsort(descending=True)[:8]
        sizes = [len(c) for c in allc]
        tops = []
        for idx in order.tolist():
            l = 0
            while idx >= sizes[l]:
                idx -= sizes[l]
                l += 1
            tops.append({"layer": l, "neuron": idx, "share": float(allc[l][idx])})
        b["top_neurons"] = tops
        print("   largest single-neuron shares: " + ", ".join(f"L{t['layer']}:F{t['neuron']} {t['share']:+.2%}" for t in tops))
        print(f"   all MLP layers 0-{sel_layer - 1} together: {sum(x['mlp_share'] for x in layers):+.1%} of the direction; "
              f"layer {args.layer}'s MLP: {layers[args.layer]['mlp_share']:+.1%}")
    else:
        print(f"B. the direction is read at layer {sel_layer}, at or before the neuron's layer {args.layer}, so the "
              f"neuron cannot have written into it; no share.")
    out = ROOT / "outputs" / "arditi"
    path = out / f"{args.model.split('/')[-1]}_residual_share_{args.tag}.json"
    path.write_text(json.dumps({"args": vars(args), "selected": {"layer": sel_layer, "pos": pos},
                                "norm": float(D.norm()), "cosine_with_saved": cos_saved, "A": a, "B": b}, indent=1))
    print(f"saved {path.relative_to(ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
