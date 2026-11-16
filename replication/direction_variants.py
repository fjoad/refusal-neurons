"""The refusal neuron against the refusal direction: variants asked for on 12 November (points 14-17).

With r = Arditi's residual-stream direction (raw difference of means, from an arditi.py .pt), v = the refusal
neuron's write vector (its column of down_proj), and the search file's top-5 candidates:

Attack runs on harmful prompts (HarmBench, 128 tokens; judge afterwards):
  none, neuron_pin, arditi_ablate            the three references
  ablate_v                                   point 17: use v as the direction, removed at every layer like r
  ablate_r_without_v                         point 16: r with its part along v taken out, removed at every layer
  top5_off                                   point 15: all five search candidates held at their harmless means
  pin_plus_direction x1, x2                  point 14a: the neuron pinned (model jailbroken), then r added back at
                                             r's layer (Arditi's activation addition): does refusal return?
  apple_and_next_off                         point 14c: the neuron and the neurons that contribute most to the
                                             direction once it is off, all held at their harmless means
Direction statistics, no generation (point 14b): the difference of means at r's layer and token recomputed with
the neuron off (harmless mean) and with it pinned: cosine with the original, length ratio, and with the neuron off
the neurons that then contribute most to it (as in residual_share.py).
Responses go only to outputs/evaluate; logs show string-match refusal rates and statistics.
Usage: python replication/direction_variants.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20 \
           --search outputs/neuron_search/Qwen3-8B_float16_arditi.json --arditi outputs/arditi/Qwen3-8B_jbb.pt
"""
import argparse
import contextlib
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arditi import intervention
from neuron_check import ROOT, build_prompts, generate, is_refusal, load


class MultiPin:
    """Pre-hooks on several layers' down_proj: holds chosen neurons at chosen values (empty = off)."""

    def __init__(self, model):
        self.values, self.records = {}, []  # {layer: {neuron: value}}; records only for generate()'s interface
        for i, block in enumerate(model.model.layers):
            block.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, layer):
        def hook(module, args):
            pins = self.values.get(layer)
            if not pins:
                return None
            h = args[0].clone()
            for n, val in pins.items():
                h[..., n] = val
            return (h,)
        return hook


@torch.no_grad()
def diff_of_means(model, tok, sets, sel_layer, pos, bs, neuron_layers=()):
    """Mean harmful-minus-harmless residual at (sel_layer input, token pos), and per-neuron gaps for neuron_layers."""
    blocks, store = model.model.layers, {}

    def resid(module, a, kw):
        h = a[0] if a else kw["hidden_states"]
        store["resid"] = store.get("resid", 0) + h[:, pos].detach().double().sum(0).cpu()

    def neuron(l):
        def hook(module, a):
            store[l] = store.get(l, 0) + a[0][:, pos].detach().double().sum(0).cpu()
        return hook

    # registered after MultiPin's hooks, so they see pinned values
    handles = [blocks[sel_layer].register_forward_pre_hook(resid, with_kwargs=True)]
    handles += [blocks[l].mlp.down_proj.register_forward_pre_hook(neuron(l)) for l in neuron_layers]
    means = []
    for prompts in sets:
        store.clear()
        for i in range(0, len(prompts), bs):
            batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**batch, logits_to_keep=1)
        means.append({k: val / len(prompts) for k, val in store.items()})
    for h in handles:
        h.remove()
    return {k: means[0][k] - means[1][k] for k in means[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--arditi", required=True)
    ap.add_argument("--eval_split", default="harmbench_val")
    ap.add_argument("--harmful", default="arditi_harmful_train")
    ap.add_argument("--harmless", default="arditi_harmless_train")
    ap.add_argument("--n_next", type=int, default=5, help="how many next-largest contributors to switch off with the neuron")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="en")
    args = ap.parse_args()

    t0 = time.time()
    search = json.loads((ROOT / args.search).read_text())
    cands = search["candidates"] + [w for w in search.get("watched", []) if w.get("mean_act_harmless") is not None]
    harmless_mean = {(c["layer"], c["neuron"]): c["mean_act_harmless"] for c in cands}
    top5 = [(c["layer"], c["neuron"]) for c in search["candidates"]]
    saved = torch.load(ROOT / args.arditi, map_location="cpu", weights_only=True)
    r_raw, sel_layer, pos = saved["direction"].double(), saved["selected"]["layer"], saved["selected"]["pos"]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    pin = MultiPin(model)
    short = args.model.split("/")[-1]

    v = model.model.layers[args.layer].mlp.down_proj.weight[:, args.neuron].detach().double().cpu()
    v_hat, r_hat = v / v.norm(), r_raw / r_raw.norm()
    cos = float(torch.dot(v_hat, r_hat))
    r_without_v = r_hat - cos * v_hat
    print(f"{args.model} L{args.layer}:F{args.neuron}; direction at layer {sel_layer}, token {pos}; cosine(v, r) = {cos:+.3f}; "
          f"the part of r along v is {cos ** 2:.1%} of its squared length")

    # ---- point 14b: the direction recomputed with the neuron off / pinned
    train = [build_prompts(tok, load(args.harmful))[0], build_prompts(tok, load(args.harmless))[0]]
    a_safe = harmless_mean[(args.layer, args.neuron)]
    stats = {"cosine_v_r": cos}
    d0 = diff_of_means(model, tok, train, sel_layer, pos, args.bs)["resid"]
    for name, values in (("neuron off (held at its harmless mean)", {args.layer: {args.neuron: a_safe}}),
                         ("neuron pinned at the attack value", {args.layer: {args.neuron: args.pin}})):
        pin.values = values
        layers = range(sel_layer) if "off" in name else ()
        d = diff_of_means(model, tok, train, sel_layer, pos, args.bs, layers)
        pin.values = {}
        dr = d["resid"]
        c = float(torch.dot(dr / dr.norm(), d0 / d0.norm()))
        print(f"direction recomputed with the {name}: cosine with the original {c:+.3f}, length {float(dr.norm() / d0.norm()):.2f}x")
        stats[name] = {"cosine_with_original": c, "length_ratio": float(dr.norm() / d0.norm())}
        if layers:
            unit = dr / dr.norm()
            best = []
            for l in layers:
                w = model.model.layers[l].mlp.down_proj.weight
                share = d[l] * (unit.to(w.device, torch.float32) @ w.float()).double().cpu() / dr.norm()
                for n in share.abs().topk(args.n_next + 1).indices.tolist():
                    best.append((abs(float(share[n])), float(share[n]), l, n))
            best = [b for b in sorted(best, reverse=True) if (b[2], b[3]) != (args.layer, args.neuron)][:args.n_next]
            nxt = [(l, n) for _, _, l, n in best]
            print("  with it off, the neurons contributing most to the direction: " + ", ".join(f"L{l}:F{n} {s:+.2%}" for _, s, l, n in best))
            stats["next_neurons"] = [{"layer": l, "neuron": n, "share": s} for _, s, l, n in best]

    # harmless means of the next neurons, for switching them off
    pin.values = {}
    safe_prompts = train[1]
    acc = {}

    def grab(l):
        def hook(module, a):
            acc[l] = acc.get(l, 0) + a[0][:, pos].detach().double().sum(0).cpu()
        return hook
    handles = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(grab(l)) for l in {l for l, _ in nxt}]
    with torch.no_grad():
        for i in range(0, len(safe_prompts), args.bs):
            batch = tok(safe_prompts[i:i + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**batch, logits_to_keep=1)
    for h in handles:
        h.remove()
    next_off = {}
    for l, n in nxt:
        next_off.setdefault(l, {})[n] = float(acc[l][n] / len(safe_prompts))
    next_off.setdefault(args.layer, {})[args.neuron] = a_safe
    top5_off = {}
    for l, n in top5:
        top5_off.setdefault(l, {})[n] = harmless_mean[(l, n)]

    # ---- attack runs
    raw = load(args.eval_split)
    prompts = build_prompts(tok, raw)[0]
    apple = {args.layer: {args.neuron: args.pin}}
    add_layer = lambda scale: intervention(model, (r_raw * scale).float(), layer=sel_layer)
    runs = [  # mode, neuron pins, context manager factory, pin value shown
        ("none", {}, contextlib.nullcontext, None),
        ("neuron_pin", apple, contextlib.nullcontext, args.pin),
        ("arditi_ablate", {}, lambda: intervention(model, r_raw.float()), None),
        ("ablate_v", {}, lambda: intervention(model, v.float()), None),
        ("ablate_r_without_v", {}, lambda: intervention(model, r_without_v.float()), None),
        ("top5_off", top5_off, contextlib.nullcontext, None),
        ("apple_and_next_off", next_off, contextlib.nullcontext, None),
        ("pin_plus_direction x1", apple, lambda: add_layer(1.0), args.pin),
        ("pin_plus_direction x2", apple, lambda: add_layer(2.0), args.pin),
    ]
    out = ROOT / "outputs" / "evaluate" / f"{short}_L{args.layer}_dirvar_{args.tag}.json"
    result = {"args": vars(args), "model": args.model, "prompts": raw, "stats": stats, "top5": top5, "runs": []}
    for mode, pins, ctx, shown in runs:
        pin.values = pins
        with ctx():
            texts = generate(model, tok, pin, prompts, args.bs, args.max_new)
        pin.values = {}
        rr = sum(map(is_refusal, texts)) / len(texts)
        result["runs"].append({"layer": args.layer, "neuron": args.neuron, "pin": shown, "mode": mode,
                               "refusal_rate_stringmatch": rr, "responses": texts})
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{mode}: string-match refusal {rr:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"saved {out.relative_to(ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
