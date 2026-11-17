"""The MLP refusal direction removed at every layer, and the reverse route run blind (17 November meeting).

The 5 November MLP slide removed the MLP direction in one layer only (the refusal neuron's), while Nanda's
(Arditi's) ablation acts at every layer. This script runs the every-layer versions on harmful prompts:

  none, neuron_pin                    references
  mlp_ablate_one_layer                the 5 November run: the direction of the neuron's layer, removed there only
  one_direction_every_mlp             that same list of numbers removed inside every layer's MLP
  own_direction_every_mlp             each layer's own MLP direction (token t*), removed inside its own MLP
  own_direction_every_mlp_avg_token   the same with each layer's direction averaged over Table 3's tokens
                                      (uses nothing about the refusal neuron)
  nanda_mlp_outputs_only              Nanda's residual direction removed only from what the MLPs write
  nanda_attn_outputs_only             ... only from what the attention layers write
  nanda_every_layer                   Nanda's full ablation (block inputs, attention and MLP outputs)

An MLP direction is the harmful-minus-harmless mean of a layer's neuron values (before down_proj) on the search
prompts. t* is the Table 3 token where the refusal neuron's gap is largest, as in mlp_direction.py. Removing a
direction sets the activation's component along it to zero at every token, during the prompt and generation.

Reverse route, blind (no generation): every neuron of every layer is ranked without using the refusal neuron to
choose the layer or the token (directions averaged over Table 3's tokens), by two measures: its share of its own
layer's direction, and |gap| x the length of its write vector (how large a harmful-harmless difference it writes
into the residual stream, comparable across layers). Prints the top 10 and the refusal neuron's rank.

Responses go only to outputs/evaluate; logs show string-match refusal rates and the rankings.
Usage: python replication/mlp_every_layer.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20
"""
import argparse
from contextlib import contextmanager
import time

import torch

from arditi import generate, intervention as residual_intervention, project_out, tokenize_batches, unit_direction
from mlp_direction import collect_moments, mlp_intervention, normalize, read_arditi, transform_mlp, write_json


@contextmanager
def every_mlp(model, units):
    """Remove units[layer] (a unit vector over that layer's neurons) inside each listed MLP, at every token."""
    handles = []

    def make(unit):
        def hook(module, args):
            return (transform_mlp(args[0], unit, "ablate"), *args[1:])
        return hook

    try:
        for layer, unit in units.items():
            down = model.model.layers[layer].mlp.down_proj
            handles.append(down.register_forward_pre_hook(make(unit.to(device=down.weight.device, dtype=torch.float32))))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def residual_part(model, direction, part):
    """Nanda's direction removed only from the MLP outputs (part='mlp') or only from the attention outputs ('attn')."""
    unit, handles = unit_direction(direction), []

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (project_out(output[0], unit), *output[1:])
        return project_out(output, unit)

    try:
        for block in model.model.layers:
            handles.append((block.mlp if part == "mlp" else block.self_attn).register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def blind_ranking(model, gaps, layer, neuron):
    """Rank every neuron of every layer without using the refusal neuron to choose the layer or token."""
    n_layers, search_layers = len(gaps), 2 * len(gaps) // 3
    avg = torch.stack([gaps[l].mean(0) for l in range(n_layers)])  # [layers, neurons]
    share = avg.square() / avg.square().sum(1, keepdim=True)
    norms = torch.stack([model.model.layers[l].mlp.down_proj.weight.float().norm(dim=0).double().cpu()
                         for l in range(n_layers)])
    written = avg.abs() * norms
    report = {}
    for name, score in (("share of its own layer's direction", share), ("|gap| x write-vector length", written)):
        for scope, limit in (("all layers", n_layers), ("first two thirds of layers", search_layers)):
            flat = score[:limit].flatten()
            order = torch.argsort(flat, descending=True, stable=True)
            width = score.shape[1]
            top = [(int(i) // width, int(i) % width, float(flat[i])) for i in order[:10]]
            rank = int((order == layer * width + neuron).nonzero()[0, 0]) + 1 if layer < limit else None
            print(f"blind ranking by {name}, {scope}: refusal neuron L{layer}:F{neuron} is rank {rank} of {len(flat)}")
            print("  top 10: " + ", ".join(f"L{l}:F{n} {v:.3g}" for l, n, v in top), flush=True)
            report[f"{name}; {scope}"] = {"refusal_neuron_rank": rank,
                                          "top10": [{"layer": l, "neuron": n, "score": v} for l, n, v in top]}
    per_layer = []
    for l in range(n_layers):
        n = int(share[l].argmax())
        per_layer.append({"layer": l, "top_neuron": n, "share": float(share[l, n]),
                          "direction_norm": float(avg[l].norm())})
    print("top neuron of each layer's direction (averaged over tokens): "
          + ", ".join(f"L{p['layer']}:F{p['top_neuron']} {p['share']:.0%}" for p in per_layer), flush=True)
    report["per_layer"] = per_layer
    return report


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_check import ROOT, build_prompts, is_refusal, load, paper_score_tokens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True)
    ap.add_argument("--harmful", default="arditi_harmful_train")
    ap.add_argument("--harmless", default="arditi_harmless_train")
    ap.add_argument("--eval_split", default="harmbench_val")
    ap.add_argument("--arditi", help="Arditi .pt path (default: outputs/arditi/<model short>_jbb.pt)")
    ap.add_argument("--n", type=int, default=0, help="first n evaluation prompts (0 = all)")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="en")
    args = ap.parse_args()

    t0 = time.time()
    short = args.model.rstrip("/").split("/")[-1]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    arditi = read_arditi(ROOT / (args.arditi or f"outputs/arditi/{short}_jbb.pt"), model.config.hidden_size, n_layers)

    # ---- each layer's MLP direction, from the search prompts
    p_harm, n_post, post_tokens = build_prompts(tok, load(args.harmful))
    p_safe = build_prompts(tok, load(args.harmless))[0]
    n_score = paper_score_tokens(args.model, post_tokens)
    post_ids = tok.convert_tokens_to_ids(post_tokens)
    layers = list(range(n_layers))
    harm = collect_moments(model, tokenize_batches(tok, p_harm, args.bs, post_ids), layers, n_post, n_score)
    safe = collect_moments(model, tokenize_batches(tok, p_safe, args.bs, post_ids), layers, n_post, n_score)
    gaps = {l: harm[l].mean - safe[l].mean for l in layers}  # [scored tokens, neurons]
    token = int(gaps[args.layer][:, args.neuron].abs().argmax())
    print(f"{args.model}: {n_layers} layers, {gaps[0].shape[1]} neurons per MLP; t* = token {token} "
          f"('{post_tokens[token]}') of {n_score} scored ({(time.time() - t0) / 60:.1f} min)", flush=True)
    blind = blind_ranking(model, gaps, args.layer, args.neuron)

    one = normalize(gaps[args.layer][token])
    own = {l: normalize(gaps[l][token]) for l in layers}
    own_avg = {l: normalize(gaps[l].mean(0)) for l in layers}
    r = arditi["direction"] if arditi is not None else None

    # ---- attack runs
    raw = load(args.eval_split)
    raw = raw[:args.n] if args.n else raw
    batches = tokenize_batches(tok, build_prompts(tok, raw)[0], args.bs, post_ids)
    runs = [("none", lambda: mlp_intervention(model, args.layer, args.neuron), None),
            ("neuron_pin", lambda: mlp_intervention(model, args.layer, args.neuron, pin=args.pin), args.pin),
            ("mlp_ablate_one_layer", lambda: every_mlp(model, {args.layer: one}), None),
            ("one_direction_every_mlp", lambda: every_mlp(model, {l: one for l in layers}), None),
            ("own_direction_every_mlp", lambda: every_mlp(model, own), None),
            ("own_direction_every_mlp_avg_token", lambda: every_mlp(model, own_avg), None)]
    if r is not None:
        runs += [("nanda_mlp_outputs_only", lambda: residual_part(model, r, "mlp"), None),
                 ("nanda_attn_outputs_only", lambda: residual_part(model, r, "attn"), None),
                 ("nanda_every_layer", lambda: residual_intervention(model, r), None)]
    else:
        print("Arditi file missing: skipping the three Nanda runs.", flush=True)
    out = ROOT / "outputs" / "evaluate" / f"{short}_L{args.layer}_mlpall_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "model": args.model, "prompts": raw, "t_star": token, "blind_ranking": blind,
              "runs": []}
    for mode, ctx, shown in runs:
        with ctx():
            texts = generate(model, tok, batches, args.max_new)
        rr = sum(map(is_refusal, texts)) / len(texts)
        result["runs"].append({"layer": args.layer, "neuron": args.neuron, "pin": shown, "mode": mode,
                               "refusal_rate_stringmatch": rr, "responses": texts})
        write_json(out, result)
        print(f"{mode}: string-match refusal {rr:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"saved {out.relative_to(ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
