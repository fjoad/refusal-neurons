"""Mean harmful-minus-harmless refusal direction inside the MLP, before down_proj.

Status (2026-10-27): written, not yet reviewed line by line or run. Review it before use.

Part A measures h = act_fn(gate_proj(x)) * up_proj(x) on Table 3's tokens.
The main direction uses t*, the token with the largest absolute gap for the
published neuron; a second direction averages the gaps over scored tokens.
With --all_layers, every layer uses that same t* for a comparable snapshot.
Indices are zero-based; ranks are one-based, with ties broken by neuron index.

Part B tests, at every token during prefill and greedy generation:
  none: the model's ordinary refusal rate.
  neuron_pin: the published neuron's chosen attack pin, as a reference.
  mlp_ablate: remove the activation component along the MLP direction.
  mlp_ablate_without_neuron: remove that component using the other neurons only.
  neuron_mean_ablate: reset just the neuron to its harmless training mean.
  mlp_steer: subtract alpha times the raw gap, moving away from harmful examples.
  arditi_ablate: remove Arditi's residual direction at every layer.
  arditi_ablate_neuron_held: do the same while holding the neuron at its harmful
    mean. Restored refusal would support a role for this neuron in that effect.

Statistics use unfiltered training splits, CPU float64 batch-merged moments and
pooled within-group sample variance: (M2_harmful + M2_harmless) / (n_h + n_s - 2).
Z-normalisation floors the pooled standard deviation at 1e-12 and reports how
many coordinates needed the floor. A zero direction gives a no-op projection;
its norm shares, participation ratio and cosines are undefined (JSON null).
Responses go only to outputs/evaluate JSON, never to logs.

Usage: python replication/mlp_direction.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import torch

from arditi import generate, intervention as residual_intervention, on_input_device, tokenize_batches


STD_FLOOR = 1e-12


class Moments:
    """CPU float64 mean and sum of squared deviations, shaped [scored tokens, neurons]."""

    def __init__(self, n_score, width):
        self.n = 0
        self.mean = torch.zeros(n_score, width, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, values):
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 3 or values.shape[1:] != self.mean.shape or not len(values):
            raise ValueError("Expected nonempty [batch, scored tokens, neurons] activations")
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite MLP activations; try --dtype float32")
        count = len(values)
        mean = values.mean(0)
        delta = mean - self.mean
        total = self.n + count
        # Chan's batch merge avoids subtracting two large, nearly equal raw moments.
        self.m2.add_(((values - mean) ** 2).sum(0) + delta.square() * (self.n * count / total))
        self.mean.add_(delta * (count / total))
        self.n = total


def pooled_variance(harmful, harmless):
    """Pool within-group variance; the harmful/harmless mean gap is not noise."""
    if harmful.n < 2 or harmless.n < 2 or harmful.mean.shape != harmless.mean.shape:
        raise ValueError("Both groups need at least two observations and matching shapes")
    return ((harmful.m2 + harmless.m2) / (harmful.n + harmless.n - 2)).clamp_min(0)


def normalize(direction):
    """Normalize in CPU float64; preserve an exactly zero vector as a no-op."""
    vector = direction.detach().to(device="cpu", dtype=torch.float64)
    if vector.ndim != 1 or not vector.numel() or not torch.isfinite(vector).all():
        raise ValueError("Expected a finite, nonempty direction vector")
    norm = vector.norm()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite direction norm")
    return vector / norm if norm > 0 else torch.zeros_like(vector)


def without_neuron(direction, neuron):
    vector = direction.detach().clone()
    if not 0 <= neuron < vector.numel():
        raise ValueError("Neuron is outside the direction width")
    vector[neuron] = 0
    return normalize(vector)


def decomposition(direction, neuron=None):
    """Signed top ten, squared-norm shares, participation ratio and optional watched rank."""
    unit = normalize(direction)
    values = direction.detach().to(device="cpu", dtype=torch.float64)
    order = torch.argsort(values.abs(), descending=True, stable=True)
    shares = unit.square()
    nonzero = bool(values.norm() > 0)

    def entry(index, rank):
        return {"neuron": index, "rank": rank, "value": float(values[index]),
                "squared_norm_share": float(shares[index]) if nonzero else None}

    watched = None
    if neuron is not None:
        if not 0 <= neuron < len(values):
            raise ValueError("Watched neuron is outside the direction width")
        watched = entry(neuron, int((order == neuron).nonzero()[0, 0]) + 1)
    top = [entry(index, rank) for rank, index in enumerate(order[:10].tolist(), 1)]
    return {"norm": float(values.norm()), "zero_direction": not nonzero,
            "watched": watched, "top_neuron": top[0], "top10": top,
            "participation_ratio": float(1 / shares.square().sum()) if nonzero else None,
            "top_squared_norm_share": {
                str(k): float(shares[order[:k]].sum()) if nonzero else None for k in (1, 10, 100)}}


def summarize_layer(harmful, harmless, token, neuron=None):
    gap = harmful.mean - harmless.mean
    variance = pooled_variance(harmful, harmless)
    std = variance[token].sqrt()
    direction, averaged = gap[token].clone(), gap.mean(0)
    z_direction = direction / std.clamp_min(STD_FLOOR)
    report = {"raw": decomposition(direction, neuron), "z_normalized": decomposition(z_direction, neuron),
              "averaged_raw": decomposition(averaged, neuron),
              "std_floor": STD_FLOOR, "std_floored_neurons": int((std < STD_FLOOR).sum())}
    vectors = {"direction": direction, "unit_direction": normalize(direction),
               "averaged_direction": averaged, "z_direction": z_direction,
               "mean_harmful": harmful.mean, "mean_harmless": harmless.mean,
               "m2_harmful": harmful.m2, "m2_harmless": harmless.m2,
               "pooled_variance": variance, "n_harmful": harmful.n, "n_harmless": harmless.n}
    return report, vectors


def cosine(first, second):
    if first.shape != second.shape:
        raise ValueError("Cosine vectors must have matching shapes")
    a, b = normalize(first), normalize(second)
    if not a.any() or not b.any():
        return None
    return float(torch.dot(a, b).clamp(-1, 1))


@torch.no_grad()
def write_vector(weight, direction, chunk_size=512):
    """Compute W_down @ d without copying the full weight matrix to CPU float64."""
    if weight.ndim != 2 or direction.shape != (weight.shape[1],) or chunk_size < 1:
        raise ValueError("Expected weight [d_model, d_ff] and direction [d_ff]")
    vector = direction.detach().to(device="cpu", dtype=torch.float64)
    result = torch.zeros(weight.shape[0], dtype=torch.float64)
    for start in range(0, weight.shape[1], chunk_size):
        columns = weight[:, start:start + chunk_size].detach().to(device="cpu", dtype=torch.float64)
        result.add_(columns @ vector[start:start + columns.shape[1]])
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite MLP write vector")
    return result


@torch.no_grad()
def collect_moments(model, batches, layers, n_post, n_score):
    """One forward per batch collects all requested layers; never retain full activations."""
    if not 1 <= n_score <= n_post or not batches:
        raise ValueError("Expected batches and a nonempty scored prefix of the suffix")
    moments, handles = {}, []

    def collector(stats):
        def hook(module, args):
            stats.update(args[0][:, -n_post:][:, :n_score])
        return hook

    try:
        for layer in layers:
            down = model.model.layers[layer].mlp.down_proj
            moments[layer] = Moments(n_score, down.in_features)
            handles.append(down.register_forward_pre_hook(collector(moments[layer])))
        for batch in batches:
            model(**on_input_device(model, batch), logits_to_keep=1, use_cache=False,
                  output_hidden_states=False, output_attentions=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    expected = sum(batch["input_ids"].shape[0] for batch in batches)
    if any(stats.n != expected for stats in moments.values()):
        raise RuntimeError("Each monitored MLP must be called exactly once per forward")
    return moments


def analyze(model, tok, model_name, harmful, harmless, layer, neuron, bs=4, all_layers=False,
            arditi=None):
    """Part A: return JSON-ready metrics and CPU tensors, without generation or file writes."""
    from neuron_check import build_prompts, paper_score_tokens

    if bs < 1 or not 0 <= layer < len(model.model.layers):
        raise ValueError("Invalid batch size or layer")
    if not 0 <= neuron < model.model.layers[layer].mlp.down_proj.in_features:
        raise ValueError("Neuron is outside the selected MLP width")
    for split in (harmful, harmless):
        if len(split) < 2 or any(not isinstance(s, str) or not s for s in split):
            raise ValueError("Training splits must each contain at least two nonempty instructions")
    p_harm, n_post, post_tokens = build_prompts(tok, harmful)
    p_safe, n_safe, safe_tokens = build_prompts(tok, harmless)
    if n_post < 1 or (n_post, post_tokens) != (n_safe, safe_tokens):
        raise ValueError("Training splits must share a nonempty post-instruction suffix")
    n_score = paper_score_tokens(model_name, post_tokens)
    post_ids = tok.convert_tokens_to_ids(post_tokens)
    layers = sorted(set(range(2 * len(model.model.layers) // 3)) | {layer}) if all_layers else [layer]
    harm = collect_moments(model, tokenize_batches(tok, p_harm, bs, post_ids), layers, n_post, n_score)
    safe = collect_moments(model, tokenize_batches(tok, p_safe, bs, post_ids), layers, n_post, n_score)
    gap = harm[layer].mean - safe[layer].mean
    token = int(gap[:, neuron].abs().argmax())  # first scored token wins an exact tie
    selected = {"layer": layer, "neuron": neuron, "post_token_index": token,
                "pos": token - n_post, "post_token": post_tokens[token]}
    report = {"selected": selected, "post_tokens": post_tokens, "n_score": n_score,
              "n_harmful": len(harmful), "n_harmless": len(harmless), "layers": [],
              "token_selection": "largest absolute gap of the specified neuron; first token breaks ties",
              "all_layers_token": "same t* selected at the specified layer/neuron",
              "variance": "within-group pooled sample variance (M2_h + M2_s) / (n_h + n_s - 2)",
              "rank_convention": "1-based descending absolute value; ties by neuron index",
              "neuron_per_token": [
                  {"post_token_index": j, "post_token": post_tokens[j],
                   "harmful": float(harm[layer].mean[j, neuron]),
                   "harmless": float(safe[layer].mean[j, neuron]), "gap": float(gap[j, neuron])}
                  for j in range(n_score)]}
    tensors = {"selected": selected, "post_tokens": post_tokens, "n_score": n_score, "layers": {}}
    for index in layers:
        metrics, vectors = summarize_layer(harm[index], safe[index], token, neuron if index == layer else None)
        report["layers"].append({"layer": index, "post_token_index": token, **metrics})
        tensors["layers"][index] = vectors
    main = tensors["layers"][layer]
    weight = model.model.layers[layer].mlp.down_proj.weight
    write = write_vector(weight, main["direction"])
    own_write = weight[:, neuron].detach().to(device="cpu", dtype=torch.float64)
    report["alignment"] = {"mlp_layer": layer, "cosine_neuron_write": cosine(write, own_write),
                           "arditi_selected": arditi["selected"] if arditi is not None else None,
                           "cosine_arditi": cosine(write, arditi["direction"]) if arditi is not None else None}
    tensors.update(direction=main["direction"], unit_direction=main["unit_direction"],
                   averaged_direction=main["averaged_direction"], write_vector=write,
                   unit_without_neuron=without_neuron(main["direction"], neuron))
    return report, tensors


def transform_mlp(activation, vector, mode, alpha=1.0):
    """Apply an already-unit ablation vector or a raw steering gap at every token."""
    dtype = torch.float64 if activation.dtype == torch.float64 else torch.float32
    h, v = activation.to(dtype=dtype), vector.to(device=activation.device, dtype=dtype)
    if v.shape != (h.shape[-1],):
        raise ValueError("MLP direction width differs from activations")
    if mode == "ablate":
        result = h - (h @ v).unsqueeze(-1) * v
    elif mode == "steer":
        result = h - alpha * v
    else:
        raise ValueError("Expected ablate or steer")
    return result.to(dtype=activation.dtype)


@contextmanager
def mlp_intervention(model, layer, neuron, pin=None, vector=None, mode="ablate", alpha=1.0):
    """A removable down_proj pre-hook; scalar pins reuse the replication's Pin."""
    if pin is not None and vector is not None:
        raise ValueError("Use either a pin or a direction in one MLP intervention")
    if pin is None and vector is None:
        yield
        return
    vectors = {}
    if pin is not None:
        from intervene import Pin
        hook = Pin(neuron)
        hook.values = pin
    else:
        def hook(module, args):
            h = args[0]
            dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
            key = (h.device, dtype)
            if key not in vectors:
                vectors[key] = vector.to(device=h.device, dtype=dtype)
            return (transform_mlp(h, vectors[key], mode, alpha), *args[1:])
    handle = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def evaluate(model, tok, raw, tensors, layer, neuron, pin, alphas, out_path, metadata,
             bs=4, max_new=128, arditi=None):
    """Part B: save evaluate.py-compatible JSON after each run; return its result."""
    from neuron_check import build_prompts, is_refusal

    if bs < 1 or max_new < 1 or not raw or any(not isinstance(s, str) or not s for s in raw):
        raise ValueError("Expected positive generation limits and nonempty evaluation instructions")
    prompts, _, post_tokens = build_prompts(tok, raw)
    if post_tokens != tensors["post_tokens"]:
        raise ValueError("Evaluation template suffix differs from training")
    batches = tokenize_batches(tok, prompts, bs, tok.convert_tokens_to_ids(post_tokens))
    token = tensors["selected"]["post_token_index"]
    stats = tensors["layers"][layer]
    harmless_mean = float(stats["mean_harmless"][token, neuron])
    harmful_mean = float(stats["mean_harmful"][token, neuron])
    # mode, pin, MLP vector, transform, alpha, residual ablation
    runs = [("none", None, None, "ablate", None, False),
            ("neuron_pin", pin, None, "ablate", None, False),
            ("mlp_ablate", None, tensors["unit_direction"], "ablate", None, False),
            ("mlp_ablate_without_neuron", None, tensors["unit_without_neuron"], "ablate", None, False),
            ("neuron_mean_ablate", harmless_mean, None, "ablate", None, False)]
    runs += [("mlp_steer", None, tensors["direction"], "steer", alpha, False) for alpha in alphas]
    if arditi is not None:
        runs += [("arditi_ablate", None, None, "ablate", None, True),
                 ("arditi_ablate_neuron_held", harmful_mean, None, "ablate", None, True)]
    else:
        print("Arditi file missing: skipping arditi_ablate and arditi_ablate_neuron_held.", flush=True)
    result = {**metadata, "prompts": raw, "selected": tensors["selected"], "runs": []}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for mode, value, vector, transform, alpha, residual in runs:
        run_start = time.time()
        direction = arditi["direction"] if residual else None
        # The held pin is inside down_proj; Arditi's MLP-output hook still runs
        # afterwards. Nested contexts clean up both sets of hooks on any exception.
        with residual_intervention(model, direction):
            with mlp_intervention(model, layer, neuron, value, vector, transform,
                                  alpha if alpha is not None else 1.0):
                responses = generate(model, tok, batches, max_new)
        rr = sum(map(is_refusal, responses)) / len(responses)
        params = {"post_token_index": token, "applies_to": "all tokens"}
        if alpha is not None:
            params["alpha"] = alpha
        if vector is not None:
            params["zero_direction_noop"] = not bool(vector.any())
        if mode == "neuron_mean_ablate" or mode == "arditi_ablate_neuron_held":
            params["mean_source"] = "harmful" if residual else "harmless"
        if residual:
            params.update(arditi_file=arditi["path"], arditi_selected=arditi["selected"],
                          residual_layers="all")
        run_layer = None if mode == "none" else layer
        if mode == "arditi_ablate":
            run_layer = arditi["selected"]["layer"]
        result["runs"].append({"layer": run_layer,
                               "neuron": None if mode == "arditi_ablate" else neuron,
                               "pin": value, "mode": mode, "k": None,
                               "params": params, "refusal_rate_stringmatch": rr, "responses": responses})
        result["minutes"] = (time.time() - started) / 60
        write_json(out_path, result)
        label = mode if alpha is None else f"{mode} alpha={alpha:g}"
        print(f"{label}: string-match refusal {rr:.3f} ({(time.time() - run_start) / 60:.1f} min)", flush=True)
    return result


def read_arditi(path, width, n_layers):
    if not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    direction, selected = saved.get("direction"), saved.get("selected")
    if not isinstance(direction, torch.Tensor) or direction.shape != (width,):
        raise ValueError("Arditi file lacks a direction with this model's residual width")
    if not isinstance(selected, dict) or not isinstance(selected.get("layer"), int):
        raise ValueError("Arditi file lacks selected.layer metadata")
    if not 0 <= selected["layer"] < n_layers or not normalize(direction).any():
        raise ValueError("Arditi source layer is invalid or its direction is zero")
    return {"direction": direction.to(dtype=torch.float64), "selected": selected, "path": str(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_check import ROOT, load

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True, help="m*, signed (from the rerank)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--harmful", default="arditi_harmful_train")
    ap.add_argument("--harmless", default="arditi_harmless_train")
    ap.add_argument("--all_layers", action="store_true", help="also analyze the first floor(2*n_layers/3) layers")
    ap.add_argument("--arditi", help="Arditi .pt path (default: outputs/arditi/<model short>_jbb.pt)")
    ap.add_argument("--eval_split", default="harmbench_val")
    ap.add_argument("--n", type=int, default=0, help="first n evaluation prompts (0 = all); training is unchanged")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1, 2, 4])
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--tag", default="train")
    args = ap.parse_args()
    if args.bs < 1 or args.max_new < 1 or args.n < 0 or min(args.layer, args.neuron) < 0:
        ap.error("--bs/--max_new must be positive; --n/--layer/--neuron must be nonnegative")
    if not math.isfinite(args.pin) or any(not math.isfinite(a) or a < 0 for a in args.alphas):
        ap.error("--pin must be finite; --alphas must be finite and nonnegative")
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                           for c in args.tag):
        ap.error("--tag must contain only letters, digits, underscores, dots or hyphens")

    started = time.time()
    short = args.model.rstrip("/").split("/")[-1]
    arditi_path = ROOT / (args.arditi or f"outputs/arditi/{short}_jbb.pt")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Resident weights required (multiple GPUs supported; disk offload is not)")
    arditi = read_arditi(arditi_path, model.config.hidden_size, len(model.model.layers))
    report, tensors = analyze(model, tok, args.model, load(args.harmful), load(args.harmless),
                              args.layer, args.neuron, args.bs, args.all_layers, arditi)
    out_dir = ROOT / "outputs" / "mlp_direction"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{short}_L{args.layer}_{args.tag}"
    metadata = {"args": vars(args), "model": args.model, "arditi_file": str(arditi_path),
                "arditi_available": arditi is not None}
    report.update(metadata, minutes=(time.time() - started) / 60)
    tensors.update(metadata)
    torch.save(tensors, out_dir / (stem + ".pt"))
    write_json(out_dir / (stem + ".json"), report)
    for row in report["layers"]:
        top, watched = row["raw"]["top_neuron"], row["raw"]["watched"]
        print(f"L{row['layer']}: top F{top['neuron']} rank={top['rank']} share={top['squared_norm_share']}", flush=True)
        if watched is not None:
            print(f"  F{args.neuron}: rank={watched['rank']} share={watched['squared_norm_share']}", flush=True)
    print(f"saved outputs/mlp_direction/{stem}.json / .pt ({(time.time() - started) / 60:.1f} min)", flush=True)
    if args.no_eval:
        if arditi is None:
            print(f"Arditi file missing: {arditi_path}; residual cosine unavailable.", flush=True)
        return
    raw = load(args.eval_split)
    raw = raw[:args.n] if args.n else raw
    eval_path = ROOT / "outputs" / "evaluate" / f"{short}_L{args.layer}_mlpdir_{args.tag}.json"
    evaluate(model, tok, raw, tensors, args.layer, args.neuron, args.pin, args.alphas, eval_path,
             {**metadata, "direction_file": str(out_dir / (stem + ".pt"))}, args.bs, args.max_new, arditi)


if __name__ == "__main__":
    main()
