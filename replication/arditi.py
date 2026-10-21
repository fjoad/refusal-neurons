"""Arditi refusal-direction baseline and Kazemi et al. Appendix G alignment.

Uses all post-instruction tokens, including Qwen3's empty thinking suffix, with
the repo's chat templates. Scores first-response-token log odds, not the full
refusal phrases used by neuron_search.py. Layer/neuron indices are zero-based;
--pos accepts either a suffix index (0..n_post-1) or Arditi's negative position.
--layer and --pos together recompute that training candidate and skip selection.

Reference: https://github.com/andyrdt/refusal_direction/tree/main/pipeline
Selection follows submodules/select_direction.py, including its floor-rounded
layer cutoff and epsilon-smoothed KL(baseline || ablated). Float64 CPU means
and last-position-only logits keep accelerator memory bounded. Ablation vectors
are normalized once in float64, then cast per device/dtype, rather than repeatedly
renormalized in activation precision as in utils/hook_utils.py.

Responses are saved only to outputs/evaluate; logs contain metrics, never text.
Usage: python replication/arditi.py --model Qwen/Qwen3-1.7B --watch 13:3270 --tag jbb
"""
import argparse
from contextlib import contextmanager
import json
import math
import time

import torch


REFUSAL_TOKENS = {"qwen3": [40, 2121], "llama": [40]}
TABLE7 = {
    "qwen3-1.7b": {"layer": 13, "neuron": 3270, "abs_cosine": 0.186, "expected_max": 0.096},
    "qwen3-8b": {"layer": 20, "neuron": 8719, "abs_cosine": 0.200, "expected_max": 0.070},
    "llama-3.1-8b": {"layer": 12, "neuron": 5760, "abs_cosine": 0.148, "expected_max": 0.071},
}


def model_family(name):
    for family in REFUSAL_TOKENS:
        if family in name.lower():
            return family
    raise ValueError("Model name must identify qwen3 or llama (Llama-3 family)")


def unit_direction(direction):
    vector = direction.detach().to(device="cpu", dtype=torch.float64)
    norm = vector.norm()
    if vector.ndim != 1 or not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Direction must be a finite, nonzero vector")
    return vector / norm


def project_out(activation, unit):
    """Remove the component along an already normalized direction, at every token."""
    unit = unit.to(activation)
    return activation - (activation @ unit).unsqueeze(-1) * unit


def refusal_score(logits, token_ids, epsilon=1e-8):
    """Arditi's log odds of summed refusal-token mass; logits are [batch, vocab]."""
    probs = logits.double().softmax(dim=-1)
    p = probs[:, token_ids].sum(dim=-1)
    return (p + epsilon).log() - (1 - p + epsilon).log()


def kl_score(baseline_logits, ablated_logits, epsilon=1e-6):
    """Per-example KL(baseline || ablated), with the reference's log smoothing."""
    p = baseline_logits.double().softmax(dim=-1)
    q = ablated_logits.double().softmax(dim=-1)
    return (p * ((p + epsilon).log() - (q + epsilon).log())).sum(dim=-1)


def rejection_reasons(bypass, induce, kl, layer, n_layers):
    reasons = []
    if not all(math.isfinite(x) for x in (bypass, induce, kl)):
        reasons.append("nonfinite")
    # The reference floors the cutoff, rather than comparing to a fractional layer.
    if layer >= int(n_layers * 0.8):
        reasons.append("last_20_percent")
    if kl > 0.1:
        reasons.append("kl_above_0.1")
    if induce < 0:
        reasons.append("induce_below_0")
    return reasons


def gaussian_alignment_null(c, d_model, d_ff):
    """Appendix G Eq. 8-9, under the stated Gaussian null (an upper bound, not an exact p)."""
    if d_model <= 0 or d_ff <= 0 or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError("Expected positive dimensions and an absolute cosine in [0, 1]")
    expected = math.sqrt(2 * math.log(2 * d_ff) / d_model)
    log_bound = (math.log(d_ff) + 0.5 * math.log(2 / (math.pi * d_model))
                 - math.log(c) - c * c * d_model / 2) if c else None
    return {"expected_max": expected,
            "union_bound_p": math.exp(min(0.0, log_bound)) if c else 1.0,
            "log_union_bound_unclipped": log_bound}


@torch.no_grad()
def write_cosines(weight, direction, chunk_size=1024):
    """Signed cosines of down_proj columns; transfer only a small weight chunk at a time."""
    unit = unit_direction(direction)
    if weight.ndim != 2 or weight.shape[0] != len(unit):
        raise ValueError("Expected down_proj.weight shaped [d_model, d_ff]")
    cosines = torch.zeros(weight.shape[1], dtype=torch.float64)
    zero_columns = 0
    for start in range(0, weight.shape[1], chunk_size):
        columns = weight[:, start:start + chunk_size].detach().to(device="cpu", dtype=torch.float64)
        norms = columns.norm(dim=0)
        if not torch.isfinite(columns).all():
            raise FloatingPointError("Nonfinite down_proj weights")
        zero_columns += int((norms == 0).sum())
        # Zero write vectors have no direction; assign zero alignment explicitly.
        cosines[start:start + columns.shape[1]] = ((unit @ columns) / norms.clamp_min(1e-300)).clamp(-1, 1)
    return cosines, zero_columns


def cosine_ranking(cosines, watch=None):
    order = torch.argsort(cosines.abs(), descending=True, stable=True)
    top = [{"neuron": i, "cosine": float(cosines[i]), "abs_cosine": float(cosines[i].abs())}
           for i in order[:5].tolist()]
    watched = None
    if watch is not None:
        if not 0 <= watch < len(cosines):
            raise ValueError("Watched neuron is outside down_proj width")
        watched = {"neuron": watch, "cosine": float(cosines[watch]),
                   "rank": int((order == watch).nonzero()[0, 0]) + 1}
    return top, watched


@contextmanager
def intervention(model, direction=None, layer=None):
    """All-layer ablation, or raw activation addition at one layer's input only.

    Support both positional and keyword hidden_states (HF decoder versions differ).
    Preserve attention tuple tails and remove every hook even on a failed forward.
    """
    handles, vectors = [], {}
    if direction is None:
        yield
        return
    vector = unit_direction(direction) if layer is None else direction.detach()

    def transform(h):
        key = (h.device, h.dtype)
        if key not in vectors:
            vectors[key] = vector.to(h)
        return project_out(h, vectors[key]) if layer is None else h + vectors[key]

    def pre_hook(module, args, kwargs):
        if args:
            return (transform(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": transform(kwargs["hidden_states"])}

    def output_hook(module, args, output):
        if isinstance(output, tuple):
            return (transform(output[0]), *output[1:])
        return transform(output)

    try:
        blocks = model.model.layers
        indices = range(len(blocks)) if layer is None else [layer]
        for i in indices:
            block = blocks[i]
            handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
            if layer is None:
                handles.append(block.self_attn.register_forward_hook(output_hook))
                handles.append(block.mlp.register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def tokenize_batches(tok, prompts, bs, post_ids):
    batches = []
    suffix = torch.tensor(post_ids)
    for start in range(0, len(prompts), bs):
        batch = tok(prompts[start:start + bs], return_tensors="pt", padding=True,
                    truncation=False, add_special_tokens=False)
        if (batch.input_ids.shape[1] < len(post_ids)
                or not torch.all(batch.input_ids[:, -len(post_ids):] == suffix)
                or not torch.all(batch.attention_mask[:, -len(post_ids):])):
            raise ValueError("Post-instruction suffix differs across prompts or includes padding")
        batches.append(dict(batch))
    return batches


def on_input_device(model, batch):
    device = model.get_input_embeddings().weight.device
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def last_logits(model, batch):
    # Avoid allocating [batch, prompt_length, vocab] logits; no KV cache during search.
    hidden = model.model(**on_input_device(model, batch), use_cache=False,
                         return_dict=True).last_hidden_state[:, -1, :]
    head = model.get_output_embeddings()
    return head(hidden.to(head.weight.device)).detach().cpu()


@torch.no_grad()
def score_batches(model, batches, token_ids):
    return torch.cat([refusal_score(last_logits(model, batch), token_ids) for batch in batches])


@torch.no_grad()
def mean_activations(model, batches, n_post):
    blocks = model.model.layers
    n = sum(batch["input_ids"].shape[0] for batch in batches)
    means = torch.zeros(n_post, len(blocks), model.config.hidden_size, dtype=torch.float64)
    handles = []

    def collector(layer):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            suffix = h[:, -n_post:].detach().to(device="cpu", dtype=torch.float64)
            means[:, layer].add_(suffix.sum(dim=0) / n)
        return hook

    try:
        for i, block in enumerate(blocks):
            handles.append(block.register_forward_pre_hook(collector(i), with_kwargs=True))
        for batch in batches:
            model.model(**on_input_device(model, batch), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(means).all():
        raise FloatingPointError("Nonfinite residual means; try --dtype float32")
    return means


@torch.no_grad()
def select_direction(model, candidates, harmful, harmless, token_ids):
    n_pos, n_layers, _ = candidates.shape
    baseline_logits = [last_logits(model, batch) for batch in harmless]
    baseline = {"harmful": float(score_batches(model, harmful, token_ids).mean()),
                "harmless": float(torch.cat([refusal_score(x, token_ids) for x in baseline_logits]).mean())}
    scores = []
    started = time.time()
    for pos in range(-n_pos, 0):
        for layer in range(n_layers):
            direction = candidates[pos, layer]
            record = {"pos": pos, "post_token_index": pos + n_pos, "layer": layer}
            if direction.norm() == 0:
                record.update(bypass_score=None, induce_score=None, kl_score=None,
                              rejection_reasons=["zero_direction"], passes_filter=False)
            else:
                with intervention(model, direction):
                    bypass = float(score_batches(model, harmful, token_ids).mean())
                    kl = float(torch.cat([kl_score(base, last_logits(model, batch))
                                          for base, batch in zip(baseline_logits, harmless)]).mean())
                with intervention(model, direction, layer=layer):
                    induce = float(score_batches(model, harmless, token_ids).mean())
                reasons = rejection_reasons(bypass, induce, kl, layer, n_layers)
                record.update(bypass_score=bypass if math.isfinite(bypass) else None,
                              induce_score=induce if math.isfinite(induce) else None,
                              kl_score=kl if math.isfinite(kl) else None,
                              rejection_reasons=reasons, passes_filter=not reasons)
            scores.append(record)
            if len(scores) % 10 == 0 or len(scores) == n_pos * n_layers:
                print(f"selection: {len(scores)}/{n_pos * n_layers} candidates"
                      f" ({(time.time() - started) / 60:.1f} min)", flush=True)
    valid = sorted((r for r in scores if r["passes_filter"]), key=lambda r: r["bypass_score"])
    print("Best passing candidates: pos  layer  bypass  induce  KL", flush=True)
    for r in valid[:5]:
        print(f"  {r['pos']:3d}  {r['layer']:3d}  {r['bypass_score']:+.5f}"
              f"  {r['induce_score']:+.5f}  {r['kl_score']:.6f}", flush=True)
    return valid[0] if valid else None, scores, baseline


def alignment_report(model, direction, selected_layer, watch):
    weight = model.model.layers[selected_layer].mlp.down_proj.weight
    cosines, zeros = write_cosines(weight, direction)
    watched_id = watch[1] if watch is not None and watch[0] == selected_layer else None
    top, watched = cosine_ranking(cosines, watched_id)
    report = {"layer": selected_layer, "d_model": weight.shape[0], "d_ff": weight.shape[1],
              "top5": top, "watch_in_selected_layer": watched, "zero_write_vectors": zeros,
              "rank_convention": "1-based, descending absolute cosine; ties by neuron index",
              **gaussian_alignment_null(top[0]["abs_cosine"], *weight.shape)}
    print(f"Alignment at L{selected_layer} (largest |cosine|):")
    for row in top:
        print(f"  F{row['neuron']}: cosine={row['cosine']:+.6f}")
    print(f"  E[max]={report['expected_max']:.6f}; Gaussian union-bound p <= {report['union_bound_p']:.6g}")
    if watch is not None:
        if watched is None:
            other = model.model.layers[watch[0]].mlp.down_proj.weight
            other_cosines, _ = write_cosines(other, direction)
            _, watched = cosine_ranking(other_cosines, watch[1])
        report["watch_own_layer"] = {"layer": watch[0], **watched,
                                     "direction_source_layer": selected_layer}
        print(f"  watch L{watch[0]}:F{watch[1]} against selected direction:"
              f" cosine={watched['cosine']:+.6f}, own-layer rank={watched['rank']}")
    return report


@torch.no_grad()
def generate(model, tok, batches, max_new):
    responses = []
    for batch in batches:
        inputs = on_input_device(model, batch)
        ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1,
                             use_cache=True, pad_token_id=tok.pad_token_id,
                             return_dict_in_generate=False, output_scores=False,
                             output_logits=False, logits_to_keep=1)
        responses.extend(tok.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return responses


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_check import ROOT, build_prompts, is_refusal, load

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--watch", metavar="L:F", help="report this neuron's rank by absolute cosine")
    ap.add_argument("--layer", type=int, help="with --pos, recompute this candidate and skip selection")
    ap.add_argument("--pos", type=int, help="suffix index or negative position (-1 = final prompt token)")
    ap.add_argument("--eval_split", default="jailbreakbench")
    ap.add_argument("--n", type=int, default=0, help="first n evaluation prompts (0 = all); does not limit train/val")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--no_eval", action="store_true", help="skip response generation")
    ap.add_argument("--tag", default="jbb")
    args = ap.parse_args()
    if args.bs < 1 or args.n < 0 or args.max_new < 1:
        ap.error("--bs/--max_new must be positive and --n nonnegative")
    if (args.layer is None) != (args.pos is None):
        ap.error("--layer and --pos must be supplied together")
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                           for c in args.tag):
        ap.error("--tag must contain only letters, digits, underscores, dots or hyphens")
    try:
        family = model_family(args.model)
        watch = tuple(map(int, args.watch.split(":"))) if args.watch else None
        if watch is not None and (len(watch) != 2 or min(watch) < 0):
            raise ValueError("--watch expects nonnegative L:F")
    except ValueError as error:
        ap.error(str(error))

    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    token_ids = REFUSAL_TOKENS[family]
    # Check that a differently named local tokenizer has not silently changed these IDs.
    expected = ["I", "As"] if family == "qwen3" else ["I"]
    if [tok.decode([i]) for i in token_ids] != expected:
        raise ValueError("Tokenizer refusal-token IDs differ from Arditi's model-family IDs")

    formatted, post_tokens = {}, None
    for name, size in (("harmful_train", 128), ("harmless_train", 128),
                       ("harmful_val", 32), ("harmless_val", 32)):
        raw = load("arditi_" + name)
        if not isinstance(raw, list) or len(raw) != size or any(not isinstance(x, str) or not x for x in raw):
            raise ValueError(f"arditi_{name}: expected {size} nonempty instructions")
        prompts, n_post, tokens = build_prompts(tok, raw)
        if n_post < 1 or (post_tokens is not None and tokens != post_tokens):
            raise ValueError("Train/val splits must share a nonempty template suffix")
        formatted[name], post_tokens = prompts, tokens
    n_post = len(post_tokens)
    post_ids = tok.convert_tokens_to_ids(post_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Resident weights required (multiple GPUs supported; disk offload is not)")
    layers = model.model.layers
    if args.layer is not None and not (0 <= args.layer < len(layers) and -n_post <= args.pos < n_post):
        ap.error("--layer or --pos is outside the model/template range")
    if watch is not None and (watch[0] >= len(layers) or watch[1] >= layers[watch[0]].mlp.down_proj.in_features):
        ap.error("--watch is outside the model's layer/neuron range")

    batches, filtering = {}, {}
    for name, prompts in formatted.items():
        original_batches = tokenize_batches(tok, prompts, args.bs, post_ids)
        scores = score_batches(model, original_batches, token_ids)
        if not torch.isfinite(scores).all():
            raise FloatingPointError(f"Nonfinite refusal scores for {name}")
        keep = scores > 0 if name.startswith("harmful_") else scores < 0
        indices = keep.nonzero().flatten().tolist()
        filtering[name] = {"before": len(prompts), "after": len(indices),
                           "kept_indices": indices, "refusal_scores": scores.tolist()}
        print(f"{name}: kept {len(indices)}/{len(prompts)}", flush=True)
        if not indices:
            raise ValueError(f"Filtering emptied {name}; cannot estimate or select a direction")
        batches[name] = tokenize_batches(tok, [prompts[i] for i in indices], args.bs, post_ids)

    candidates = (mean_activations(model, batches["harmful_train"], n_post)
                  - mean_activations(model, batches["harmless_train"], n_post))
    print(f"candidate directions: {list(candidates.shape)} ({(time.time() - started) / 60:.1f} min)", flush=True)
    if args.layer is None:
        selected, scores, baseline = select_direction(
            model, candidates, batches["harmful_val"], batches["harmless_val"], token_ids)
    else:
        pos = args.pos if args.pos < 0 else args.pos - n_post
        selected = {"pos": pos, "post_token_index": pos + n_post, "layer": args.layer}
        scores, baseline = [], None

    out_dir = ROOT / "outputs" / "arditi"
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.rstrip("/").split("/")[-1]
    stem = f"{short}_{args.tag}"
    metadata = {"args": vars(args), "model": args.model, "family": family,
                "refusal_token_ids": token_ids, "post_tokens": post_tokens, "post_token_ids": post_ids,
                "candidate_shape": list(candidates.shape), "filtering": filtering,
                "selection": {"skipped": args.layer is not None, "objective": "minimize bypass_score",
                              "kl_threshold": 0.1, "induce_threshold": 0.0,
                              "prune_from_layer": int(len(layers) * 0.8),
                              "kl_convention": "KL(baseline || ablated), log epsilon=1e-6"},
                "baseline_val_refusal_scores": baseline, "candidates": scores, "selected": selected}
    tensors = {"candidate_directions": candidates, "post_tokens": post_tokens,
               "selected": selected, "candidate_scores": scores}
    if selected is None:
        torch.save(tensors, out_dir / (stem + ".pt"))
        write_json(out_dir / (stem + ".json"), metadata)
        raise RuntimeError("All candidates failed Arditi's filters; candidate scores saved, no direction selected")
    direction = candidates[selected["pos"], selected["layer"]].clone()
    tensors.update(direction=direction, unit_direction=unit_direction(direction))
    torch.save(tensors, out_dir / (stem + ".pt"))
    print(f"selected: pos={selected['pos']}, layer={selected['layer']}", flush=True)
    metadata["alignment"] = alignment_report(model, direction, selected["layer"], watch)
    target = next((value for key, value in TABLE7.items() if key in args.model.lower()), None)
    metadata["table7_target"] = target
    if target is not None:
        print(f"Table 7 target: L{target['layer']}:F{target['neuron']},"
              f" |cosine|={target['abs_cosine']:.3f}, E[max]={target['expected_max']:.3f}")
    metadata["minutes"] = (time.time() - started) / 60
    write_json(out_dir / (stem + ".json"), metadata)
    print(f"saved outputs/arditi/{stem}.pt / .json", flush=True)
    if args.no_eval:
        return

    raw = load(args.eval_split)
    raw = raw[:args.n] if args.n else raw
    if not raw or any(not isinstance(x, str) or not x for x in raw):
        raise ValueError("Evaluation split must contain nonempty instructions")
    prompts, _, tokens = build_prompts(tok, raw)
    if tokens != post_tokens:
        raise ValueError("Evaluation template suffix differs from training")
    eval_batches = tokenize_batches(tok, prompts, args.bs, post_ids)
    eval_dir = ROOT / "outputs" / "evaluate"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{short}_arditi_{args.tag}.json"
    result = {"args": vars(args), "model": args.model, "prompts": raw, "runs": [],
              "direction_file": str((out_dir / (stem + ".pt")).relative_to(ROOT)), "selected": selected}
    for mode in ("none", "arditi"):
        run_start = time.time()
        with intervention(model, direction if mode == "arditi" else None):
            responses = generate(model, tok, eval_batches, args.max_new)
        rr = sum(map(is_refusal, responses)) / len(responses)
        result["runs"].append({"layer": selected["layer"] if mode == "arditi" else None,
                               "neuron": None, "pin": None, "mode": mode, "k": None,
                               "refusal_rate_stringmatch": rr, "responses": responses})
        result["minutes"] = (time.time() - started) / 60
        write_json(eval_path, result)
        print(f"{mode}: string-match refusal {rr:.3f} ({(time.time() - run_start) / 60:.1f} min)", flush=True)
    print(f"saved outputs/evaluate/{eval_path.name} ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
