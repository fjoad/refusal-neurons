"""Milestone 2: localize refusal features using the two training splits.

Hooks observe h = SiLU(W_gate x) * W_up x before down_proj in the first
floor(2 * n_layers / 3) layers. With --space residual, observe decoder-layer inputs
(resid_pre) instead; F / neuron then indexes a residual-stream dimension.
All layer/feature/token indices are zero-based. Residual output names add _residual.
Refusal probability is the mass of the paper's Table 4 refusal phrases for the model
family (or --refusal_phrases), with duplicate/overlapping token prefixes removed
before summing phrase probabilities.

Sign convention: the README requests L = -log(p / (1-p)) and
score = (mean_grad_harmful + mean_grad_harmless) * (mean_act_harmful - mean_act_harmless).
Refusal features have NEGATIVE scores under this convention: smaller is better,
including when selecting the best token. This is equivalent to maximizing the
paper's Eq. 4, which uses the opposite activation gap (harmless - harmful).

Usage: python replication/neuron_search.py --model Qwen/Qwen3-1.7B --watch 13:3270
       python replication/neuron_search.py --model meta-llama/Llama-3.1-8B-Instruct --space residual --watch 12:291
This script only ranks candidates; it does not intervene or run the ASR sweep.
"""
import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, build_prompts, load, paper_score_tokens


def refusal_prefixes(tok, phrases):
    """Return a prefix-free token set, so the probability mass is a valid union."""
    encoded = {tuple(tok(p, add_special_tokens=False).input_ids) for p in phrases}
    if not encoded or () in encoded:
        raise ValueError("Refusal phrases must contain at least one token each")
    prefixes = []
    for ids in sorted(encoded, key=lambda x: (len(x), x)):
        if not any(ids[:len(prefix)] == prefix for prefix in prefixes):
            prefixes.append(ids)
    return prefixes


class SearchHook:
    """Pre-down_proj or resid_pre hook with CPU sums of shape [post_tokens, features].

    A tensor backward hook collects dL/dh without retaining a full h.grad tensor.
    Only the first otherwise-untracked h is made a leaf; subsequent layers stay
    connected, preserving all indirect gradient paths through later layers.
    """

    def __init__(self, n_post, width, grad_scale):
        self.n_post, self.grad_scale = n_post, grad_scale
        self.act_sum = torch.zeros(n_post, width, dtype=torch.float32)
        self.grad_sum = torch.zeros_like(self.act_sum)
        self.prompt_len = None
        self.record_acts = False
        self.backward_calls = 0

    def reset(self):
        self.act_sum.zero_()
        self.grad_sum.zero_()
        self.backward_calls = 0

    def __call__(self, module, args, kwargs=None):
        h = args[0] if args else kwargs["hidden_states"]  # [batch, tokens, features]
        start, stop = self.prompt_len - self.n_post, self.prompt_len
        if self.record_acts:
            self.act_sum.add_(h[:, start:stop].detach().float().sum(0).cpu())
        if torch.is_grad_enabled():
            if not h.requires_grad:
                h.requires_grad_(True)

            def collect(grad):
                # Capture this forward's slice, not mutable state from a later batch.
                value = grad[:, start:stop].detach().float().sum(0)
                self.grad_sum.add_((value / self.grad_scale).cpu())
                self.backward_calls += 1

            h.register_hook(collect)


def phrase_log_probability(model, batch, phrase):
    """Log P(complete phrase | prompt), one value per prompt, with teacher forcing.

    Append all but the last target token. The logits at prompt_len - 1 predict
    the first target, so prompt-suffix hook positions never include phrase tokens.
    Apply lm_head only at prediction positions to avoid [batch, prompt_len, vocab]
    logits. Qwen3 and Llama use the same model.model / output-embedding interface.
    """
    input_ids, mask = batch.input_ids, batch.attention_mask
    prompt_len = input_ids.shape[1]
    target = torch.tensor(phrase, device=input_ids.device, dtype=torch.long)
    if len(phrase) > 1:
        continuation = target[:-1].unsqueeze(0).expand(input_ids.shape[0], -1)
        input_ids = torch.cat((input_ids, continuation), dim=1)
        mask = torch.cat((mask, torch.ones_like(continuation)), dim=1)
    hidden = model.model(input_ids=input_ids, attention_mask=mask,
                         use_cache=False, return_dict=True).last_hidden_state
    logits = model.get_output_embeddings()(hidden[:, prompt_len - 1:]).float()
    targets = target.to(logits.device).view(1, -1, 1).expand(logits.shape[0], -1, -1)
    return logits.log_softmax(-1).gather(-1, targets).squeeze(-1).sum(-1)


def split_means(model, tok, hooks, prompts, post_ids, prefixes, bs, label):
    """Accumulate means without storing examples' activations or gradient graphs.

    First evaluate each phrase without a graph to get p = sum_j P_j. Then
    recompute one phrase at a time and backpropagate -w_j * log(P_j), where
    w_j = (P_j / p) / (1-p) is detached. The summed gradients are exactly
    d[-log(p/(1-p))]/dh, including shared-prefix probability normalization.
    Two passes keep only one phrase's backward graph alive at a time.
    """
    for hook in hooks:
        hook.reset()
    start_time = time.time()
    expected_backwards = 0
    for offset in range(0, len(prompts), bs):
        batch = tok(prompts[offset:offset + bs], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
        prompt_len = batch.input_ids.shape[1]
        suffix = torch.tensor(post_ids, device=batch.input_ids.device)
        if prompt_len < len(post_ids) or not torch.all(batch.input_ids[:, -len(post_ids):] == suffix):
            raise ValueError("Chat-template suffix tokenization differs across prompts")
        if not torch.all(batch.attention_mask[:, -len(post_ids):]):
            raise ValueError("Post-instruction positions must not include padding")
        for hook in hooks:
            hook.prompt_len = prompt_len

        # CPU float64 makes logsumexp / 1-p stable even when mass is near one.
        log_probs = []
        with torch.no_grad():
            for j, phrase in enumerate(prefixes):
                for hook in hooks:
                    hook.record_acts = j == 0  # prompt activations counted once
                log_probs.append(phrase_log_probability(model, batch, phrase).double().cpu())
        for hook in hooks:
            hook.record_acts = False
        log_probs = torch.stack(log_probs)  # [phrases, batch]
        log_p = torch.logsumexp(log_probs, dim=0)
        if not torch.isfinite(log_probs).all() or not torch.all(log_p < 0):
            raise FloatingPointError("Invalid refusal probability; try --dtype float32")
        weights = (log_probs - log_p).exp() / (-torch.expm1(log_p))
        for phrase, weight in zip(prefixes, weights):
            log_prob = phrase_log_probability(model, batch, phrase)
            # Sum, not mean: each example contributes once, including short batches.
            # Fixed loss scaling protects small fp16 activation gradients. Hooks
            # undo this scale before accumulating signed float32 gradients.
            loss = -(log_prob * weight.to(device=log_prob.device, dtype=log_prob.dtype)).sum()
            (loss * hooks[0].grad_scale).backward()
            del log_prob, loss  # never retain one graph into the next phrase
            expected_backwards += 1
        for hook in hooks:
            if hook.backward_calls != expected_backwards:
                raise RuntimeError("A monitored layer did not receive exactly one gradient per phrase")
            if not torch.isfinite(hook.act_sum).all() or not torch.isfinite(hook.grad_sum).all():
                raise FloatingPointError(
                    "Non-finite activations/gradients; try --dtype float32 or a smaller --grad_scale")
        done = min(offset + bs, len(prompts))
        if offset == 0 or done % (10 * bs) == 0 or done == len(prompts):
            print(f"{label}: {done}/{len(prompts)} prompts ({(time.time() - start_time) / 60:.1f} min)",
                  flush=True)
    # Independent CPU means survive reset() before the next split.
    return ([hook.act_sum / len(prompts) for hook in hooks],
            [hook.grad_sum / len(prompts) for hook in hooks])


def top_candidates(harmful, harmless, post_tokens, k=5):
    """Choose the best signed-score token, filter at that token, then rank globally."""
    act_harm, grad_harm = harmful
    act_safe, grad_safe = harmless
    candidates = []
    for layer, (ah, gh, ash, gs) in enumerate(zip(act_harm, grad_harm, act_safe, grad_safe)):
        combined_grad = gh + gs
        scores = combined_grad * (ah - ash)  # [post_tokens, neurons]
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite neuron score; try --dtype float32")
        # The requested sign is the negative of the paper's score: argmin == best.
        # Apply the magnitude filter AFTER selecting the token, as in Section 2.3.
        best_score, best_token = scores.min(dim=0)
        neurons = torch.arange(scores.shape[1])
        valid = ah[best_token, neurons].abs() > ash[best_token, neurons].abs()
        valid_neurons = neurons[valid]
        order = torch.argsort(best_score[valid], stable=True)[:k]
        for neuron in valid_neurons[order].tolist():
            token = int(best_token[neuron])
            candidates.append({
                "layer": layer, "neuron": neuron, "score": float(best_score[neuron]),
                "post_token_index": token, "post_token": post_tokens[token],
                "mean_act_harmful": float(ah[token, neuron]),
                "mean_act_harmless": float(ash[token, neuron]),
                "mean_grad_harmful": float(gh[token, neuron]),
                "mean_grad_harmless": float(gs[token, neuron]),
                "combined_grad": float(combined_grad[token, neuron]),
            })
    return sorted(candidates, key=lambda x: (x["score"], x["layer"], x["neuron"]))[:k]


def watch_ranks(harmful, harmless, post_tokens, watch):
    """Global rank (1 = best) of each watched (layer, neuron) under the same score and filter."""
    act_harm, grad_harm = harmful
    act_safe, grad_safe = harmless
    best, valid = [], []
    for ah, gh, ash, gs in zip(act_harm, grad_harm, act_safe, grad_safe):
        score, token = ((gh + gs) * (ah - ash)).min(dim=0)
        neurons = torch.arange(score.shape[0])
        best.append((score, token))
        valid.append(ah[token, neurons].abs() > ash[token, neurons].abs())
    pool = torch.cat([s[v] for (s, _), v in zip(best, valid)])
    out = []
    for layer, neuron in watch:
        if layer >= len(best):
            out.append({"layer": layer, "neuron": neuron, "searched": False})
            continue
        score, token = best[layer][0][neuron], int(best[layer][1][neuron])
        out.append({
            "layer": layer, "neuron": neuron, "searched": True, "score": float(score),
            "rank": int((pool < score).sum()) + 1, "passes_filter": bool(valid[layer][neuron]),
            "post_token_index": token, "post_token": post_tokens[token],
            "mean_act_harmful": float(act_harm[layer][token, neuron]),
            "mean_act_harmless": float(act_safe[layer][token, neuron]),
            "combined_grad": float(grad_harm[layer][token, neuron] + grad_safe[layer][token, neuron]),
        })
    return out


# Table 4 of Kazemi et al. (2026): the log-odds targets per model family.
PAPER_PHRASES = {
    "qwen3": ["I'm sorry, but I can't help with that request.", "I'm unable to assist"],
    "llama": ["I can't help with that.", "I'm unable to assist"],
}


def paper_phrases(model):
    for family, phrases in PAPER_PHRASES.items():
        if family in model.lower():
            return phrases
    raise ValueError(f"No Table 4 phrases for {model}; pass --refusal_phrases")



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--space", choices=["mlp", "residual"], default="mlp",
                    help="rank MLP neurons (default) or decoder-input residual dimensions")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bs", type=int, default=1, help="prompt batch size; backward uses more memory than inference")
    ap.add_argument("--harmful", default="arditi_harmful_train", help="split name in data/splits/ (default: Arditi's exact sample)")
    ap.add_argument("--harmless", default="arditi_harmless_train", help="split name in data/splits/")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. ar")
    ap.add_argument("--grad_scale", type=float, default=128.0, help="fixed backward scale, undone in gradient hooks")
    ap.add_argument("--refusal_phrases", nargs="+", default=None,
                    help="response-prefix targets; defaults to the paper's Table 4 phrases for the model family")
    ap.add_argument("--score_tokens", type=int, default=None,
                    help="score only the first N post-instruction tokens (default: Table 3 for the family; 0 = all)")
    ap.add_argument("--watch", nargs="*", default=[], metavar="L:F",
                    help="also report layer:feature ranks; F is a dimension with --space residual")
    args = ap.parse_args()
    if args.bs < 1 or not math.isfinite(args.grad_scale) or args.grad_scale <= 0:
        ap.error("--bs must be positive and --grad_scale must be finite and positive")
    if args.refusal_phrases is None:
        args.refusal_phrases = paper_phrases(args.model)
    try:
        watch = [tuple(int(x) for x in w.split(":")) for w in args.watch]
    except ValueError:
        ap.error("--watch expects layer:feature pairs, e.g. 14:7924")

    t0 = time.time()
    harmful, harmless = load(args.harmful), load(args.harmless)
    for name, split in ((args.harmful, harmful), (args.harmless, harmless)):
        if not isinstance(split, list) or not split or any(not isinstance(s, str) or not s for s in split):
            raise ValueError(f"{name}: expected a nonempty list of nonempty instructions")
    # Search only locally cached models: running this script cannot fetch weights.
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    p_harm, n_post, post_tokens = build_prompts(tok, harmful)
    p_safe, n_safe, safe_tokens = build_prompts(tok, harmless)
    if n_post < 1 or (n_post, post_tokens) != (n_safe, safe_tokens):
        raise ValueError("Both splits must have the same nonempty post-instruction template suffix")
    post_ids = tok.convert_tokens_to_ids(post_tokens)
    if args.score_tokens is None:
        args.score_tokens = paper_score_tokens(args.model, post_tokens)
    n_score = args.score_tokens or len(post_tokens)
    prefixes = refusal_prefixes(tok, args.refusal_phrases)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    model.requires_grad_(False)  # no weight gradients or optimizer state
    model.config.use_cache = False
    model.gradient_checkpointing_disable()  # hooks must run exactly once per forward
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Search needs resident model weights; CPU/disk offload is not supported")
    layers = model.model.layers
    n_layers = 2 * len(layers) // 3
    if n_layers < 1:
        raise ValueError("Model has too few layers to search its first two-thirds")
    hooks, handles = [], []
    try:
        for layer in layers[:n_layers]:
            width = model.config.hidden_size if args.space == "residual" else layer.mlp.down_proj.in_features
            hook = SearchHook(n_post, width, args.grad_scale)
            hooks.append(hook)
            if args.space == "residual":
                # Transformers 4.51.3 passes hidden_states positionally; accept keywords too.
                handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
            else:
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        print(f"model: {args.model}; layers: 0..{n_layers - 1}; train: {len(harmful)} harmful / {len(harmless)} harmless")
        print(f"post-instruction tokens: {post_tokens}")
        print(f"refusal prefixes: {len(prefixes)} (full phrase probabilities, two passes per batch)")
        print("score = (grad_harmful + grad_harmless) * (act_harmful - act_harmless); smaller is better")
        harm_means = split_means(model, tok, hooks, p_harm, post_ids, prefixes, args.bs, "harmful")
        safe_means = split_means(model, tok, hooks, p_safe, post_ids, prefixes, args.bs, "harmless")
    finally:
        for handle in handles:
            handle.remove()
    # Means cover the whole suffix; score only the leading Table 3 tokens.
    head = lambda split: ([t[:n_score] for t in split[0]], [t[:n_score] for t in split[1]])
    print(f"scored tokens: {post_tokens[:n_score]}")
    candidates = top_candidates(head(harm_means), head(safe_means), post_tokens[:n_score])
    print(f"Top {len(candidates)} candidates for {args.model} (zero-based; most negative score first):")
    for rank, candidate in enumerate(candidates, 1):
        print(f"  {rank}. (layer={candidate['layer']}, neuron={candidate['neuron']}, score={candidate['score']:+.8g})"
              f"  L{candidate['layer']}:F{candidate['neuron']}"
              f"  token[{candidate['post_token_index']}]={candidate['post_token']!r}")
    watched = watch_ranks(head(harm_means), head(safe_means), post_tokens[:n_score], watch)
    for w in watched:
        if not w["searched"]:
            print(f"  watch L{w['layer']}:F{w['neuron']}: layer not searched")
            continue
        print(f"  watch L{w['layer']}:F{w['neuron']}: rank {w['rank']}, score={w['score']:+.6g},"
              f" filter={'pass' if w['passes_filter'] else 'FAIL'}, token[{w['post_token_index']}]={w['post_token']!r},"
              f" act harmful/harmless={w['mean_act_harmful']:+.3f}/{w['mean_act_harmless']:+.3f}")
    out = ROOT / "outputs" / "neuron_search"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    space = "_residual" if args.space == "residual" else ""
    name = f"{args.model.rstrip('/').split('/')[-1]}_{args.dtype}{tag}{space}.json"
    # Per-layer mean activations and gradients, so other scores or neurons can be checked without rerunning.
    torch.save({"harmful": harm_means, "harmless": safe_means, "post_tokens": post_tokens},
               out / name.replace(".json", "_means.pt"))
    (out / name).write_text(json.dumps({
        "args": vars(args), "n_layers_searched": n_layers,
        "n_harmful": len(harmful), "n_harmless": len(harmless),
        "post_tokens": post_tokens, "refusal_prefix_ids": prefixes,
        "refusal_prefixes": [tok.decode(ids) for ids in prefixes],
        "loss": "-log(p_refusal / (1 - p_refusal))",
        "score": "(mean_grad_harmful + mean_grad_harmless) * (mean_act_harmful - mean_act_harmless)",
        "ranking": "ascending; select minimum-score token, then apply magnitude filter",
        "candidates": candidates, "watched": watched, "minutes": (time.time() - t0) / 60,
    }, indent=1, ensure_ascii=False, allow_nan=False))
    print(f"saved outputs/neuron_search/{name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
